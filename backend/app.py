import logging
import os
import subprocess
import traceback

import requests
from flask import (
    Flask,
    jsonify,
    request,
    send_from_directory,
)

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend import config
from backend.retrieval_system import VideoRetrievalSystem
from utils.query_processing import decompose_query
from utils.video_metadata import load_video_metadata

log_file = "system.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] - %(message)s",
    handlers=[
        logging.FileHandler(log_file, encoding='utf-8'),
        logging.StreamHandler()
    ],
)
logger = logging.getLogger(__name__)

# Fix console encoding for Windows
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'replace')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'replace')

# Flask app với template/static paths từ thư mục gốc
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRONTEND_DIST = os.path.join(ROOT_DIR, "frontend", "dist")
app = Flask(__name__, 
            template_folder=os.path.join(ROOT_DIR, 'templates'),
            static_folder=os.path.join(ROOT_DIR, 'static'))

VIDEO_METADATA = load_video_metadata(config.VIDEOS_DIR)
runtime_evaluation_config = {
    "sessionId": config.SESSION_ID,
    "evaluationId": config.EVALUATION_ID,
    "evalServerUrl": config.EVAL_SERVER_URL.rstrip("/"),
}

try:
    import psutil
    mem = psutil.virtual_memory()
    logger.info(f"Available memory: {mem.available / (1024**3):.2f} GB / {mem.total / (1024**3):.2f} GB")
    if mem.available < 2 * (1024**3):  # Less than 2GB available
        logger.warning("⚠️ Low memory detected! System may crash during initialization.")
        logger.warning("   Consider closing other applications or increasing virtual memory.")
except ImportError:
    logger.warning("psutil not installed - cannot check memory status")

try:
    search_system = VideoRetrievalSystem(re_ingest=False)
    logger.info("Search system initialized successfully!")
except MemoryError as e:
    logger.error(f"❌ MEMORY ERROR during initialization: {e}")
    logger.error("   SOLUTION: Increase Windows virtual memory (page file)")
    logger.error("   See: https://www.windowscentral.com/how-change-virtual-memory-size-windows-10")
    logger.error("   Recommended: Set page file to 8-16 GB on your SSD")
    search_system = None
except Exception as e:
    logger.error(f"Failed to initialize search system: {e}")
    logger.error(traceback.format_exc())
    search_system = None


@app.route("/")
def home():
    index_path = os.path.join(FRONTEND_DIST, "index.html")
    if os.path.exists(index_path):
        return send_from_directory(FRONTEND_DIST, "index.html")
    return jsonify({
        "message": "Frontend build not found. Run `cd frontend && npm install && npm run build`, or use Vite dev server on port 5173."
    }), 200


@app.route("/assets/<path:filename>")
def frontend_assets(filename):
    return send_from_directory(os.path.join(FRONTEND_DIST, "assets"), filename)


@app.route("/api/health")
def health_api():
    return jsonify({
        "ok": search_system is not None,
        "search_system": search_system is not None,
        "milvus": {
            "host": config.MILVUS_HOST,
            "port": config.MILVUS_PORT,
            "collection": config.KEYFRAME_COLLECTION_NAME,
            "vector_dimension": config.VECTOR_DIMENSION,
            "text_collection": config.TEXT_COLLECTION_NAME,
            "text_vector_dimension": config.TEXT_VECTOR_DIMENSION,
        },
        "elasticsearch": {
            "url": config.ELASTICSEARCH_URL,
            "index": config.TRANSCRIPT_INDEX,
        },
        "models": {
            "visual_provider": config.VISUAL_MODEL_PROVIDER,
            "visual_model": config.VISUAL_MODEL_NAME,
            "visual_truncate_dim": config.VISUAL_TRUNCATE_DIM,
            "dense_text_enabled": config.ENABLE_DENSE_TEXT_RETRIEVAL,
            "text_provider": config.TEXT_MODEL_PROVIDER,
            "text_model": config.TEXT_MODEL_NAME,
            "query_translation_enabled": config.ENABLE_QUERY_TRANSLATION,
            "query_translation_provider": config.QUERY_TRANSLATION_PROVIDER,
            "query_translation_model": config.QUERY_TRANSLATION_MODEL,
            "query_translation_direction": f"{config.QUERY_TRANSLATION_SRC_LANG}->{config.QUERY_TRANSLATION_TGT_LANG}",
            "ocr_engine": config.OCR_ENGINE,
            "ocr_languages": config.OCR_LANGUAGES,
            "asr_model": config.ASR_MODEL,
            "asr_language": config.ASR_LANGUAGE,
            "rerank_provider": config.RERANK_MODEL_PROVIDER,
            "rerank_model": config.RERANK_MODEL_NAME,
        },
    })


@app.route("/search", methods=["POST"])
def search_api():
    if not search_system:
        return jsonify({"error": "Search system is not available."}), 500

    query_data = request.get_json()
    if not query_data:
        return jsonify({"error": "Invalid input: No JSON data received."}), 400

    logger.info(f"Received search request: {query_data}")

    try:
        if query_data.get("fusion") == "intersection":
            description = query_data.get("description", "")
            result_sets = []
            if description:
                result_sets.append(
                    search_system.clip_search(
                        description,
                        max_results=config.VISUAL_MAX_RESULTS,
                    )
                )

            transcript_text = query_data.get("transcript") or query_data.get("audio")
            if transcript_text:
                result_sets.append(search_system.transcript_search(transcript_text))

        # Giao các tập kết quả
            results = search_system.intersect(result_sets)
        else:
            if query_data.get("rerank_top_k") is not None:
                try:
                    query_data["rerank_top_k"] = int(query_data["rerank_top_k"])
                except (TypeError, ValueError):
                    query_data.pop("rerank_top_k", None)
            results = search_system.hybrid_search(query_data)

        for item in results:
            vid = item.get("video_id")
            # Lấy FPS từ Cache RAM, mặc định 25 nếu không tìm thấy
            item["fps"] = item.get("fps") or VIDEO_METADATA.get(vid, config.DEFAULT_FALLBACK_FPS)

        logger.info(f"Search completed. Number of results: {len(results)}")
        return jsonify(results)
    except Exception as e:
        logger.error(f"An error occurred during search: {e}", exc_info=True)
        return jsonify({"error": "An internal error occurred during search."}), 500


def _candidate_reason(item: dict) -> str:
    sources = item.get("sources") or [item.get("source_type") or item.get("doc_type")]
    source_text = ", ".join(str(source) for source in sources if source)
    evidence = item.get("ocr_text") or item.get("transcript_text") or item.get("caption_text") or item.get("text")
    if evidence:
        return f"Matched by {source_text}: {str(evidence)[:180]}"
    return f"Matched by {source_text or 'retrieval signals'}."


@app.route("/api/agent/solve", methods=["POST"])
def agent_solve_api():
    """Assistant-style retrieval endpoint for automatic-mode experiments."""

    if not search_system:
        return jsonify({"error": "Search system is not available."}), 500

    data = request.get_json(silent=True) or {}
    query = data.get("query") or data.get("description") or ""
    if not str(query).strip():
        return jsonify({"error": "Missing query"}), 400

    auto_submit = bool(data.get("auto_submit", False))
    if auto_submit:
        return jsonify({
            "error": "auto_submit is disabled by default for safety. Review candidates manually first."
        }), 400

    top_k = data.get("top_k", 10)
    try:
        top_k = max(1, min(50, int(top_k)))
    except (TypeError, ValueError):
        top_k = 10

    plan = decompose_query(str(query))
    payload = {
        "description": plan.visual_query,
        "ocr": data.get("ocr") or plan.ocr_query,
        "transcript": data.get("transcript") or plan.transcript_query,
        "negative": data.get("negative") or plan.negative_query,
        "neighbor_seconds": data.get("neighbor_seconds") or [-5, -3, 0, 3, 5],
        "rerank_top_k": data.get("rerank_top_k", config.RERANK_TOP_K),
    }
    results = search_system.hybrid_search(payload)[:top_k]
    candidates = []
    for item in results:
        candidates.append({
            "video_id": item.get("video_id"),
            "keyframe_index": item.get("keyframe_index"),
            "frame_number": item.get("frame_number"),
            "start_seconds": item.get("start_seconds", item.get("start")),
            "time_ms": item.get("time_ms"),
            "fps": item.get("fps"),
            "score": item.get("rerank_score", item.get("fusion_score")),
            "sources": item.get("sources", []),
            "source_scores": item.get("source_scores", {}),
            "neighbors": item.get("neighbors", []),
            "reason": _candidate_reason(item),
        })

    return jsonify({
        "query_plan": {
            "visual_query": plan.visual_query,
            "ocr_query": plan.ocr_query,
            "transcript_query": plan.transcript_query,
            "negative_query": plan.negative_query,
        },
        "candidates": candidates,
    })


@app.route("/keyframes/<string:video_id>/keyframe_<int:keyframe_index>.png")
def serve_frame_image(video_id, keyframe_index):
    try:
        keyframe_dir = os.path.join(config.KEYFRAMES_DIR, video_id)
        filename = f"keyframe_{keyframe_index}.png"
        return send_from_directory(keyframe_dir, filename)
    except FileNotFoundError:
        return "Keyframe not found", 404


@app.route("/keyframes/<string:video_id>/keyframe_<int:keyframe_index>.webp")
def serve_legacy_frame_image(video_id, keyframe_index):
    try:
        keyframe_dir = os.path.join(config.KEYFRAMES_DIR, video_id)
        png_filename = f"keyframe_{keyframe_index}.png"
        if os.path.exists(os.path.join(keyframe_dir, png_filename)):
            return send_from_directory(keyframe_dir, png_filename)
        return send_from_directory(keyframe_dir, f"keyframe_{keyframe_index}.webp")
    except FileNotFoundError:
        return "Keyframe not found", 404


@app.route("/frontend/<path:path>")
def frontend_spa(path):
    target_path = os.path.join(FRONTEND_DIST, path)
    if os.path.exists(target_path) and os.path.isfile(target_path):
        return send_from_directory(FRONTEND_DIST, path)
    return send_from_directory(FRONTEND_DIST, "index.html")


@app.route("/videos/<path:video_id>")
def serve_video_file(video_id):
    try:
        filename = f"{video_id}.mp4"
        return send_from_directory(config.VIDEOS_DIR, filename, as_attachment=False)
    except FileNotFoundError:
        return "Video not found", 404


HLS_DIR = os.path.join(os.getcwd(), "data", "hls")


@app.route("/hls/<string:video_id>/<path:filename>")
def serve_hls(video_id, filename):
    """
    API phục vụ file playlist (.m3u8) và segment (.ts)
    """
    try:
        video_hls_path = os.path.join(HLS_DIR, video_id)
        response = send_from_directory(video_hls_path, filename)

        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"

        return response
    except FileNotFoundError:
        return "File not found", 404


@app.route("/api/login", methods=["POST"])
def login_proxy():
    """
    Trả về Session ID và Evaluation ID từ config
    Không cần call server evaluation nữa
    """
    try:
        # Lấy trực tiếp từ config
        session_id = runtime_evaluation_config.get("sessionId") or config.SESSION_ID
        evaluation_id = runtime_evaluation_config.get("evaluationId") or config.EVALUATION_ID
        
        logger.info(f"[LOGIN] Returning session ID: {session_id}")
        logger.info(f"[LOGIN] Returning evaluation ID: {evaluation_id}")

        return jsonify({
            "message": "Connected",
            "sessionId": session_id,
            "evaluationId": evaluation_id,
            "evalServerUrl": runtime_evaluation_config.get("evalServerUrl") or config.EVAL_SERVER_URL,
        })

    except Exception as e:
        logger.error(f"Login error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/evaluation-config", methods=["GET", "POST"])
def evaluation_config_api():
    """Read/update evaluation credentials without restarting backend."""

    if request.method == "GET":
        return jsonify(runtime_evaluation_config)

    data = request.get_json(silent=True) or {}
    session_id = str(data.get("sessionId") or "").strip()
    evaluation_id = str(data.get("evaluationId") or "").strip()
    eval_server_url = str(data.get("evalServerUrl") or "").strip()

    if session_id:
        runtime_evaluation_config["sessionId"] = session_id
    if evaluation_id:
        runtime_evaluation_config["evaluationId"] = evaluation_id
    if eval_server_url:
        runtime_evaluation_config["evalServerUrl"] = eval_server_url.rstrip("/")

    logger.info(
        "[EVAL_CONFIG] Updated runtime evaluation config: evaluationId=%s, server=%s",
        runtime_evaluation_config.get("evaluationId"),
        runtime_evaluation_config.get("evalServerUrl"),
    )

    return jsonify({
        "message": "Evaluation config updated",
        **runtime_evaluation_config,
    })


@app.route("/api/submit", methods=["POST"])
def submit_proxy():
    """
    Gửi kết quả submit
    """
    try:
        data = request.get_json()
        logger.info(f"[SUBMIT] Received data: {data}")
        
        session_id = data.get("sessionId") or runtime_evaluation_config.get("sessionId")
        evaluation_id = data.get("evaluationId") or runtime_evaluation_config.get("evaluationId")
        video_id = data.get("videoId")
        time_ms = data.get("timeMs")  # Thời gian tính bằng milliseconds

        logger.info(f"[SUBMIT] Parsed fields - sessionId: {session_id}, evaluationId: {evaluation_id}, videoId: {video_id}, timeMs: {time_ms}")

        if not all([session_id, evaluation_id, video_id, time_ms is not None]):
            missing = []
            if not session_id: missing.append("sessionId")
            if not evaluation_id: missing.append("evaluationId")
            if not video_id: missing.append("videoId")
            if time_ms is None: missing.append("timeMs")
            logger.error(f"[SUBMIT] Missing required fields: {missing}")
            return jsonify({"error": f"Missing required fields: {', '.join(missing)}"}), 400

        eval_server_url = (
            data.get("evalServerUrl")
            or runtime_evaluation_config.get("evalServerUrl")
            or config.EVAL_SERVER_URL
        ).rstrip("/")
        submit_url = f"{eval_server_url}/api/v2/submit/{evaluation_id}"

        payload = {
            "answerSets": [
                {
                    "answers": [
                        {
                            "mediaItemName": video_id,
                            "start": str(int(time_ms)),  # timeMs đã là milliseconds
                            "end": str(int(time_ms)),
                        }
                    ]
                }
            ]
        }

        logger.info(f"[SUBMIT] Sending payload to {submit_url}: {payload}")
        logger.info(f"[SUBMIT] With session param: {session_id}")

        # Gửi request lên server đánh giá
        try:
            response = requests.post(
                submit_url,
                json=payload,
                params={"session": session_id},
                timeout=(4, 8),  # connect timeout 4s, read timeout 8s to avoid hanging
            )

            if response.status_code == 200:
                logger.info(f"[SUBMIT] Success! Server response: {response.json()}")
                return jsonify({"success": True, "remote_response": response.json()})
            else:
                logger.error(f"[SUBMIT] Server returned {response.status_code}: {response.text}")
                return (
                    jsonify({"success": False, "error": response.text}),
                    response.status_code,
                )
        except requests.exceptions.Timeout:
            logger.error(f"[SUBMIT] Timeout connecting to {eval_server_url}")
            return jsonify({
                "error": f"Evaluation server timeout. Cannot submit to {eval_server_url}"
            }), 503
        except requests.exceptions.ConnectionError as e:
            logger.error(f"[SUBMIT] Connection error: {e}")
            return jsonify({
                "error": f"Cannot connect to evaluation server at {eval_server_url}. Please check network or server status."
            }), 503

    except Exception as e:
        logger.error(f"Submit proxy error: {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
