import logging
import os
import subprocess
import traceback

import requests
from flask import (
    Flask,
    Response,
    jsonify,
    render_template,
    request,
    send_from_directory,
)

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend import config
from backend.retrieval_system import VideoRetrievalSystem
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
app = Flask(__name__, 
            template_folder=os.path.join(ROOT_DIR, 'templates'),
            static_folder=os.path.join(ROOT_DIR, 'static'))

VIDEO_METADATA = load_video_metadata(config.VIDEOS_DIR)

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
    return render_template("index.html")


@app.route("/test")
def test_page():
    return render_template("test_search.html")


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


@app.route("/keyframes/<string:video_id>/keyframe_<int:keyframe_index>.webp")
def serve_frame_image(video_id, keyframe_index):
    try:
        keyframe_dir = os.path.join(config.KEYFRAMES_DIR, video_id)
        filename = f"keyframe_{keyframe_index}.webp"
        return send_from_directory(keyframe_dir, filename)
    except FileNotFoundError:
        return send_from_directory("static", "placeholder.png"), 404


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
        session_id = config.SESSION_ID
        evaluation_id = config.EVALUATION_ID
        
        logger.info(f"[LOGIN] Returning session ID: {session_id}")
        logger.info(f"[LOGIN] Returning evaluation ID: {evaluation_id}")

        return jsonify({
            "message": "Connected",
            "sessionId": session_id,
            "evaluationId": evaluation_id
        })

    except Exception as e:
        logger.error(f"Login error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/submit", methods=["POST"])
def submit_proxy():
    """
    Gửi kết quả submit
    """
    try:
        data = request.get_json()
        logger.info(f"[SUBMIT] Received data: {data}")
        
        session_id = data.get("sessionId")
        evaluation_id = data.get("evaluationId")
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

        submit_url = f"{config.EVAL_SERVER_URL}/api/v2/submit/{evaluation_id}"

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
            logger.error(f"[SUBMIT] Timeout connecting to {config.EVAL_SERVER_URL}")
            return jsonify({
                "error": f"Evaluation server timeout. Cannot submit to {config.EVAL_SERVER_URL}"
            }), 503
        except requests.exceptions.ConnectionError as e:
            logger.error(f"[SUBMIT] Connection error: {e}")
            return jsonify({
                "error": f"Cannot connect to evaluation server at {config.EVAL_SERVER_URL}. Please check network or server status."
            }), 503

    except Exception as e:
        logger.error(f"Submit proxy error: {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
