import argparse
import logging
import os
import sys
import json
import cv2  # Import thư viện OpenCV để đọc metadata video
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import torch
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
from pymilvus import Collection, CollectionSchema, DataType, FieldSchema, connections, utility

from backend import config
from utils.elasticsearch_client import (
    get_elasticsearch_client,
    recreate_transcript_index,
)

BULK_CHUNK_SIZE = 2000
logger = logging.getLogger(__name__)

# --- Helper Functions ---

def get_video_fps(video_id: str) -> float:
    """
    Lấy FPS thực tế của video dựa trên ID.
    Nếu không tìm thấy hoặc lỗi, trả về giá trị mặc định 25.0.
    """
    video_path = os.path.join(config.VIDEOS_DIR, f"{video_id}.mp4")
    if os.path.exists(video_path):
        try:
            cap = cv2.VideoCapture(video_path)
            if cap.isOpened():
                fps = cap.get(cv2.CAP_PROP_FPS)
                cap.release()
                if fps and fps > 0:
                    return float(fps)
        except Exception as e:
            logger.warning(f"Error reading FPS for {video_id}: {e}")
            
    return 25.0  # Fallback default value

def _load_keyframe_map(video_id: str):
    maps_dir = Path(config.KEYFRAMES_DIR) / "maps"
    map_file = maps_dir / f"{video_id}_map.csv"
    if not map_file.exists():
        return None

    try:
        df = pd.read_csv(map_file, usecols=["FrameID", "Seconds"])
        df = df.dropna(subset=["FrameID", "Seconds"]).sort_values("Seconds")
        if df.empty:
            return None
        frame_ids = df["FrameID"].to_numpy(dtype=np.int32)
        seconds = df["Seconds"].to_numpy(dtype=np.float32)
        return seconds, frame_ids
    except Exception as exc:
        logger.warning(f"Failed to load keyframe map for {video_id}: {exc}")
        return None


def _resolve_frames_from_map(mapping, target_seconds: np.ndarray):
    if mapping is None or target_seconds.size == 0:
        return None, None

    seconds, frame_ids = mapping
    if seconds.size == 0:
        return None, None

    positions = np.searchsorted(seconds, target_seconds, side="left")
    right_idx = np.clip(positions, 0, len(seconds) - 1)
    left_idx = np.clip(positions - 1, 0, len(seconds) - 1)

    diff_left = np.abs(target_seconds - seconds[left_idx])
    diff_right = np.abs(target_seconds - seconds[right_idx])
    best_idx = np.where(diff_left <= diff_right, left_idx, right_idx)

    return frame_ids[best_idx].astype(int), seconds[best_idx].astype(float)


# --- Ingestion Functions ---

def setup_milvus_collection(collection_name, schema, index_field, index_params, skip_if_exists=True):
    if utility.has_collection(collection_name):
        collection = Collection(collection_name)
        collection.load()
        count = collection.num_entities
        existing_dim = _collection_vector_dim(collection, index_field)
        if existing_dim is not None and existing_dim != config.VECTOR_DIMENSION:
            raise ValueError(
                f"Milvus collection '{collection_name}' vector dim is {existing_dim}, "
                f"but config.VECTOR_DIMENSION is {config.VECTOR_DIMENSION}. "
                "Drop/recreate the collection or restore the matching model config."
            )
        
        if skip_if_exists and count > 0:
            logger.info(f"Collection '{collection_name}' already exists with {count} entities. Skipping recreation.")
            return collection
        else:
            logger.warning(f"Collection '{collection_name}' already exists. Dropping.")
            utility.drop_collection(collection_name)
    
    collection = Collection(collection_name, schema)
    logger.info(f"Collection '{collection_name}' created.")
    
    logger.info(f"Creating index for field '{index_field}'...")
    collection.create_index(field_name=index_field, index_params=index_params)
    collection.flush()
    logger.info("Index created and data flushed.")
    return collection


def _collection_vector_dim(collection: Collection, vector_field: str) -> int | None:
    try:
        for field in collection.schema.fields:
            if field.name == vector_field:
                dim = field.params.get("dim") if hasattr(field, "params") else None
                return int(dim) if dim is not None else None
    except Exception as exc:
        logger.warning("Could not inspect Milvus collection schema: %s", exc)
    return None

def ingest_keyframe_data(collection: Collection, skip_if_has_data=True):
    # Check if collection already has data
    if skip_if_has_data:
        count = collection.num_entities
        if count > 0:
            logger.info(f"Collection already has {count} keyframes. Skipping ingestion.")
            return
    
    logger.info("Ingesting keyframe data into Milvus...")
    root = Path(config.CLIP_FEATURES_DIR)
    
    # Kiểm tra thư mục embedding có tồn tại không
    if not root.exists():
        logger.error(f"Embeddings directory not found: {root}")
        return

    for video_path in list(root.iterdir()):
        if not video_path.is_dir():
            continue
            
        video_id = video_path.name
        vectors = []
        frame_indices = []
        
        for pt_file in list(video_path.glob("*.pt")):
            try:
                frame_idx = int(pt_file.stem.split("_")[-1])
                vec = torch.load(str(pt_file), map_location="cpu").numpy().astype(np.float32)
                vec = vec.reshape(1, -1)
                if vec.shape[-1] != config.VECTOR_DIMENSION:
                    raise ValueError(
                        f"Embedding dimension mismatch in {pt_file}: got {vec.shape[-1]}, "
                        f"expected {config.VECTOR_DIMENSION}."
                    )
                vectors.append(vec)
                frame_indices.append(frame_idx)
            except Exception as e:
                logger.error(f"Error processing {pt_file}: {e}")
                continue

        if vectors:
            vectors = np.vstack(vectors)
            if vectors.shape[-1] != config.VECTOR_DIMENSION:
                raise ValueError(
                    f"Stacked embeddings for {video_id} have dim {vectors.shape[-1]}, "
                    f"expected {config.VECTOR_DIMENSION}."
                )
            num_vectors = len(vectors)
            entities = [[video_id] * num_vectors, frame_indices, vectors]
            collection.insert(entities)
            
    collection.flush()
    logger.info("Keyframe data ingestion complete.")

def ingest_transcript_data(es_client: Elasticsearch, folder_path: str) -> None:
    logger.info("Ingesting transcript data into Elasticsearch...")

    transcripts_dir = Path(folder_path)
    if not transcripts_dir.exists():
        logger.error(f"Transcript directory not found: {transcripts_dir}")
        return

    # Hỗ trợ cả CSV (legacy) và JSON (Whisper format)
    csv_files = sorted(transcripts_dir.glob("*.csv"))
    json_files = sorted(transcripts_dir.glob("*.json"))
    
    if not csv_files and not json_files:
        logger.warning("No transcript files (CSV or JSON) found.")
        return

    # Biến để đếm tổng số docs đã ingest
    total_docs = 0
    map_cache: dict[str, tuple[np.ndarray, np.ndarray] | None] = {}

    # --- Process JSON files (Whisper format) ---
    for json_path in json_files:
        video_id = json_path.stem
        fps = get_video_fps(video_id)
        
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as exc:
            logger.error(f"Failed to read {json_path}: {exc}")
            continue
        
        if "segments" not in data:
            logger.warning(f"JSON file {json_path} missing 'segments' field; skipping")
            continue
        
        segments = data["segments"]
        if not segments:
            continue
        
        # Load keyframe map for this video
        if video_id not in map_cache:
            map_cache[video_id] = _load_keyframe_map(video_id)
        frame_map = map_cache[video_id]
        
        actions = []
        for seg in segments:
            start_sec = float(seg.get("start", 0.0))
            end_sec = float(seg.get("end", start_sec))
            text = seg.get("text", "").strip()
            seg_id = seg.get("id", 0)
            
            if not text:
                continue
            
            # Calculate frame index
            start_frame = max(0, int(round(start_sec * fps)))
            
            # Resolve to nearest keyframe if map exists
            resolved_frame = start_frame
            resolved_start = start_sec
            
            if frame_map:
                resolved = _resolve_frames_from_map(frame_map, np.array([start_sec]))
                if resolved[0] is not None and len(resolved[0]) > 0:
                    resolved_frame = int(resolved[0][0])
                    resolved_start = float(resolved[1][0])
            
            action = {
                "_index": config.TRANSCRIPT_INDEX,
                "_id": f"{video_id}_{resolved_frame}_{seg_id}",
                "_source": {
                    "video_id": video_id,
                    "keyframe_index": resolved_frame,
                    "start": round(resolved_start, 3),
                    "end": round(end_sec, 3),
                    "text": text,
                    "doc_type": "transcript",
                    "source_type": "asr",
                    "language": data.get("language"),
                    "model_name": data.get("model_name"),
                },
            }
            actions.append(action)
            
            if len(actions) >= BULK_CHUNK_SIZE:
                success, _ = bulk(es_client, actions, refresh=False)
                total_docs += success
                actions.clear()
        
        if actions:
            success, _ = bulk(es_client, actions, refresh=False)
            total_docs += success

    # --- Process CSV files (legacy format) ---
    for csv_path in csv_files:
        video_id = csv_path.stem
        
        # --- CẬP NHẬT: Lấy FPS thực tế thay vì dùng config cứng ---
        fps = get_video_fps(video_id)
        # --------------------------------------------------------

        try:
            df = pd.read_csv(csv_path)
        except Exception as exc:
            logger.error(f"Failed to read {csv_path}: {exc}")
            continue

        df.columns = [col.strip().title() for col in df.columns]
        required_columns = {"Start", "End", "Text"}
        if not required_columns.issubset(df.columns):
            logger.warning(f"Transcript file {csv_path} missing required columns; skipping")
            continue

        df = df.dropna(subset=["Text"])
        df["Text"] = df["Text"].astype(str).str.strip()
        df = df[df["Text"] != ""]
        if df.empty:
            continue

        start_secs = pd.to_numeric(df["Start"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
        end_secs = pd.to_numeric(df["End"], errors="coerce").to_numpy(dtype=np.float32)
        end_secs = np.where(np.isnan(end_secs), start_secs, end_secs)
        end_secs = np.maximum(end_secs, start_secs)

        # Tính toán frame dựa trên FPS thực tế của từng video
        start_frames = np.maximum(0, np.rint(start_secs * fps).astype(np.int32))

        if video_id not in map_cache:
            map_cache[video_id] = _load_keyframe_map(video_id)
        frame_map = map_cache[video_id]

        resolved_frames = start_frames
        resolved_starts = start_secs
        
        # Nếu có map, cố gắng khớp với keyframe có sẵn
        if frame_map:
            resolved = _resolve_frames_from_map(frame_map, start_secs)
            if resolved[0] is not None:
                resolved_frames = resolved[0]
                resolved_starts = resolved[1]

        texts = df["Text"].tolist()
        row_ids = df.index.to_numpy()

        actions = []
        for idx in range(len(texts)):
            action = {
                "_index": config.TRANSCRIPT_INDEX,
                "_id": f"{video_id}_{resolved_frames[idx]}_{row_ids[idx]}",
                "_source": {
                    "video_id": video_id,
                    "keyframe_index": int(resolved_frames[idx]),
                    "start": float(round(resolved_starts[idx], 3)),
                    "end": float(round(end_secs[idx], 3)),
                    "text": texts[idx],
                    "doc_type": "transcript",
                    "source_type": "transcript_csv",
                },
            }
            actions.append(action)

            if len(actions) >= BULK_CHUNK_SIZE:
                success, _ = bulk(es_client, actions, refresh=False)
                total_docs += success
                actions.clear()

        if actions:
            success, _ = bulk(es_client, actions, refresh=False)
            total_docs += success

    es_client.indices.refresh(index=config.TRANSCRIPT_INDEX)
    logger.info(f"Transcript ingestion complete. Total documents: {total_docs}")


def _parse_keyframe_index_from_name(name: str) -> int | None:
    try:
        stem = Path(name).stem
        return int(stem.split("_")[-1])
    except (ValueError, AttributeError):
        return None


def _resolve_keyframe_start(video_id: str, keyframe_index: int, fallback_seconds: float = 0.0) -> float:
    mapping = _load_keyframe_map(video_id)
    if mapping is None:
        return float(fallback_seconds)

    seconds, frame_ids = mapping
    if frame_ids.size == 0:
        return float(fallback_seconds)

    matches = np.where(frame_ids == int(keyframe_index))[0]
    if matches.size:
        return float(seconds[matches[0]])
    return float(fallback_seconds)


def ingest_ocr_data(es_client: Elasticsearch, folder_path: str) -> None:
    """Ingest OCR JSON artifacts into the shared multimodal text index.

    Supported current format:
    data/ocr_result/<video_id>/keyframe_0.json
    {
      "keyframe": "keyframe_0.webp",
      "time_seconds": 0.0,
      "ocr_results": [{"text": "...", "confidence": 0.95}]
    }
    """

    ocr_dir = Path(folder_path)
    if not ocr_dir.exists():
        logger.warning("OCR directory not found: %s", ocr_dir)
        return

    actions = []
    total_docs = 0

    for video_dir in sorted(p for p in ocr_dir.iterdir() if p.is_dir()):
        video_id = video_dir.name
        for json_path in sorted(video_dir.glob("*.json")):
            try:
                with json_path.open("r", encoding="utf-8") as handle:
                    data = json.load(handle)
            except Exception as exc:
                logger.warning("Failed to read OCR file %s: %s", json_path, exc)
                continue

            keyframe_index = _parse_keyframe_index_from_name(
                data.get("keyframe") or json_path.name
            )
            if keyframe_index is None:
                continue

            fallback_seconds = float(data.get("time_seconds") or 0.0)
            start_seconds = _resolve_keyframe_start(video_id, keyframe_index, fallback_seconds)

            raw_results = data.get("ocr_results") or data.get("results") or []
            text_parts = []
            confidences = []
            for result in raw_results:
                if isinstance(result, dict):
                    text = str(result.get("text") or "").strip()
                    confidence = result.get("confidence")
                elif isinstance(result, (list, tuple)) and len(result) >= 2:
                    text = str(result[1] or "").strip()
                    confidence = result[2] if len(result) >= 3 else None
                else:
                    continue

                if text:
                    text_parts.append(text)
                try:
                    if confidence is not None:
                        confidences.append(float(confidence))
                except (TypeError, ValueError):
                    pass

            ocr_text = " ".join(text_parts).strip()
            if not ocr_text:
                continue

            avg_confidence = float(np.mean(confidences)) if confidences else None
            actions.append(
                {
                    "_index": config.TRANSCRIPT_INDEX,
                    "_id": f"{video_id}_{keyframe_index}_ocr",
                    "_source": {
                        "video_id": video_id,
                        "keyframe_index": int(keyframe_index),
                        "start": round(start_seconds, 3),
                        "end": round(start_seconds, 3),
                        "text": ocr_text,
                        "ocr_text": ocr_text,
                        "doc_type": "ocr",
                        "source_type": "ocr",
                        "model_name": data.get("model_name") or config.OCR_ENGINE,
                        "confidence": avg_confidence,
                        "metadata": {
                            "artifact": str(json_path),
                            "num_boxes": len(text_parts),
                        },
                    },
                }
            )

            if len(actions) >= BULK_CHUNK_SIZE:
                success, _ = bulk(es_client, actions, refresh=False)
                total_docs += success
                actions.clear()

    if actions:
        success, _ = bulk(es_client, actions, refresh=False)
        total_docs += success

    if total_docs:
        es_client.indices.refresh(index=config.TRANSCRIPT_INDEX)
    logger.info("OCR ingestion complete. Total documents: %s", total_docs)


def ingest_caption_data(es_client: Elasticsearch, folder_path: str) -> None:
    """Ingest optional caption/VLM tag artifacts into Elasticsearch."""

    captions_dir = Path(folder_path)
    if not captions_dir.exists():
        logger.warning("Caption directory not found: %s", captions_dir)
        return

    actions = []
    total_docs = 0

    json_files = sorted(captions_dir.glob("*.json")) + sorted(captions_dir.glob("*/*.json"))
    for json_path in json_files:
        try:
            with json_path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
        except Exception as exc:
            logger.warning("Failed to read caption file %s: %s", json_path, exc)
            continue

        video_id = data.get("video_id") or json_path.parent.name
        keyframe_index = data.get("keyframe_index")
        if keyframe_index is None:
            keyframe_index = _parse_keyframe_index_from_name(data.get("keyframe") or json_path.name)
        if keyframe_index is None:
            continue

        caption = str(data.get("caption") or data.get("text") or "").strip()
        if not caption:
            tags = data.get("tags") or []
            caption = " ".join(str(tag) for tag in tags if tag).strip()
        if not caption:
            continue

        start_seconds = float(data.get("start") or data.get("time_seconds") or 0.0)
        start_seconds = _resolve_keyframe_start(video_id, int(keyframe_index), start_seconds)
        actions.append(
            {
                "_index": config.TRANSCRIPT_INDEX,
                "_id": f"{video_id}_{keyframe_index}_caption",
                "_source": {
                    "video_id": video_id,
                    "keyframe_index": int(keyframe_index),
                    "start": round(start_seconds, 3),
                    "end": round(start_seconds, 3),
                    "text": caption,
                    "caption": caption,
                    "doc_type": "caption",
                    "source_type": "vlm_caption",
                    "model_name": data.get("model_name"),
                    "metadata": data.get("metadata") or {},
                },
            }
        )

        if len(actions) >= BULK_CHUNK_SIZE:
            success, _ = bulk(es_client, actions, refresh=False)
            total_docs += success
            actions.clear()

    if actions:
        success, _ = bulk(es_client, actions, refresh=False)
        total_docs += success

    if total_docs:
        es_client.indices.refresh(index=config.TRANSCRIPT_INDEX)
    logger.info("Caption ingestion complete. Total documents: %s", total_docs)

def main():
    parser = argparse.ArgumentParser(description="Ingest keyframe embeddings và transcripts.")
    parser.add_argument(
        "--skip-transcripts",
        action="store_true",
        help="Bỏ qua ingest transcript."
    )
    parser.add_argument(
        "--append-transcripts",
        action="store_true",
        help="Chỉ thêm transcript mới, không xóa index cũ."
    )
    parser.add_argument(
        "--skip-milvus",
        action="store_true",
        help="Bỏ qua ingest keyframe embeddings vào Milvus."
    )
    parser.add_argument(
        "--skip-ocr",
        action="store_true",
        help="Bỏ qua ingest OCR artifacts."
    )
    parser.add_argument(
        "--skip-captions",
        action="store_true",
        help="Bỏ qua ingest caption/VLM artifacts."
    )
    args = parser.parse_args()

    if args.skip_transcripts and args.append_transcripts:
        parser.error("Không thể vừa --skip-transcripts vừa --append-transcripts")

    # Cấu hình logging để hiện ra console khi chạy trực tiếp
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] - %(message)s",
        handlers=[logging.StreamHandler()]
    )

    # --- Elasticsearch Ingestion ---
    if not args.skip_transcripts:
        es_client = get_elasticsearch_client()
        if args.append_transcripts:
            if not es_client.indices.exists(index=config.TRANSCRIPT_INDEX):
                logger.info("Transcript index '%s' chưa tồn tại. Tạo mới.", config.TRANSCRIPT_INDEX)
                recreate_transcript_index(es_client)
            else:
                logger.info("Append mode: giữ nguyên dữ liệu transcript sẵn có.")
        else:
            logger.info("Recreate mode: xóa và tạo lại transcript index '%s'.", config.TRANSCRIPT_INDEX)
            recreate_transcript_index(es_client)

        ingest_transcript_data(es_client, config.TRANSCRIPTS_DIR)
        if not args.skip_ocr:
            ingest_ocr_data(es_client, config.OCR_RESULTS_DIR)
        if not args.skip_captions:
            ingest_caption_data(es_client, config.CAPTIONS_DIR)
    else:
        logger.info("Bỏ qua ingest transcript theo yêu cầu.")

    # --- Milvus Ingestion ---
    if not args.skip_milvus:
        connections.connect("default", host=config.MILVUS_HOST, port=config.MILVUS_PORT)
        kf_fields = [
            FieldSchema(name="pk", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="video_id", dtype=DataType.VARCHAR, max_length=20),
            FieldSchema(name="keyframe_index", dtype=DataType.INT64),
            FieldSchema(name="keyframe_vector", dtype=DataType.FLOAT_VECTOR, dim=config.VECTOR_DIMENSION)
        ]
        kf_schema = CollectionSchema(kf_fields, "Keyframe vectors")
        kf_index_params = {"metric_type": "COSINE", "index_type": "IVF_FLAT", "params": {"nlist": 128}}
        
        kf_collection = setup_milvus_collection(
            config.KEYFRAME_COLLECTION_NAME,
            kf_schema,
            "keyframe_vector",
            kf_index_params
        )
        ingest_keyframe_data(kf_collection)
    else:
        logger.info("Bỏ qua ingest Milvus theo yêu cầu.")

    logger.info("--- DATA INGESTION COMPLETE ---")

if __name__ == "__main__":
    main()
