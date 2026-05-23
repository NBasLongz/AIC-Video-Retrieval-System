"""Validate data contracts across the retrieval pipeline.

Run this before Milvus/Elasticsearch ingest or after changing models:

    python -m scripts.validate_pipeline
    python -m scripts.validate_pipeline --check-services --strict

It checks shape/dimension compatibility, keyframe timestamp maps, transcript JSON,
OCR artifacts, and optional database schemas.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
import importlib.util
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    import numpy as np
except ImportError:
    np = None
try:
    import cv2
except ImportError:
    cv2 = None

try:
    import torch
except ImportError:
    torch = None

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] - %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class ValidationReport:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    info: list[str] = field(default_factory=list)

    def error(self, message: str):
        logger.error(message)
        self.errors.append(message)

    def warn(self, message: str):
        logger.warning(message)
        self.warnings.append(message)

    def note(self, message: str):
        logger.info(message)
        self.info.append(message)


def _load_map(video_id: str, report: ValidationReport) -> list[dict[str, Any]]:
    map_path = Path(config.KEYFRAMES_DIR) / "maps" / f"{video_id}_map.csv"
    if not map_path.exists():
        report.warn(f"{video_id}: missing keyframe map {map_path}")
        return []

    rows = []
    try:
        with map_path.open("r", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                rows.append(
                    {
                        "FrameID": int(row["FrameID"]),
                        "Seconds": float(row["Seconds"]),
                        "OriginalFrame": int(row.get("OriginalFrame") or 0),
                    }
                )
    except Exception as exc:
        report.error(f"{video_id}: invalid keyframe map {map_path}: {exc}")
        return []
    return rows


def validate_videos_and_keyframes(report: ValidationReport):
    if cv2 is None:
        report.warn("opencv-python is not installed; skipping video FPS/frame validation")
        return

    videos_dir = Path(config.VIDEOS_DIR)
    keyframes_dir = Path(config.KEYFRAMES_DIR)
    if not videos_dir.exists():
        report.warn(f"Videos directory not found: {videos_dir}")
        return
    if not keyframes_dir.exists():
        report.warn(f"Keyframes directory not found: {keyframes_dir}")

    videos = sorted(videos_dir.glob("*.mp4"))
    report.note(f"Found {len(videos)} videos")

    for video_path in videos:
        video_id = video_path.stem
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            report.error(f"{video_id}: cannot open video {video_path}")
            continue

        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        duration = total_frames / fps if fps > 0 else 0
        cap.release()

        if fps <= 0:
            report.error(f"{video_id}: invalid FPS={fps}")
        if total_frames <= 0:
            report.warn(f"{video_id}: total frame count is {total_frames}")

        frame_dir = keyframes_dir / video_id
        keyframe_files = sorted(frame_dir.glob("keyframe_*.webp")) if frame_dir.exists() else []
        rows = _load_map(video_id, report)
        if rows and keyframe_files and len(rows) != len(keyframe_files):
            report.warn(
                f"{video_id}: keyframe files ({len(keyframe_files)}) != map rows ({len(rows)})"
            )

        last_seconds = -1.0
        seen_ids = set()
        for row in rows:
            frame_id = row["FrameID"]
            seconds = row["Seconds"]
            original_frame = row["OriginalFrame"]
            if frame_id in seen_ids:
                report.error(f"{video_id}: duplicate FrameID={frame_id} in map")
            seen_ids.add(frame_id)
            if seconds < last_seconds:
                report.error(f"{video_id}: map Seconds is not monotonic at FrameID={frame_id}")
            last_seconds = seconds
            if duration and seconds > duration + 1:
                report.warn(f"{video_id}: keyframe time {seconds:.2f}s exceeds duration {duration:.2f}s")
            if fps > 0 and abs((seconds * fps) - original_frame) > max(2, fps * 0.1):
                report.warn(
                    f"{video_id}: OriginalFrame mismatch at FrameID={frame_id}: "
                    f"seconds*fps={seconds * fps:.1f}, OriginalFrame={original_frame}"
                )


def validate_embeddings(report: ValidationReport):
    if torch is None or np is None:
        report.warn("torch/numpy is not installed; skipping embedding tensor validation")
        return

    embeddings_root = Path(config.CLIP_FEATURES_DIR)
    if not embeddings_root.exists():
        report.warn(f"Embeddings directory not found: {embeddings_root}")
        return

    checked = 0
    for video_dir in sorted(path for path in embeddings_root.iterdir() if path.is_dir()):
        metadata_path = video_dir / "_metadata.json"
        if metadata_path.exists():
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                meta_dim = int(metadata.get("vector_dimension"))
                if meta_dim != config.VECTOR_DIMENSION:
                    report.error(
                        f"{video_dir.name}: metadata dim={meta_dim} != config.VECTOR_DIMENSION={config.VECTOR_DIMENSION}"
                    )
                if metadata.get("provider") != config.VISUAL_MODEL_PROVIDER:
                    report.error(
                        f"{video_dir.name}: metadata provider={metadata.get('provider')} "
                        f"!= config.VISUAL_MODEL_PROVIDER={config.VISUAL_MODEL_PROVIDER}"
                    )
                if metadata.get("model_name") != config.VISUAL_MODEL_NAME:
                    report.error(
                        f"{video_dir.name}: metadata model={metadata.get('model_name')} "
                        f"!= config.VISUAL_MODEL={config.VISUAL_MODEL_NAME}"
                    )
                if int(metadata.get("truncate_dim") or metadata.get("vector_dimension") or 0) != config.VISUAL_TRUNCATE_DIM:
                    report.error(
                        f"{video_dir.name}: metadata truncate_dim={metadata.get('truncate_dim')} "
                        f"!= config.VISUAL_TRUNCATE_DIM={config.VISUAL_TRUNCATE_DIM}"
                    )
            except Exception as exc:
                report.warn(f"{video_dir.name}: invalid embedding metadata: {exc}")
        else:
            report.warn(f"{video_dir.name}: missing _metadata.json; recompute embeddings to record model contract")

        for pt_path in sorted(video_dir.glob("keyframe_*.pt")):
            try:
                tensor = torch.load(str(pt_path), map_location="cpu")
                arr = tensor.detach().cpu().numpy() if hasattr(tensor, "detach") else np.asarray(tensor)
                arr = arr.reshape(1, -1)
            except Exception as exc:
                report.error(f"Cannot load embedding {pt_path}: {exc}")
                continue

            dim = arr.shape[-1]
            if dim != config.VECTOR_DIMENSION:
                report.error(
                    f"{pt_path}: dim={dim} != config.VECTOR_DIMENSION={config.VECTOR_DIMENSION}"
                )
            norm = float(np.linalg.norm(arr[0]))
            if not np.isfinite(norm):
                report.error(f"{pt_path}: non-finite embedding norm")
            elif abs(norm - 1.0) > 0.05:
                report.warn(f"{pt_path}: embedding norm {norm:.4f} is not close to 1.0")
            checked += 1

        keyframe_dir = Path(config.KEYFRAMES_DIR) / video_dir.name
        if keyframe_dir.exists():
            keyframe_count = len(list(keyframe_dir.glob("keyframe_*.webp")))
            embedding_count = len(list(video_dir.glob("keyframe_*.pt")))
            if keyframe_count != embedding_count:
                report.error(
                    f"{video_dir.name}: embedding files ({embedding_count}) != keyframes ({keyframe_count}); "
                    "rerun scripts.compute_embeddings without --keep-existing"
                )

    report.note(f"Checked {checked} embedding files")


def validate_model_config(report: ValidationReport):
    if config.VISUAL_MODEL_PROVIDER == "siglip2":
        if "siglip2" not in config.VISUAL_MODEL_NAME.lower():
            report.warn(
                "VISUAL_MODEL_PROVIDER=siglip2 is tuned for Google SigLIP2 checkpoints; "
                f"current model={config.VISUAL_MODEL_NAME}"
            )
        if "naflex" in config.VISUAL_MODEL_NAME.lower() and config.VISUAL_TRUNCATE_DIM != config.VECTOR_DIMENSION:
            report.error(
                f"VISUAL_TRUNCATE_DIM={config.VISUAL_TRUNCATE_DIM} must match "
                f"VECTOR_DIMENSION={config.VECTOR_DIMENSION}"
            )
        for package in ("transformers", "timm", "PIL"):
            if importlib.util.find_spec(package) is None:
                report.warn(f"{package} is not installed; SigLIP2 cannot run in this environment")

    if config.VISUAL_MODEL_PROVIDER in {"jina_clip", "jina"}:
        if config.VISUAL_MODEL_NAME != "jinaai/jina-clip-v2":
            report.warn(
                "VISUAL_MODEL_PROVIDER=jina_clip is tuned for jinaai/jina-clip-v2; "
                f"current model={config.VISUAL_MODEL_NAME}"
            )
        if not config.MODEL_TRUST_REMOTE_CODE:
            report.error("jina-clip-v2 requires MODEL_TRUST_REMOTE_CODE=true")
        if config.VISUAL_TRUNCATE_DIM != config.VECTOR_DIMENSION:
            report.error(
                f"VISUAL_TRUNCATE_DIM={config.VISUAL_TRUNCATE_DIM} must match "
                f"VECTOR_DIMENSION={config.VECTOR_DIMENSION}"
            )
        for package in ("transformers", "einops", "timm", "PIL"):
            if importlib.util.find_spec(package) is None:
                report.warn(f"{package} is not installed; jina-clip-v2 cannot run in this environment")

    if config.RERANK_MODEL_PROVIDER not in {"", "none", "disabled", "off"}:
        if importlib.util.find_spec("sentence_transformers") is None:
            report.warn("sentence-transformers is not installed; reranker will disable itself at runtime")

    if config.ENABLE_DENSE_TEXT_RETRIEVAL:
        if config.TEXT_MODEL_PROVIDER in {"", "none", "off", "disabled"}:
            report.error("ENABLE_DENSE_TEXT_RETRIEVAL=true but TEXT_MODEL_PROVIDER is disabled")
        if config.TEXT_VECTOR_DIMENSION != 1024 and "bge-m3" in config.TEXT_MODEL_NAME.lower():
            report.error(
                f"TEXT_MODEL={config.TEXT_MODEL_NAME} outputs 1024d dense vectors, "
                f"but TEXT_VECTOR_DIMENSION={config.TEXT_VECTOR_DIMENSION}"
            )
        if importlib.util.find_spec("sentence_transformers") is None:
            report.warn("sentence-transformers is not installed; dense text retrieval cannot run")

    if config.ENABLE_QUERY_TRANSLATION:
        for package in ("transformers", "sentencepiece"):
            if importlib.util.find_spec(package) is None:
                report.warn(f"{package} is not installed; query translation cannot run in this environment")

    if config.OCR_ENGINE == "paddleocr" and importlib.util.find_spec("paddleocr") is None:
        report.warn("paddleocr is not installed; OCR script will fall back to EasyOCR if available")

    if config.ASR_MODEL and importlib.util.find_spec("whisper") is None:
        report.warn("openai-whisper is not installed; ASR extraction cannot run in this environment")


def validate_transcripts(report: ValidationReport):
    transcripts_dir = Path(config.TRANSCRIPTS_DIR)
    if not transcripts_dir.exists():
        report.warn(f"Transcripts directory not found: {transcripts_dir}")
        return

    count = 0
    for json_path in sorted(transcripts_dir.glob("*.json")):
        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
        except Exception as exc:
            report.error(f"{json_path}: invalid JSON: {exc}")
            continue

        segments = data.get("segments")
        if not isinstance(segments, list):
            report.error(f"{json_path}: missing segments list")
            continue

        last_start = -1.0
        for idx, seg in enumerate(segments):
            try:
                start = float(seg.get("start"))
                end = float(seg.get("end"))
            except (TypeError, ValueError):
                report.error(f"{json_path}: segment {idx} has invalid start/end")
                continue
            text = str(seg.get("text") or "").strip()
            if start < last_start:
                report.warn(f"{json_path}: segment {idx} start is not monotonic")
            if end < start:
                report.error(f"{json_path}: segment {idx} end < start")
            if not text:
                report.warn(f"{json_path}: segment {idx} has empty text")
            last_start = start
        count += 1

    report.note(f"Checked {count} transcript JSON files")


def validate_ocr(report: ValidationReport):
    ocr_dir = Path(config.OCR_RESULTS_DIR)
    if not ocr_dir.exists():
        report.warn(f"OCR directory not found: {ocr_dir}")
        return

    count = 0
    for json_path in sorted(ocr_dir.glob("*/*.json")):
        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
        except Exception as exc:
            report.error(f"{json_path}: invalid JSON: {exc}")
            continue

        if data.get("keyframe_index") is None and not data.get("keyframe"):
            report.warn(f"{json_path}: missing keyframe_index/keyframe")
        results = data.get("ocr_results") or []
        if not isinstance(results, list):
            report.error(f"{json_path}: ocr_results must be a list")
        count += 1

    report.note(f"Checked {count} OCR JSON files")


def validate_services(report: ValidationReport):
    try:
        from pymilvus import Collection, connections, utility

        connections.connect("default", host=config.MILVUS_HOST, port=config.MILVUS_PORT, timeout=5)
        if utility.has_collection(config.KEYFRAME_COLLECTION_NAME):
            collection = Collection(config.KEYFRAME_COLLECTION_NAME)
            dim = None
            for field in collection.schema.fields:
                if field.name == "keyframe_vector":
                    dim = int(field.params.get("dim"))
            if dim != config.VECTOR_DIMENSION:
                report.error(
                    f"Milvus dim={dim} != config.VECTOR_DIMENSION={config.VECTOR_DIMENSION}"
                )
            else:
                report.note(f"Milvus collection dim OK: {dim}")
        else:
            report.warn(f"Milvus collection does not exist: {config.KEYFRAME_COLLECTION_NAME}")

        if config.ENABLE_DENSE_TEXT_RETRIEVAL:
            if utility.has_collection(config.TEXT_COLLECTION_NAME):
                collection = Collection(config.TEXT_COLLECTION_NAME)
                dim = None
                for field in collection.schema.fields:
                    if field.name == "text_vector":
                        dim = int(field.params.get("dim"))
                if dim != config.TEXT_VECTOR_DIMENSION:
                    report.error(
                        f"Milvus text dim={dim} != TEXT_VECTOR_DIMENSION={config.TEXT_VECTOR_DIMENSION}"
                    )
                else:
                    report.note(f"Milvus text collection dim OK: {dim}")
            else:
                report.warn(f"Milvus text collection does not exist: {config.TEXT_COLLECTION_NAME}")
    except Exception as exc:
        report.warn(f"Could not validate Milvus: {exc}")

    try:
        from utils.elasticsearch_client import get_elasticsearch_client

        es = get_elasticsearch_client()
        if es.indices.exists(index=config.TRANSCRIPT_INDEX):
            count = es.count(index=config.TRANSCRIPT_INDEX).get("count")
            report.note(f"Elasticsearch index {config.TRANSCRIPT_INDEX} exists with {count} docs")
        else:
            report.warn(f"Elasticsearch index does not exist: {config.TRANSCRIPT_INDEX}")
    except Exception as exc:
        report.warn(f"Could not validate Elasticsearch: {exc}")


def main():
    parser = argparse.ArgumentParser(description="Validate retrieval pipeline contracts")
    parser.add_argument("--check-services", action="store_true", help="Also validate Milvus and Elasticsearch")
    parser.add_argument("--strict", action="store_true", help="Exit non-zero on warnings as well as errors")
    args = parser.parse_args()

    report = ValidationReport()
    report.note(f"Configured visual model: {config.VISUAL_MODEL_PROVIDER}:{config.VISUAL_MODEL_NAME}")
    report.note(f"Configured vector dimension: {config.VECTOR_DIMENSION}")
    report.note(f"Configured visual truncate dim: {config.VISUAL_TRUNCATE_DIM}")
    report.note(
        f"Configured dense text model: {config.TEXT_MODEL_PROVIDER}:{config.TEXT_MODEL_NAME} "
        f"dim={config.TEXT_VECTOR_DIMENSION} enabled={config.ENABLE_DENSE_TEXT_RETRIEVAL}"
    )
    report.note(f"Configured ASR model: {config.ASR_MODEL} language={config.ASR_LANGUAGE}")
    report.note(
        "Configured translator: "
        f"{config.QUERY_TRANSLATION_PROVIDER}:{config.QUERY_TRANSLATION_MODEL} "
        f"{config.QUERY_TRANSLATION_SRC_LANG}->{config.QUERY_TRANSLATION_TGT_LANG}"
    )
    report.note(f"Configured reranker: {config.RERANK_MODEL_PROVIDER}:{config.RERANK_MODEL_NAME}")
    report.note(f"Configured text index: {config.TRANSCRIPT_INDEX}")

    validate_model_config(report)
    validate_videos_and_keyframes(report)
    validate_embeddings(report)
    validate_transcripts(report)
    validate_ocr(report)
    if args.check_services:
        validate_services(report)

    logger.info("Validation summary: %s errors, %s warnings", len(report.errors), len(report.warnings))
    if report.errors or (args.strict and report.warnings):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
