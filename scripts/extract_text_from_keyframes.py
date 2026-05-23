"""Extract OCR text from keyframes.

Default engine follows the README direction: PaddleOCR for full-corpus OCR,
with EasyOCR as a lightweight fallback when PaddleOCR is not installed.

Output format is intentionally simple and ingest_data.py can consume it:
data/ocr_result/<video_id>/keyframe_0.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] - %(message)s")
logger = logging.getLogger(__name__)


class OCRExtractor:
    def __init__(self, engine: str = "paddleocr", languages: list[str] | None = None, use_gpu: bool = True):
        self.engine = engine.lower()
        self.languages = languages or config.OCR_LANGUAGES
        self.use_gpu = use_gpu
        self.reader = self._load_engine()

    def _load_engine(self):
        if self.engine == "paddleocr":
            try:
                from paddleocr import PaddleOCR

                # PaddleOCR uses a single language code in many releases.
                # "en" is safest; multilingual configs can be passed via OCR_LANGUAGES.
                lang = self.languages[0] if self.languages else "en"
                logger.info("Loading PaddleOCR lang=%s gpu=%s", lang, self.use_gpu)
                return PaddleOCR(use_angle_cls=True, lang=lang, use_gpu=self.use_gpu)
            except Exception as exc:
                logger.warning("Failed to load PaddleOCR (%s). Falling back to EasyOCR.", exc)
                self.engine = "easyocr"

        if self.engine == "easyocr":
            try:
                import easyocr

                logger.info("Loading EasyOCR languages=%s gpu=%s", self.languages, self.use_gpu)
                return easyocr.Reader(self.languages, gpu=self.use_gpu)
            except Exception as exc:
                raise RuntimeError(
                    "No OCR engine available. Install paddleocr or easyocr, "
                    "or run with --engine none to skip OCR."
                ) from exc

        if self.engine == "none":
            return None

        raise ValueError(f"Unsupported OCR engine: {self.engine}")

    def readtext(self, image_path: Path) -> list[dict[str, Any]]:
        if self.engine == "none":
            return []

        if self.engine == "paddleocr":
            raw = self.reader.ocr(str(image_path), cls=True)
            results = []
            for page in raw or []:
                for item in page or []:
                    if not item or len(item) < 2:
                        continue
                    bbox = item[0]
                    text_score = item[1]
                    text = text_score[0] if text_score else ""
                    confidence = text_score[1] if len(text_score) > 1 else None
                    if text:
                        results.append({"text": text, "confidence": confidence, "bbox": bbox})
            return results

        raw = self.reader.readtext(str(image_path))
        return [
            {"text": item[1], "confidence": item[2], "bbox": item[0]}
            for item in raw
            if len(item) >= 3 and item[1]
        ]


def _keyframe_seconds(video_id: str, keyframe_index: int) -> float:
    map_path = Path(config.KEYFRAMES_DIR) / "maps" / f"{video_id}_map.csv"
    if not map_path.exists():
        return float(keyframe_index * 2)

    try:
        import csv

        with map_path.open("r", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if int(row.get("FrameID", -1)) == keyframe_index:
                    return float(row.get("Seconds", 0.0))
    except Exception:
        return float(keyframe_index * 2)

    return float(keyframe_index * 2)


def extract_text_from_keyframes(
    keyframes_dir: str = config.KEYFRAMES_DIR,
    output_dir: str = config.OCR_RESULTS_DIR,
    engine: str = config.OCR_ENGINE,
    languages: list[str] | None = None,
    use_gpu: bool = True,
    video_id: str | None = None,
    skip_existing: bool = True,
):
    keyframes_root = Path(keyframes_dir)
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    if not keyframes_root.exists():
        logger.error("Keyframes directory not found: %s", keyframes_root)
        return

    extractor = OCRExtractor(engine=engine, languages=languages, use_gpu=use_gpu)
    video_dirs = [keyframes_root / video_id] if video_id else [
        path for path in keyframes_root.iterdir() if path.is_dir() and path.name != "maps"
    ]

    for video_dir in sorted(video_dirs):
        if not video_dir.exists():
            logger.warning("Video keyframe directory not found: %s", video_dir)
            continue

        current_video_id = video_dir.name
        video_result_dir = output_root / current_video_id
        video_result_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Processing OCR for video: %s", current_video_id)

        keyframe_files = sorted(
            video_dir.glob("keyframe_*.png"),
            key=lambda path: int(path.stem.split("_")[-1]),
        )
        for keyframe_path in keyframe_files:
            keyframe_index = int(keyframe_path.stem.split("_")[-1])
            result_file = video_result_dir / f"{keyframe_path.stem}.json"
            if skip_existing and result_file.exists():
                continue

            results = extractor.readtext(keyframe_path)
            artifact = {
                "video_id": current_video_id,
                "keyframe": keyframe_path.name,
                "keyframe_index": keyframe_index,
                "time_seconds": _keyframe_seconds(current_video_id, keyframe_index),
                "model_name": extractor.engine,
                "ocr_results": results,
            }

            with result_file.open("w", encoding="utf-8") as handle:
                json.dump(artifact, handle, ensure_ascii=False, indent=2)

    logger.info("OCR extraction complete.")


def main():
    parser = argparse.ArgumentParser(description="Extract OCR text from keyframes")
    parser.add_argument("--engine", default=config.OCR_ENGINE, choices=["paddleocr", "easyocr", "none"])
    parser.add_argument("--video", help="Only process one video id, e.g. L01_V001")
    parser.add_argument("--keyframes-dir", default=config.KEYFRAMES_DIR)
    parser.add_argument("--output-dir", default=config.OCR_RESULTS_DIR)
    parser.add_argument("--languages", default=",".join(config.OCR_LANGUAGES))
    parser.add_argument("--cpu", action="store_true", help="Disable GPU for OCR engine")
    parser.add_argument("--force", action="store_true", help="Overwrite existing OCR JSON files")
    args = parser.parse_args()

    languages = [lang.strip() for lang in args.languages.split(",") if lang.strip()]
    extract_text_from_keyframes(
        keyframes_dir=args.keyframes_dir,
        output_dir=args.output_dir,
        engine=args.engine,
        languages=languages,
        use_gpu=not args.cpu,
        video_id=args.video,
        skip_existing=not args.force,
    )


if __name__ == "__main__":
    main()
