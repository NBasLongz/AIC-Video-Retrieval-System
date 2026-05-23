"""
Script to extract keyframes from videos and save as WebP images with mapping CSV.

Usage:
    python scripts/extract_keyframes.py --method interval --interval 1.0
    python scripts/extract_keyframes.py --method uniform --count 100
    python scripts/extract_keyframes.py --video L01_V001
"""

import argparse
import csv
import logging
import os
import re
import shutil
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

from backend import config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] - %(message)s",
)
logger = logging.getLogger(__name__)


def extract_keyframes_interval(
    video_path: str,
    output_dir: Path,
    map_file: Path,
    interval_seconds: float = 1.0,
    resume: bool = False,
    overwrite: bool = False,
):
    """
    Extract keyframes at regular time intervals.
    
    Args:
        video_path: Path to video file
        output_dir: Directory to save keyframe images
        map_file: Path to save mapping CSV
        interval_seconds: Time interval between keyframes in seconds
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Cannot open video: {video_path}")
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if fps <= 0:
        logger.warning("Invalid FPS for %s, using fallback %.3f", video_path, config.DEFAULT_FALLBACK_FPS)
        fps = config.DEFAULT_FALLBACK_FPS
    if interval_seconds <= 0:
        raise ValueError("interval_seconds must be > 0")
    
    logger.info(f"Video FPS: {fps}, Total frames: {total_frames}")

    if overwrite:
        if output_dir.exists():
            shutil.rmtree(output_dir)
        if map_file.exists():
            map_file.unlink()
        resume = False
    
    output_dir.mkdir(parents=True, exist_ok=True)
    map_file.parent.mkdir(parents=True, exist_ok=True)
    
    keyframe_data = []
    keyframe_index = 0

    # Determine resume start
    start_seconds = 0.0
    if resume and output_dir.exists():
        # Try reading last row from map_file
        if map_file.exists():
            try:
                with open(map_file, newline="", encoding="utf-8") as f:
                    rows = list(csv.DictReader(f))
                    if rows:
                        last = rows[-1]
                        try:
                            last_id = int(last.get("FrameID", 0))
                            last_seconds = float(last.get("Seconds", 0.0))
                            start_seconds = (int(last_seconds / interval_seconds) + 1) * interval_seconds
                            keyframe_index = last_id + 1
                        except Exception:
                            start_seconds = 0.0
            except Exception:
                start_seconds = 0.0

        # fallback: inspect existing keyframe files
        if start_seconds == 0:
            pattern = re.compile(r"keyframe_(\d+)\.webp$")
            max_idx = -1
            if output_dir.exists():
                for p in output_dir.iterdir():
                    m = pattern.search(p.name)
                    if m:
                        idx = int(m.group(1))
                        if idx > max_idx:
                            max_idx = idx
            if max_idx >= 0:
                keyframe_index = max_idx + 1
                start_seconds = keyframe_index * interval_seconds

    # If start_frame beyond total, nothing to do
    start_frame = int(round(start_seconds * fps))
    if start_frame >= total_frames:
        logger.info("Nothing to resume, already extracted all keyframes.")
        cap.release()
        return

    duration_seconds = total_frames / fps if fps > 0 else 0.0
    target_seconds = np.arange(start_seconds, duration_seconds, interval_seconds, dtype=np.float64)

    with tqdm(total=len(target_seconds), desc="Extracting keyframes") as pbar:
        for target_second in target_seconds:
            target_frame = int(round(float(target_second) * fps))
            if target_frame >= total_frames:
                break

            cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
            ret, frame = cap.read()
            if not ret:
                logger.warning("Failed to read frame %s at %.3fs", target_frame, target_second)
                pbar.update(1)
                continue

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)

            output_path = output_dir / f"keyframe_{keyframe_index}.webp"
            if output_path.exists():
                logger.debug(f"Skipping existing keyframe file {output_path}")
            else:
                img.save(output_path, "WebP", quality=90)

            actual_seconds = target_frame / fps
            keyframe_data.append({
                "FrameID": keyframe_index,
                "Seconds": round(actual_seconds, 3),
                "OriginalFrame": target_frame,
            })
            keyframe_index += 1
            pbar.update(1)
    
    cap.release()
    
    # Write mapping CSV (append in resume mode)
    write_mode = "a" if resume and map_file.exists() else "w"
    with open(map_file, write_mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["FrameID", "Seconds", "OriginalFrame"])
        if write_mode == "w":
            writer.writeheader()
        writer.writerows(keyframe_data)
    
    logger.info(f"Extracted {keyframe_index} keyframes to {output_dir}")
    logger.info(f"Mapping saved to {map_file}")


def extract_keyframes_uniform(
    video_path: str, output_dir: Path, map_file: Path, count: int = 100, resume: bool = False, overwrite: bool = False
):
    """
    Extract a fixed number of uniformly distributed keyframes.
    
    Args:
        video_path: Path to video file
        output_dir: Directory to save keyframe images
        map_file: Path to save mapping CSV
        count: Number of keyframes to extract
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Cannot open video: {video_path}")
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if fps <= 0:
        logger.warning("Invalid FPS for %s, using fallback %.3f", video_path, config.DEFAULT_FALLBACK_FPS)
        fps = config.DEFAULT_FALLBACK_FPS
    
    if total_frames < count:
        logger.warning(f"Video has only {total_frames} frames, extracting all")
        count = total_frames
    
    logger.info(f"Video FPS: {fps}, Total frames: {total_frames}")
    logger.info(f"Extracting {count} uniformly distributed keyframes")

    if overwrite:
        if output_dir.exists():
            shutil.rmtree(output_dir)
        if map_file.exists():
            map_file.unlink()
        resume = False
    
    output_dir.mkdir(parents=True, exist_ok=True)
    map_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Calculate frame indices to extract
    if count == 1:
        frame_indices = [0]
    else:
        frame_indices = np.linspace(0, total_frames - 1, count, dtype=int)
    
    keyframe_data = []
    keyframe_index = 0

    # If resuming, read existing OriginalFrame values to skip
    existing_frames = set()
    if resume and map_file.exists():
        try:
            with open(map_file, newline="", encoding="utf-8") as f:
                for r in csv.DictReader(f):
                    try:
                        existing_frames.add(int(r.get("OriginalFrame", -1)))
                    except Exception:
                        pass
        except Exception:
            existing_frames = set()

    for target_frame in tqdm(frame_indices, desc="Extracting keyframes"):
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
        ret, frame = cap.read()
        
        if not ret:
            logger.warning(f"Failed to read frame {target_frame}")
            continue
        if resume and target_frame in existing_frames:
            logger.debug(f"Skipping already extracted frame {target_frame}")
            continue
        
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        
        # Save as WebP
        output_path = output_dir / f"keyframe_{keyframe_index}.webp"
        img.save(output_path, "WebP", quality=90)
        
        # Record mapping
        seconds = target_frame / fps
        keyframe_data.append({
            "FrameID": keyframe_index,
            "Seconds": round(seconds, 3),
            "OriginalFrame": int(target_frame)
        })
        
        keyframe_index += 1
    
    cap.release()

    # Write mapping CSV (append in resume mode)
    write_mode = "a" if resume and map_file.exists() else "w"
    with open(map_file, write_mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["FrameID", "Seconds", "OriginalFrame"])
        if write_mode == "w":
            writer.writeheader()
        writer.writerows(keyframe_data)
    
    logger.info(f"Extracted {keyframe_index} keyframes to {output_dir}")
    logger.info(f"Mapping saved to {map_file}")


def process_video(video_id: str, method: str = "interval", **kwargs):
    """Process a single video to extract keyframes."""
    video_path = Path(config.VIDEOS_DIR) / f"{video_id}.mp4"
    
    if not video_path.exists():
        logger.error(f"Video not found: {video_path}")
        return False
    
    output_dir = Path(config.KEYFRAMES_DIR) / video_id
    map_file = Path(config.KEYFRAMES_DIR) / "maps" / f"{video_id}_map.csv"
    
    logger.info(f"Processing video: {video_id}")
    
    if method == "interval":
        interval = kwargs.get("interval", config.KEYFRAME_INTERVAL_SECONDS)
        resume = kwargs.get("resume", False)
        overwrite = kwargs.get("overwrite", False)
        extract_keyframes_interval(str(video_path), output_dir, map_file, interval, resume=resume, overwrite=overwrite)
    elif method == "uniform":
        count = kwargs.get("count", 100)
        resume = kwargs.get("resume", False)
        overwrite = kwargs.get("overwrite", False)
        extract_keyframes_uniform(str(video_path), output_dir, map_file, count, resume=resume, overwrite=overwrite)
    else:
        logger.error(f"Unknown method: {method}")
        return False
    
    return True


def process_all_videos(method: str = "interval", **kwargs):
    """Process all videos in the VIDEOS_DIR."""
    videos_dir = Path(config.VIDEOS_DIR)
    
    if not videos_dir.exists():
        logger.error(f"Videos directory not found: {videos_dir}")
        return
    
    video_files = list(videos_dir.glob("*.mp4"))
    logger.info(f"Found {len(video_files)} videos to process")
    
    for video_path in video_files:
        video_id = video_path.stem
        process_video(video_id, method, **kwargs)


def main():
    parser = argparse.ArgumentParser(
        description="Extract keyframes from videos"
    )
    parser.add_argument(
        "--method",
        choices=["interval", "uniform"],
        default="interval",
        help="Extraction method: 'interval' (every N seconds) or 'uniform' (fixed count)",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=config.KEYFRAME_INTERVAL_SECONDS,
        help=f"Time interval in seconds (for interval method). Default: {config.KEYFRAME_INTERVAL_SECONDS}",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=100,
        help="Number of keyframes to extract (for uniform method)",
    )
    parser.add_argument(
        "--video",
        type=str,
        help="Process only specific video ID (e.g., L01_V001). If not provided, processes all videos.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume extraction for videos when keyframes/map already exist",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete existing keyframes/map for the selected videos before extraction. Use this after changing interval.",
    )
    
    args = parser.parse_args()
    
    if args.video:
        process_video(
            args.video,
            method=args.method,
            interval=args.interval,
            count=args.count,
            resume=args.resume,
            overwrite=args.overwrite,
        )
    else:
        process_all_videos(
            method=args.method,
            interval=args.interval,
            count=args.count,
            resume=args.resume,
            overwrite=args.overwrite,
        )


if __name__ == "__main__":
    main()
