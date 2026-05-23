"""
Script to extract keyframes from videos and save as PNG images with mapping CSV.

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
import subprocess
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from backend import config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] - %(message)s",
)
logger = logging.getLogger(__name__)


LOSSY_CODECS = {"av1", "vp9", "vp8", "hevc"}
PNG_COMPRESSION = 3


def _run_command(command: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(command, check=False, capture_output=True, text=True)


def _is_valid_image(path: Path) -> bool:
    if not path.exists() or path.stat().st_size <= 0:
        return False
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    return image is not None and image.size > 0


def _save_frame_png(frame: np.ndarray, output_path: Path) -> None:
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    ok, encoded = cv2.imencode(
        ".png",
        frame,
        [cv2.IMWRITE_PNG_COMPRESSION, PNG_COMPRESSION],
    )
    if not ok:
        raise RuntimeError(f"Failed to encode PNG for {output_path}")

    try:
        encoded.tofile(str(tmp_path))
        if not _is_valid_image(tmp_path):
            raise RuntimeError(f"Encoded PNG is not readable: {tmp_path}")
        os.replace(tmp_path, output_path)
        if not _is_valid_image(output_path):
            raise RuntimeError(f"Saved PNG is not readable: {output_path}")
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def _next_keyframe_index(output_dir: Path) -> int:
    pattern = re.compile(r"keyframe_(\d+)\.png$")
    max_idx = -1
    if output_dir.exists():
        for path in output_dir.iterdir():
            match = pattern.search(path.name)
            if match and _is_valid_image(path):
                max_idx = max(max_idx, int(match.group(1)))
    return max_idx + 1


def _probe_video_codec(video_path: Path) -> str | None:
    if not shutil.which("ffprobe"):
        return None
    result = _run_command([
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=codec_name",
        "-of",
        "default=nokey=1:noprint_wrappers=1",
        str(video_path),
    ])
    if result.returncode != 0:
        logger.warning("ffprobe failed for %s: %s", video_path, result.stderr.strip())
        return None
    return result.stdout.strip().lower() or None


def _ensure_compatible_video(video_path: Path, compatible_dir: Path, force: bool = False) -> Path:
    codec = _probe_video_codec(video_path)
    needs_transcode = force or codec in LOSSY_CODECS
    if not needs_transcode:
        return video_path
    if not shutil.which("ffmpeg"):
        raise RuntimeError(
            f"Video {video_path.name} uses codec '{codec}', but ffmpeg is not available. "
            "Install ffmpeg or convert the video to H.264 before extracting keyframes."
        )

    compatible_dir.mkdir(parents=True, exist_ok=True)
    output_path = compatible_dir / video_path.name
    if output_path.exists() and output_path.stat().st_size > 0 and not force:
        logger.info("Using existing compatible video: %s", output_path)
        return output_path

    logger.warning(
        "Video %s uses codec '%s'. Transcoding to H.264 for reliable frame reads: %s",
        video_path.name,
        codec or "unknown",
        output_path,
    )
    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-map",
        "0:v:0",
        "-map",
        "0:a?",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "20",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-b:a",
        "128k",
        str(output_path),
    ]
    result = _run_command(command)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg transcode failed for {video_path}: {result.stderr[-2000:]}")
    return output_path


def extract_keyframes_interval(
    video_path: str,
    output_dir: Path,
    map_file: Path,
    interval_seconds: float = 1.0,
    resume: bool = False,
    overwrite: bool = False,
    max_read_failures: int = 10,
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
    read_failures = 0

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
            keyframe_index = _next_keyframe_index(output_dir)
            if keyframe_index > 0:
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
                read_failures += 1
                logger.warning("Failed to read frame %s at %.3fs", target_frame, target_second)
                if read_failures > max_read_failures:
                    cap.release()
                    raise RuntimeError(
                        f"Too many failed frame reads for {video_path} ({read_failures}). "
                        "This is often caused by AV1/VP9 decode issues. Rerun with "
                        "--ensure-compatible to transcode to H.264 before extraction."
                    )
                pbar.update(1)
                continue

            output_path = output_dir / f"keyframe_{keyframe_index}.png"
            if _is_valid_image(output_path):
                logger.debug(f"Skipping existing keyframe file {output_path}")
            else:
                if output_path.exists():
                    logger.warning("Replacing corrupt or empty keyframe file: %s", output_path)
                _save_frame_png(frame, output_path)

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
    video_path: str,
    output_dir: Path,
    map_file: Path,
    count: int = 100,
    resume: bool = False,
    overwrite: bool = False,
    max_read_failures: int = 10,
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
    read_failures = 0

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
        keyframe_index = _next_keyframe_index(output_dir)

    for target_frame in tqdm(frame_indices, desc="Extracting keyframes"):
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
        ret, frame = cap.read()
        
        if not ret:
            read_failures += 1
            logger.warning(f"Failed to read frame {target_frame}")
            if read_failures > max_read_failures:
                cap.release()
                raise RuntimeError(
                    f"Too many failed frame reads for {video_path} ({read_failures}). "
                    "Rerun with --ensure-compatible to transcode to H.264 before extraction."
                )
            continue
        if resume and target_frame in existing_frames:
            logger.debug(f"Skipping already extracted frame {target_frame}")
            continue
        
        # Save as PNG
        output_path = output_dir / f"keyframe_{keyframe_index}.png"
        _save_frame_png(frame, output_path)
        
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

    if kwargs.get("ensure_compatible", False):
        video_path = _ensure_compatible_video(
            video_path,
            Path(kwargs.get("compatible_dir", Path(config.DATA_DIR) / "videos_compatible")),
            force=kwargs.get("force_transcode", False),
        )
    
    output_dir = Path(config.KEYFRAMES_DIR) / video_id
    map_file = Path(config.KEYFRAMES_DIR) / "maps" / f"{video_id}_map.csv"
    
    logger.info(f"Processing video: {video_id}")
    
    if method == "interval":
        interval = kwargs.get("interval", config.KEYFRAME_INTERVAL_SECONDS)
        resume = kwargs.get("resume", False)
        overwrite = kwargs.get("overwrite", False)
        max_read_failures = kwargs.get("max_read_failures", 10)
        extract_keyframes_interval(
            str(video_path),
            output_dir,
            map_file,
            interval,
            resume=resume,
            overwrite=overwrite,
            max_read_failures=max_read_failures,
        )
    elif method == "uniform":
        count = kwargs.get("count", 100)
        resume = kwargs.get("resume", False)
        overwrite = kwargs.get("overwrite", False)
        max_read_failures = kwargs.get("max_read_failures", 10)
        extract_keyframes_uniform(
            str(video_path),
            output_dir,
            map_file,
            count,
            resume=resume,
            overwrite=overwrite,
            max_read_failures=max_read_failures,
        )
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
    parser.add_argument(
        "--ensure-compatible",
        action="store_true",
        help="Transcode AV1/VP9/HEVC videos to H.264 before extraction to avoid OpenCV decode failures.",
    )
    parser.add_argument(
        "--compatible-dir",
        type=str,
        default=str(Path(config.DATA_DIR) / "videos_compatible"),
        help="Directory for H.264 compatible video copies when --ensure-compatible is enabled.",
    )
    parser.add_argument(
        "--force-transcode",
        action="store_true",
        help="Always recreate the compatible H.264 video copy.",
    )
    parser.add_argument(
        "--max-read-failures",
        type=int,
        default=10,
        help="Abort extraction after this many failed frame reads. Prevents silent broken keyframe maps.",
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
            ensure_compatible=args.ensure_compatible,
            compatible_dir=args.compatible_dir,
            force_transcode=args.force_transcode,
            max_read_failures=args.max_read_failures,
        )
    else:
        process_all_videos(
            method=args.method,
            interval=args.interval,
            count=args.count,
            resume=args.resume,
            overwrite=args.overwrite,
            ensure_compatible=args.ensure_compatible,
            compatible_dir=args.compatible_dir,
            force_transcode=args.force_transcode,
            max_read_failures=args.max_read_failures,
        )


if __name__ == "__main__":
    main()
