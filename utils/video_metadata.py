import os
import cv2
import glob
import logging
from pathlib import Path

from backend import config

logger = logging.getLogger(__name__)

def load_video_metadata(videos_dir: str) -> dict:
    """
    Quét toàn bộ file .mp4 trong thư mục videos_dir.
    Trả về một dictionary: { 'video_id': fps, ... }
    """
    metadata_cache = {}
    
    if not os.path.exists(videos_dir):
        logger.error(f"Directory not found: {videos_dir}")
        return metadata_cache

    video_files = glob.glob(os.path.join(videos_dir, "*.mp4"))
    logger.info(f"Loading metadata for {len(video_files)} videos into RAM...")

    for video_path in video_files:
        try:
            video_id = Path(video_path).stem # Lấy tên file làm ID (VD: L01_V001)
            cap = cv2.VideoCapture(video_path)
            
            if cap.isOpened():
                fps = cap.get(cv2.CAP_PROP_FPS)
                # Fallback nếu không đọc được FPS, mặc định 25
                if fps <= 0 or fps is None:
                    fps = config.DEFAULT_FALLBACK_FPS
                
                metadata_cache[video_id] = fps
                cap.release()
            else:
                logger.warning(f"Could not open video: {video_path}")
                metadata_cache[video_id] = config.DEFAULT_FALLBACK_FPS

        except Exception as e:
            logger.error(f"Error reading metadata for {video_path}: {e}")
            metadata_cache[video_id] = config.DEFAULT_FALLBACK_FPS

    logger.info(f"Metadata loaded successfully. Cached FPS for {len(metadata_cache)} videos.")
    return metadata_cache
