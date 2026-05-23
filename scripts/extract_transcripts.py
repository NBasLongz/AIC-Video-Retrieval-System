"""
Script để extract transcripts từ video sử dụng OpenAI Whisper.
Hỗ trợ:
- Tự động detect ngôn ngữ hoặc chỉ định ngôn ngữ cụ thể
- Xuất ra JSON format với timestamps
- Hỗ trợ batch processing
- Tùy chọn model size (tiny, base, small, medium, large)
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

try:
    import whisper
except ImportError:
    whisper = None
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend import config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] - %(message)s"
)
logger = logging.getLogger(__name__)

DEFAULT_VIETNAMESE_PROMPT = (
    "Đây là âm thanh tiếng Việt trong video. Hãy chép lại chính xác tiếng Việt có dấu, "
    "giữ tên riêng, địa danh, số, đơn vị đo, thương hiệu và từ tiếng Anh nếu có."
)


class WhisperTranscriptExtractor:
    """Class để extract transcripts từ video sử dụng Whisper"""
    
    def __init__(
        self,
        model_size: str = "base",
        language: Optional[str] = None,
        device: Optional[str] = None,
        beam_size: int = 5,
        best_of: int = 5,
        temperature: str = "0,0.2,0.4",
        initial_prompt: Optional[str] = None,
        condition_on_previous_text: bool = True,
        fp16: Optional[bool] = None,
    ):
        """
        Args:
            model_size: Kích thước model ('tiny', 'base', 'small', 'medium', 'large')
            language: Mã ngôn ngữ (VD: 'en', 'vi', 'ja'). None = auto detect
            device: 'cuda', 'cpu', hoặc None (auto detect)
        """
        self.model_size = model_size
        self.language = language
        self.device = device
        self.beam_size = beam_size
        self.best_of = best_of
        self.temperature = tuple(float(value.strip()) for value in temperature.split(",") if value.strip())
        self.initial_prompt = initial_prompt
        self.condition_on_previous_text = condition_on_previous_text
        self.fp16 = fp16
        
        logger.info(f"Loading Whisper model '{model_size}'...")
        if whisper is None:
            raise RuntimeError(
                "openai-whisper is not installed. Install requirements.txt or use "
                "the faster-whisper path when implemented."
            )
        self.model = whisper.load_model(model_size, device=device)
        logger.info(f"Model loaded successfully. Device: {self.model.device}")
    
    @staticmethod
    def _seconds_to_timestamp(seconds: float) -> str:
        """Chuyển đổi seconds thành HH:MM:SS format"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    
    def extract_transcript(
        self,
        video_path: str,
        output_json_path: Optional[str] = None
    ) -> Dict:
        """
        Extract transcript từ một video file
        
        Args:
            video_path: Đường dẫn đến file video
            output_json_path: Đường dẫn output JSON. None = không lưu file
            
        Returns:
            Dictionary chứa transcript data với format:
            {
                "video_id": str,
                "language": str,
                "segments": [
                    {
                        "id": int,
                        "start": float,
                        "end": float,
                        "text": str
                    },
                    ...
                ]
            }
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        video_id = video_path.stem
        logger.info(f"Transcribing video: {video_id}")
        
        # Whisper transcription options
        transcribe_options = {
            "verbose": False,
            "language": self.language,
            "beam_size": self.beam_size,
            "best_of": self.best_of,
            "temperature": self.temperature,
            "condition_on_previous_text": self.condition_on_previous_text,
            "task": "transcribe",  # 'transcribe' hoặc 'translate'
        }
        
        if self.fp16 is not None:
            transcribe_options["fp16"] = self.fp16
        if self.initial_prompt:
            transcribe_options["initial_prompt"] = self.initial_prompt

        # Run Whisper
        result = self.model.transcribe(str(video_path), **transcribe_options)
        
        # Format output
        transcript_data = {
            "video_id": video_id,
            "language": result.get("language", "unknown"),
            "duration": round(result.get("duration", 0), 3),
            "model_name": self.model_size,
            "asr_options": {
                "beam_size": self.beam_size,
                "best_of": self.best_of,
                "temperature": list(self.temperature),
                "condition_on_previous_text": self.condition_on_previous_text,
                "initial_prompt": self.initial_prompt,
            },
            "segments": []
        }
        
        for segment in result["segments"]:
            transcript_data["segments"].append({
                "id": segment["id"],
                "start": round(segment["start"], 3),
                "end": round(segment["end"], 3),
                "text": segment["text"].strip(),
                "timestamp": self._seconds_to_timestamp(segment["start"])
            })
        
        # Save to JSON if output path provided
        if output_json_path:
            output_path = Path(output_json_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(transcript_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Transcript saved to: {output_path}")
        
        return transcript_data
    
    def batch_extract(
        self,
        videos_dir: str,
        output_dir: str,
        video_pattern: str = "*.mp4",
        skip_existing: bool = True
    ) -> List[str]:
        """
        Batch extract transcripts từ nhiều video
        
        Args:
            videos_dir: Thư mục chứa video files
            output_dir: Thư mục output cho JSON files
            video_pattern: Glob pattern cho video files
            skip_existing: Bỏ qua video đã có transcript
            
        Returns:
            List các video_id đã được xử lý
        """
        videos_dir = Path(videos_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        video_files = sorted(videos_dir.glob(video_pattern))
        
        if not video_files:
            logger.warning(f"No video files found in {videos_dir} matching '{video_pattern}'")
            return []
        
        logger.info(f"Found {len(video_files)} video files to process")
        
        processed_ids = []
        
        for video_file in tqdm(video_files, desc="Extracting transcripts"):
            video_id = video_file.stem
            output_json = output_dir / f"{video_id}.json"
            
            # Skip if already exists
            if skip_existing and output_json.exists():
                logger.info(f"Skipping {video_id} (transcript already exists)")
                continue
            
            try:
                self.extract_transcript(
                    video_path=str(video_file),
                    output_json_path=str(output_json)
                )
                processed_ids.append(video_id)
                
            except Exception as e:
                logger.error(f"Error processing {video_id}: {e}")
                continue
        
        logger.info(f"Batch extraction complete. Processed {len(processed_ids)} videos.")
        return processed_ids


def main():
    parser = argparse.ArgumentParser(
        description="Extract transcripts từ video sử dụng Whisper"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default=config.ASR_MODEL,
        choices=["tiny", "base", "small", "medium", "large", "large-v2", "large-v3"],
        help="Whisper model size (default: base)"
    )
    
    parser.add_argument(
        "--language",
        type=str,
        default=config.ASR_LANGUAGE or None,
        help="Language code (e.g., 'en', 'vi', 'ja'). None = auto detect"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cuda", "cpu"],
        help="Device to run on. None = auto detect"
    )
    
    parser.add_argument(
        "--videos-dir",
        type=str,
        default=config.VIDEOS_DIR,
        help=f"Directory containing video files (default: {config.VIDEOS_DIR})"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default=config.TRANSCRIPTS_DIR,
        help=f"Output directory for JSON files (default: {config.TRANSCRIPTS_DIR})"
    )
    
    parser.add_argument(
        "--video-pattern",
        type=str,
        default="*.mp4",
        help="Glob pattern for video files (default: *.mp4)"
    )
    
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        default=True,
        help="Skip videos that already have transcripts"
    )
    
    parser.add_argument(
        "--single-video",
        type=str,
        default=None,
        help="Process only a single video (video_id or full path)"
    )
    parser.add_argument(
        "--beam-size",
        type=int,
        default=5,
        help="Beam size for Whisper decoding (higher can improve Vietnamese accuracy)."
    )
    parser.add_argument(
        "--best-of",
        type=int,
        default=5,
        help="Number of candidates for non-beam sampling fallback."
    )
    parser.add_argument(
        "--temperature",
        default="0,0.2,0.4",
        help="Comma-separated temperature fallback values."
    )
    parser.add_argument(
        "--initial-prompt",
        default=None,
        help="Prompt to bias transcription vocabulary."
    )
    parser.add_argument(
        "--vietnamese-prompt",
        action="store_true",
        help="Use a default Vietnamese prompt to reduce common word/name errors."
    )
    parser.add_argument(
        "--no-condition-on-previous-text",
        action="store_true",
        help="Disable conditioning on previous text; useful when hallucinations repeat."
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Force fp16 decoding."
    )
    parser.add_argument(
        "--fp32",
        action="store_true",
        help="Force fp32 decoding."
    )
    
    args = parser.parse_args()
    
    # Initialize extractor
    extractor = WhisperTranscriptExtractor(
        model_size=args.model,
        language=args.language,
        device=args.device,
        beam_size=args.beam_size,
        best_of=args.best_of,
        temperature=args.temperature,
        initial_prompt=DEFAULT_VIETNAMESE_PROMPT if args.vietnamese_prompt else args.initial_prompt,
        condition_on_previous_text=not args.no_condition_on_previous_text,
        fp16=True if args.fp16 else (False if args.fp32 else None),
    )
    
    # Single video mode
    if args.single_video:
        video_path = Path(args.single_video)
        
        # If only video_id provided, construct full path
        if not video_path.exists():
            video_path = Path(args.videos_dir) / f"{args.single_video}.mp4"
        
        if not video_path.exists():
            logger.error(f"Video not found: {video_path}")
            return
        
        output_json = Path(args.output_dir) / f"{video_path.stem}.json"
        
        extractor.extract_transcript(
            video_path=str(video_path),
            output_json_path=str(output_json)
        )
        
        logger.info(f"✅ Transcript extracted successfully for {video_path.stem}")
    
    # Batch mode
    else:
        processed = extractor.batch_extract(
            videos_dir=args.videos_dir,
            output_dir=args.output_dir,
            video_pattern=args.video_pattern,
            skip_existing=args.skip_existing
        )
        
        logger.info(f"✅ Batch extraction complete. {len(processed)} videos processed.")


if __name__ == "__main__":
    main()
