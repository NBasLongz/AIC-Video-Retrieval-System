# Data Directory

This directory is committed only as a project scaffold. Runtime files are intentionally ignored by Git.

Keep local/private files here:

```text
data/videos/        input videos, not committed
data/keyframes/     extracted keyframes and timestamp maps, not committed
data/embeddings/    SigLIP2 keyframe vectors, not committed
data/transcripts/   Whisper/ASR transcripts, not committed
data/ocr_result/    OCR JSON artifacts, not committed
data/captions/      optional VLM captions/tags, not committed
```

The repository tracks `.gitkeep` files so the expected folder structure exists after clone, but it does not track videos, generated keyframes, embeddings, transcripts, OCR outputs, captions, or benchmark run artifacts.
