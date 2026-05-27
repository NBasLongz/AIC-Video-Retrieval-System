# Quickstart

This guide is written for a fresh clone and for daily competition startup. Commands are portable and should be run from the repository root unless noted.

## 1. Requirements

Install these first:

- Python 3.10 or 3.11
- Node.js 20+
- Docker Desktop or Docker Engine
- Git

Recommended:

- NVIDIA GPU and CUDA for faster preprocessing/search warm-up
- At least 30 GB free space on the drive used for model cache and data

Check versions:

```powershell
python --version
node --version
npm --version
docker --version
```

If your machine has more than one Python installation, use the full path to the Python you want, or create a virtual environment with that Python.

## 2. First-Time Setup

### 2.1. Clone and enter the repository

```powershell
git clone <repo-url>
cd Multi_Retrieval_System
```

### 2.2. Create a Python virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Linux/macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

After activation, verify that `python` points to the virtual environment:

```powershell
python -c "import sys; print(sys.executable)"
```

### 2.3. Install frontend dependencies

```powershell
cd frontend
npm install
cd ..
```

### 2.4. Configure model cache location

The backend auto-selects a cache directory and prefers the repository drive when it has enough free space. To force a cache location, set `AIC_CACHE_DIR`.

Windows PowerShell example:

```powershell
$env:AIC_CACHE_DIR = "D:\aic_cache"
$env:AIC_CACHE_MIN_GB = "30"
```

Linux/macOS example:

```bash
export AIC_CACHE_DIR=/data/aic_cache
export AIC_CACHE_MIN_GB=30
```

If not set, the backend chooses a suitable drive automatically.

### 2.5. Start database services

```powershell
docker compose up -d etcd minio standalone elasticsearch redis
docker compose ps
```

Expected services:

- `milvus-standalone`
- `milvus-etcd`
- `milvus-minio`
- `es01`
- `retrieval-redis`

### 2.6. Add videos

Put videos here:

```text
data/videos/<video_id>.mp4
```

Example:

```text
data/videos/L03_V004.mp4
data/videos/L03_V006.mp4
data/videos/L03_V013.mp4
```

The file name without `.mp4` is used as `video_id`.

## 3. Build or Rebuild Data

Run this section only when setting up data for the first time or after adding/replacing videos.

### 3.1. Extract keyframes

```powershell
python -m scripts.extract_keyframes --method interval --interval 1.0 --overwrite --ensure-compatible
```

### 3.2. Extract OCR

```powershell
python -m scripts.extract_text_from_keyframes --engine paddleocr --languages en,vi
```

### 3.3. Extract transcripts

CPU:

```powershell
python -m scripts.extract_transcripts --model large-v3 --language vi --vietnamese-prompt
```

CUDA:

```powershell
python -m scripts.extract_transcripts --model large-v3 --language vi --vietnamese-prompt --device cuda
```

### 3.4. Compute visual embeddings

CPU:

```powershell
python -m scripts.compute_embeddings --batch-size 8 --device cpu
```

CUDA:

```powershell
python -m scripts.compute_embeddings --batch-size 32 --device cuda
```

### 3.5. Ingest into Milvus and Elasticsearch

```powershell
python -m backend.ingest_data
```

### 3.6. Validate services and data

```powershell
python -m scripts.validate_pipeline --check-services
```

## 4. Daily Competition Startup

Use this section every time you want to run the system. If data is already built and ingested, do not rerun extraction or embedding.

### 4.1. Open Docker and start services

```powershell
docker compose up -d etcd minio standalone elasticsearch redis
docker compose ps
```

### 4.2. Activate Python environment

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Linux/macOS:

```bash
source .venv/bin/activate
```

### 4.3. Start backend

Windows PowerShell:

```powershell
$env:VISUAL_MIN_AVAILABLE_MEMORY_GB = "0"
$env:ENABLE_VISUAL_RETRIEVAL = "true"
$env:ENABLE_VISUAL_TEXT_FALLBACK = "true"
$env:VISUAL_TEXT_ONLY_MODEL = "true"
$env:VISUAL_STREAM_SAFE_LOAD = "true"
$env:VISUAL_MODEL_DTYPE = "float16"
$env:VISUAL_LOW_CPU_MEM_USAGE = "true"
$env:ENABLE_DENSE_TEXT_RETRIEVAL = "false"
$env:ENABLE_QUERY_TRANSLATION = "false"
$env:RERANK_MODEL_PROVIDER = "none"
$env:RERANK_TOP_K = "0"
$env:FLASK_DEBUG = "0"

python backend/app.py
```

Linux/macOS:

```bash
export VISUAL_MIN_AVAILABLE_MEMORY_GB=0
export ENABLE_VISUAL_RETRIEVAL=true
export ENABLE_VISUAL_TEXT_FALLBACK=true
export VISUAL_TEXT_ONLY_MODEL=true
export VISUAL_STREAM_SAFE_LOAD=true
export VISUAL_MODEL_DTYPE=float16
export VISUAL_LOW_CPU_MEM_USAGE=true
export ENABLE_DENSE_TEXT_RETRIEVAL=false
export ENABLE_QUERY_TRANSLATION=false
export RERANK_MODEL_PROVIDER=none
export RERANK_TOP_K=0
export FLASK_DEBUG=0

python backend/app.py
```

Backend URL:

```text
http://localhost:5000
```

Health check:

```powershell
Invoke-RestMethod http://localhost:5000/api/health
```

### 4.4. Start frontend realtime

Open a second terminal.

```powershell
cd frontend
npm run dev
```

Frontend URL:

```text
http://localhost:5173
```

The Vite dev server proxies API calls to `http://localhost:5000`.

### 4.5. Optional warm-up before competition

The first visual query after backend restart can be slow because the visual text encoder loads model weights into memory. Run one warm-up query before competition starts.

PowerShell:

```powershell
$body = @{
  fusion = "rrf"
  rerank_top_k = 0
  neighbor_seconds = @(-4,-3,-2,-1,0,1,2,3,4)
  explain = $true
  description = "a person is standing in front of the display"
} | ConvertTo-Json -Compress

Invoke-RestMethod -Uri http://localhost:5000/search -Method Post -ContentType "application/json" -Body $body -TimeoutSec 300
```

Bash:

```bash
curl -X POST http://localhost:5000/search \
  -H "Content-Type: application/json" \
  -d '{"fusion":"rrf","rerank_top_k":0,"neighbor_seconds":[-4,-3,-2,-1,0,1,2,3,4],"explain":true,"description":"a person is standing in front of the display"}'
```

## 5. Search Modes

- `Visual`: use English visual scene descriptions. This runs visual search only.
- `OCR`: use this for text visible in frames.
- `Transcript`: use this for spoken or transcript text.
- `Hybrid`: combines signals. It can help for mixed clues but may be slower or noisier.
- `Audio`: use when searching audio transcript fields.

For fastest competition use:

- Keep rerank off unless needed.
- Keep query translation off if you search in English.
- Prefer `Visual`, `OCR`, or `Transcript` over `Hybrid` when the clue clearly belongs to one signal.

## 6. Static Frontend Build

For production-like static serving through Flask:

```powershell
cd frontend
npm run build
cd ..
python backend/app.py
```

Open:

```text
http://localhost:5000
```

For active development and UI testing, use `http://localhost:5173`.

## 7. Troubleshooting

### Backend starts with the wrong Python

Check:

```powershell
python -c "import sys; print(sys.executable)"
```

If it does not point to `.venv`, activate the virtual environment again.

### Search returns no result or keeps loading

Check backend:

```powershell
Invoke-RestMethod http://localhost:5000/api/health
```

Check logs:

```powershell
Get-Content system.log -Tail 80
```

If the log says the text encoder is loading, wait for the first warm-up query to finish.

### Port already in use

Windows:

```powershell
Get-NetTCPConnection -LocalPort 5000,5173 -ErrorAction SilentlyContinue
```

Linux/macOS:

```bash
lsof -i :5000
lsof -i :5173
```

### Docker services are down

```powershell
docker compose ps
docker compose restart standalone elasticsearch redis
```

### Drive C is filling up on Windows

Set `AIC_CACHE_DIR` to a drive with more free space before starting backend.

Example:

```powershell
$env:AIC_CACHE_DIR = "F:\aic_cache"
```

Model cache should not stay in:

```text
C:\Users\<user>\.cache\huggingface
```
