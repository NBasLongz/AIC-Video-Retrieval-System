# Project Structure

```text
Multi_Retrieval_System/
├── backend/
│   ├── app.py                  # Flask API, React dist serving, submit proxy
│   ├── retrieval_system.py     # Hybrid retrieval, RRF fusion, optional rerank
│   ├── ingest_data.py          # Milvus/Elasticsearch ingestion
│   └── config.py               # Environment-driven runtime config
├── frontend/
│   ├── src/
│   │   ├── app/                # App shell
│   │   ├── pages/              # Retrieval page
│   │   ├── features/retrieval/ # AIC retrieval UI feature
│   │   ├── components/ui/      # Shared UI primitives
│   │   ├── hooks/              # Global React hooks
│   │   ├── lib/                # API client and config helpers
│   │   └── styles/             # Tailwind globals
│   ├── package.json
│   └── vite.config.ts
├── scripts/
│   ├── extract_keyframes.py
│   ├── compute_embeddings.py
│   ├── extract_text_from_keyframes.py
│   ├── extract_transcripts.py
│   └── validate_pipeline.py
├── utils/
│   ├── elasticsearch_client.py
│   ├── text_encoder.py
│   └── video_metadata.py
├── data/
│   ├── videos/
│   ├── keyframes/
│   ├── embeddings/
│   ├── transcripts/
│   ├── ocr/
│   └── captions/
├── docker-compose.yml
├── requirements.txt
├── README.md
└── STRUCTURE.md
```

## Runtime Flow

```text
React/Vite UI
  -> POST /search
  -> backend/app.py
  -> backend/retrieval_system.py
  -> Milvus dense retrieval + Elasticsearch OCR/transcript/caption retrieval
  -> RRF fusion + optional rerank_top_k
  -> ranked frames with score breakdown
  -> React modal, nearby frames, shortlist, submit history
  -> POST /api/submit
```

## Frontend Feature Layout

```text
frontend/src/features/retrieval/
├── api/
│   ├── retrievalApi.ts
│   ├── submitApi.ts
│   └── videoApi.ts
├── components/
│   ├── SearchHeader.tsx
│   ├── ModeTabs.tsx
│   ├── QuickAssistBar.tsx
│   ├── VideoList.tsx
│   ├── ResultGrid.tsx
│   ├── ResultCard.tsx
│   ├── VideoModal.tsx
│   ├── NearbyFrames.tsx
│   ├── ScoreBreakdown.tsx
│   └── OcrChips.tsx
├── hooks/
├── types/
├── constants/
└── utils/
```

## When To Edit What

| Goal | Main files |
| --- | --- |
| Change ranking/fusion | `backend/retrieval_system.py` |
| Add/adjust API payload | `backend/app.py`, `frontend/src/features/retrieval/api/` |
| Change frontend retrieval UI | `frontend/src/features/retrieval/components/` |
| Change frontend page state | `frontend/src/pages/RetrievalPage.tsx` |
| Validate data contracts | `scripts/validate_pipeline.py` |
| Change embedding model/dimension | `backend/config.py`, `scripts/compute_embeddings.py`, `backend/ingest_data.py` |
