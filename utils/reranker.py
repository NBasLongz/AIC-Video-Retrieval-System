"""Optional reranking stage.

The default provider is "none" to keep the current repo lightweight. Set
RERANK_MODEL_PROVIDER=sentence_transformers and RERANK_MODEL to a CrossEncoder
model such as BAAI/bge-reranker-v2-m3 to enable it.
"""

from __future__ import annotations

import logging
from typing import Any

from backend import config

logger = logging.getLogger(__name__)


def _candidate_context(item: dict[str, Any]) -> str:
    parts = [
        f"video_id: {item.get('video_id')}",
        f"time: {item.get('start_seconds', item.get('start', 0))}",
    ]
    for field in ("ocr_text", "transcript_text", "caption_text", "text"):
        value = item.get(field)
        if value:
            parts.append(f"{field}: {value}")
    return "\n".join(parts)


class OptionalReranker:
    def __init__(self):
        self.provider = config.RERANK_MODEL_PROVIDER
        self.model_name = config.RERANK_MODEL_NAME
        self._model = None

    @property
    def enabled(self) -> bool:
        return self.provider not in {"", "none", "disabled", "off"}

    def _ensure_loaded(self):
        if not self.enabled or self._model is not None:
            return

        if self.provider in {"sentence_transformers", "cross_encoder", "bge"}:
            try:
                from sentence_transformers import CrossEncoder
            except ImportError as exc:
                logger.warning("Reranker disabled; sentence-transformers is not installed: %s", exc)
                self.provider = "none"
                return

            logger.info("Loading reranker model: %s", self.model_name)
            self._model = CrossEncoder(self.model_name)
            return

        logger.warning("Unsupported reranker provider '%s'; reranker disabled.", self.provider)
        self.provider = "none"

    def rerank(self, query: str, candidates: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
        if not query or not candidates or not self.enabled:
            return candidates

        self._ensure_loaded()
        if self._model is None:
            return candidates

        head = candidates[:top_k]
        tail = candidates[top_k:]
        pairs = [(query, _candidate_context(item)) for item in head]

        try:
            scores = self._model.predict(pairs)
        except Exception as exc:
            logger.warning("Reranker prediction failed; returning fused order: %s", exc)
            return candidates

        for item, score in zip(head, scores):
            item["rerank_score"] = float(score)

        head.sort(
            key=lambda item: (
                float(item.get("rerank_score") or 0.0),
                float(item.get("fusion_score") or 0.0),
            ),
            reverse=True,
        )
        return head + tail

