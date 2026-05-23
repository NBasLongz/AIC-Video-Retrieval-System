from __future__ import annotations

import logging
from typing import Iterable

import numpy as np
import torch

from backend import config

logger = logging.getLogger(__name__)


class DenseTextEncoder:
    """SentenceTransformer encoder for transcript/OCR/caption dense retrieval."""

    def __init__(self, device: str = "cuda"):
        self.device = device
        self.provider = config.TEXT_MODEL_PROVIDER
        self.model_name = config.TEXT_MODEL_NAME
        self._model = None
        logger.info(
            "DenseTextEncoder created (provider=%s, model=%s, lazy loading enabled)",
            self.provider,
            self.model_name,
        )

    @property
    def enabled(self) -> bool:
        return self.provider not in {"", "none", "off", "disabled"}

    def _ensure_loaded(self):
        if self._model is not None:
            return
        if not self.enabled:
            raise RuntimeError("Dense text retrieval is disabled by TEXT_MODEL_PROVIDER")
        if self.provider not in {"sentence_transformers", "sentence-transformers", "bge_m3", "bge-m3"}:
            raise ValueError(f"Unsupported TEXT_MODEL_PROVIDER={self.provider!r}")

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "Dense text retrieval requires sentence-transformers. "
                "Install requirements.txt or disable ENABLE_DENSE_TEXT_RETRIEVAL."
            ) from exc

        logger.info("Loading dense text model '%s' on '%s'...", self.model_name, self.device)
        self._model = SentenceTransformer(
            self.model_name,
            device=self.device if torch.cuda.is_available() else "cpu",
            trust_remote_code=config.MODEL_TRUST_REMOTE_CODE,
        )

    def encode(self, texts: str | Iterable[str]) -> np.ndarray:
        self._ensure_loaded()
        is_single = isinstance(texts, str)
        batch = [texts] if is_single else list(texts)
        if not batch:
            return np.empty((0, config.TEXT_VECTOR_DIMENSION), dtype=np.float32)

        vectors = self._model.encode(
            batch,
            batch_size=32,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        vectors = np.asarray(vectors, dtype=np.float32).reshape(len(batch), -1)
        dim = vectors.shape[-1]
        if dim != config.TEXT_VECTOR_DIMENSION:
            raise ValueError(
                f"Dense text vector dimension mismatch: got {dim}, "
                f"expected TEXT_VECTOR_DIMENSION={config.TEXT_VECTOR_DIMENSION}."
            )
        return vectors[:1] if is_single else vectors
