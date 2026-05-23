"""Optional query translation for visual retrieval.

The UI can stay Vietnamese while dense visual search receives an English query.
OCR and transcript search still keep the original text so brand names, signs,
and spoken Vietnamese remain searchable.
"""

from __future__ import annotations

import logging
from functools import lru_cache

import torch

from backend import config
from utils.query_processing import detect_language_hint, normalize_query

logger = logging.getLogger(__name__)


class QueryTranslator:
    def __init__(self):
        self.provider = config.QUERY_TRANSLATION_PROVIDER
        self.model_name = config.QUERY_TRANSLATION_MODEL
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._tokenizer = None
        self._model = None

    @property
    def enabled(self) -> bool:
        return config.ENABLE_QUERY_TRANSLATION and self.provider not in {"", "none", "off", "disabled"}

    def _ensure_loaded(self):
        if not self.enabled or self._model is not None:
            return

        if self.provider != "nllb":
            logger.warning("Unsupported translation provider '%s'; translation disabled.", self.provider)
            self.provider = "none"
            return

        try:
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        except ImportError as exc:
            logger.warning("Translation disabled; transformers is not installed: %s", exc)
            self.provider = "none"
            return

        logger.info("Loading query translator: %s on %s", self.model_name, self.device)
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            src_lang=config.QUERY_TRANSLATION_SRC_LANG,
        )
        self._model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name).to(self.device)
        self._model.eval()

    @lru_cache(maxsize=4096)
    def translate_vi_to_en(self, query: str) -> str:
        normalized = normalize_query(query)
        if not normalized or not self.enabled:
            return normalized

        if detect_language_hint(normalized) != "vi":
            return normalized

        self._ensure_loaded()
        if self._model is None or self._tokenizer is None:
            return normalized

        try:
            inputs = self._tokenizer(normalized, return_tensors="pt", truncation=True).to(self.device)
            forced_bos_token_id = self._tokenizer.convert_tokens_to_ids(config.QUERY_TRANSLATION_TGT_LANG)
            with torch.no_grad():
                outputs = self._model.generate(
                    **inputs,
                    forced_bos_token_id=forced_bos_token_id,
                    max_length=config.QUERY_TRANSLATION_MAX_LENGTH,
                )
            translated = self._tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
            translated = normalize_query(translated)
            logger.info("Translated visual query: %r -> %r", normalized, translated)
            return translated or normalized
        except Exception as exc:
            logger.warning("Query translation failed; using original query: %s", exc)
            return normalized
