"""Lightweight query preprocessing for competition retrieval.

This module intentionally avoids heavyweight translation dependencies. It keeps
the original query and provides a hook for future VI-EN translation services.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass


_VIETNAMESE_MARKS = set(
    "ăâđêôơư"
    "ĂÂĐÊÔƠƯ"
    "áàảãạắằẳẵặấầẩẫậéèẻẽẹếềểễệ"
    "íìỉĩịóòỏõọốồổỗộớờởỡợ"
    "úùủũụứừửữựýỳỷỹỵ"
    "ÁÀẢÃẠẮẰẲẴẶẤẦẨẪẬÉÈẺẼẸẾỀỂỄỆ"
    "ÍÌỈĨỊÓÒỎÕỌỐỒỔỖỘỚỜỞỠỢ"
    "ÚÙỦŨỤỨỪỬỮỰÝỲỶỸỴ"
)


@dataclass(frozen=True)
class QueryPlan:
    raw: str
    normalized: str
    language_hint: str
    variants: tuple[str, ...]


def normalize_query(query: str) -> str:
    text = unicodedata.normalize("NFC", query or "")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def detect_language_hint(query: str) -> str:
    """Return a cheap language hint: vi, en, or unknown."""

    text = query or ""
    if any(ch in _VIETNAMESE_MARKS for ch in text):
        return "vi"

    # Combining marks catch Vietnamese vowels entered in decomposed form.
    normalized = unicodedata.normalize("NFD", text)
    if any(unicodedata.category(ch) == "Mn" for ch in normalized):
        return "vi"

    if re.search(r"[a-zA-Z]", text):
        return "en"
    return "unknown"


def build_query_plan(query: str, translated_query: str | None = None) -> QueryPlan:
    normalized = normalize_query(query)
    variants: list[str] = []
    if normalized:
        variants.append(normalized)

    translated = normalize_query(translated_query or "")
    if translated and translated.lower() != normalized.lower():
        variants.append(translated)

    return QueryPlan(
        raw=query or "",
        normalized=normalized,
        language_hint=detect_language_hint(normalized),
        variants=tuple(variants),
    )
