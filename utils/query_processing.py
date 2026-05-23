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


@dataclass(frozen=True)
class QueryDecomposition:
    raw_query: str
    visual_query: str
    ocr_query: str
    transcript_query: str
    negative_query: str


def normalize_query(query: str) -> str:
    text = unicodedata.normalize("NFC", query or "")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def strip_vietnamese_accents(text: str) -> str:
    normalized = unicodedata.normalize("NFD", text or "")
    without_marks = "".join(ch for ch in normalized if unicodedata.category(ch) != "Mn")
    return without_marks.replace("đ", "d").replace("Đ", "D")


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


def _extract_after_marker(query: str, markers: tuple[str, ...]) -> tuple[str, str]:
    text = query or ""
    lowered = strip_vietnamese_accents(text).lower()
    for marker in markers:
        marker_plain = strip_vietnamese_accents(marker).lower()
        pos = lowered.find(marker_plain)
        if pos < 0:
            continue
        before = text[:pos].strip(" ,.;:")
        after = text[pos + len(marker):].strip(" ,.;:")
        if not after:
            continue
        after = re.split(r"\b(?:và|and|nhưng|but|ở|trong|gần|near)\b", after, maxsplit=1, flags=re.IGNORECASE)[0]
        return before.strip(), after.strip(" ,.;:")
    return text, ""


def decompose_query(query: str) -> QueryDecomposition:
    """Rule-based split for competition queries.

    This is intentionally conservative: it only extracts obvious OCR/transcript
    hints and leaves the remaining text as the visual query.
    """

    normalized = normalize_query(query)
    visual_query = normalized
    negative_query = ""

    negative_match = re.search(r"\b(?:kh[oô]ng ph[aả]i|not|without|except)\b(.+)$", normalized, flags=re.IGNORECASE)
    if negative_match:
        negative_query = negative_match.group(1).strip(" ,.;:")
        visual_query = normalized[:negative_match.start()].strip(" ,.;:")

    visual_query, ocr_query = _extract_after_marker(
        visual_query,
        (
            "có chữ",
            "co chu",
            "chữ",
            "chu",
            "biển hiệu",
            "bien hieu",
            "logo",
            "text",
            "sign says",
            "word",
        ),
    )
    visual_query, transcript_query = _extract_after_marker(
        visual_query,
        (
            "nói rằng",
            "noi rang",
            "nghe thấy",
            "nghe thay",
            "lời thoại",
            "loi thoai",
            "transcript",
            "spoken",
            "says",
        ),
    )

    quoted = re.findall(r"[\"'“”‘’]([^\"'“”‘’]{2,80})[\"'“”‘’]", normalized)
    if quoted and not ocr_query:
        ocr_query = quoted[0].strip()

    return QueryDecomposition(
        raw_query=normalized,
        visual_query=visual_query,
        ocr_query=ocr_query,
        transcript_query=transcript_query,
        negative_query=negative_query,
    )
