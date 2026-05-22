"""Ranking utilities for multimodal retrieval.

The important rule here is: never compare raw scores from different engines
directly. CLIP/SigLIP cosine scores, BM25 scores, and reranker logits all have
different scales. Reciprocal Rank Fusion (RRF) uses ranks instead of raw scores,
which makes it a good default for competition systems.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, Iterable, Mapping


Result = Dict[str, Any]


def result_key(item: Mapping[str, Any]) -> tuple[str, int | str]:
    """Return the stable identity used to merge hits from multiple sources."""

    video_id = str(item.get("video_id") or item.get("videoId") or "")
    keyframe_index = item.get("keyframe_index")

    if keyframe_index is None:
        # Fallback for text-only hits that only know the approximate timestamp.
        start_seconds = item.get("start_seconds", item.get("start", 0))
        try:
            keyframe_index = int(round(float(start_seconds)))
        except (TypeError, ValueError):
            keyframe_index = "unknown"

    try:
        keyframe_index = int(keyframe_index)
    except (TypeError, ValueError):
        keyframe_index = str(keyframe_index)

    return video_id, keyframe_index


def _merge_value(existing: Any, incoming: Any) -> Any:
    if existing in (None, "", [], {}):
        return incoming
    return existing


def rrf_fusion(
    result_sets: Mapping[str, Iterable[Result]],
    *,
    k: int = 60,
    weights: Mapping[str, float] | None = None,
    limit: int | None = None,
) -> list[Result]:
    """Fuse ranked result lists with weighted Reciprocal Rank Fusion."""

    if not result_sets:
        return []

    weights = weights or {}
    buckets: dict[tuple[str, int | str], Result] = {}
    source_scores: dict[tuple[str, int | str], dict[str, float]] = defaultdict(dict)
    source_ranks: dict[tuple[str, int | str], dict[str, int]] = defaultdict(dict)

    for source, raw_results in result_sets.items():
        weight = float(weights.get(source, 1.0))
        if weight <= 0:
            continue

        for rank, item in enumerate(raw_results or [], start=1):
            key = result_key(item)
            if not key[0]:
                continue

            bucket = buckets.setdefault(
                key,
                {
                    "video_id": key[0],
                    "keyframe_index": key[1],
                    "fusion_score": 0.0,
                    "sources": [],
                    "source_ranks": {},
                    "source_scores": {},
                },
            )

            bucket["fusion_score"] += weight / (k + rank)
            if source not in bucket["sources"]:
                bucket["sources"].append(source)
            source_ranks[key][source] = rank

            for field_name, value in item.items():
                if field_name in {"fusion_score", "sources", "source_ranks", "source_scores"}:
                    continue
                bucket[field_name] = _merge_value(bucket.get(field_name), value)

            for score_field in (
                "clip_score",
                "visual_score",
                "transcript_score",
                "ocr_score",
                "caption_score",
                "text_score",
                "rerank_score",
            ):
                if score_field in item and item[score_field] is not None:
                    source_scores[key][score_field] = item[score_field]

    for key, bucket in buckets.items():
        bucket["source_ranks"] = source_ranks[key]
        bucket["source_scores"] = source_scores[key]

    fused = sorted(
        buckets.values(),
        key=lambda item: (
            float(item.get("fusion_score") or 0.0),
            float(item.get("clip_score") or item.get("visual_score") or 0.0),
        ),
        reverse=True,
    )

    if limit is not None:
        return fused[:limit]
    return fused


def legacy_intersection(result_sets: list[list[Result]]) -> list[Result]:
    """Keep the old strict intersection behavior as an explicit option."""

    if not result_sets:
        return []
    if len(result_sets) == 1:
        return result_sets[0]

    lookup_map = {result_key(item): item for item in result_sets[0]}
    intersecting_ids = set(lookup_map.keys())

    for other_list in result_sets[1:]:
        intersecting_ids &= {result_key(item) for item in other_list}
        if not intersecting_ids:
            break

    return [lookup_map[key] for key in intersecting_ids]

