import csv
import logging
import os
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

import torch
from elasticsearch import Elasticsearch
from pymilvus import Collection, connections

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend import config
from utils.elasticsearch_client import get_elasticsearch_client
from utils.fusion import legacy_intersection, rrf_fusion
from utils.query_processing import build_query_plan
from utils.reranker import OptionalReranker
from utils.text_encoder import TextEncoder
from utils.video_metadata import load_video_metadata

logger = logging.getLogger(__name__)

_KEYFRAME_MAP_DIR = Path(config.KEYFRAMES_DIR) / "maps"


@lru_cache(maxsize=2048)
def _load_keyframe_seconds_map(video_id: str):
    """Load mapping of keyframe index to seconds for a given video."""

    map_path = _KEYFRAME_MAP_DIR / f"{video_id}_map.csv"
    if not map_path.exists():
        return None

    mapping = {}
    try:
        with map_path.open("r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                try:
                    frame_id = int(row.get("FrameID", "").strip())
                    seconds = float(row.get("Seconds", "").strip())
                    original_frame_raw = row.get("OriginalFrame", "")
                    original_frame = (
                        int(original_frame_raw.strip())
                        if original_frame_raw is not None and original_frame_raw.strip() != ""
                        else None
                    )
                except (ValueError, AttributeError):
                    continue
                mapping[frame_id] = (seconds, original_frame)
    except Exception as exc:  # noqa: BLE001 - log and fallback
        logger.warning("Failed to read keyframe map for %s: %s", video_id, exc)
        return None

    return mapping or None


class VideoRetrievalSystem:
    """Multimodal retrieval facade used by Flask routes.

    The class keeps dense visual retrieval, sparse text retrieval, fusion, and
    future reranking separated so each stage can be benchmarked independently.
    """

    def __init__(self, re_ingest: bool = False):
        if re_ingest:
            from backend.ingest_data import main

            main()

        logger.info("Initializing Video Retrieval System...")

        self.video_fps = load_video_metadata(config.VIDEOS_DIR)

        connections.connect("default", host=config.MILVUS_HOST, port=config.MILVUS_PORT)
        logger.info("Successfully connected to Milvus.")
        self.keyframes_collection = Collection(config.KEYFRAME_COLLECTION_NAME)
        self.collection_vector_dim = self._collection_vector_dim("keyframe_vector")
        if self.collection_vector_dim and self.collection_vector_dim != config.VECTOR_DIMENSION:
            raise ValueError(
                f"Milvus collection vector dim={self.collection_vector_dim} but "
                f"config.VECTOR_DIMENSION={config.VECTOR_DIMENSION}. Recompute embeddings "
                "or recreate the collection with the matching model dimension."
            )
        logger.info("Milvus collection ready.")

        self.es_client: Elasticsearch = get_elasticsearch_client()
        logger.info("Successfully connected to Elasticsearch.")

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.encoder = TextEncoder(device=self.device)
        self.reranker = OptionalReranker()

    def _collection_vector_dim(self, vector_field: str) -> int | None:
        try:
            for field in self.keyframes_collection.schema.fields:
                if field.name == vector_field:
                    dim = field.params.get("dim") if hasattr(field, "params") else None
                    return int(dim) if dim is not None else None
        except Exception as exc:
            logger.warning("Could not inspect Milvus collection schema: %s", exc)
        return None

    def _resolve_frame_info(self, video_id: str, keyframe_index: int) -> tuple[float, int]:
        try:
            key_idx = int(keyframe_index)
        except (TypeError, ValueError):
            key_idx = keyframe_index

        mapping = _load_keyframe_seconds_map(video_id)
        seconds_value = None
        original_frame_value = None
        if mapping and key_idx in mapping:
            try:
                seconds_candidate, original_candidate = mapping[key_idx]
                if seconds_candidate is not None:
                    seconds_value = float(seconds_candidate)
                if original_candidate is not None:
                    original_frame_value = int(original_candidate)
            except (TypeError, ValueError):
                seconds_value = None
                original_frame_value = None

        fps = self.video_fps.get(video_id, config.DEFAULT_FALLBACK_FPS)
        try:
            fps_value = float(fps)
        except (TypeError, ValueError):
            fps_value = config.DEFAULT_FALLBACK_FPS

        if fps_value <= 0:
            fps_value = config.DEFAULT_FALLBACK_FPS

        if seconds_value is None:
            try:
                seconds_value = float(key_idx) / fps_value
            except (TypeError, ValueError):
                seconds_value = 0.0

        if original_frame_value is None:
            try:
                original_frame_value = int(round(seconds_value * fps_value))
            except (TypeError, ValueError):
                original_frame_value = 0

        return seconds_value, original_frame_value

    def _with_video_metadata(self, item: dict[str, Any]) -> dict[str, Any]:
        video_id = item.get("video_id")
        fps = self.video_fps.get(video_id, config.DEFAULT_FALLBACK_FPS)
        item["fps"] = fps

        if item.get("keyframe_index") is not None:
            try:
                keyframe_index = int(item.get("keyframe_index"))
            except (TypeError, ValueError):
                return item
            start_seconds, original_frame = self._resolve_frame_info(
                str(video_id),
                keyframe_index,
            )
            item.setdefault("start", start_seconds)
            item.setdefault("start_seconds", start_seconds)
            item.setdefault("frame_number", original_frame)
        return item

    def clip_search(self, query: str = "", max_results: int | None = None) -> list[dict[str, Any]]:
        """Dense visual retrieval over keyframe embeddings."""

        max_results = max_results or config.VISUAL_MAX_RESULTS
        logger.info("Dense visual search query=%r max_results=%s", query, max_results)

        if not query:
            return []

        query_vector = self.encoder.encode(query)
        query_dim = int(query_vector.reshape(1, -1).shape[-1])
        expected_dim = self.collection_vector_dim or config.VECTOR_DIMENSION
        if query_dim != expected_dim:
            logger.error(
                "Query vector dimension mismatch: got %s, expected %s. "
                "Check VISUAL_MODEL/VECTOR_DIMENSION and Milvus collection schema.",
                query_dim,
                expected_dim,
            )
            return []
        search_params = {"metric_type": "COSINE", "params": {"nprobe": 10}}

        try:
            search_results = self.keyframes_collection.search(
                data=query_vector,
                anns_field="keyframe_vector",
                param=search_params,
                limit=max_results,
                output_fields=["video_id", "keyframe_index"],
            )
        except Exception as exc:
            logger.error("Milvus visual search failed: %s", exc, exc_info=True)
            return []

        results: list[dict[str, Any]] = []
        if search_results:
            for hit in search_results[0]:
                video_id = hit.entity.get("video_id")
                keyframe_index = hit.entity.get("keyframe_index")
                start_seconds, original_frame = self._resolve_frame_info(video_id, keyframe_index)
                score = float(hit.distance)
                results.append(
                    {
                        "video_id": video_id,
                        "keyframe_index": keyframe_index,
                        "frame_number": original_frame,
                        "start": start_seconds,
                        "start_seconds": start_seconds,
                        "clip_score": score,
                        "visual_score": score,
                        "source_type": "visual",
                    }
                )

        logger.info("Dense visual search found %s keyframes.", len(results))
        return results

    def _text_search(
        self,
        query: str,
        *,
        doc_types: list[str] | None = None,
        max_results: int | None = None,
    ) -> list[dict[str, Any]]:
        """Sparse retrieval over transcript/OCR/caption text in Elasticsearch."""

        if not query:
            return []

        max_results = max_results or config.TEXT_MAX_RESULTS
        should_queries: list[dict[str, Any]] = [
            {"match": {"text": {"query": query, "fuzziness": "AUTO", "boost": 2.0}}},
            {"match_phrase": {"text": {"query": query, "boost": 3.0}}},
            {"match": {"text.as_you_type": {"query": query}}},
            {"match": {"ocr_text": {"query": query, "fuzziness": "AUTO", "boost": 2.0}}},
            {"match_phrase": {"ocr_text": {"query": query, "boost": 3.0}}},
            {"match": {"caption": {"query": query, "fuzziness": "AUTO", "boost": 1.5}}},
            {"match_phrase": {"caption": {"query": query, "boost": 2.0}}},
        ]

        filters = []
        if doc_types:
            filters.append({"terms": {"doc_type": doc_types}})

        body_query: dict[str, Any] = {
            "bool": {
                "should": should_queries,
                "minimum_should_match": 1,
            }
        }
        if filters:
            body_query["bool"]["filter"] = filters

        try:
            response = self.es_client.search(
                index=config.TRANSCRIPT_INDEX,
                size=max_results,
                query=body_query,
                _source=[
                    "video_id",
                    "keyframe_index",
                    "start",
                    "end",
                    "text",
                    "ocr_text",
                    "caption",
                    "doc_type",
                    "source_type",
                    "language",
                    "confidence",
                    "metadata",
                ],
            )
        except Exception as exc:
            logger.error("Elasticsearch text search failed: %s", exc, exc_info=True)
            return []

        hits: list[dict[str, Any]] = []
        for hit in response.get("hits", {}).get("hits", []):
            source = hit.get("_source", {})
            doc_type = source.get("doc_type") or source.get("source_type") or "text"
            text = source.get("text") or source.get("ocr_text") or source.get("caption") or ""
            start_time = source.get("start", 0)
            keyframe_index = source.get("keyframe_index")

            item = {
                "video_id": source.get("video_id"),
                "keyframe_index": keyframe_index,
                "start": start_time,
                "start_seconds": start_time,
                "end": source.get("end"),
                "text": text,
                "doc_type": doc_type,
                "source_type": doc_type,
                "text_score": hit.get("_score"),
                "language": source.get("language"),
                "confidence": source.get("confidence"),
                "metadata": source.get("metadata") or {},
            }

            if doc_type == "transcript":
                item["transcript_text"] = text
                item["transcript_score"] = hit.get("_score")
            elif doc_type == "ocr":
                item["ocr_text"] = text
                item["ocr_score"] = hit.get("_score")
            elif doc_type == "caption":
                item["caption_text"] = text
                item["caption_score"] = hit.get("_score")

            hits.append(self._with_video_metadata(item))

        logger.info(
            "Sparse text search found %s hits for doc_types=%s.",
            len(hits),
            doc_types or "all",
        )
        return hits

    def transcript_search(self, query: str = "", max_results: int | None = None) -> list[dict[str, Any]]:
        return self._text_search(query, doc_types=["transcript"], max_results=max_results)

    def ocr_search(self, query: str = "", max_results: int | None = None) -> list[dict[str, Any]]:
        return self._text_search(query, doc_types=["ocr"], max_results=max_results)

    def caption_search(self, query: str = "", max_results: int | None = None) -> list[dict[str, Any]]:
        return self._text_search(query, doc_types=["caption"], max_results=max_results)

    def hybrid_search(self, query_data: dict[str, Any]) -> list[dict[str, Any]]:
        """Run dense + sparse retrieval, then fuse results with RRF."""

        description = query_data.get("description") or query_data.get("query") or ""
        transcript = query_data.get("transcript") or query_data.get("audio") or ""
        ocr = query_data.get("ocr") or ""
        caption = query_data.get("caption") or ""
        translated = query_data.get("translated") or query_data.get("query_en")

        primary_query = description or transcript or ocr or caption
        query_plan = build_query_plan(primary_query, translated_query=translated)

        result_sets: dict[str, list[dict[str, Any]]] = {}

        if description:
            visual_hits: list[dict[str, Any]] = []
            for variant in query_plan.variants or (description,):
                visual_hits.extend(self.clip_search(variant, max_results=config.VISUAL_MAX_RESULTS))
            result_sets["visual"] = visual_hits

        text_query = transcript or description or ocr or caption
        if text_query:
            all_text_hits = self._text_search(
                text_query,
                doc_types=None,
                max_results=config.TEXT_MAX_RESULTS,
            )
            transcript_hits = [item for item in all_text_hits if item.get("doc_type") == "transcript"]
            ocr_hits = [item for item in all_text_hits if item.get("doc_type") == "ocr"]
            caption_hits = [item for item in all_text_hits if item.get("doc_type") == "caption"]
            generic_hits = [
                item
                for item in all_text_hits
                if item.get("doc_type") not in {"transcript", "ocr", "caption"}
            ]

            if transcript_hits or transcript:
                result_sets["transcript"] = transcript_hits or self.transcript_search(text_query)
            if ocr_hits or ocr or description:
                result_sets["ocr"] = ocr_hits
            if caption_hits or caption or description:
                result_sets["caption"] = caption_hits
            if generic_hits:
                result_sets["text"] = generic_hits

        weights = {
            "visual": config.WEIGHT_VISUAL,
            "transcript": config.WEIGHT_TRANSCRIPT,
            "ocr": config.WEIGHT_OCR,
            "caption": config.WEIGHT_CAPTION,
            "text": 0.8,
        }

        fused = rrf_fusion(
            result_sets,
            k=config.RRF_K,
            weights=weights,
            limit=config.FINAL_MAX_RESULTS,
        )

        for item in fused:
            self._with_video_metadata(item)
            item.setdefault("query_language", query_plan.language_hint)

        fused = self.reranker.rerank(
            query_plan.normalized,
            fused,
            top_k=config.RERANK_TOP_K,
        )

        logger.info(
            "Hybrid search completed. sources=%s fused_results=%s",
            {key: len(value) for key, value in result_sets.items()},
            len(fused),
        )
        return fused

    def intersect(self, list_results: list[list[dict[str, Any]]]) -> list[dict[str, Any]]:
        """Legacy strict intersection kept for debugging/ablation."""

        return legacy_intersection(list_results)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    searcher = VideoRetrievalSystem()
    print(searcher.hybrid_search({"description": "person walking"})[:3])
