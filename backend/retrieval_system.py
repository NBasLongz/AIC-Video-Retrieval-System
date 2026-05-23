import csv
import logging
import os
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

import torch
from elasticsearch import Elasticsearch
from pymilvus import Collection, connections, utility

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend import config
from utils.elasticsearch_client import get_elasticsearch_client
from utils.fusion import legacy_intersection, rrf_fusion
from utils.dense_text_encoder import DenseTextEncoder
from utils.query_processing import build_query_plan
from utils.reranker import OptionalReranker
from utils.text_encoder import TextEncoder
from utils.translator import QueryTranslator
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
        self.collection_vector_dim = self._collection_vector_dim(self.keyframes_collection, "keyframe_vector")
        if self.collection_vector_dim and self.collection_vector_dim != config.VECTOR_DIMENSION:
            raise ValueError(
                f"Milvus collection vector dim={self.collection_vector_dim} but "
                f"config.VECTOR_DIMENSION={config.VECTOR_DIMENSION}. Recompute embeddings "
                "or recreate the collection with the matching model dimension."
            )
        logger.info("Milvus collection ready.")

        self.text_collection = None
        self.text_collection_vector_dim = None
        if config.ENABLE_DENSE_TEXT_RETRIEVAL and utility.has_collection(config.TEXT_COLLECTION_NAME):
            self.text_collection = Collection(config.TEXT_COLLECTION_NAME)
            self.text_collection_vector_dim = self._collection_vector_dim(self.text_collection, "text_vector")
            if self.text_collection_vector_dim and self.text_collection_vector_dim != config.TEXT_VECTOR_DIMENSION:
                raise ValueError(
                    f"Text Milvus collection vector dim={self.text_collection_vector_dim} but "
                    f"TEXT_VECTOR_DIMENSION={config.TEXT_VECTOR_DIMENSION}. Recreate "
                    "the dense text collection with backend.ingest_data."
                )
            logger.info("Dense text Milvus collection ready.")
        elif config.ENABLE_DENSE_TEXT_RETRIEVAL:
            logger.warning(
                "Dense text retrieval enabled but collection '%s' does not exist yet. "
                "Run backend.ingest_data to build it.",
                config.TEXT_COLLECTION_NAME,
            )

        self.es_client: Elasticsearch = get_elasticsearch_client()
        logger.info("Successfully connected to Elasticsearch.")

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.encoder = TextEncoder(device=self.device)
        self.text_encoder = DenseTextEncoder(device=self.device)
        self.reranker = OptionalReranker()
        self.translator = QueryTranslator()

    def _collection_vector_dim(self, collection: Collection, vector_field: str) -> int | None:
        try:
            for field in collection.schema.fields:
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

    def dense_text_search(self, query: str, max_results: int | None = None) -> list[dict[str, Any]]:
        """Dense retrieval over transcript/OCR/caption embeddings in Milvus."""

        if not query or not config.ENABLE_DENSE_TEXT_RETRIEVAL or self.text_collection is None:
            return []

        max_results = max_results or config.TEXT_DENSE_MAX_RESULTS
        try:
            query_vector = self.text_encoder.encode(query)
        except Exception as exc:
            logger.error("Dense text query encoding failed: %s", exc, exc_info=True)
            return []

        query_dim = int(query_vector.reshape(1, -1).shape[-1])
        expected_dim = self.text_collection_vector_dim or config.TEXT_VECTOR_DIMENSION
        if query_dim != expected_dim:
            logger.error(
                "Dense text query dimension mismatch: got %s, expected %s. "
                "Check TEXT_MODEL/TEXT_VECTOR_DIMENSION and Milvus text collection schema.",
                query_dim,
                expected_dim,
            )
            return []

        try:
            search_results = self.text_collection.search(
                data=query_vector,
                anns_field="text_vector",
                param={"metric_type": "COSINE", "params": {"nprobe": 10}},
                limit=max_results,
                output_fields=["video_id", "keyframe_index", "start", "end", "doc_type", "text"],
            )
        except Exception as exc:
            logger.error("Milvus dense text search failed: %s", exc, exc_info=True)
            return []

        hits: list[dict[str, Any]] = []
        if not search_results:
            return hits

        for hit in search_results[0]:
            doc_type = hit.entity.get("doc_type") or "text"
            text = hit.entity.get("text") or ""
            score = float(hit.distance)
            item = {
                "video_id": hit.entity.get("video_id"),
                "keyframe_index": hit.entity.get("keyframe_index"),
                "start": hit.entity.get("start"),
                "start_seconds": hit.entity.get("start"),
                "end": hit.entity.get("end"),
                "text": text,
                "doc_type": doc_type,
                "source_type": doc_type,
                "text_dense_score": score,
                "text_score": score,
            }
            if doc_type == "transcript":
                item["transcript_text"] = text
                item["transcript_score"] = score
            elif doc_type == "ocr":
                item["ocr_text"] = text
                item["ocr_score"] = score
            elif doc_type == "caption":
                item["caption_text"] = text
                item["caption_score"] = score

            hits.append(self._with_video_metadata(item))

        logger.info("Dense text search found %s hits.", len(hits))
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
        if not translated and description and config.ENABLE_QUERY_TRANSLATION:
            translated = self.translator.translate_vi_to_en(description)

        primary_query = description or transcript or ocr or caption
        query_plan = build_query_plan(primary_query, translated_query=translated)
        visual_query = translated or description

        result_sets: dict[str, list[dict[str, Any]]] = {}

        if description:
            visual_hits: list[dict[str, Any]] = []
            if visual_query:
                visual_hits.extend(self.clip_search(visual_query, max_results=config.VISUAL_MAX_RESULTS))
            result_sets["visual"] = visual_hits

        text_queries = []
        for value in (ocr, transcript, caption, description, translated):
            value = (value or "").strip()
            if value and value.lower() not in {item.lower() for item in text_queries}:
                text_queries.append(value)

        if text_queries:
            all_text_hits = []
            all_dense_text_hits = []
            per_query_limit = max(50, config.TEXT_MAX_RESULTS // max(1, len(text_queries)))
            per_dense_query_limit = max(30, config.TEXT_DENSE_MAX_RESULTS // max(1, len(text_queries)))
            for text_query in text_queries:
                all_text_hits.extend(
                    self._text_search(
                        text_query,
                        doc_types=None,
                        max_results=per_query_limit,
                    )
                )
                all_dense_text_hits.extend(
                    self.dense_text_search(
                        text_query,
                        max_results=per_dense_query_limit,
                    )
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
                result_sets["transcript"] = transcript_hits
            if ocr_hits or ocr or description:
                result_sets["ocr"] = ocr_hits
            if caption_hits or caption or description:
                result_sets["caption"] = caption_hits
            if generic_hits:
                result_sets["text"] = generic_hits
            if all_dense_text_hits:
                result_sets["text_dense"] = all_dense_text_hits

        weights = {
            "visual": config.WEIGHT_VISUAL,
            "transcript": config.WEIGHT_TRANSCRIPT,
            "ocr": config.WEIGHT_OCR,
            "caption": config.WEIGHT_CAPTION,
            "text": 0.8,
            "text_dense": config.WEIGHT_TEXT_DENSE,
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
            item.setdefault("query_original", primary_query)
            item.setdefault("query_visual", visual_query)
            if translated:
                item.setdefault("query_translated", translated)

        rerank_top_k = query_data.get("rerank_top_k", config.RERANK_TOP_K)
        try:
            rerank_top_k = max(0, int(rerank_top_k))
        except (TypeError, ValueError):
            rerank_top_k = config.RERANK_TOP_K

        fused = self.reranker.rerank(
            visual_query or query_plan.normalized,
            fused,
            top_k=rerank_top_k,
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
