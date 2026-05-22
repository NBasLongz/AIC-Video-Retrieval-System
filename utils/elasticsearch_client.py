from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any, Dict

from elasticsearch import BadRequestError, Elasticsearch

from backend import config

logger = logging.getLogger(__name__)

@lru_cache(maxsize=1)
def get_elasticsearch_client() -> Elasticsearch:
    """Return a cached Elasticsearch client configured via config.py."""

    host = {
        "host": config.ELASTIC_HOST,
        "port": int(config.ELASTIC_PORT),
        "scheme": config.ELASTIC_SCHEME,
    }
    return Elasticsearch(hosts=[host], request_timeout=30)


def transcript_index_mapping() -> Dict[str, Any]:
    """Default mapping/settings for the transcript index."""

    return {
        "settings": {
            "analysis": {
                "analyzer": {
                    "transcript_text": {
                        "type": "custom",
                        "tokenizer": "standard",
                        "filter": ["lowercase", "asciifolding"],
                    }
                }
            }
        },
        "mappings": {
            "properties": {
                "video_id": {"type": "keyword"},
                "keyframe_index": {"type": "integer"},
                "start": {"type": "float"},
                "end": {"type": "float"},
                "doc_type": {"type": "keyword"},
                "source_type": {"type": "keyword"},
                "language": {"type": "keyword"},
                "model_name": {"type": "keyword"},
                "confidence": {"type": "float"},
                "metadata": {"type": "object", "enabled": True},
                "text": {
                    "type": "text",
                    "analyzer": "transcript_text",
                    "fields": {"as_you_type": {"type": "search_as_you_type"}},
                },
                "caption": {
                    "type": "text",
                    "analyzer": "transcript_text",
                    "fields": {"as_you_type": {"type": "search_as_you_type"}},
                },
                "ocr_text": {
                    "type": "text",
                    "analyzer": "transcript_text",
                    "fields": {"as_you_type": {"type": "search_as_you_type"}},
                },
            }
        },
    }


def recreate_transcript_index(client: Elasticsearch) -> None:
    """Drop and recreate the transcript index for deterministic ingestion."""

    mapping = transcript_index_mapping()

    try:
        client.indices.delete(
            index=config.TRANSCRIPT_INDEX,
            ignore_unavailable=True,
        )
    except BadRequestError as exc:
        logger.warning(
            "Failed to delete transcript index '%s' before recreate: %s",
            config.TRANSCRIPT_INDEX,
            exc,
        )

    try:
        client.indices.create(index=config.TRANSCRIPT_INDEX, body=mapping)
    except BadRequestError as exc:
        logger.error(
            "Failed to create transcript index '%s': %s",
            config.TRANSCRIPT_INDEX,
            exc,
        )
        raise
