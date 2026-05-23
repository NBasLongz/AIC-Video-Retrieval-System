"""Measure retrieval API latency without ground-truth annotations.

Run after the Flask backend is up:

    python -m scripts.benchmark_latency
    python -m scripts.benchmark_latency --url http://localhost:5000 --queries "người áo trắng" "chữ pharmacy"
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any

try:
    import requests
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Install requests first: pip install requests") from exc


DEFAULT_QUERIES = [
    "người đứng gần xe màu đỏ",
    "biển hiệu có chữ pharmacy",
    "người mặc áo trắng đang nói chuyện",
]


def load_queries(path: str | None, inline_queries: list[str]) -> list[str]:
    if inline_queries:
        return inline_queries
    if not path:
        return DEFAULT_QUERIES

    query_path = Path(path)
    queries: list[str] = []
    for line in query_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("{"):
            queries.append(json.loads(line)["query"])
        else:
            queries.append(line)
    return queries or DEFAULT_QUERIES


def post_json(url: str, payload: dict[str, Any], timeout: float) -> tuple[int, Any, float]:
    started = time.perf_counter()
    response = requests.post(url, json=payload, timeout=timeout)
    elapsed_ms = (time.perf_counter() - started) * 1000
    try:
        body = response.json()
    except ValueError:
        body = response.text
    return response.status_code, body, elapsed_ms


def main():
    parser = argparse.ArgumentParser(description="Benchmark retrieval API latency")
    parser.add_argument("--url", default="http://localhost:5000", help="Backend base URL")
    parser.add_argument("--query-file", default=None, help="Text file or JSONL file with queries")
    parser.add_argument("--queries", nargs="*", default=[], help="Inline query list")
    parser.add_argument("--timeout", type=float, default=30.0, help="HTTP timeout seconds")
    parser.add_argument("--top-k", type=int, default=50, help="Rerank top-K for rerank run")
    args = parser.parse_args()

    base_url = args.url.rstrip("/")
    queries = load_queries(args.query_file, args.queries)
    rows: list[dict[str, Any]] = []

    health = requests.get(f"{base_url}/api/health", timeout=args.timeout)
    print(f"health_status={health.status_code}")

    for query in queries:
        for label, rerank_top_k in (("no_rerank", 0), ("rerank", args.top_k)):
            payload = {
                "description": query,
                "rerank_top_k": rerank_top_k,
                "neighbor_seconds": [-5, -3, 0, 3, 5],
            }
            status, body, elapsed_ms = post_json(f"{base_url}/search", payload, args.timeout)
            result_count = len(body) if isinstance(body, list) else 0
            row = {
                "query": query,
                "mode": label,
                "rerank_top_k": rerank_top_k,
                "status": status,
                "num_results": result_count,
                "latency_ms": round(elapsed_ms, 2),
            }
            rows.append(row)
            print(json.dumps(row, ensure_ascii=False))

    latencies = [row["latency_ms"] for row in rows if row["status"] == 200]
    if latencies:
        print(
            json.dumps(
                {
                    "count": len(latencies),
                    "avg_ms": round(statistics.mean(latencies), 2),
                    "p50_ms": round(statistics.median(latencies), 2),
                    "max_ms": round(max(latencies), 2),
                },
                ensure_ascii=False,
            )
        )


if __name__ == "__main__":
    main()
