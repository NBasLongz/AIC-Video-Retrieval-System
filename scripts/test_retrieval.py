"""Smoke-test local retrieval endpoints.

Usage:
    python scripts/test_retrieval.py
    python scripts/test_retrieval.py --base-url http://localhost:5000

The visual semantic model is very large. On low-memory machines this project can
fall back to OCR/transcript text expansion to keep visual-mode queries useful
without crashing the backend.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any


DEFAULT_TIMEOUT = 30


@dataclass(frozen=True)
class SearchCase:
    name: str
    payload: dict[str, Any]
    min_results: int = 1
    max_seconds: float = 10.0
    allow_zero_results: bool = False


def request_json(
    method: str,
    url: str,
    payload: dict[str, Any] | None = None,
    timeout: int = DEFAULT_TIMEOUT,
) -> tuple[Any, float, int]:
    body = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        body = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"

    request = urllib.request.Request(url, data=body, headers=headers, method=method)
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            elapsed = time.perf_counter() - started
            raw = response.read().decode("utf-8")
            if not raw:
                return None, elapsed, response.status
            return json.loads(raw), elapsed, response.status
    except urllib.error.HTTPError as exc:
        elapsed = time.perf_counter() - started
        raw = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} from {url}: {raw[:500]}") from exc
    except urllib.error.URLError as exc:
        elapsed = time.perf_counter() - started
        raise RuntimeError(f"Cannot reach {url} after {elapsed:.2f}s: {exc}") from exc


def request_text(url: str, timeout: int = DEFAULT_TIMEOUT) -> tuple[str, float, int]:
    started = time.perf_counter()
    with urllib.request.urlopen(url, timeout=timeout) as response:
        elapsed = time.perf_counter() - started
        return response.read().decode("utf-8", errors="replace"), elapsed, response.status


def result_summary(results: list[dict[str, Any]]) -> str:
    if not results:
        return "-"
    first = results[0]
    video_id = first.get("video_id", "")
    keyframe = first.get("keyframe_index", "")
    source = first.get("source_type") or first.get("doc_type") or ",".join(first.get("sources") or [])
    score = (
        first.get("rerank_score")
        or first.get("fusion_score")
        or first.get("ocr_score")
        or first.get("transcript_score")
        or first.get("visual_score")
    )
    if isinstance(score, (float, int)):
        return f"{video_id}/keyframe_{keyframe}/{source}/score={score:.4f}"
    return f"{video_id}/keyframe_{keyframe}/{source}"


def build_cases() -> list[SearchCase]:
    base_payload = {
        "fusion": "rrf",
        "rerank_top_k": 0,
        "neighbor_seconds": [-5, -3, 0, 3, 5],
        "explain": True,
    }
    return [
        SearchCase(
            name="ocr_htv",
            payload={**base_payload, "ocr": "HTV"},
            min_results=20,
        ),
        SearchCase(
            name="ocr_vietnam",
            payload={**base_payload, "ocr": "Vietnam"},
            min_results=1,
        ),
        SearchCase(
            name="transcript_ha_noi",
            payload={**base_payload, "transcript": "Ha Noi"},
            min_results=1,
        ),
        SearchCase(
            name="audio_ha_noi",
            payload={**base_payload, "audio": "Ha Noi"},
            min_results=1,
        ),
        SearchCase(
            name="hybrid_long_bien",
            payload={**base_payload, "description": "Long Bien", "ocr": "Long Bien", "transcript": "Long Bien"},
            min_results=1,
        ),
        SearchCase(
            name="intersection_transcript",
            payload={**base_payload, "fusion": "intersection", "transcript": "Ha Noi"},
            min_results=1,
        ),
        SearchCase(
            name="visual_person_smoke",
            payload={**base_payload, "description": "person"},
            min_results=1,
            max_seconds=8.0,
        ),
    ]


def run(args: argparse.Namespace) -> int:
    base_url = args.base_url.rstrip("/")
    failures: list[str] = []

    print(f"Testing retrieval app at {base_url}")

    health, elapsed, status = request_json("GET", f"{base_url}/api/health", timeout=args.timeout)
    models = health.get("models", {}) if isinstance(health, dict) else {}
    print(f"health: status={status} time={elapsed:.3f}s ok={health.get('ok') if isinstance(health, dict) else None}")
    print(
        "models: "
        f"rerank={models.get('rerank_provider')} "
        f"translate={models.get('query_translation_enabled')} "
        f"dense_text={models.get('dense_text_enabled')} "
        f"visual={models.get('visual_enabled')} "
        f"visual_text_fallback={models.get('visual_text_fallback_enabled')} "
        f"visual_min_ram_gb={models.get('visual_min_available_memory_gb')}"
    )
    if not isinstance(health, dict) or not health.get("ok"):
        failures.append("health endpoint did not report ok=true")

    html, page_elapsed, page_status = request_text(f"{base_url}/", timeout=args.timeout)
    page_ok = page_status == 200 and "AIC Video Retrieval" in html
    print(f"frontend: status={page_status} time={page_elapsed:.3f}s title_present={page_ok}")
    if not page_ok:
        failures.append("frontend page did not load the expected app shell")

    print()
    print(f"{'case':24} {'status':8} {'time':>8} {'count':>7} first_result")
    print("-" * 90)

    for case in build_cases():
        try:
            results, elapsed, status = request_json(
                "POST",
                f"{base_url}/search",
                payload=case.payload,
                timeout=args.timeout,
            )
            if not isinstance(results, list):
                raise RuntimeError(f"expected list response, got {type(results).__name__}")
            count = len(results)
            too_slow = elapsed > case.max_seconds
            too_few = count < case.min_results and not (case.allow_zero_results and count == 0)
            ok = status == 200 and not too_slow and not too_few
            verdict = "PASS" if ok else "FAIL"
            print(f"{case.name:24} {verdict:8} {elapsed:8.3f} {count:7d} {result_summary(results)}")
            if too_slow:
                failures.append(f"{case.name}: slow response {elapsed:.3f}s > {case.max_seconds:.3f}s")
            if too_few:
                failures.append(f"{case.name}: expected at least {case.min_results} results, got {count}")
        except Exception as exc:  # noqa: BLE001 - report every case cleanly
            print(f"{case.name:24} FAIL     {'-':>8} {'-':>7} {exc}")
            failures.append(f"{case.name}: {exc}")

    if failures:
        print()
        print("Failures:")
        for failure in failures:
            print(f" - {failure}")
        return 1

    print()
    print("All required retrieval checks passed.")
    print("Visual smoke should use dense visual when SigLIP2 is warm; fallback is only a crash guard.")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test local AIC retrieval endpoints.")
    parser.add_argument("--base-url", default="http://localhost:5000", help="Backend/frontend base URL.")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT, help="HTTP timeout in seconds.")
    return parser.parse_args()


if __name__ == "__main__":
    sys.exit(run(parse_args()))
