#!/usr/bin/env python3
"""Load test a running aws_capsolver server with AwsWafClassification tasks.

The tasks are built from a locally stored grid problem (data/samples/problem_visual.json,
the raw /problem reply), so nothing but the local server is contacted.

For each concurrency level (default 1, 4, 16) N requests are sent by that many concurrent
workers; the script reports p50 / p95 / max latency, throughput and error codes.

  .venv/bin/python -m aws_capsolver serve --open &          # or with a real key
  .venv/bin/python tools/loadtest.py --requests 32
  .venv/bin/python tools/loadtest.py --url http://127.0.0.1:8790 --key MY-KEY --levels 1,8 --json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple
from urllib.parse import urlparse

import httpx

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SAMPLE = ROOT / "data" / "samples" / "problem_visual.json"
LOCAL_HOSTS = {"127.0.0.1", "localhost", "::1", "0.0.0.0"}


def build_task(sample: Path) -> Tuple[Dict[str, Any], str]:
    """Turn a stored /problem reply into an AwsWafClassification task."""
    problem = json.loads(sample.read_text(encoding="utf-8"))
    assets = problem["assets"]
    images = json.loads(assets["images"]) if isinstance(assets["images"], str) else assets["images"]
    target = assets["target"]
    target = json.loads(target)[0] if isinstance(target, str) and target.startswith("[") else target
    if isinstance(target, list):
        target = target[0]
    task = {"type": "AwsWafClassification", "question": f"aws:grid:{target}", "images": images}
    return task, str(problem.get("problem_type", "?"))


def percentile(values: List[float], pct: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    k = (len(ordered) - 1) * pct / 100.0
    lo, hi = int(k), min(int(k) + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (k - lo)


async def one_request(client: httpx.AsyncClient, url: str, body: Dict[str, Any]) -> Tuple[float, str]:
    started = time.perf_counter()
    try:
        resp = await client.post(url + "/createTask", json=body)
        data = resp.json() if resp.status_code == 200 else {}
        if resp.status_code != 200:
            outcome = f"HTTP_{resp.status_code}"
        elif data.get("errorId"):
            outcome = str(data.get("errorCode"))
        elif data.get("status") == "ready" and "solution" in data:
            outcome = "ok"
        else:
            outcome = "UNEXPECTED_REPLY"
    except httpx.TimeoutException:
        outcome = "CLIENT_TIMEOUT"
    except (httpx.HTTPError, ValueError) as exc:
        outcome = f"CLIENT_{exc.__class__.__name__}"
    return time.perf_counter() - started, outcome


async def run_level(url: str, body: Dict[str, Any], concurrency: int, total: int,
                    timeout: float) -> Dict[str, Any]:
    queue: asyncio.Queue[int] = asyncio.Queue()
    for i in range(total):
        queue.put_nowait(i)
    latencies: List[float] = []
    outcomes: Counter = Counter()

    limits = httpx.Limits(max_connections=concurrency, max_keepalive_connections=concurrency)
    async with httpx.AsyncClient(timeout=timeout, limits=limits) as client:
        async def worker() -> None:
            while True:
                try:
                    queue.get_nowait()
                except asyncio.QueueEmpty:
                    return
                latency, outcome = await one_request(client, url, body)
                outcomes[outcome] += 1
                if outcome == "ok":
                    latencies.append(latency)

        started = time.perf_counter()
        await asyncio.gather(*(worker() for _ in range(concurrency)))
        wall = time.perf_counter() - started

    return {
        "concurrency": concurrency,
        "requests": total,
        "ok": outcomes.get("ok", 0),
        "errors": {k: v for k, v in outcomes.items() if k != "ok"},
        "p50_ms": round(percentile(latencies, 50) * 1000, 1),
        "p95_ms": round(percentile(latencies, 95) * 1000, 1),
        "max_ms": round(max(latencies) * 1000, 1) if latencies else float("nan"),
        "mean_ms": round(statistics.fmean(latencies) * 1000, 1) if latencies else float("nan"),
        "throughput_rps": round(outcomes.get("ok", 0) / wall, 2) if wall > 0 else 0.0,
        "wall_s": round(wall, 2),
    }


async def main_async(args: argparse.Namespace) -> int:
    url = args.url.rstrip("/")
    host = urlparse(url).hostname or ""
    if host not in LOCAL_HOSTS and not args.allow_remote:
        print(f"refusing to load test non-local host {host!r} (pass --allow-remote)", file=sys.stderr)
        return 2
    task, ptype = build_task(Path(args.sample))
    body = {"clientKey": args.key, "task": task}
    levels = [int(x) for x in args.levels.split(",") if x.strip()]

    async with httpx.AsyncClient(timeout=10) as client:
        try:
            health = (await client.get(url + "/health")).json()
        except (httpx.HTTPError, ValueError) as exc:
            print(f"server not reachable at {url}: {exc}", file=sys.stderr)
            return 2
    if not args.json:
        print(f"server {url} version={health.get('version')} vision={health.get('vision')}")
        print(f"task: {task['question']} ({ptype}, {len(task['images'])} tiles)")

    # warm-up: the first classification may load the model (tens of seconds on CPU)
    warm = await run_level(url, body, 1, max(1, args.warmup), args.timeout)
    if warm["ok"] == 0:
        print(f"warm-up failed: {warm['errors']}", file=sys.stderr)
        if not args.continue_on_error:
            return 1
    async with httpx.AsyncClient(timeout=args.timeout) as client:
        sample_reply = (await client.post(url + "/createTask", json=body)).json()

    results = []
    for level in levels:
        total = max(level, args.requests if args.requests else level * args.per_worker)
        results.append(await run_level(url, body, level, total, args.timeout))
        if not args.json:
            r = results[-1]
            print(f"c={r['concurrency']:>3}  n={r['requests']:>4}  ok={r['ok']:>4}  "
                  f"p50={r['p50_ms']:>8.1f} ms  p95={r['p95_ms']:>8.1f} ms  "
                  f"max={r['max_ms']:>8.1f} ms  {r['throughput_rps']:>6.2f} req/s"
                  + (f"  errors={r['errors']}" if r["errors"] else ""))
    if args.json:
        print(json.dumps({"url": url, "question": task["question"], "warmup": warm,
                          "sample_reply": sample_reply, "levels": results}, indent=2))
    else:
        print(f"sample reply: {json.dumps(sample_reply)[:200]}")
    return 0 if all(r["ok"] == r["requests"] for r in results) else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://127.0.0.1:8790")
    ap.add_argument("--key", default="loadtest", help="clientKey (any value with serve --open)")
    ap.add_argument("--sample", default=str(DEFAULT_SAMPLE), help="stored /problem JSON")
    ap.add_argument("--levels", default="1,4,16", help="comma-separated concurrency levels")
    ap.add_argument("--requests", type=int, default=0,
                    help="requests per level (default: --per-worker x concurrency)")
    ap.add_argument("--per-worker", type=int, default=4)
    ap.add_argument("--warmup", type=int, default=2, help="sequential warm-up requests")
    ap.add_argument("--timeout", type=float, default=180.0, help="per-request timeout, seconds")
    ap.add_argument("--json", action="store_true", help="print machine-readable results")
    ap.add_argument("--continue-on-error", action="store_true")
    ap.add_argument("--allow-remote", action="store_true", help="allow a non-localhost --url")
    return asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    sys.exit(main())
