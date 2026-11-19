#!/usr/bin/env python3
"""Collect AWS WAF CAPTCHA problems (read-only GET /problem) for building a vision dataset.

Each grid problem is stored as data/grids/<id>/{tile_0..8.jpg, meta.json}.
Non-grid visual problems are stored under data/other/<id>/ for inspection.

Usage:
    .venv/bin/python tools/collect_samples.py --max 300 --delay 3.5
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import logging
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
UA = ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
      "Chrome/130.0.0.0 Safari/537.36")
log = logging.getLogger("collect")


@dataclass(frozen=True)
class Source:
    name: str
    base: str     # captcha base URL (without trailing /problem)
    domain: str   # protected host name sent as ?domain=


# Only this endpoint is an approved sample source (read-only GET /problem at a polite rate).
# Do not add third-party sites, and never submit answers (/verify) to a real site's WAF.
SOURCES = [
    Source("blspoland",
           "https://c7be39fcf8e1.3c71dd20.eu-central-1.captcha.awswaf.com/c7be39fcf8e1/3ad2c8da74d4/c957f8af1db3",
           "appointment.blspolandvisa.com"),
]


class FetchError(RuntimeError):
    """Raised when a problem cannot be fetched or parsed."""


def fetch_problem(session: requests.Session, src: Source, timeout: float = 20.0) -> dict:
    headers = {"Origin": f"https://{src.domain}", "Referer": f"https://{src.domain}/",
               "Accept": "application/json"}
    params = {"kind": "visual", "domain": src.domain, "locale": "en-us"}
    try:
        r = session.get(f"{src.base}/problem", params=params, headers=headers, timeout=timeout)
    except requests.RequestException as exc:
        raise FetchError(f"{src.name}: network error: {exc}") from exc
    if r.status_code != 200:
        raise FetchError(f"{src.name}: HTTP {r.status_code}")
    try:
        return r.json()
    except ValueError as exc:
        raise FetchError(f"{src.name}: invalid JSON") from exc


def save_problem(problem: dict, src: Source, out_root: Path) -> tuple[str, str] | None:
    ptype = problem.get("problem_type", "")
    assets = problem.get("assets") or {}
    raw_images = assets.get("images")
    images = json.loads(raw_images) if isinstance(raw_images, str) else (raw_images or [])
    target_raw = assets.get("target")
    targets = json.loads(target_raw) if isinstance(target_raw, str) else (target_raw or [])
    blobs = [base64.b64decode(b) for b in images]
    if not blobs:
        return None
    gid = hashlib.sha1(b"".join(blobs)).hexdigest()[:12]
    is_grid = ptype.startswith("gridcaptcha") and len(blobs) == 9
    d = out_root / ("grids" if is_grid else "other") / gid
    if d.exists():
        return gid, "dup"
    d.mkdir(parents=True)
    for i, b in enumerate(blobs):
        (d / (f"tile_{i}.jpg" if is_grid else f"image_{i}.jpg")).write_bytes(b)
    meta = {
        "id": gid, "source": src.name, "problem_type": ptype, "target": targets,
        "localized_target": (problem.get("localized_assets") or {}).get("target0"),
        "fetched_at": int(time.time()),
        "extra_assets": {k: v for k, v in assets.items() if k not in ("images",)},
    }
    (d / "meta.json").write_text(json.dumps(meta, indent=1))
    return gid, ptype


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--max", type=int, default=300, help="max number of problems to fetch")
    ap.add_argument("--cap", type=int, default=1000, help="stop when data/grids holds this many grids")
    ap.add_argument("--delay", type=float, default=3.5, help="min seconds between requests (>=3)")
    ap.add_argument("--sources", default=",".join(s.name for s in SOURCES))
    ap.add_argument("--out", default=str(ROOT / "data"))
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    delay = max(3.0, args.delay)
    srcs = [s for s in SOURCES if s.name in args.sources.split(",")]
    out_root = Path(args.out)
    (out_root / "grids").mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers["User-Agent"] = UA
    failures = {s.name: 0 for s in srcs}
    for n in range(args.max):
        if len(list((out_root / "grids").glob("*/meta.json"))) >= args.cap:
            log.info("grid cap %d reached", args.cap)
            break
        alive = [s for s in srcs if failures[s.name] < 5]
        if not alive:
            log.error("all sources failing, stopping")
            return 1
        src = alive[n % len(alive)]
        try:
            res = save_problem(fetch_problem(session, src), src, out_root)
            failures[src.name] = 0
            log.info("[%d] %s -> %s", n, src.name, res)
        except FetchError as exc:
            failures[src.name] += 1
            log.warning("[%d] %s", n, exc)
        except (ValueError, KeyError, TypeError) as exc:  # malformed payload
            log.warning("[%d] %s: bad payload: %s", n, src.name, exc)
        time.sleep(delay + random.uniform(0, 1.0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
