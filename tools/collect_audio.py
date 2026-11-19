#!/usr/bin/env python3
"""Collect AWS WAF audio CAPTCHA clips (GET /problem?kind=audio, read-only).

Only the BLS Poland endpoint is used, at a polite rate. Nothing is ever submitted
(/verify is not called): labels are produced offline by tools/bench_audio.py
(multi-model agreement) plus manual review.

Layout: data/audio/clips/<id>/{audio.aac, meta.json}; meta.json carries
    {"source", "problem_type", "fetched_at"} and, once labelled, {"label", "label_how"}.

Usage:
    .venv/bin/python tools/collect_audio.py --max 50 --delay 3.5
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import logging
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
CLIPS = ROOT / "data" / "audio" / "clips"
UA = ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
      "Chrome/130.0.0.0 Safari/537.36")
TIMEOUT = 20.0
MIN_DELAY = 3.0
BASE = ("https://c7be39fcf8e1.3c71dd20.eu-central-1.captcha.awswaf.com/"
        "c7be39fcf8e1/3ad2c8da74d4/c957f8af1db3")
DOMAIN = "appointment.blspolandvisa.com"
log = logging.getLogger("collect_audio")


class CollectError(RuntimeError):
    """A problem could not be fetched or parsed."""


def fetch_problem(session: requests.Session) -> dict:
    params = {"kind": "audio", "domain": DOMAIN, "locale": "en-us"}
    headers = {"Origin": f"https://{DOMAIN}", "Referer": f"https://{DOMAIN}/", "Accept": "*/*"}
    try:
        r = session.get(f"{BASE}/problem", params=params, headers=headers, timeout=TIMEOUT)
    except requests.RequestException as exc:
        raise CollectError(f"network error: {exc}") from exc
    if r.status_code != 200:
        raise CollectError(f"/problem HTTP {r.status_code}: {r.text[:200]}")
    try:
        return r.json()
    except ValueError as exc:
        raise CollectError("/problem returned non-JSON") from exc


def save_clip(problem: dict) -> Path:
    assets = problem.get("assets")
    audio = assets.get("audio") if isinstance(assets, dict) else None
    if not audio:
        raise CollectError(f"problem without audio (type={problem.get('problem_type')})")
    try:
        blob = base64.b64decode(audio, validate=True)
    except ValueError as exc:
        raise CollectError("audio is not valid base64") from exc
    d = CLIPS / hashlib.sha1(blob).hexdigest()[:12]
    d.mkdir(parents=True, exist_ok=True)
    (d / "audio.aac").write_bytes(blob)
    meta_path = d / "meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    meta.update({"source": "blspoland", "problem_type": problem.get("problem_type"),
                 "fetched_at": int(time.time())})
    meta_path.write_text(json.dumps(meta, indent=1))
    return d


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--max", type=int, default=20)
    ap.add_argument("--delay", type=float, default=3.5)
    args = ap.parse_args()
    delay = max(args.delay, MIN_DELAY)
    session = requests.Session()
    session.headers["User-Agent"] = UA
    failures = 0
    for i in range(args.max):
        try:
            problem = fetch_problem(session)
            d = save_clip(problem)
            failures = 0
            log.info("[%d/%d] %s %s", i + 1, args.max, problem.get("problem_type"), d.name)
        except CollectError as exc:
            failures += 1
            log.warning("[%d/%d] %s", i + 1, args.max, exc)
            if failures >= 5:
                raise SystemExit("5 consecutive failures; stopping")
        time.sleep(delay * (1 + failures))


if __name__ == "__main__":
    main()
