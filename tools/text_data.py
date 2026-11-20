"""Real BLS Poland captcha samples: labels, the frozen train/test split and a collector.

Real images live in data/text/real/ (gitignored); their labels are versioned in
tools/text_real_labels.csv (filename,label), labelled by eye.

Split rule (frozen, never change it): a file is TEST when the first byte of
sha1(filename) is < 192 (~75 %), else TRAIN. It depends only on the name, so new
samples land in a stable split.

Collector (read-only, polite): ``python tools/text_data.py collect --max 400`` GETs the
login page, extracts the captcha <img src>, downloads the image, waits >= 4 s between
requests and stops at the first 403 / WAF response. It never submits anything.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import re
import secrets
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REAL_DIR = ROOT / "data" / "text" / "real"
LABELS = ROOT / "tools" / "text_real_labels.csv"
BASE_URL = "https://appointment.blspolandvisa.com"
LOGIN_URL = BASE_URL + "/appointment/login"


def split_of(filename: str) -> str:
    return "test" if hashlib.sha1(filename.encode()).digest()[0] < 192 else "train"


def load_labels() -> dict[str, str]:
    with open(LABELS, newline="") as fh:
        return {r["filename"]: r["label"] for r in csv.DictReader(fh)}


def load_real(split: str | None = None) -> list[tuple[str, bytes, str]]:
    """[(filename, jpeg bytes, label)] for the labelled real samples in ``split``."""
    out = []
    for name, label in sorted(load_labels().items()):
        if split and split_of(name) != split:
            continue
        p = REAL_DIR / name
        if p.exists():
            out.append((name, p.read_bytes(), label))
    return out


def collect(max_images: int, delay: float) -> None:
    import requests

    REAL_DIR.mkdir(parents=True, exist_ok=True)
    s = requests.Session()
    s.headers["User-Agent"] = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                               "(KHTML, like Gecko) Chrome/131.0 Safari/537.36")
    got = 0
    while got < max_images:
        r = s.get(LOGIN_URL, timeout=15)
        if r.status_code != 200 or "awswaf" in r.text.lower():
            print(f"login page answered HTTP {r.status_code}; stopping (no bypass attempted)")
            return
        m = re.search(r'<img[^>]+src="([^"]*captcha[^"]*)"', r.text, re.IGNORECASE)
        if not m:
            print("no captcha <img> on the login page; stopping")
            return
        src = m.group(1)
        url = src if src.startswith("http") else BASE_URL + "/" + src.lstrip("/")
        time.sleep(delay)
        img = s.get(url, timeout=15)
        if img.status_code != 200 or len(img.content) < 100:
            print(f"captcha image answered HTTP {img.status_code}; stopping")
            return
        name = f"cap-{secrets.token_hex(6)}.jpg"
        (REAL_DIR / name).write_bytes(img.content)
        got += 1
        print(f"[{got}/{max_images}] {name}")
        time.sleep(delay)


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect")
    c.add_argument("--max", type=int, default=400)
    c.add_argument("--delay", type=float, default=4.0)
    sub.add_parser("splits")
    args = ap.parse_args()
    if args.cmd == "collect":
        collect(min(args.max, 400), max(args.delay, 4.0))
    else:
        for split in ("train", "test"):
            rows = load_real(split)
            print(split, len(rows), " ".join(f"{n}={l}" for n, _, l in rows))


if __name__ == "__main__":
    sys.exit(main())
