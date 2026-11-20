"""Evaluate BLS Poland readers on the labelled real samples.

Readers:
  ddddocr        one plain ``classification()`` call, digits kept
  ddddocr_vote   the BlsPolandBott solver: 16 preprocessing variants, keep 5-digit
                 readings, majority vote (utils/captcha_helper.py, replicated here)
  ours           aws_capsolver.text.bls_poland (weights/bls_poland_ocr.pt or --weights)

Reports exact-match accuracy, character accuracy, n and the Wilson 95 % lower bound,
plus single-image latency for ours.

    .venv/bin/python tools/text_eval.py --split test
    .venv/bin/python tools/text_eval.py --split all --readers ours --weights data/text/ckpt/x.pt
"""

from __future__ import annotations

import argparse
import io
import math
import os
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import text_data  # noqa: E402


def wilson_lower(k: int, n: int, z: float = 1.96) -> float:
    if n == 0:
        return 0.0
    p = k / n
    den = 1 + z * z / n
    centre = p + z * z / (2 * n)
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (centre - margin) / den


_dddd = None


def _ocr():
    global _dddd
    if _dddd is None:
        import ddddocr
        _dddd = ddddocr.DdddOcr(show_ad=False)
    return _dddd


def ddddocr_plain(jpeg: bytes) -> str:
    return "".join(c for c in _ocr().classification(jpeg) if c.isdigit())


def _variants(img_bytes: bytes) -> list[bytes]:
    from PIL import Image, ImageEnhance, ImageFilter

    def b(im):
        buf = io.BytesIO()
        im.save(buf, format="PNG")
        return buf.getvalue()

    img = Image.open(io.BytesIO(img_bytes)).convert("L")
    sharp = img.filter(ImageFilter.SHARPEN)
    scaled = img.resize((img.width * 2, img.height * 2), Image.LANCZOS)
    inv = img.point(lambda x: 255 - x)
    return [
        img_bytes,
        b(ImageEnhance.Contrast(img).enhance(2.0)),
        b(ImageEnhance.Contrast(sharp).enhance(1.5)),
        b(img.point(lambda x: 255 if x > 128 else 0)),
        b(img.point(lambda x: 255 if x > 100 else 0)),
        b(ImageEnhance.Contrast(ImageEnhance.Brightness(img).enhance(1.3)).enhance(2.0)),
        b(scaled),
        b(ImageEnhance.Contrast(scaled).enhance(2.0)),
        b(scaled.point(lambda x: 255 if x > 128 else 0)),
        b(ImageEnhance.Contrast(img.filter(ImageFilter.MedianFilter(3))).enhance(2.0)),
        b(img),
        b(img.point(lambda x: 255 if x > 150 else 0)),
        b(ImageEnhance.Contrast(img.filter(ImageFilter.EDGE_ENHANCE)).enhance(1.5)),
        b(img.resize((img.width * 3, img.height * 3), Image.LANCZOS)),
        b(inv.point(lambda x: 255 if x > 128 else 0)),
        b(ImageEnhance.Contrast(sharp.filter(ImageFilter.SHARPEN)).enhance(2.0)),
    ]


def ddddocr_vote(jpeg: bytes) -> str:
    results = []
    for v in _variants(jpeg):
        r = "".join(c for c in (_ocr().classification(v) or "") if c.isdigit())
        if len(r) == 5:
            results.append(r)
    return Counter(results).most_common(1)[0][0] if results else ""


def ours_factory(weights: str | None):
    if weights:
        os.environ["AWS_CAPSOLVER_BLS_POLAND_WEIGHTS"] = str(Path(weights).resolve())
    from aws_capsolver.text import bls_poland

    return lambda jpeg: bls_poland.solve(jpeg)[0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=("test", "train", "all"), default="test")
    ap.add_argument("--readers", default="ddddocr,ddddocr_vote,ours")
    ap.add_argument("--weights")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    rows = text_data.load_real(None if args.split == "all" else args.split)
    readers = {"ddddocr": ddddocr_plain, "ddddocr_vote": ddddocr_vote}
    if "ours" in args.readers:
        readers["ours"] = ours_factory(args.weights)
    print(f"split={args.split} n={len(rows)}")
    print(f"{'reader':<14}{'exact':>8}{'k/n':>8}{'wilson95':>10}{'char':>8}{'ms/img':>9}")
    for name in args.readers.split(","):
        fn = readers[name]
        fn(rows[0][1])  # warm-up (model load) outside the timing
        k = chars = 0
        t0 = time.perf_counter()
        for fname, jpeg, label in rows:
            pred = fn(jpeg)
            k += pred == label
            chars += sum(a == b for a, b in zip(pred, label)) if len(pred) == 5 else 0
            if args.verbose and pred != label:
                print(f"  {name}: {fname} {label} -> {pred}")
        ms = (time.perf_counter() - t0) / max(1, len(rows)) * 1000
        n = len(rows)
        print(f"{name:<14}{k / n:>8.3f}{f'{k}/{n}':>8}{wilson_lower(k, n):>10.3f}"
              f"{chars / (5 * n):>8.3f}{ms:>9.1f}")


if __name__ == "__main__":
    main()
