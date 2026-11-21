#!/usr/bin/env python3
"""Held-out accuracy + latency of the GeeTest solvers on real samples from GeeTest's own demos.

Nothing here was used to fit a model (the slide and gobang solvers have no trained
weights; the icon models are trained on synthetic pictures only; nine uses zero-shot
CLIP). Duplicated challenges (the demos sometimes serve the same picture twice) are
counted once.

* slide v3  - data/geetest/v3_slide: truth = the piece aligned on |bg - fullbg| (exact),
              scored for bg+piece (the normal case), bg+piece+fullbg and bg only.
* slide v4  - data/geetest/v4_slide with a truth from tools/geetest_slide_truth.py (piece
              content matched on the per-picture median of other samples; strict
              thresholds, plus relaxed ones whose disagreements with the solver were
              checked by eye: tools/geetest_labels/v4_slide_hand.json), bg+piece.
* nine      - tools/geetest_labels/v4_nine.json, split "test" (hand verified cells).
* icon      - tools/geetest_labels/v4_icon.json, split "test" (hand verified clicks).
* gobang    - the 100 collected v4 ``ques`` boards rendered with their own piece
              images (canvas and widget look, 4 scales / JPEG levels) + matrix input.

    .venv/bin/python tools/geetest_bench.py [--types slide,nine,icon,gobang] [--threads 8]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
DATA = ROOT / "data" / "geetest"
LABELS = ROOT / "tools" / "geetest_labels"


def wilson_lower(k: int, n: int, z: float = 1.96) -> float:
    if n == 0:
        return 0.0
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    r = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c - r) / d


def report(name: str, oks: list[bool], lat: list[float]) -> dict:
    n, k = len(oks), sum(oks)
    row = {"type": name, "n": n, "correct": k, "acc": k / n if n else 0.0, "wilson95_lower": wilson_lower(k, n)}
    if lat:
        s = sorted(lat)
        row["lat_median_ms"] = 1000 * statistics.median(s)
        row["lat_p95_ms"] = 1000 * s[min(len(s) - 1, int(0.95 * len(s)))]
    print(f"{name:28s} n={n:4d}  acc={row['acc']*100:6.2f}%  wilson95>={row['wilson95_lower']*100:6.2f}%"
          + (f"  latency median {row['lat_median_ms']:.0f} ms, p95 {row['lat_p95_ms']:.0f} ms" if lat else ""))
    return row


def _md5(p: Path) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest()


def _timed(fn, *a, **kw):
    t0 = time.perf_counter()
    r = fn(*a, **kw)
    return r, time.perf_counter() - t0


def bench_slide(tol: int = 4) -> list[dict]:
    import aws_capsolver.geetest as G
    from aws_capsolver.geetest import slide

    rows = []
    res = {"piece": [], "piece+full": [], "bg only": []}
    lat = []
    seen = set()
    for d in sorted((DATA / "v3_slide").iterdir()):
        if not (d / "meta.json").is_file():
            continue
        h = _md5(d / "bg.jpg")
        if h in seen:
            continue
        seen.add(h)
        meta = json.loads((d / "meta.json").read_text())
        bg, piece, full = (d / "bg.jpg").read_bytes(), (d / "slice.png").read_bytes(), (d / "fullbg.jpg").read_bytes()
        truth = slide.solve(bg, piece, full, y=meta["ypos"])
        x, t = _timed(G.slide_offset, bg, piece)
        lat.append(t)
        res["piece"].append(abs(x - truth.x) <= tol)
        res["piece+full"].append(abs(G.slide_offset(bg, piece, full) - truth.x) <= tol)
        res["bg only"].append(abs(G.slide_offset(bg) - truth.gap_x) <= tol)
    rows.append(report(f"slide v3 bg+piece (+-{tol}px)", res["piece"], lat))
    rows.append(report(f"slide v3 bg+piece+fullbg", res["piece+full"], []))
    rows.append(report(f"slide v3 bg only (gap edge)", res["bg only"], []))

    hand = json.loads((LABELS / "v4_slide_hand.json").read_text()) if (LABELS / "v4_slide_hand.json").is_file() else {}
    oks, oks_y, lat, seen = [], [], [], set()
    for d in sorted((DATA / "v4_slide").iterdir()):
        # strict content truth, else hand-checked label, else relaxed content truth
        if (d / "truth.json").is_file():
            t = json.loads((d / "truth.json").read_text())
        elif d.name in hand:
            t = hand[d.name]
        elif (d / "truth_relaxed.json").is_file():
            t = json.loads((d / "truth_relaxed.json").read_text())
        else:
            continue
        h = _md5(d / "bg.png")
        if h in seen:
            continue
        seen.add(h)
        bg, piece = (d / "bg.png").read_bytes(), (d / "slice.png").read_bytes()
        x, dt = _timed(G.slide_offset, bg, piece)
        lat.append(dt)
        oks.append(abs(x - t["x"]) <= tol)
        ypos = json.loads((d / "meta.json").read_text())["ypos"]
        oks_y.append(abs(G.slide_offset(bg, piece, y=ypos) - t["x"]) <= tol)
    rows.append(report(f"slide v4 bg+piece (+-{tol}px)", oks, lat))
    rows.append(report(f"slide v4 bg+piece+ypos", oks_y, []))
    return rows


def bench_gobang() -> list[dict]:
    import aws_capsolver.geetest as G
    from aws_capsolver.geetest.board import moves
    from geetest_render import load_gobang_pieces, render_gobang

    oks, lat, mat = [], [], []
    for f in sorted((DATA / "v4_gobang").glob("*/meta.json")):
        q = json.loads(f.read_text())["ques"]
        truth = moves(q, 0)[0]
        mat.append(G.gobang(q) == truth)
        pieces = load_gobang_pieces(f.parent)
        for style in ("canvas", "widget"):
            for scale, jpeg in ((1.0, None), (0.6, 70), (1.3, 85), (0.4, 60)):
                img = render_gobang(q, pieces, style, scale=scale, jpeg=jpeg)
                try:
                    r, dt = _timed(G.gobang, img)
                except G.GeeTestError:
                    r, dt = None, 0.0
                oks.append(r == truth)
                lat.append(dt)
    rows = [report("gobang matrix (ques)", mat, []), report("gobang picture (rendered)", oks, lat)]
    shot = ROOT / "tests" / "fixtures" / "geetest" / "v4_gobang_widget.png"
    if shot.is_file():
        rows.append(report("gobang real widget screenshot", [G.gobang(shot.read_bytes()) == ((0, 4), (2, 1))], []))
    return rows


def bench_nine(split: str = "test") -> list[dict]:
    import aws_capsolver.geetest as G

    labels = json.loads((LABELS / "v4_nine.json").read_text())
    oks, lat, seen = [], [], set()
    for lot, lab in sorted(labels.items()):
        if lab["split"] != split:
            continue
        d = DATA / "v4_nine" / lot
        h = _md5(d / "image.jpg") + _md5(d / "hint_0.png")
        if h in seen:
            continue
        seen.add(h)
        cells, dt = _timed(G.nine_grid, (d / "image.jpg").read_bytes(), (d / "hint_0.png").read_bytes())
        lat.append(dt)
        oks.append(sorted(cells) == sorted(lab["cells"]))
    return [report(f"nine ({split})", oks, lat[1:] if len(lat) > 1 else lat)]


def bench_icon(split: str = "test") -> list[dict]:
    import aws_capsolver.geetest as G

    path = LABELS / "v4_icon.json"
    if not path.is_file():
        print("icon: no labels yet")
        return []
    labels = json.loads(path.read_text())
    oks, lat, seen = [], [], set()
    for lot, lab in sorted(labels.items()):
        if lab["split"] != split:
            continue
        d = DATA / "v4_icon" / lot
        h = _md5(d / "image.jpg")
        if h in seen:
            continue
        seen.add(h)
        hints = [(d / f"hint_{i}.png").read_bytes() for i in range(len(lab["boxes"]))]
        try:
            pts, dt = _timed(G.click_points, (d / "image.jpg").read_bytes(), "icon", hints)
        except G.GeeTestError:
            pts, dt = [], 0.0
        lat.append(dt)
        ok = len(pts) == len(lab["boxes"]) and all(
            b[0] <= x <= b[2] and b[1] <= y <= b[3] for (x, y), b in zip(pts, lab["boxes"]))
        oks.append(ok)
    return [report(f"icon click v4 ({split})", oks, lat[1:] if len(lat) > 1 else lat)]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--types", default="slide,gobang,nine,icon")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--json", help="also write the rows to this file")
    args = ap.parse_args()
    os.environ.setdefault("AWS_CAPSOLVER_TORCH_THREADS", str(args.threads))
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    rows = []
    for t in [t.strip() for t in args.types.split(",") if t.strip()]:
        rows += {"slide": bench_slide, "gobang": bench_gobang, "nine": bench_nine, "icon": bench_icon}[t]()
    if args.json:
        Path(args.json).write_text(json.dumps(rows, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
