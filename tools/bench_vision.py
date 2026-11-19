#!/usr/bin/env python3
"""Benchmark the grid classifier on the labelled grids.

Reports per-tile and per-grid accuracy (a grid counts as correct only if the selected set
equals the labelled set exactly) on the frozen held-out split, plus warm latency of a full
``classify_grid`` call (decode + embed + score) for 9 tiles.

Examples:
  .venv/bin/python tools/bench_vision.py                          # default engine (+head)
  .venv/bin/python tools/bench_vision.py --models ViT-B-32:laion2b_s34b_b79k --no-head
  .venv/bin/python tools/bench_vision.py --compare                # zero-shot model sweep
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import vision_data as vd  # noqa: E402

COMPARE = [
    "ViT-B-32:laion2b_s34b_b79k",
    "ViT-B-16:datacomp_xl_s13b_b90k",
    "ViT-B-16-SigLIP2:webli",
    "ViT-L-14:datacomp_xl_s13b_b90k",
    "ViT-L-14-quickgelu:dfn2b",
    "ViT-SO400M-14-SigLIP2:webli",
]


def wilson_lower(k: int, n: int, z: float = 1.96) -> float:
    """95% Wilson score lower bound of a binomial proportion."""
    if n == 0:
        return 0.0
    p = k / n
    den = 1 + z * z / n
    centre = p + z * z / (2 * n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (centre - half) / den


def novel_tiles(engine, gids: list[str], train: list[str], cos: float = 0.95) -> dict[str, np.ndarray]:
    """Per grid, a bool mask of tiles without a near-duplicate (cosine > ``cos``) in train."""
    emb = vd.embeddings(engine, sorted(set(gids) | set(train)), 0)
    if not train:
        return {g: np.ones(9, bool) for g in gids}
    T = np.concatenate([emb[g] for g in train])
    return {g: (emb[g] @ T.T).max(1) <= cos for g in gids}


def evaluate(engine, gids: list[str], threshold: float | None, use_head: bool, count: int | None = 5) -> dict:
    from aws_capsolver.vision.grid import select_tiles, normalize_target

    labels = vd.train_labels()
    per_bb = [vd.embeddings(engine, gids, k) for k in range(len(engine.specs))]
    tiles_ok = grids_ok = 0
    errors, margins, ok_by_grid, tile_ok = [], [], {}, {}
    for g in gids:
        t = normalize_target(vd.target(g))
        feats = [pb[g] for pb in per_bb]
        s = engine.head_probs(feats, t) if use_head else None
        method = "head"
        if s is None:
            s, method = engine.zero_shot_probs(feats, t), "zeroshot"
        if threshold is None and count:
            pred = set(select_tiles(s, count=count))
        else:
            thr = threshold if threshold is not None else (engine.default_threshold(t) if use_head else 0.5)
            pred = set(select_tiles(s, thr))
        truth = set(vd.grid_answer(g, labels[g]))
        srt = np.sort(s)[::-1]
        if count and len(srt) > count:
            margins.append(float(srt[count - 1] - srt[count]))
        tiles_ok += sum((i in pred) == (i in truth) for i in range(9))
        ok_by_grid[g] = pred == truth
        tile_ok[g] = np.array([(i in pred) == (i in truth) for i in range(9)])
        if pred == truth:
            grids_ok += 1
        else:
            errors.append({"grid": g, "target": t, "pred": sorted(pred), "truth": sorted(truth),
                           "method": method, "scores": [round(float(x), 3) for x in s]})
    n = len(gids)
    return {"grids": n, "grids_ok": grids_ok, "tile_acc": tiles_ok / (9 * n) if n else 0.0,
            "grid_acc": grids_ok / n if n else 0.0, "errors": errors, "ok": ok_by_grid, "tile_ok": tile_ok,
            "min_margin": min(margins) if margins else None,
            "p1_margin": float(np.percentile(margins, 1)) if margins else None}


def latency(engine, n: int = 10) -> float:
    from aws_capsolver.vision.grid import select_tiles

    gids = (vd.splits()["test"] or vd.list_grids())[:n]
    engine.scores(vd.tiles(gids[0]), vd.target(gids[0]))  # warm-up
    t0 = time.perf_counter()
    for g in gids:
        s, _ = engine.scores(vd.tiles(g), vd.target(g))
        select_tiles(s, count=5)
    return (time.perf_counter() - t0) / len(gids)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", help="comma separated model:pretrained (default: engine default)")
    ap.add_argument("--compare", action="store_true", help="zero-shot sweep over candidate backbones")
    ap.add_argument("--no-head", action="store_true")
    ap.add_argument("--tta", action="store_true", help="embed tile + mirror image")
    ap.add_argument("--threshold", type=float, help="threshold mode instead of top-K")
    ap.add_argument("--count", type=int, default=5, help="select top-K tiles (0: threshold mode)")
    ap.add_argument("--split", choices=["test", "val", "train", "eval"], default="test",
                    help="eval = val + test (manually labelled grids only)")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--show-errors", type=int, default=10)
    args = ap.parse_args()

    from aws_capsolver.vision.grid import GridEngine

    sp = vd.splits()
    gids = sp["val"] + sp["test"] if args.split == "eval" else sp[args.split]
    runs = [[m] for m in COMPARE] if args.compare else [args.models.split(",") if args.models else None]
    rows = []
    for specs in runs:
        eng = GridEngine(specs=specs, head_path="none" if args.no_head else None,
                         tta=args.tta, threads=args.threads or None)
        if eng.head is not None and eng.head.get("trained_on", "train") != "train" and args.split != "train":
            print(f"WARNING: head was trained on {eng.head['trained_on']}; {args.split} is NOT held out",
                  file=sys.stderr)
        res = evaluate(eng, gids, args.threshold, use_head=not args.no_head, count=args.count)
        novel = novel_tiles(eng, gids, sp["train"])
        n_nov = sum(int(m.sum()) for m in novel.values())
        ok_nov = sum(int(res["tile_ok"][g][m].sum()) for g, m in novel.items())
        lat = latency(eng)
        mode = f"top{args.count}" if args.count and args.threshold is None else "threshold"
        row = {"models": eng.specs, "head": bool(eng.head) and not args.no_head, "tta": eng.tta,
               "split": args.split, "mode": mode, "grids": res["grids"], "grids_ok": res["grids_ok"],
               "grid_acc": round(res["grid_acc"], 4),
               "grid_acc_wilson95_low": round(wilson_lower(res["grids_ok"], res["grids"]), 4),
               "tile_acc": round(res["tile_acc"], 4),
               "novel_tiles": n_nov, "novel_tile_acc": round(ok_nov / n_nov, 4) if n_nov else None,
               "min_top5_margin": None if res["min_margin"] is None else round(res["min_margin"], 4),
               "latency_s": round(lat, 3)}
        rows.append(row)
        print(json.dumps(row), flush=True)
        for e in res["errors"][:args.show_errors]:
            print("   miss", json.dumps(e))
        del eng
    if len(rows) > 1:
        print("\nmodel | tile acc | grid acc | latency/grid")
        for r in rows:
            print(f"{'+'.join(r['models'])} | {r['tile_acc']:.4f} | {r['grid_acc']:.4f} | {r['latency_s']:.2f}s")


if __name__ == "__main__":
    main()
