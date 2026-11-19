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


def evaluate(engine, gids: list[str], threshold: float | None, use_head: bool, count: int | None = 5) -> dict:
    from aws_capsolver.vision.grid import select_tiles, normalize_target

    tl = vd.train_labels()
    per_bb = [vd.embeddings(engine, gids, k) for k in range(len(engine.specs))]
    tiles_ok = grids_ok = 0
    errors = []
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
        truth = set(vd.grid_answer(g, tl[g]))
        tiles_ok += sum((i in pred) == (i in truth) for i in range(9))
        if pred == truth:
            grids_ok += 1
        else:
            errors.append({"grid": g, "target": t, "pred": sorted(pred), "truth": sorted(truth),
                           "method": method, "scores": [round(float(x), 3) for x in s]})
    n = len(gids)
    return {"grids": n, "tile_acc": tiles_ok / (9 * n) if n else 0.0,
            "grid_acc": grids_ok / n if n else 0.0, "errors": errors}


def latency(engine, n: int = 10) -> float:
    from aws_capsolver.vision.grid import select_tiles

    gids = vd.list_grids()[:n]
    engine.scores(vd.tiles(gids[0]), vd.target(gids[0]))  # warm-up
    t0 = time.perf_counter()
    for g in gids:
        s, _ = engine.scores(vd.tiles(g), vd.target(g))
        select_tiles(s, 0.5)
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
        res = evaluate(eng, gids, args.threshold, use_head=not args.no_head, count=args.count)
        lat = latency(eng)
        row = {"models": eng.specs, "head": bool(eng.head) and not args.no_head, "tta": eng.tta,
               "split": args.split, "mode": f"top{args.count}" if args.count and args.threshold is None else "threshold", "grids": res["grids"], "tile_acc": round(res["tile_acc"], 4),
               "grid_acc": round(res["grid_acc"], 4), "latency_s": round(lat, 3)}
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
