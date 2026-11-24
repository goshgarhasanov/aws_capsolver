"""Reproduce the reCAPTCHA v2 classifier evaluation.

Fast mode (default) scores cached embeddings (tools/recaptcha_embed.py) with the trained
head exactly as the engine does at inference time:
  * bx 3x3 / 4x4 ``test`` split (machine labels: Qwen per tile / SAM 3) and
    ``review`` split (labels corrected by a human-level reviewer, the most trustworthy),
  * DannyLuna ``val`` single tiles (size-1 hasObject decision, positives vs. negatives).
Reported: per-tile accuracy and per-challenge exact match (every tile right) with the
Wilson 95 % lower bound, overall and per class.

``--e2e N`` additionally runs the public ``classify()`` on N raw review images of each grid
size (decode + split + embed + head) and reports accuracy and latency.

  .venv/bin/python tools/recaptcha_bench.py [--head weights/recaptcha_head.pt] [--e2e 100]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from recaptcha_train import BX_MAP, DANNY_MAP, TILE_CLASSES, answers_set, may_cooccur  # noqa: E402

from aws_capsolver.recaptcha.model import Engine  # noqa: E402
from recaptcha_embed import out_path  # noqa: E402

DATA = ROOT / "data" / "recaptcha"


def wilson_lb(k: int, n: int, z: float = 1.96) -> float:
    if n == 0:
        return 0.0
    p = k / n
    den = 1 + z * z / n
    centre = p + z * z / (2 * n)
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (centre - margin) / den


def fmt(k: int, n: int) -> str:
    return f"{k}/{n} = {100 * k / max(n, 1):.2f}% (LB95 {100 * wilson_lb(k, n):.2f}%)"


def load(specs, what, split):
    parts = [dict(np.load(out_path(s, what, split), allow_pickle=False)) for s in specs]
    d = parts[0]
    d["emb"] = np.concatenate([p["emb"].astype(np.float32) for p in parts], axis=-1)
    return d


def truth(row, n):
    a = np.zeros(n, bool)
    a[list(answers_set(row))] = True
    return a


def eval_grid(eng: Engine, specs, grid: str, split: str, label_key: str = "answers") -> dict:
    what = "bx3" if grid == "3x3" else "bx4"
    p = out_path(specs[0], what, split)
    if not p.exists():
        return {}
    b = load(specs, what, split)
    n_cells = 9 if grid == "3x3" else 16
    per = defaultdict(lambda: [0, 0, 0, 0])  # exact, challenges, tiles_ok, tiles
    for r in range(len(b["target"])):
        tgt = BX_MAP.get(str(b["target"][r]))
        if tgt is None:
            continue
        if grid == "3x3":
            prob, thr, _ = eng.tile_probs(b["emb"][b["tile_idx"][r]], tgt)
        elif eng.knows_cell(tgt):
            prob, thr, _ = eng.cell_probs(b["emb"][r], tgt)
        elif eng.cell_specs == eng.tile_specs and eng.knows_tile(tgt):
            prob, thr, _ = eng.tile_probs(b["emb"][r][:16], tgt)
        else:
            continue  # no cached features for this fallback path
        pred = prob >= thr
        a = truth(b[label_key][r], n_cells)
        for key in (tgt, "ALL"):
            s = per[key]
            s[0] += int((pred == a).all()); s[1] += 1; s[2] += int((pred == a).sum()); s[3] += n_cells
    return dict(per)


def eval_dense(path: Path, split: str) -> dict:
    """4x4 dense head on cached bx4dense (+ bx4 crop) features."""
    import torch

    from aws_capsolver.recaptcha.dense import build

    raw = torch.load(path, map_location="cpu", weights_only=True)
    if not out_path(raw["dense_backbone"], "bx4dense", split).exists():
        return {}
    net = build(int(raw["dim"]), len(raw["classes"]), hidden=int(raw["hidden"]), cell_dim=int(raw["cell_dim"]))
    net.load_state_dict(raw["state"])
    net.eval()
    b = np.load(out_path(raw["dense_backbone"], "bx4dense", split))
    C = np.load(out_path(raw["crop_backbone"], "bx4", split))["emb"] if raw.get("crop_backbone") else None
    classes = list(raw["classes"])
    per = defaultdict(lambda: [0, 0, 0, 0])
    with torch.inference_mode():
        for s0 in range(0, len(b["emb"]), 128):
            x = torch.from_numpy(b["emb"][s0:s0 + 128].astype(np.float32))
            c = torch.from_numpy(C[s0:s0 + 128].astype(np.float32)) if C is not None else None
            P = torch.sigmoid(net(x, c)).numpy()
            for j in range(len(P)):
                r = s0 + j
                tgt = BX_MAP.get(str(b["target"][r]))
                if tgt not in classes:
                    continue
                pred = P[j, :, classes.index(tgt)] >= raw["thresholds"].get(tgt, 0.5)
                a = truth(b["answers"][r], 16)
                for key in (tgt, "ALL"):
                    st = per[key]
                    st[0] += int((pred == a).all()); st[1] += 1; st[2] += int((pred == a).sum()); st[3] += 16
    return dict(per)


def eval_single(eng: Engine, specs) -> dict:
    p = out_path(specs[0], "danny", "val")
    if not p.exists():
        return {}
    d = load(specs, "danny", "val")
    names = [DANNY_MAP[str(c)] for c in d["classes"]]
    per = defaultdict(lambda: [0, 0])
    X = d["emb"]
    for c in TILE_CLASSES:
        prob, thr, _ = eng.tile_probs(X, c)
        for i, yi in enumerate(d["y"]):
            a = names[yi]
            if a is not None and a != c and may_cooccur(a, c):
                continue  # Danny gives no reliable negative label here
            ok = (prob[i] >= thr) == (a == c)
            for key in (c, "ALL"):
                per[key][0] += int(ok); per[key][1] += 1
    return dict(per)


def e2e(n: int) -> None:
    import aws_capsolver.recaptcha as R

    for grid, size in (("3x3", 3), ("4x4", 4)):
        d = DATA / "bx" / "data" / grid / "review"
        rows = [json.loads(line) for line in open(d / "metadata.jsonl")][:n]
        R.warm()
        ok = tot = 0
        inferred_ok = 0
        lat = []
        for r in rows:
            img = (d / r["file_name"]).read_bytes()
            t0 = time.perf_counter()
            res = R.classify(img, r["target"])  # size inferred from the image
            lat.append(time.perf_counter() - t0)
            inferred_ok += int(res["size"] == size)
            ok += int(sorted(res["objects"]) == sorted(r["correct_answers"]))
            tot += 1
        lat_ms = np.array(lat) * 1000
        print(f"e2e {grid} review: exact {fmt(ok, tot)}; size inferred right {inferred_ok}/{tot}; "
              f"latency p50 {np.median(lat_ms):.0f} ms p90 {np.percentile(lat_ms, 90):.0f} ms")


def live(path: Path) -> None:
    """Hand-labelled live 2captcha-demo challenges (data/recaptcha/live/labels_manual.json)."""
    import aws_capsolver.recaptcha as R

    items = json.loads(path.read_text())["items"]
    R.warm()
    per = defaultdict(lambda: [0, 0])
    lat = defaultdict(list)
    for it in items:
        img = (path.parent / it["image"]).read_bytes()
        t0 = time.perf_counter()
        res = R.classify(img, it["question"], it["size"])
        lat[it["size"]].append(time.perf_counter() - t0)
        ok = res["hasObject"] == it["hasObject"] if it["size"] == 1 else sorted(res["objects"]) == sorted(it["objects"])
        q = R.normalize_question(it["question"])
        for key in (f"size {it['size']}", f"size {it['size']} {q}"):
            per[key][0] += int(ok); per[key][1] += 1
    for k in sorted(per):
        print(f"live {k:28s} {fmt(*per[k])}")
    for size, v in sorted(lat.items()):
        print(f"live latency size {size}: p50 {1000 * np.median(v):.0f} ms p90 {1000 * np.percentile(v, 90):.0f} ms")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--head", default=str(ROOT / "weights" / "recaptcha_head.pt"))
    ap.add_argument("--e2e", type=int, default=0, help="also run classify() on N raw review images per grid")
    ap.add_argument("--live", action="store_true", help="also score the hand-labelled live challenges")
    ap.add_argument("--dense", default=str(ROOT / "weights" / "recaptcha_dense.pt"), help="4x4 dense head")
    a = ap.parse_args()
    import os

    os.environ["RECAPTCHA_HEAD"] = a.head  # e2e / live runs use the same heads
    os.environ["RECAPTCHA_DENSE"] = a.dense
    eng = Engine(a.head, load_backbones=False)
    if eng.head is None:
        sys.exit("no head: train it first (tools/recaptcha_train.py)")
    specs = eng.tile_specs
    print("head:", eng.head["trained_on"], "| tile:", eng.tile_specs, "| cell:", eng.cell_specs)
    for grid in ("3x3", "4x4"):
        specs = eng.tile_specs if grid == "3x3" else eng.cell_specs
        for split, key, name in (("test", "answers", "test (machine labels)"),
                                 ("review", "answers", "review (reviewed labels)"),
                                 ("review", "machine", "review (machine labels)")):
            per = eval_grid(eng, specs, grid, split, key)
            if not per:
                continue
            s = per["ALL"]
            print(f"\n{grid} {name}: challenge exact {fmt(s[0], s[1])}; tile acc {100 * s[2] / s[3]:.2f}%")
            for c in sorted(k for k in per if k != "ALL"):
                e = per[c]
                print(f"    {c:14s} exact {fmt(e[0], e[1])}  tile {100 * e[2] / e[3]:.2f}%")
    dense_path = Path(a.dense)
    if dense_path.is_file():
        for split in ("test", "review"):
            per = eval_dense(dense_path, split)
            if not per:
                continue
            s_ = per["ALL"]
            print(f"\n4x4 DENSE head {split}: challenge exact {fmt(s_[0], s_[1])}; tile acc {100 * s_[2] / s_[3]:.2f}%")
            for c in sorted(k for k in per if k != "ALL"):
                e = per[c]
                print(f"    {c:14s} exact {fmt(e[0], e[1])}  tile {100 * e[2] / e[3]:.2f}%")
    per = eval_single(eng, eng.tile_specs)
    if per:
        s = per["ALL"]
        print(f"\nsingle tile (Danny val, hasObject): {fmt(s[0], s[1])}")
        for c in sorted(k for k in per if k != "ALL"):
            print(f"    {c:14s} {fmt(*per[c])}")
    if a.e2e:
        print()
        e2e(a.e2e)
    lp = DATA / "live" / "labels_manual.json"
    if a.live and lp.is_file():
        print()
        live(lp)


if __name__ == "__main__":
    main()
