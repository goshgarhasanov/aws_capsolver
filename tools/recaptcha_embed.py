"""Embed reCAPTCHA training / evaluation data with a frozen CLIP backbone (cached .npz).

Datasets (downloaded into data/recaptcha, see aws_capsolver/recaptcha/README.md):
  danny  DannyLuna/recaptcha-57k-images-dataset: single tiles in class folders
         -> emb (n, d), y (class index), classes, split ("train" / "val")
  bx3    bxding/recaptcha-67k 3x3: unique tiles (deduplicated by pixel hash)
         -> emb (u, d), tile_idx (challenges, 9), target, answers, uncertain, ids
  bx4    bxding/recaptcha-67k 4x4: 16 cells + whole photo per challenge
         -> emb (challenges, 17, d), target, answers, ids

Only challenges whose image is present on disk are embedded. Usage:
  .venv/bin/python tools/recaptcha_embed.py --backbone ViT-B-32:laion2b_s34b_b79k@224 \
      --what bx3 --split test
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from aws_capsolver.recaptcha import features  # noqa: E402
from aws_capsolver.recaptcha.backbone import get_encoder  # noqa: E402

DATA = ROOT / "data" / "recaptcha"
BATCH = 96


def tag(spec: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.@-]", "_", spec)


def out_path(spec: str, what: str, split: str) -> Path:
    return DATA / "emb" / tag(spec) / f"{what}_{split}.npz"


def _open(path):
    from PIL import Image

    with Image.open(path) as im:
        return im.convert("RGB")


def embed_pils(enc, pils_iter, total: int, label: str) -> np.ndarray:
    out, done, t0 = [], 0, time.time()
    buf = []
    for p in pils_iter:
        buf.append(p)
        if len(buf) == BATCH:
            out.append(enc(features.to_tensor(buf, enc.res)).astype(np.float16))
            done += len(buf)
            buf = []
            if done % (BATCH * 20) == 0:
                el = time.time() - t0
                print(f"  {label}: {done}/{total} {done / el:.0f} img/s eta {(total - done) / max(1e-6, done / el) / 60:.1f} min",
                      flush=True)
    if buf:
        out.append(enc(features.to_tensor(buf, enc.res)).astype(np.float16))
    return np.concatenate(out) if out else np.zeros((0, 0), np.float16)


def _prefetch(fn, items, workers=6):
    with ThreadPoolExecutor(workers) as ex:
        yield from ex.map(fn, items)


def run_danny(enc, split: str):
    base = DATA / "danny" / "dataset_cls_full_57k" / split
    classes = sorted(os.listdir(DATA / "danny" / "dataset_cls_full_57k" / "train"))
    files, ys = [], []
    for i, c in enumerate(classes):
        d = base / c
        if d.is_dir():
            for f in sorted(d.iterdir()):
                files.append(f)
                ys.append(i)
    emb = embed_pils(enc, _prefetch(_open, files), len(files), f"danny/{split}")
    return dict(emb=emb, y=np.array(ys), classes=np.array(classes),
                files=np.array([str(f.relative_to(DATA)) for f in files]))


def _rows(grid: str, split: str):
    d = DATA / "bx" / "data" / grid / split
    rows = [json.loads(line) for line in open(d / "metadata.jsonl")]
    return d, [r for r in rows if (d / r["file_name"]).is_file()]


def _pad(lists, width):
    a = np.full((len(lists), width), -1, np.int16)
    for i, l in enumerate(lists):
        l = sorted(set(int(v) for v in l))[:width]
        a[i, :len(l)] = l
    return a


def run_bx3(enc, split: str):
    d, rows = _rows("3x3", split)
    uniq: dict[bytes, int] = {}
    tiles = []
    tile_idx = np.zeros((len(rows), 9), np.int32)

    def load(r):
        im = _open(d / r["file_name"])
        return features.model_crops(im, 3)

    for ci, crops in enumerate(_prefetch(load, rows)):
        for k, t in enumerate(crops):
            h = hashlib.md5(t.tobytes()).digest()
            j = uniq.get(h)
            if j is None:
                j = uniq[h] = len(tiles)
                tiles.append(t)
            tile_idx[ci, k] = j
    print(f"  bx3/{split}: {len(rows)} challenges, {len(tiles)} unique tiles", flush=True)
    emb = embed_pils(enc, iter(tiles), len(tiles), f"bx3/{split}")
    hashes = np.array([h.hex() for h in uniq.keys()])
    return dict(emb=emb, tile_idx=tile_idx, hashes=hashes, target=np.array([r["target"] for r in rows]),
                answers=_pad([r["correct_answers"] for r in rows], 9),
                machine=_pad([r["machine_answers"] for r in rows], 9),
                uncertain=_pad([r.get("uncertain_cells") or [] for r in rows], 9),
                ids=np.array([r["id"] for r in rows]))


def run_bx4(enc, split: str):
    d, rows = _rows("4x4", split)

    def load(r):
        return features.model_crops(_open(d / r["file_name"]), 4)

    def flat():
        for crops in _prefetch(load, rows):
            yield from crops

    emb = embed_pils(enc, flat(), len(rows) * 17, f"bx4/{split}")
    emb = emb.reshape(len(rows), 17, -1)
    return dict(emb=emb, target=np.array([r["target"] for r in rows]),
                answers=_pad([r["correct_answers"] for r in rows], 16),
                machine=_pad([r["machine_answers"] for r in rows], 16),
                ids=np.array([r["id"] for r in rows]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", required=True)
    ap.add_argument("--what", required=True, choices=["danny", "bx3", "bx4"])
    ap.add_argument("--split", required=True, nargs="+")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    import torch

    torch.set_num_threads(a.threads)
    enc = get_encoder(a.backbone)
    for split in a.split:
        p = out_path(a.backbone, a.what, split)
        if p.exists() and not a.force:
            print(f"skip {p} (exists)")
            continue
        t0 = time.time()
        res = {"danny": run_danny, "bx3": run_bx3, "bx4": run_bx4}[a.what](enc, split)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp.npz")
        np.savez(tmp, **res)
        os.replace(tmp, p)
        print(f"wrote {p} in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
