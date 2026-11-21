#!/usr/bin/env python3
"""Build ground truth for collected v4 slide samples (independent of the solver's method).

GeeTest v4 draws its gaps on a small pool of pictures. Samples that share a picture are
grouped (thumbnail distance), the per-pixel median of a group (>= 3 members, gaps at
different places) is the clean picture, and the truth is found exactly like v3's
``fullbg`` case: the piece's alpha shape aligned on |bg - clean| and refined on the
piece's own pixels. Writes truth.json next to each sample that got one.

    .venv/bin/python tools/geetest_slide_truth.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
MIN_NCC = float(os.environ.get("TRUTH_MIN_NCC", "0.7"))    # minimum high-pass NCC of the piece on the clean picture
MIN_GROUP = int(os.environ.get("TRUTH_MIN_GROUP", "3"))
MIN_MARGIN = float(os.environ.get("TRUTH_MIN_MARGIN", "0.1")) # over the best match elsewhere
DATA = ROOT / "data" / "geetest" / "v4_slide"


def _hp(a: np.ndarray) -> np.ndarray:
    a = a.astype(np.float32)
    return a - cv2.GaussianBlur(a, (0, 0), 2.0)


def content_truth(clean: np.ndarray, piece: np.ndarray, ypos: int | None):
    """(x, y, ncc) of the piece IMAGE's top-left on the clean picture by its own pixels.

    Decoy gaps can share the real gap's shape and row, so the truth must come from the
    piece content, not from the shape: on the clean picture the piece's fine texture
    (high-pass, which ignores the piece's bevel lighting) matches only where it was cut
    out. Rejected (None) when the match is weak or not unique."""
    core = cv2.erode((piece[:, :, 3] > 250).astype(np.uint8), np.ones((9, 9), np.uint8))
    if core.sum() < 50:
        return None
    s = cv2.matchTemplate(_hp(clean), np.ascontiguousarray(_hp(piece[:, :, :3])),
                          cv2.TM_CCOEFF_NORMED, mask=np.repeat(core[:, :, None], 3, 2).astype(np.float32))
    s = np.nan_to_num(s, nan=-1.0, posinf=-1.0, neginf=-1.0)
    if ypos is not None:
        band = np.full_like(s, -1.0)
        lo, hi = max(0, ypos - 2), min(s.shape[0], ypos + 3)
        band[lo:hi] = s[lo:hi]
        s = band
    yy, xx = np.unravel_index(int(np.argmax(s)), s.shape)
    best = float(s[yy, xx])
    rest = s.copy()
    rest[max(0, yy - 6):yy + 7, max(0, xx - 6):xx + 7] = -1.0
    if best < MIN_NCC or best - float(rest.max()) < MIN_MARGIN:
        return None
    return int(xx), int(yy), best


def main() -> int:
    dirs = sorted(p for p in DATA.iterdir() if (p / "bg.png").is_file())
    imgs = [np.asarray(Image.open(p / "bg.png").convert("RGB")) for p in dirs]
    th = np.array([np.asarray(Image.fromarray(a).convert("L").resize((30, 20)), float).ravel() for a in imgs])
    n = len(dirs)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    D = np.abs(th[:, None] - th[None]).mean(2)
    for i in range(n):
        for j in range(i + 1, n):
            if D[i, j] < 6 and imgs[i].shape == imgs[j].shape:
                # confirm at full resolution: most pixels identical
                same = (np.abs(imgs[i].astype(int) - imgs[j].astype(int)).max(2) < 12).mean()
                if same > 0.8:
                    parent[find(i)] = find(j)
    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    done = skipped = 0
    for members in groups.values():
        # the demo sometimes serves the very same challenge twice: count distinct pictures only
        uniq = list({imgs[i].tobytes(): i for i in members}.values())
        if len(uniq) < MIN_GROUP:
            skipped += len(members)
            continue
        clean = np.median(np.stack([imgs[i] for i in uniq]), axis=0).astype(np.uint8)
        for i in members:
            meta = json.loads((dirs[i] / "meta.json").read_text())
            r = content_truth(clean, np.asarray(Image.open(dirs[i] / "slice.png").convert("RGBA")),
                              meta.get("ypos"))
            if r is None:
                skipped += 1
                continue
            x, y, score = r
            (dirs[i] / "truth.json").write_text(json.dumps(
                {"x": x, "y": y, "score": score, "group_size": len(uniq), "method": "pool-median content"}))
            done += 1
    print(f"truth for {done} samples, {skipped} without a big enough picture group")
    return 0


if __name__ == "__main__":
    sys.exit(main())
