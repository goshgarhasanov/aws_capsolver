#!/usr/bin/env python3
"""Build weights/geetest_nine_icons.pt: the v4 "nine" prompt icons -> what they show.

The prompt of a v4 nine-grid is a black line icon (e.g. a fish); the cells to pick are
photos of that thing. The icons come from a fixed set: every icon collected from the
official demo (data/geetest/_static/static_resources/nine/...) was named by hand once
(LABELS, in the order of the sorted resource paths). The solver recognises a prompt icon
by its shape (32x32 alpha mask) and then scores the cells with CLIP zero-shot on the
label; unknown icons fall back to CLIP image similarity.

    .venv/bin/python tools/geetest_nine_icons.py
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# hand labels, index = position in the sorted list of collected prompt-icon paths
LABELS = [
    "shuttlecock", "power strip", "pen", "dog", "railway track", "tire", "lipstick", "helmet", "smoking pipe",
    "glasses", "sheep", "penguin", "seashell", "socks", "zipper", "deer", "soccer ball", "keyboard", "key",
    "computer mouse", "wallet", "comb", "tiger", "baseball cap", "calculator", "manhole cover", "ladle",
    "stapler", "traffic light", "giraffe", "balloons", "toothbrush", "lion", "rooster", "kangaroo", "globe",
    "fountain", "scissors", "book", "fish", "binoculars", "desk lamp", "suitcase", "refrigerator", "bridge",
    "bowl", "axe", "crocodile", "butterfly", "chair", "kettle", "camel", "rabbit", "printer", "table",
    "bird", "stethoscope", "fork", "umbrella", "motorcycle", "necktie", "bull", "electric drill",
    "steering wheel", "turtle", "swing", "button", "panda", "flashlight", "monkey", "cat", "bus",
    "wristwatch", "fly swatter", "billiards", "camera", "ship", "ambulance", "wheelchair", "cooking pot",
    "roller coaster", "loudspeaker", "pig", "toy blocks", "gears", "rocket",
]


def main() -> int:
    import numpy as np
    import torch
    from PIL import Image

    from aws_capsolver.geetest.nine import icon_signature

    paths = sorted({json.load(open(f))["ques"][0] for f in glob.glob(str(ROOT / "data/geetest/v4_nine/*/meta.json"))})
    if len(paths) != len(LABELS):
        print(f"{len(paths)} icons collected but {len(LABELS)} labels: label the new ones first")
        return 1
    sigs = []
    for p in paths:
        im = Image.open(ROOT / "data/geetest/_static" / p).convert("RGBA")
        sigs.append(icon_signature(im))
    out = ROOT / "weights" / "geetest_nine_icons.pt"
    torch.save({"labels": LABELS, "signatures": torch.from_numpy(np.stack(sigs)), "version": 1}, out)
    print(f"wrote {out} ({len(LABELS)} icons)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
