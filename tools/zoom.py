#!/usr/bin/env python3
"""Render tiles of review sheets at full size: zoom.py SHEET:POS [SHEET:POS ...]"""
import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sel = [a.split(":") for a in sys.argv[1:]]
W = Image.new("RGB", (len(sel) * 152, 165), "white")
d = ImageDraw.Draw(W)
for k, (s, p) in enumerate(sel):
    g, i = json.loads((ROOT / "data" / "sheets" / f"sheet_{s}.json").read_text())["tiles"][p]
    W.paste(Image.open(ROOT / "data" / "grids" / g / f"tile_{i}.jpg").resize((150, 150)), (k * 152, 15))
    d.text((k * 152 + 2, 1), f"{s}:{p}", fill="red")
W.save(ROOT / "data" / "sheets" / "zoom.png")
