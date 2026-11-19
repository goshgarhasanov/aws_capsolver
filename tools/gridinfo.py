#!/usr/bin/env python3
"""Show the grid of a sheet tile: gridinfo.py SHEET:POS -> data/sheets/grid.png + target/pseudo."""
import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
s, p = sys.argv[1].split(":")
g, i = json.loads((ROOT / "data" / "sheets" / f"sheet_{s}.json").read_text())["tiles"][p]
meta = json.loads((ROOT / "data" / "grids" / g / "meta.json").read_text())
pseudo = json.loads((ROOT / "data" / "grids" / "pseudo.json").read_text()).get(g)
W = Image.new("RGB", (3 * 122, 3 * 122 + 14), "white")
d = ImageDraw.Draw(W)
d.text((2, 1), f"{g} target={meta['target']} tile={i}", fill="red")
for k in range(9):
    W.paste(Image.open(ROOT / "data" / "grids" / g / f"tile_{k}.jpg").resize((120, 120)),
            ((k % 3) * 122, 14 + (k // 3) * 122))
    d.text(((k % 3) * 122 + 2, 16 + (k // 3) * 122), str(k), fill="yellow" if k != i else "red")
W.save(ROOT / "data" / "sheets" / "grid.png")
print(g, meta["target"], "tile", i, [c for c, _ in pseudo] if pseudo else None)
