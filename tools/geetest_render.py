"""Render GeeTest-like pictures from collected official assets (used by bench + tests).

* gobang boards: the ``ques`` matrix of a collected v4 "winlinze" sample drawn with the
  sample's own piece images, in the two looks a solver sees - the extension's canvas
  (pieces on white, empty cells blank) and a widget screenshot (grey board, empty cells
  as faint grey discs) - optionally rescaled and JPEG compressed.
"""
from __future__ import annotations

import io
import random
from pathlib import Path

from PIL import Image, ImageDraw


def render_gobang(ques: list[list[int]], pieces: list[Image.Image], style: str = "canvas",
                  cell: int = 80, scale: float = 1.0, jpeg: int | None = None, seed: int = 0) -> bytes:
    rnd = random.Random(seed)
    n = len(ques)
    if style == "canvas":
        board = Image.new("RGBA", (n * cell, n * cell), (255, 255, 255, 255))
    else:
        board = Image.new("RGBA", (n * cell, n * cell), (220, 226, 236, 255))
    d = ImageDraw.Draw(board)
    for r in range(n):
        for c in range(n):
            v = ques[r][c]
            x, y = c * cell, r * cell
            if v == 0:
                if style != "canvas":
                    m = int(cell * 0.1)
                    d.ellipse([x + m, y + m, x + cell - m, y + cell - m], fill=(205, 212, 224, 255))
                continue
            p = pieces[v].convert("RGBA")
            size = int(cell * (0.8 if style != "canvas" else 1.0))
            p = p.resize((size, size), Image.BILINEAR)
            off = (cell - size) // 2
            board.alpha_composite(p, (x + off, y + off))
    im = board.convert("RGB")
    if abs(scale - 1.0) > 1e-3:
        im = im.resize((max(20, int(im.width * scale)), max(20, int(im.height * scale))), Image.BILINEAR)
    buf = io.BytesIO()
    if jpeg:
        im.save(buf, "JPEG", quality=jpeg)
    else:
        im.save(buf, "PNG")
    return buf.getvalue()


def load_gobang_pieces(sample_dir: Path) -> list[Image.Image]:
    return [Image.open(sample_dir / f"piece_{i}.png").convert("RGBA") for i in range(5)]
