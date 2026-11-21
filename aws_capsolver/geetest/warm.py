"""Warm-up: one dummy inference per solver so the first real request is fast."""
from __future__ import annotations

import io
import logging

log = logging.getLogger(__name__)


def _png(size, color, mode="RGB") -> bytes:
    from PIL import Image

    buf = io.BytesIO()
    Image.new(mode, size, color).save(buf, "PNG")
    return buf.getvalue()


def run() -> None:
    from PIL import Image, ImageDraw

    from . import board, slide
    from .errors import GeeTestError

    bg = Image.new("RGB", (300, 200), (120, 160, 200))
    ImageDraw.Draw(bg).rectangle([150, 60, 199, 109], fill=(60, 80, 100))
    buf = io.BytesIO()
    bg.save(buf, "PNG")
    piece = Image.new("RGBA", (60, 60), (0, 0, 0, 0))
    ImageDraw.Draw(piece).rectangle([5, 5, 54, 54], fill=(120, 160, 200, 255))
    pbuf = io.BytesIO()
    piece.save(pbuf, "PNG")
    slide.solve(buf.getvalue(), pbuf.getvalue())
    board.solve_matrix([[1, 1, 1, 1, 0], [2, 2, 0, 2, 1], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]])
    for name in ("nine", "click"):
        try:
            mod = __import__(f"{__package__}.{name}", fromlist=["warm"])
            if hasattr(mod, "warm"):
                mod.warm()
        except GeeTestError as exc:
            log.info("geetest %s warm-up skipped: %s", name, exc)
