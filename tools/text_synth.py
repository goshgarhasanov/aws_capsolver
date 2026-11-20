"""Synthetic generator for the BLS Poland login captcha.

The site uses the CodeIgniter 3 ``captcha_helper`` (``create_captcha``) output:
160x55 truecolor GD image, white background, a 20-ring "spiral" of 1px lines,
5 digits written one by one with ``imagettftext`` (angle 0 for words shorter
than 6 chars) at a random baseline in [h/2, h-3], saved with ``imagejpeg``
at the IJG default quality (75).

Measured on the real samples (tools/text_data.py ``measure``):
  * glyphs are Open Sans Regular (Noto Sans is metrically near-identical),
    digit height 20 px  ->  em ~= 28 px (21 pt at GD's 96 dpi);
  * pen advance per character ~= 22.9 px (not 21 as stock CI would give, so the
    site passes a slightly different font_size / advance; we sample 22..24);
  * first pen x in [0, 48]  (= mt_rand(0, img_width / (length / 1.5)));
  * glyph bottom row in [27, 52]  (= mt_rand(img_height / 2, img_height - 3));
  * no border (border colour equals the background).

The spiral is a direct port of the CI3 loop::

    theta=1; thetac=7; radius=16; circles=20; points=32
    for i in 0 .. circles*points-2:
        theta += thetac; rad = radius*(i/points)
        x = rad*cos(theta)+x_axis; y = rad*sin(theta)+y_axis
        theta += thetac; rad1 = radius*((i+1)/points)
        x1 = rad1*cos(theta)+x_axis; y1 = rad1*sin(theta)+y_axis
        imageline(x, y, x1, y1, grid); theta -= thetac

with x_axis = mt_rand(6, 360/length - 16) and y_axis = mt_rand(h, w) for angle >= 0.

Usage:
    .venv/bin/python tools/text_synth.py --n 16 --out /tmp/x   # dump samples
"""

from __future__ import annotations

import argparse
import io
import math
import os
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

ROOT = Path(__file__).resolve().parents[1]
FONT_DIR = ROOT / "data" / "text" / "fonts"

W, H = 160, 55
LENGTH = 5
DIGITS = "0123456789"

# Candidate fonts in preference order; the first is the measured match.
_FONT_CANDIDATES = [
    FONT_DIR / "OpenSans-Regular.ttf",
    Path("/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf"),
    Path("/usr/share/fonts/truetype/adwaita/AdwaitaSans-Regular.ttf"),
]


def available_fonts() -> list[Path]:
    fonts = [p for p in _FONT_CANDIDATES if p.exists()]
    if not fonts:
        raise FileNotFoundError(
            "no font found; copy OpenSans-Regular.ttf into data/text/fonts/")
    return fonts


_font_cache: dict[tuple[str, int], ImageFont.FreeTypeFont] = {}


def _font(path: Path, px: int) -> ImageFont.FreeTypeFont:
    key = (str(path), px)
    f = _font_cache.get(key)
    if f is None:
        f = _font_cache[key] = ImageFont.truetype(str(path), px)
    return f


def _spiral(draw: ImageDraw.ImageDraw, rng: random.Random, color) -> None:
    x_axis = rng.randint(6, int(360 / LENGTH) - 16)
    y_axis = rng.randint(H, W)
    theta, thetac, radius, circles, points = 1.0, 7.0, 16.0, 20, 32
    for i in range(circles * points - 1):
        theta += thetac
        rad = radius * (i / points)
        x = rad * math.cos(theta) + x_axis
        y = rad * math.sin(theta) + y_axis
        theta += thetac
        rad1 = radius * ((i + 1) / points)
        x1 = rad1 * math.cos(theta) + x_axis
        y1 = rad1 * math.sin(theta) + y_axis
        # GD casts the float coordinates to int (truncation toward zero).
        draw.line((int(x), int(y), int(x1), int(y1)), fill=color, width=1)
        theta -= thetac


def render(text: str, rng: random.Random, *, faithful: bool = False) -> bytes:
    """Render one captcha and return JPEG bytes.

    ``faithful=True`` uses only the measured parameters (for side-by-side
    comparison); otherwise nuisance parameters are jittered for robustness.
    """
    fonts = available_fonts()
    if faithful:
        font_path, em, adv, quality = fonts[0], 28, 23, 75
        grid = (220, 30, 40)
    else:
        font_path = fonts[0] if rng.random() < 0.7 else rng.choice(fonts)
        em = rng.choice((27, 28, 28, 28, 29))
        adv = rng.uniform(21.5, 24.0)
        quality = rng.choice((60, 70, 75, 75, 75, 80, 90))
        grid = (rng.randint(180, 255), rng.randint(0, 70), rng.randint(0, 80))

    img = Image.new("RGB", (W, H), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    _spiral(draw, rng, grid)
    if not faithful and rng.random() < 0.3:
        _spiral(draw, rng, grid)  # denser background, harder negatives

    font = _font(font_path, em)
    ascent = font.getmetrics()[0]
    x = float(rng.randint(0, int(W / (LENGTH / 1.5))))
    for ch in text:
        baseline = rng.randint(H // 2, H - 3)
        if not faithful:
            baseline += rng.choice((0, 0, 0, -1, 1))
        # PIL anchors text at the ascender line; shift so ``baseline`` is the baseline.
        draw.text((int(x), baseline - ascent), ch, font=font, fill=(0, 0, 0))
        x += adv

    if not faithful and rng.random() < 0.15:
        img = img.filter(ImageFilter.GaussianBlur(rng.uniform(0.2, 0.5)))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def random_text(rng: random.Random) -> str:
    return "".join(rng.choice(DIGITS) for _ in range(LENGTH))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--faithful", action="store_true")
    args = ap.parse_args()
    rng = random.Random(args.seed)
    os.makedirs(args.out, exist_ok=True)
    for i in range(args.n):
        t = random_text(rng)
        Path(args.out, f"{i:05d}_{t}.jpg").write_bytes(render(t, rng, faithful=args.faithful))


if __name__ == "__main__":
    main()
