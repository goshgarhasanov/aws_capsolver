"""
Synthetic text-captcha generator for the general model (aws_capsolver/text/general.py).

One call to `Generator.sample()` returns (PIL RGB image, label). Every sample draws a
style family and then randomises everything inside it, so the stream covers:

  pycaptcha     the python `captcha` library's ImageCaptcha with random fonts / colours
  claptcha      claptcha-like: plain text, white-noise blend, squashing resize
  gd            PHP GD imagestring(): bitmap font scaled up, random pixels and lines
  codeigniter   CodeIgniter create_captcha(): grid + circles, letters on a sine baseline
  securimage    Securimage-like: wave-warped text, arcs, signature curves, textures
  simple        SimpleCaptcha-like: outlined / hollow glyphs on a gradient
  botdetect     BotDetect-like: mosaic / chess / fragmented / stripe-masked glyphs
  general       per-character random fonts, rotation, overlap, scaling, and a random
                stack of noise lines, curves, dots, grids, strike-through, warps,
                coloured / textured backgrounds, inversion, blur and JPEG artefacts

Text: 3-10 characters, digits / lower / upper / mixed alphabets, sometimes without the
ambiguous glyphs. Fonts: every Latin-capable system font (fc-list) plus the OFL fonts in
data/text_general/fonts (tools/general_data.py fonts downloads them).

    .venv/bin/python tools/general_synth.py --out /tmp/synth --n 64   # preview grid
"""

import argparse
import functools
import io
import json
import math
import os
import random
import string
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageOps

ROOT = Path(__file__).resolve().parents[1]
FONT_DIR = ROOT / "data" / "text_general" / "fonts"
FONT_CACHE = ROOT / "data" / "text_general" / "fonts.json"

AMBIGUOUS = set("0Oo1lIi")
# fonts that do not draw Latin letters as letters
_FONT_BLOCKLIST = ("fontawesome", "customtkinter", "opensymbol", "standardsymbols",
                   "mathjax_ams", "mathjax_size", "mathjax_vector", "mathjax_winie6",
                   "mathjax_winchrome", "d050000l", "symbol", "emoji", "z003",
                   "mathjax_script", "mathjax_fraktur", "mathjax_caligraphic", "unifont")


# fonts -------------------------------------------------------------------------
def _system_fonts() -> List[str]:
    try:
        out = subprocess.run(["fc-list", "--format", "%{file}\n"], capture_output=True,
                             text=True, timeout=30).stdout.split()
    except (OSError, subprocess.SubprocessError):
        out = []
    files = [f for f in out if f.lower().endswith((".ttf", ".otf"))]
    files += [str(p) for p in FONT_DIR.glob("*.ttf")] + [str(p) for p in FONT_DIR.glob("*.otf")]
    return sorted(set(files))


def _font_ok(path: str) -> bool:
    """True when the font draws distinct, non-empty glyphs for the whole alphabet."""
    name = os.path.basename(path).lower()
    if any(b in name for b in _FONT_BLOCKLIST):
        return False
    try:
        font = ImageFont.truetype(path, 32)
    except OSError:
        return False
    seen = set()
    for ch in string.ascii_letters + string.digits:
        img = Image.new("L", (48, 48), 0)
        ImageDraw.Draw(img).text((8, 4), ch, font=font, fill=255)
        arr = np.asarray(img)
        if arr.sum() < 255 * 8:
            return False
        seen.add(arr.tobytes())
    # tofu boxes render identical glyphs; case-identical fonts (small caps) are fine
    return len(seen) >= 50


def load_fonts() -> List[str]:
    """Usable fonts, cached in data/text_general/fonts.json; Noto is thinned out."""
    if FONT_CACHE.exists():
        fonts = [f for f in json.loads(FONT_CACHE.read_text()) if os.path.exists(f)]
        if fonts:
            return fonts
    fonts = []
    noto = 0
    for path in _system_fonts():
        base = os.path.basename(path)
        if base.startswith("Noto"):
            # hundreds of script-specific Noto fonts share the same Latin glyphs
            if not any(k in base for k in ("NotoSans-", "NotoSerif-", "NotoSansMono-",
                                           "NotoSansDisplay", "NotoSerifDisplay")):
                continue
            noto += 1
            if noto > 12:
                continue
        if _font_ok(path):
            fonts.append(path)
    FONT_CACHE.parent.mkdir(parents=True, exist_ok=True)
    FONT_CACHE.write_text(json.dumps(fonts, indent=0))
    return fonts


@functools.lru_cache(maxsize=4096)
def _font(path: str, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(path, size)


# text --------------------------------------------------------------------------
def random_text(rng: random.Random) -> Tuple[str, bool]:
    """(text, case_matters). Lengths 3..10, mostly 4..6."""
    length = rng.choices(range(3, 11), weights=[5, 22, 26, 22, 10, 6, 5, 4])[0]
    mode = rng.choices(["digits", "lower", "upper", "mixed", "letters_lower",
                        "letters_upper", "letters_mixed"],
                       weights=[16, 26, 18, 18, 8, 6, 8])[0]
    pool = {
        "digits": string.digits,
        "lower": string.ascii_lowercase + string.digits,
        "upper": string.ascii_uppercase + string.digits,
        "mixed": string.ascii_letters + string.digits,
        "letters_lower": string.ascii_lowercase,
        "letters_upper": string.ascii_uppercase,
        "letters_mixed": string.ascii_letters,
    }[mode]
    if mode != "digits" and rng.random() < 0.35:
        pool = "".join(c for c in pool if c not in AMBIGUOUS)
    text = "".join(rng.choice(pool) for _ in range(length))
    return text, mode in ("mixed", "letters_mixed")


# colours and backgrounds -------------------------------------------------------
def _rand_color(rng, lo=0, hi=255):
    return tuple(rng.randint(lo, hi) for _ in range(3))


def _lum(c):
    return 0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2]


def _contrasting(rng, bg, min_diff=90):
    if isinstance(bg, Image.Image):
        bg = tuple(int(v) for v in np.asarray(bg, dtype=np.float32).reshape(-1, 3).mean(0))
    for _ in range(30):
        c = _rand_color(rng)
        if abs(_lum(c) - _lum(bg)) >= min_diff:
            return c
    return (0, 0, 0) if _lum(bg) > 127 else (255, 255, 255)


def background(rng, w, h, dark: bool = False) -> Image.Image:
    kind = rng.choices(["solid", "gradient", "noise", "texture", "stripes", "shapes",
                        "blotches"], weights=[34, 20, 12, 12, 8, 7, 7])[0]
    base = _rand_color(rng, 0, 70) if dark else _rand_color(rng, 170, 255)
    if kind == "solid":
        return Image.new("RGB", (w, h), base)
    if kind == "gradient":
        other = _rand_color(rng, 0, 90) if dark else _rand_color(rng, 150, 255)
        t = np.linspace(0, 1, w if rng.random() < 0.7 else h, dtype=np.float32)
        if len(t) == w:
            t = np.tile(t[None, :], (h, 1))
        else:
            t = np.tile(t[:, None], (1, w))
        arr = (np.array(base)[None, None] * (1 - t[..., None])
               + np.array(other)[None, None] * t[..., None])
        return Image.fromarray(arr.astype(np.uint8))
    if kind == "noise":
        amp = rng.randint(10, 45)
        arr = np.array(base, dtype=np.float32)[None, None] + np.random.default_rng(
            rng.getrandbits(32)).normal(0, amp, (h, w, 1 if rng.random() < 0.5 else 3))
        return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8).repeat(
            3 // arr.shape[2], axis=2) if arr.shape[2] == 1 else np.clip(arr, 0, 255)
            .astype(np.uint8))
    if kind == "texture":
        # smooth random field (upsampled low-res noise), coloured
        g = np.random.default_rng(rng.getrandbits(32))
        small = g.random((max(2, h // rng.randint(4, 16)), max(2, w // rng.randint(4, 16)), 3))
        field = np.asarray(Image.fromarray((small * 255).astype(np.uint8)).resize(
            (w, h), Image.BICUBIC), dtype=np.float32) / 255.0
        amp = rng.uniform(0.15, 0.45)
        arr = np.array(base, dtype=np.float32)[None, None] * (1 - amp) + field * 255 * amp
        return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    img = Image.new("RGB", (w, h), base)
    d = ImageDraw.Draw(img)
    if kind == "stripes":
        step = rng.randint(3, 12)
        col = _rand_color(rng, 0, 120) if dark else _rand_color(rng, 120, 240)
        slope = rng.uniform(-1.5, 1.5)
        for x in range(-2 * h, w + 2 * h, step):
            d.line([(x, 0), (x + slope * h, h)], fill=col, width=rng.randint(1, 2))
        return img
    if kind == "shapes":
        for _ in range(rng.randint(4, 16)):
            col = _rand_color(rng, 0, 80) if dark else _rand_color(rng, 165, 255)
            x, y = rng.randint(-10, w), rng.randint(-10, h)
            r = rng.randint(3, max(4, h // 2))
            if rng.random() < 0.5:
                d.ellipse([x, y, x + r, y + r], fill=col if rng.random() < 0.6 else None,
                          outline=col)
            else:
                d.rectangle([x, y, x + r, y + r], fill=col if rng.random() < 0.6 else None,
                            outline=col)
        return img
    # blotches: soft coloured spots
    for _ in range(rng.randint(3, 10)):
        col = _rand_color(rng, 0, 80) if dark else _rand_color(rng, 160, 255)
        x, y, r = rng.randint(0, w), rng.randint(0, h), rng.randint(h // 4, h)
        d.ellipse([x - r, y - r, x + r, y + r], fill=col)
    return img.filter(ImageFilter.GaussianBlur(rng.uniform(2, 6)))


# glyph rendering ---------------------------------------------------------------
def render_char(ch: str, font: ImageFont.FreeTypeFont, color, rng, rotate: float = 0.0,
                stroke: int = 0, stroke_fill=None, hollow: bool = False,
                shear: float = 0.0, stretch: float = 1.0) -> Image.Image:
    """One glyph on a transparent RGBA tile, tightly cropped, optionally transformed."""
    bbox = font.getbbox(ch, stroke_width=stroke)
    w = max(1, bbox[2] - bbox[0]) + 2 * stroke + 4
    h = max(1, bbox[3] - bbox[1]) + 2 * stroke + 4
    tile = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(tile)
    pos = (2 + stroke - bbox[0], 2 + stroke - bbox[1])
    if hollow:
        d.text(pos, ch, font=font, fill=(0, 0, 0, 0), stroke_width=max(1, stroke),
               stroke_fill=color + (255,))
    else:
        d.text(pos, ch, font=font, fill=color + (255,), stroke_width=stroke,
               stroke_fill=(stroke_fill or color) + (255,))
    if stretch != 1.0:
        tile = tile.resize((max(1, int(w * stretch)), h), Image.BICUBIC)
    if shear:
        tw, th = tile.size
        extra = int(abs(shear) * th)
        tile = tile.transform((tw + extra, th), Image.AFFINE,
                              (1, shear, -extra if shear > 0 else 0, 0, 1, 0), Image.BICUBIC)
    if rotate:
        tile = tile.rotate(rotate, resample=Image.BICUBIC, expand=True)
    return tile


def place_text(canvas: Image.Image, text: str, fonts: List[str], rng, *, colors=None,
               size_range=(0.55, 0.85), rot=25.0, overlap=(0.0, 0.25), per_char_font=True,
               baseline_wave: float = 0.0, stroke=0, stroke_fill=None, hollow=False,
               shear=0.0, stretch=(1.0, 1.0), shadow=None, jitter_y=0.15,
               margin_x=(0.03, 0.12)) -> None:
    """Draw `text` glyph by glyph across `canvas`, scaled to fit its width."""
    W, H = canvas.size
    font_path = rng.choice(fonts)
    tiles = []
    for i, ch in enumerate(text):
        fp = rng.choice(fonts) if per_char_font else font_path
        size = max(8, int(H * rng.uniform(*size_range)))
        col = colors[i % len(colors)] if colors else (0, 0, 0)
        sh = rng.uniform(-shear, shear) if shear else 0.0
        tiles.append(render_char(ch, _font(fp, size), col, rng, rng.uniform(-rot, rot),
                                 stroke=stroke, stroke_fill=stroke_fill, hollow=hollow,
                                 shear=sh, stretch=rng.uniform(*stretch)))
    ov = [rng.uniform(*overlap) for _ in tiles]
    total = sum(t.size[0] for t in tiles) - sum(int(t.size[0] * o) for t, o in zip(tiles[1:], ov))
    mx = int(W * rng.uniform(*margin_x))
    avail = W - 2 * mx
    max_h = max(t.size[1] for t in tiles)
    scale = min(1.0, avail / max(1, total), (H * 0.98) / max(1, max_h))
    if scale < 1.0:
        tiles = [t.resize((max(1, int(t.size[0] * scale)), max(1, int(t.size[1] * scale))),
                          Image.BICUBIC) for t in tiles]
        total = int(total * scale)
    x = mx + rng.randint(0, max(0, avail - total))
    phase, freq = rng.uniform(0, 2 * math.pi), rng.uniform(0.5, 2.0)
    for i, t in enumerate(tiles):
        tw, th = t.size
        cy = (H - th) / 2 + rng.uniform(-jitter_y, jitter_y) * H
        if baseline_wave:
            cy += baseline_wave * H * math.sin(phase + freq * 2 * math.pi * i / max(1, len(tiles)))
        y = int(min(max(cy, -th * 0.1), H - th * 0.9))
        if shadow is not None:
            sh_tile = Image.new("RGBA", t.size, shadow + (0,))
            sh_tile.putalpha(t.getchannel("A"))
            canvas.paste(sh_tile, (x + 2, y + 2), sh_tile)
        canvas.paste(t, (x, y), t)
        x += tw - (int(tiles[i + 1].size[0] * ov[i + 1]) if i + 1 < len(tiles) else 0)


# noise -------------------------------------------------------------------------
def _bezier(p, n=40):
    t = np.linspace(0, 1, n)[:, None]
    p = np.array(p, dtype=np.float32)
    if len(p) == 3:
        pts = (1 - t) ** 2 * p[0] + 2 * (1 - t) * t * p[1] + t ** 2 * p[2]
    else:
        pts = ((1 - t) ** 3 * p[0] + 3 * (1 - t) ** 2 * t * p[1] + 3 * (1 - t) * t ** 2 * p[2]
               + t ** 3 * p[3])
    return [tuple(q) for q in pts]


def noise_lines(img, rng, n, colors, width=(1, 3), kind="mixed"):
    d = ImageDraw.Draw(img)
    W, H = img.size
    for _ in range(n):
        col = rng.choice(colors)
        wdt = rng.randint(*width)
        k = kind if kind != "mixed" else rng.choice(["line", "curve", "arc", "sine"])
        if k == "line":
            d.line([(rng.randint(-5, W // 3), rng.randint(0, H)),
                    (rng.randint(2 * W // 3, W + 5), rng.randint(0, H))], fill=col, width=wdt)
        elif k == "curve":
            pts = [(rng.uniform(-5, W * 0.2), rng.uniform(0, H)),
                   (rng.uniform(0, W), rng.uniform(-H * 0.5, H * 1.5)),
                   (rng.uniform(0, W), rng.uniform(-H * 0.5, H * 1.5)),
                   (rng.uniform(W * 0.8, W + 5), rng.uniform(0, H))]
            d.line(_bezier(pts), fill=col, width=wdt, joint="curve")
        elif k == "arc":
            x0, y0 = rng.randint(-W // 2, W // 2), rng.randint(-H, H // 2)
            d.arc([x0, y0, x0 + rng.randint(W // 3, W * 2), y0 + rng.randint(H // 2, H * 2)],
                  rng.randint(0, 360), rng.randint(0, 360), fill=col, width=wdt)
        else:
            amp, f, ph = rng.uniform(0.05, 0.3) * H, rng.uniform(0.5, 3), rng.uniform(0, 6.3)
            yc = rng.uniform(0.25, 0.75) * H
            pts = [(x, yc + amp * math.sin(ph + f * 2 * math.pi * x / W)) for x in range(0, W, 3)]
            d.line(pts, fill=col, width=wdt)


def noise_dots(img, rng, density, colors, size=(1, 2)):
    d = ImageDraw.Draw(img)
    W, H = img.size
    for _ in range(int(W * H * density)):
        x, y = rng.randint(0, W - 1), rng.randint(0, H - 1)
        r = rng.randint(*size) - 1
        col = rng.choice(colors)
        if r <= 0:
            d.point((x, y), fill=col)
        else:
            d.ellipse([x - r, y - r, x + r, y + r], fill=col)


def grid(img, rng, color, step=None, width=1):
    d = ImageDraw.Draw(img)
    W, H = img.size
    step = step or rng.randint(5, 14)
    if rng.random() < 0.8:
        for x in range(rng.randint(0, step), W, step):
            d.line([(x, 0), (x, H)], fill=color, width=width)
    if rng.random() < 0.8:
        for y in range(rng.randint(0, step), H, step):
            d.line([(0, y), (W, y)], fill=color, width=width)


def strike(img, rng, color, width):
    """A line through the text band (the classic strike-through)."""
    W, H = img.size
    d = ImageDraw.Draw(img)
    y0, y1 = rng.uniform(0.35, 0.65) * H, rng.uniform(0.35, 0.65) * H
    if rng.random() < 0.5:
        d.line([(rng.uniform(0, W * 0.15), y0), (rng.uniform(W * 0.85, W), y1)],
               fill=color, width=width)
    else:
        pts = [(0, y0), (W * 0.33, rng.uniform(0.2, 0.8) * H),
               (W * 0.66, rng.uniform(0.2, 0.8) * H), (W, y1)]
        d.line(_bezier(pts, 60), fill=color, width=width, joint="curve")


# geometric distortions ---------------------------------------------------------
def wave_warp(img, rng, amp_x=0.0, amp_y=0.08, fill=None):
    import cv2
    arr = np.asarray(img)
    H, W = arr.shape[:2]
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    fx, fy = rng.uniform(0.5, 3.0), rng.uniform(0.5, 2.5)
    px, py = rng.uniform(0, 6.3), rng.uniform(0, 6.3)
    map_x = xx + amp_x * H * np.sin(py + 2 * np.pi * fy * yy / H)
    map_y = yy + amp_y * H * np.sin(px + 2 * np.pi * fx * xx / W)
    border = tuple(int(v) for v in (fill if fill is not None else arr[0, 0]))
    out = cv2.remap(arr, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                    borderValue=border)
    return Image.fromarray(out)


def elastic(img, rng, alpha=None, sigma=None):
    import cv2
    arr = np.asarray(img)
    H, W = arr.shape[:2]
    g = np.random.default_rng(rng.getrandbits(32))
    alpha = alpha if alpha is not None else rng.uniform(1.5, 5.0)
    sigma = sigma if sigma is not None else rng.uniform(3.0, 7.0)
    dx = cv2.GaussianBlur(g.uniform(-1, 1, (H, W)).astype(np.float32), (0, 0), sigma)
    dy = cv2.GaussianBlur(g.uniform(-1, 1, (H, W)).astype(np.float32), (0, 0), sigma)
    dx *= alpha / (np.abs(dx).max() + 1e-6)
    dy *= alpha / (np.abs(dy).max() + 1e-6)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    out = cv2.remap(arr, xx + dx, yy + dy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return Image.fromarray(out)


def perspective(img, rng, amount=0.08):
    W, H = img.size
    d = lambda s: rng.uniform(-amount, amount) * s  # noqa: E731
    quad = (d(W), d(H), d(W), H + d(H), W + d(W), H + d(H), W + d(W), d(H))
    return img.transform((W, H), Image.QUAD, quad, Image.BICUBIC, fillcolor=img.getpixel((0, 0)))


def finish(img, rng, p_blur=0.25, p_jpeg=0.35):
    """Camera / encoder artefacts shared by all styles."""
    if rng.random() < p_blur:
        img = img.filter(ImageFilter.GaussianBlur(rng.uniform(0.3, 1.1)))
    if rng.random() < 0.12:
        img = img.filter(ImageFilter.SHARPEN)
    if rng.random() < p_jpeg:
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=rng.randint(25, 90))
        img = Image.open(io.BytesIO(buf.getvalue())).convert("RGB")
    return img


# styles ------------------------------------------------------------------------
class Generator:
    STYLES = ("general", "pycaptcha", "claptcha", "gd", "codeigniter", "securimage",
              "simple", "botdetect")
    WEIGHTS = (40, 14, 5, 7, 8, 11, 7, 8)

    def __init__(self, seed: Optional[int] = None, fonts: Optional[List[str]] = None):
        self.rng = random.Random(seed)
        self.fonts = fonts or load_fonts()
        if not self.fonts:
            raise RuntimeError("no usable fonts found (fc-list / data/text_general/fonts)")
        self._bitmap = ImageFont.load_default_imagefont() if hasattr(
            ImageFont, "load_default_imagefont") else ImageFont.load_default()

    def sample(self, style: Optional[str] = None) -> Tuple[Image.Image, str, bool]:
        """(RGB image, label, case_matters)."""
        rng = self.rng
        style = style or rng.choices(self.STYLES, weights=self.WEIGHTS)[0]
        text, case_matters = random_text(rng)
        img = getattr(self, "_" + style)(text)
        if img.mode != "RGB":
            img = img.convert("RGB")
        return img, text, case_matters

    def _size(self, n):
        rng = self.rng
        h = rng.choice([30, 36, 40, 45, 50, 50, 60, 64, 70, 80, 90])
        per_char = rng.uniform(0.45, 0.95) * h
        w = int(max(h * 1.6, min(h * 7.5, per_char * n + rng.uniform(0.2, 0.9) * h)))
        return w, h

    # -- styles --
    def _general(self, text):
        rng = self.rng
        W, H = self._size(len(text))
        dark = rng.random() < 0.15
        bg = background(rng, W, H, dark=dark)
        bgc = bg
        mode = rng.random()
        if mode < 0.35:
            colors = [_contrasting(rng, bgc, 110)]
        elif mode < 0.7:
            colors = [_contrasting(rng, bgc, 110) for _ in text]
        else:
            colors = [(255, 255, 255) if dark else (0, 0, 0)]
        noise_cols = colors + [_contrasting(rng, bgc, 40) for _ in range(3)]
        heavy = rng.random() < 0.25   # a minority of really noisy samples
        # noise behind the text
        if rng.random() < 0.35:
            noise_lines(bg, rng, rng.randint(1, 8 if heavy else 4), noise_cols, (1, 2))
        if rng.random() < 0.2:
            grid(bg, rng, _contrasting(rng, bgc, 30))
        place_text(bg, text, self.fonts, rng, colors=colors,
                   size_range=(rng.uniform(0.45, 0.65), rng.uniform(0.7, 0.95)),
                   rot=rng.choice([0, 5, 15, 25, 35]),
                   overlap=(0.0, rng.choice([0.0, 0.1, 0.25, 0.4])),
                   per_char_font=rng.random() < 0.4,
                   baseline_wave=rng.choice([0, 0, 0.05, 0.12]),
                   stroke=rng.choice([0, 0, 0, 1, 2]),
                   hollow=rng.random() < 0.06,
                   shear=rng.choice([0, 0, 0.2, 0.4]),
                   stretch=(rng.uniform(0.7, 1.0), rng.uniform(1.0, 1.4)),
                   shadow=_rand_color(rng, 60, 160) if rng.random() < 0.08 else None)
        # noise in front of the text
        r = rng.random()
        if r < 0.55:
            noise_lines(bg, rng, rng.randint(1, 7 if heavy else 3), noise_cols,
                        (1, rng.choice([2, 3] if heavy else [1, 2])))
        if rng.random() < 0.35:
            noise_dots(bg, rng, rng.uniform(0.005, 0.06 if heavy else 0.025), noise_cols,
                       (1, rng.choice([1, 2, 3] if heavy else [1, 2])))
        if rng.random() < 0.18:
            strike(bg, rng, rng.choice(colors), rng.randint(1, 4))
        if rng.random() < 0.08:
            grid(bg, rng, _contrasting(rng, bgc, 60), width=1)
        # geometry
        g = rng.random()
        if g < 0.3:
            bg = wave_warp(bg, rng, amp_x=rng.uniform(0, 0.04), amp_y=rng.uniform(0.02, 0.09))
        elif g < 0.5:
            bg = elastic(bg, rng)
        elif g < 0.6:
            bg = perspective(bg, rng)
        if rng.random() < 0.08:
            bg = ImageOps.invert(bg)
        return finish(bg, rng)

    def _pycaptcha(self, text):
        from captcha.image import ImageCaptcha
        rng = self.rng
        h = rng.choice([50, 60, 60, 70, 80])
        w = int(h * rng.uniform(2.4, 4.0) * max(1.0, len(text) / 5))
        fonts = [rng.choice(self.fonts) for _ in range(rng.randint(1, 3))]
        gen = ImageCaptcha(width=w, height=h, fonts=fonts,
                           font_sizes=tuple(int(h * f) for f in (0.7, 0.8, 0.9)))
        state = random.getstate()
        random.seed(rng.getrandbits(32))  # the library uses the global random module
        try:
            bg = _rand_color(rng, 200, 255) if rng.random() < 0.7 else _rand_color(rng)
            fg = _contrasting(rng, bg, 80) if rng.random() < 0.8 else _rand_color(rng)
            img = gen.generate_image(text, bg_color=bg, fg_color=fg)
        finally:
            random.setstate(state)
        return finish(img, rng, p_blur=0.1, p_jpeg=0.3)

    def _claptcha(self, text):
        rng = self.rng
        W, H = rng.choice([(200, 80), (150, 50), (240, 80), (180, 60)])
        mx, my = rng.randint(5, 25), rng.randint(3, 20)
        font = _font(rng.choice(self.fonts), 64)
        bbox = font.getbbox(text)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        img = Image.new("RGB", (tw + 2 * mx * 3, th + 2 * my * 3), (255, 255, 255))
        ImageDraw.Draw(img).text((mx * 3 - bbox[0], my * 3 - bbox[1]), text, font=font,
                                 fill=(0, 0, 0))
        noise = rng.uniform(0, 0.9)
        if noise > 0.05:
            g = np.random.default_rng(rng.getrandbits(32))
            n = (g.random(img.size[::-1]) < noise).astype(np.uint8) * 255
            img = Image.blend(img, Image.fromarray(n).convert("RGB"), 0.5)
        img = img.resize((W, H), rng.choice([Image.NEAREST, Image.BILINEAR, Image.LANCZOS]))
        return finish(img, rng, p_blur=0.1, p_jpeg=0.2)

    def _gd(self, text):
        """PHP GD: imagestring() with a built-in bitmap font, scaled like a browser would."""
        rng = self.rng
        bg = _rand_color(rng, 180, 255) if rng.random() < 0.8 else _rand_color(rng, 0, 80)
        fg = _contrasting(rng, bg)
        spacing = rng.randint(0, 6)
        cw = 6 + spacing
        W, H = len(text) * cw + rng.randint(8, 30), rng.randint(16, 26)
        img = Image.new("RGB", (W, H), bg)
        d = ImageDraw.Draw(img)
        x = rng.randint(2, max(3, W - len(text) * cw - 2))
        for ch in text:
            d.text((x, rng.randint(1, max(2, H - 13)) if rng.random() < 0.5 else (H - 11) // 2),
                   ch, font=self._bitmap, fill=fg if rng.random() < 0.7 else _contrasting(rng, bg))
            x += cw
        if rng.random() < 0.7:
            noise_dots(img, rng, rng.uniform(0.01, 0.06), [_rand_color(rng) for _ in range(4)])
        if rng.random() < 0.6:
            noise_lines(img, rng, rng.randint(1, 6), [_rand_color(rng) for _ in range(4)], (1, 1),
                        kind="line")
        scale = rng.uniform(2.0, 4.0)
        img = img.resize((int(W * scale), int(H * scale)),
                         rng.choice([Image.NEAREST, Image.NEAREST, Image.BILINEAR]))
        return finish(img, rng, p_blur=0.1, p_jpeg=0.3)

    def _codeigniter(self, text):
        """create_captcha(): light background, grid lines, circles, sine-placed letters."""
        rng = self.rng
        W, H = rng.choice([(150, 30), (150, 40), (200, 50), (180, 45), (250, 60)])
        W = int(W * max(1.0, len(text) / 8))
        bg = (255, 255, 255) if rng.random() < 0.6 else _rand_color(rng, 200, 255)
        img = Image.new("RGB", (W, H), bg)
        d = ImageDraw.Draw(img)
        grid_col = _rand_color(rng, 150, 230) if rng.random() < 0.5 else (255, 40, 40)
        # spiral of circles like CodeIgniter's "create the spiral pattern"
        theta, thetac = 1.0, 7.0
        radius, circles = 16, 20
        points = 32 - (circles - 4)
        cx, cy = rng.uniform(0, W), rng.uniform(0, H)
        for i in range(int(circles * points) - 1):
            theta += thetac
            rad = radius * (i / points)
            x1, y1 = cx + rad * math.cos(theta), cy + rad * math.sin(theta)
            theta += thetac
            rad1 = radius * ((i + 1) / points)
            x2, y2 = cx + rad1 * math.cos(theta), cy + rad1 * math.sin(theta)
            d.line([(x1, y1), (x2, y2)], fill=grid_col)
            theta -= thetac
        fg = _rand_color(rng, 0, 140)
        if rng.random() < 0.4:
            place_text(img, text, self.fonts, rng, colors=[fg], size_range=(0.5, 0.7), rot=8,
                       overlap=(0.0, 0.05), per_char_font=False, baseline_wave=0.15,
                       jitter_y=0.05)
        else:
            # the TTF-less path: GD bitmap font placed with a random y per letter
            small = Image.new("RGB", (W // 2, H // 2), bg)
            sd = ImageDraw.Draw(small)
            x = rng.randint(2, 10)
            for ch in text:
                sd.text((x, rng.randint(0, max(1, H // 2 - 12))), ch, font=self._bitmap, fill=fg)
                x += rng.randint(6, 9)
            small = small.resize((W, H), Image.NEAREST)
            mask = ImageOps.invert(small.convert("L")).point(lambda v: 255 if v > 40 else 0)
            img.paste(small, (0, 0), mask)
        d = ImageDraw.Draw(img)
        d.rectangle([0, 0, W - 1, H - 1], outline=_rand_color(rng, 120, 220))
        return finish(img, rng, p_blur=0.05, p_jpeg=0.3)

    def _securimage(self, text):
        rng = self.rng
        W, H = rng.choice([(215, 80), (200, 70), (230, 80), (250, 90), (175, 60)])
        W = int(W * max(1.0, len(text) / 6))
        bg_img = background(rng, W, H) if rng.random() < 0.6 else Image.new(
            "RGB", (W, H), _rand_color(rng, 220, 255))
        fg = _rand_color(rng, 0, 120) if rng.random() < 0.6 else (0x70, 0x70, 0x70)
        layer = Image.new("RGB", (W, H), (255, 255, 255))
        place_text(layer, text, self.fonts, rng, colors=[(0, 0, 0)], size_range=(0.5, 0.7),
                   rot=rng.choice([5, 15]), overlap=(0.0, 0.1), per_char_font=False,
                   jitter_y=0.08, margin_x=(0.05, 0.15))
        line_col = fg if rng.random() < 0.7 else _rand_color(rng, 0, 160)
        if rng.random() < 0.8:
            noise_lines(layer, rng, rng.randint(2, 7), [(0, 0, 0)], (1, 3), kind="sine")
        layer = wave_warp(layer, rng, amp_x=rng.uniform(0.02, 0.08),
                          amp_y=rng.uniform(0.05, 0.15), fill=(255, 255, 255))
        mask = ImageOps.invert(layer.convert("L"))
        bg_img.paste(Image.new("RGB", (W, H), fg), (0, 0), mask)
        if rng.random() < 0.6:
            noise_dots(bg_img, rng, rng.uniform(0.01, 0.05), [line_col], (1, 2))
        if rng.random() < 0.5:
            noise_lines(bg_img, rng, rng.randint(1, 4), [line_col], (1, 2), kind="curve")
        return finish(bg_img, rng, p_blur=0.15, p_jpeg=0.3)

    def _simple(self, text):
        rng = self.rng
        W, H = rng.choice([(200, 50), (160, 50), (180, 60), (220, 60)])
        W = int(W * max(1.0, len(text) / 6))
        c1, c2 = _rand_color(rng, 150, 255), _rand_color(rng, 150, 255)
        t = np.linspace(0, 1, W, dtype=np.float32)[None, :, None]
        arr = np.array(c1)[None, None] * (1 - t) + np.array(c2)[None, None] * t
        img = Image.fromarray(np.repeat(arr, H, axis=0).astype(np.uint8))
        hollow = rng.random() < 0.6
        fg = _rand_color(rng, 0, 110)
        place_text(img, text, self.fonts, rng, colors=[fg], size_range=(0.65, 0.9), rot=10,
                   overlap=(0.0, 0.1), per_char_font=False, stroke=rng.randint(1, 2),
                   stroke_fill=_rand_color(rng, 0, 255) if not hollow else None, hollow=hollow,
                   shear=0.2)
        if rng.random() < 0.5:
            noise_lines(img, rng, rng.randint(1, 2), [fg], (1, 3), kind="curve")
        if rng.random() < 0.3:
            img = wave_warp(img, rng, amp_y=0.06)
        return finish(img, rng, p_blur=0.1, p_jpeg=0.5)

    def _botdetect(self, text):
        rng = self.rng
        W, H = rng.choice([(250, 50), (180, 50), (200, 60), (240, 70)])
        W = int(W * max(1.0, len(text) / 6))
        effect = rng.choice(["chess", "mosaic", "fragment", "stripes", "negative", "ghost",
                             "dotted", "jail"])
        bg_col = _rand_color(rng, 190, 255)
        fg = _contrasting(rng, bg_col, 100)
        mask = Image.new("L", (W, H), 0)
        place_text(mask, text, self.fonts, rng, colors=[(255, 255, 255)], size_range=(0.6, 0.9),
                   rot=rng.choice([5, 15, 25]), overlap=(0.0, 0.15),
                   per_char_font=rng.random() < 0.3, shear=0.2)
        m = np.asarray(mask, dtype=np.float32) / 255.0
        yy, xx = np.mgrid[0:H, 0:W]
        out = np.zeros((H, W, 3), np.float32) + np.array(bg_col)
        fgv = np.array(fg, np.float32)
        if effect == "chess":
            cell = rng.randint(4, 10)
            chk = ((yy // cell + xx // cell) % 2).astype(np.float32)
            alt = np.array(_rand_color(rng, 0, 255), np.float32)
            out = out * (1 - chk[..., None] * 0.35) + alt * chk[..., None] * 0.35
            txt = np.where(chk[..., None] > 0, fgv, np.array(_contrasting(rng, bg_col), np.float32))
            out = out * (1 - m[..., None]) + txt * m[..., None]
        elif effect == "mosaic":
            cell = rng.randint(2, 3)
            small = Image.fromarray((m * 255).astype(np.uint8)).resize(
                (max(1, W // cell), max(1, H // cell)), Image.BILINEAR).resize((W, H), Image.NEAREST)
            mm = np.asarray(small, np.float32)[..., None] / 255.0
            out = out * (1 - mm) + fgv * mm
        elif effect == "fragment":
            cut = (np.random.default_rng(rng.getrandbits(32)).random((H // 4 + 1, W // 4 + 1))
                   < rng.uniform(0.05, 0.18))
            cutm = np.kron(cut, np.ones((4, 4)))[:H, :W]
            mm = (m * (1 - cutm))[..., None]
            out = out * (1 - mm) + fgv * mm
        elif effect == "stripes":
            per = rng.randint(3, 7)
            st = ((yy + (xx if rng.random() < 0.5 else 0)) // per % 2).astype(np.float32)
            out = out * (1 - st[..., None] * 0.25) + fgv * st[..., None] * 0.25
            txt = np.where(st[..., None] > 0, fgv, fgv * 0.6 + np.array(bg_col) * 0.4)
            out = out * (1 - m[..., None]) + txt * m[..., None]
        elif effect == "negative":
            out[:] = fgv
            out = out * (1 - m[..., None]) + np.array(bg_col, np.float32) * m[..., None]
        elif effect == "ghost":
            sh = np.roll(m, (rng.randint(-3, 3), rng.randint(2, 6)), axis=(0, 1))
            out = out * (1 - sh[..., None] * 0.5) + fgv * sh[..., None] * 0.5
            out = out * (1 - m[..., None]) + fgv * m[..., None]
        elif effect == "dotted":
            dots = (((yy % 3) == 0) & ((xx % 3) == 0)).astype(np.float32)
            mm = (m * (1 - dots * 0.8))[..., None]
            out = out * (1 - mm) + fgv * mm
        else:  # jail: bars over the text
            out = out * (1 - m[..., None]) + fgv * m[..., None]
            img = Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))
            d = ImageDraw.Draw(img)
            step = rng.randint(8, 16)
            for x in range(rng.randint(0, step), W, step):
                d.line([(x, 0), (x, H)], fill=fg, width=rng.randint(1, 2))
            return finish(img, rng, p_blur=0.1, p_jpeg=0.2)
        img = Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))
        if rng.random() < 0.3:
            img = wave_warp(img, rng, amp_y=0.06)
        return finish(img, rng, p_blur=0.1, p_jpeg=0.2)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", required=True, help="directory for preview PNGs")
    ap.add_argument("--n", type=int, default=48)
    ap.add_argument("--style", default=None, choices=Generator.STYLES)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--grid", action="store_true", help="also write one contact sheet")
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    gen = Generator(args.seed)
    print(f"{len(gen.fonts)} fonts", file=sys.stderr)
    tiles = []
    for i in range(args.n):
        img, label, _ = gen.sample(args.style)
        img.save(out / f"{i:04d}_{label}.png")
        tiles.append((img, label))
    if args.grid:
        cols = 4
        cw, ch = 256, 84
        sheet = Image.new("RGB", (cw * cols, ch * math.ceil(len(tiles) / cols)), "white")
        d = ImageDraw.Draw(sheet)
        for i, (img, label) in enumerate(tiles):
            x, y = (i % cols) * cw, (i // cols) * ch
            sheet.paste(img.resize((cw - 4, 64)), (x + 2, y))
            d.text((x + 4, y + 66), label, fill="red")
        sheet.save(out / "sheet.png")


if __name__ == "__main__":
    main()
