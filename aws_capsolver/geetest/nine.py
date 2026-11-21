"""GeeTest v4 "nine": pick the cells of a 3x3 photo grid that show the prompt.

The prompt is either a text label or GeeTest's black line icon (e.g. a fish). Icons come
from a fixed set, named once by hand (``weights/geetest_nine_icons.pt``, built by
tools/geetest_nine_icons.py); a prompt icon is recognised by its shape and turned into
its label. Cells are then scored with the CLIP zero-shot machinery of
``aws_capsolver.vision.grid`` (label contrasted with an open vocabulary of ~1000 object
names) and the ``count`` best cells are returned. An icon that is not in the table falls
back to CLIP image-to-image similarity between the icon and the cells.
"""
from __future__ import annotations

import logging
import os
import threading
from pathlib import Path

import numpy as np

from ._io import load_image, to_bytes
from .errors import InvalidGeeTestInput, UnsupportedGeeTest

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
ICONS_PATH = Path(os.environ.get("GEETEST_NINE_ICONS", ROOT / "weights" / "geetest_nine_icons.pt"))
DEFAULT_COUNT = 3
SIG = 32               # icon signature side
MATCH_MIN = 0.80       # min correlation to accept a known icon
_MARGIN = 0.03         # cell border trimmed before embedding (white separators)

# How the photos show some icon labels (scores are the max over the names).
PHOTO_NAMES: dict[str, tuple[str, ...]] = {
    "ship": ("boat", "ship", "speedboat"),
    "billiards": ("billiard balls", "pool table"),
    "loudspeaker": ("loudspeaker", "speaker box"),
    "toy blocks": ("toy blocks", "building blocks"),
    "gears": ("gear", "cogwheel"),
    "ladle": ("ladle", "skimmer spoon", "spatula"),
    "smoking pipe": ("tobacco pipe", "smoking pipe"),
    "bull": ("bull", "ox"),
    "tire": ("tire", "car tyre"),
    "railway track": ("railway track", "train tracks"),
    "balloons": ("balloons", "party balloons"),
    "power strip": ("power strip", "extension socket"),
    "computer mouse": ("computer mouse",),
    "glasses": ("eyeglasses", "glasses"),
    "electric drill": ("electric drill", "cordless drill"),
    "bird": ("bird", "songbird"),
    "billiard": ("billiard balls",),
}

_lock = threading.Lock()
_icons: dict | None = None


def icon_signature(im) -> np.ndarray:
    """Shape signature of a line icon: 'ink' cropped to its box, padded square, 32x32, unit norm."""
    from PIL import Image

    rgba = np.asarray(im.convert("RGBA"), dtype=np.float32) / 255.0
    lum = rgba[:, :, :3].mean(axis=2)
    alpha = rgba[:, :, 3]
    if float(alpha.min()) > 0.99:     # flattened icon (no transparency): dark strokes on light
        ink = 1.0 - lum
    else:
        ink = alpha * (1.0 - lum)
    ys, xs = np.where(ink > 0.3)
    if len(xs) < 10:
        raise InvalidGeeTestInput("target icon: no visible strokes")
    ink = ink[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    h, w = ink.shape
    side = max(h, w)
    sq = np.zeros((side, side), np.float32)
    sq[(side - h) // 2:(side - h) // 2 + h, (side - w) // 2:(side - w) // 2 + w] = ink
    small = np.asarray(Image.fromarray((sq * 255).astype(np.uint8)).resize((SIG, SIG), Image.BOX), np.float32)
    small -= small.mean()
    n = float(np.linalg.norm(small))
    return (small / n).ravel() if n > 0 else small.ravel()


def _icon_table() -> dict:
    global _icons
    if _icons is None:
        with _lock:
            if _icons is None:
                table = {"labels": [], "signatures": np.zeros((0, SIG * SIG), np.float32)}
                if ICONS_PATH.is_file():
                    try:
                        import torch

                        d = torch.load(ICONS_PATH, map_location="cpu", weights_only=True)
                        table = {"labels": [str(x) for x in d["labels"]],
                                 "signatures": np.asarray(d["signatures"], np.float32)}
                    except Exception as exc:  # noqa: BLE001 - a broken table only disables icon naming
                        log.warning("cannot load %s: %s", ICONS_PATH, exc)
                else:
                    log.warning("no icon table at %s; icon prompts use image similarity", ICONS_PATH)
                _icons = table
    return _icons


def name_icon(icon_im) -> tuple[str | None, float]:
    """(label, correlation) of the closest known prompt icon; label None if none is close."""
    table = _icon_table()
    if not table["labels"]:
        return None, 0.0
    sig = icon_signature(icon_im)
    sims = table["signatures"] @ sig
    i = int(np.argmax(sims))
    best = float(sims[i])
    return (table["labels"][i] if best >= MATCH_MIN else None), best


def split_cells(im, n: int = 3) -> list:
    w, h = im.size
    cells = []
    for r in range(n):
        for c in range(n):
            x0, y0, x1, y1 = c * w / n, r * h / n, (c + 1) * w / n, (r + 1) * h / n
            mx, my = (x1 - x0) * _MARGIN, (y1 - y0) * _MARGIN
            cells.append(im.crop((int(x0 + mx), int(y0 + my), int(x1 - mx), int(y1 - my))))
    return cells


def _engine():
    try:
        from aws_capsolver.vision.errors import VisionError
        from aws_capsolver.vision.grid import get_engine
    except ImportError as exc:  # pragma: no cover - depends on the install
        raise UnsupportedGeeTest(f"nine-grid needs the CLIP vision module: {exc}") from exc
    try:
        return get_engine()
    except VisionError as exc:
        raise UnsupportedGeeTest(f"CLIP model unavailable: {exc}") from exc


def _normalize_label(text: str) -> str:
    t = " ".join(text.strip().lower().split())
    for prefix in ("please click on all ", "please click all ", "click on all ", "select all ", "click "):
        if t.startswith(prefix):
            t = t[len(prefix):]
    for article in ("the ", "a ", "an "):
        if t.startswith(article):
            t = t[len(article):]
    t = t.strip(" .:!")
    if not t or len(t) > 60:
        raise InvalidGeeTestInput("target text must be a short label such as 'fish'")
    return t


def cell_scores(image, target) -> tuple[np.ndarray, str]:
    """Per-cell scores (higher = more likely the target) and a description of the method."""
    from PIL import Image

    im = load_image(image, "image", "RGB")
    if im.width < 60 or im.height < 60:
        raise InvalidGeeTestInput("nine-grid image is too small")
    label: str | None = None
    icon = None
    if isinstance(target, str) and not _looks_like_image(target):
        label = _normalize_label(target)
    else:
        icon = load_image(target, "target", "RGBA")
        label, sim = name_icon(icon)
        log.debug("nine: prompt icon -> %s (%.2f)", label, sim)
    eng = _engine()
    feats = eng.embed(split_cells(im))
    if label is not None:
        names = PHOTO_NAMES.get(label, (label,))
        probs = np.max([eng.zero_shot_probs(feats, n, open_vocab=True) for n in names], axis=0)
        return probs, f"label:{label}"
    # unknown icon: compare the icon (on white) with the cells, per backbone
    flat = Image.new("RGB", icon.size, (255, 255, 255))
    flat.paste(icon, (0, 0), icon)
    ifeats = eng.embed([flat])
    sims = np.mean([f @ i[0] for f, i in zip(feats, ifeats)], axis=0)
    return sims, "icon-similarity"


def _looks_like_image(s: str) -> bool:
    """A long base64 string / data URL is an image, a short one is a label."""
    s = s.strip()
    if s.startswith("data:image/"):
        return True
    if len(s) < 64 or " " in s:
        return False
    try:
        data = to_bytes(s, "target")
    except InvalidGeeTestInput:
        return False
    return data[:8] == b"\x89PNG\r\n\x1a\n" or data[:3] == b"\xff\xd8\xff" or data[:4] in (b"RIFF", b"GIF8")


def nine_grid(image, target, count: int | None = None) -> list[int]:
    count = DEFAULT_COUNT if count is None else count
    scores, method = cell_scores(image, target)
    order = sorted(range(len(scores)), key=lambda i: (-float(scores[i]), i))
    log.debug("nine (%s): scores %s", method, np.round(scores, 3).tolist())
    return sorted(order[:count])


def warm() -> None:
    """Load CLIP and the icon table (called by the package warm-up)."""
    from PIL import Image

    _icon_table()
    eng = _engine()
    eng.embed([Image.new("RGB", (96, 84), (128, 128, 128))])
