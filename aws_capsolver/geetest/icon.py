"""GeeTest icon click (v4 "icon", v3 "click" with pic_type icon).

Pipeline
--------
1. ``clean_picture``: v4 pictures carry a 2-pixel "LED" texture (one coloured pixel per
   2x2 block, the rest dark grey). The coloured phase is picked out, which gives a clean
   half-resolution picture; other pictures are resized to the same working width.
2. ``IconDetector`` (small U-Net, trained on synthetic pictures made from GeeTest's own
   icon set) gives a heat map of icon centres; the strongest peaks are the candidates.
3. ``IconEncoder`` / ``HintEncoder`` embed a crop around every candidate and every hint
   icon; hints are assigned to distinct candidates by maximum total similarity.

v3 click pictures (344 x 384) have the hint icons in a strip below the 344 x 344 picture;
when no hints are passed they are cut from that strip.
"""
from __future__ import annotations

import itertools
import logging
import os
import threading
from pathlib import Path

import numpy as np

from ._io import load_image
from .errors import InvalidGeeTestInput, UnsupportedGeeTest

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
WEIGHTS = Path(os.environ.get("GEETEST_ICON_WEIGHTS", ROOT / "weights" / "geetest_icon.pt"))
WORK_WIDTH = 150          # working width of the clean picture
CROP = 40
MAX_CANDIDATES = 8
NMS_RADIUS = 7
ARROW_MIN = 0.5          # rotation-match correlation of v3 arrow hints

_lock = threading.Lock()
_models = None


# ----------------------------------------------------------------- preprocessing
def detect_texture(a: np.ndarray) -> tuple[int, int] | None:
    """Phase (row, col) of the coloured pixels of a 2x2 'LED' texture, or None."""
    h, w = a.shape[:2]
    if h < 20 or w < 20:
        return None
    g = a.astype(np.float32).mean(axis=2)
    means = np.array([[g[i::2, j::2].mean() for j in range(2)] for i in range(2)])
    best = np.unravel_index(int(np.argmax(means)), means.shape)
    others = [means[i, j] for i in range(2) for j in range(2) if (i, j) != best]
    gap = means[best] - max(others)
    if gap > 8 and gap > 2 * np.ptp(others):
        return int(best[0]), int(best[1])
    return None


def clean_picture(im) -> tuple[np.ndarray, float, float, float, float]:
    """-> (clean uint8 HxWx3, sx, sy, ox, oy): original = clean * s + o (pixel centres)."""
    from PIL import Image

    a = np.asarray(im.convert("RGB"))
    phase = detect_texture(a)
    if phase is not None:
        c = a[phase[0]::2, phase[1]::2]
        if c.shape[1] == WORK_WIDTH:
            return c, 2.0, 2.0, float(phase[1]), float(phase[0])
        a = c  # a textured picture shown at another size: resize its clean phase
    full_w, full_h = np.asarray(im).shape[1], np.asarray(im).shape[0]
    f = WORK_WIDTH / a.shape[1]
    nh = max(8, round(a.shape[0] * f))
    c = np.asarray(Image.fromarray(a).resize((WORK_WIDTH, nh), Image.BOX))
    # pixel centres: original x = (clean x + 0.5) * s - 0.5
    sx, sy = full_w / WORK_WIDTH, full_h / nh
    return c, sx, sy, 0.5 * sx - 0.5, 0.5 * sy - 0.5


def hint_mask(im, size: int = CROP) -> np.ndarray:
    """Hint icon -> size x size float mask (strokes = 1), cropped to its strokes, centred."""
    from PIL import Image

    rgba = np.asarray(im.convert("RGBA"), dtype=np.float32) / 255.0
    lum = rgba[:, :, :3].mean(axis=2)
    alpha = rgba[:, :, 3]
    if float(alpha.min()) > 0.99:
        # flattened: strokes are whichever of dark / light is the minority
        ink = 1.0 - lum if lum.mean() > 0.5 else lum
    else:
        ink = alpha * np.clip(1.0 - lum, 0.0, 1.0) if lum[alpha > 0.5].mean() < 0.5 else alpha
    ys, xs = np.where(ink > 0.3)
    if len(xs) < 8:
        raise InvalidGeeTestInput("hint icon has no visible strokes")
    ink = ink[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    h, w = ink.shape
    side = int(max(h, w) * 1.15) + 2
    sq = np.zeros((side, side), np.float32)
    sq[(side - h) // 2:(side - h) // 2 + h, (side - w) // 2:(side - w) // 2 + w] = ink
    return np.asarray(Image.fromarray((sq * 255).astype(np.uint8)).resize((size, size), Image.BOX),
                      np.float32) / 255.0


def crop_at(clean: np.ndarray, cx: float, cy: float, size: int = CROP) -> np.ndarray:
    """size x size crop centred at (cx, cy) with edge padding -> float HxWx3 in [0, 1]."""
    h, w = clean.shape[:2]
    pad = size
    p = np.pad(clean, ((pad, pad), (pad, pad), (0, 0)), mode="edge")
    x0 = int(round(cx)) - size // 2 + pad
    y0 = int(round(cy)) - size // 2 + pad
    x0 = min(max(x0, 0), w + 2 * pad - size)
    y0 = min(max(y0, 0), h + 2 * pad - size)
    return p[y0:y0 + size, x0:x0 + size].astype(np.float32) / 255.0


def split_v3_strip(im):
    """v3 click picture (344 x 384): (picture 344 x 344, [hint icon images left to right])."""
    from PIL import Image

    w, h = im.size
    pic_h = w if h > w else h
    pic = im.crop((0, 0, w, pic_h))
    strip = np.asarray(im.crop((0, pic_h, w, h)).convert("L"), dtype=np.float32)
    if strip.size == 0:
        return pic, []
    # hint icons are dark strokes on a light panel; a solid dark filler can follow the panel
    ink = strip < 140
    cols = ink.any(axis=0) & (ink.mean(axis=0) < 0.9)
    hints = []
    x = 0
    # the strip starts with a text label ("请依次点击" etc.) in some themes: keep only blobs
    # that look like square icons
    while x < len(cols):
        if not cols[x]:
            x += 1
            continue
        x1 = x
        gap = 0
        while x1 < len(cols) and gap < 2:
            gap = gap + 1 if not cols[x1] else 0
            x1 += 1
        x1 -= gap
        rows = ink[:, x:x1].any(axis=1)
        ys = np.where(rows)[0]
        if len(ys) and x1 - x >= 8 and (ys[-1] - ys[0]) >= 8:
            box = (x, int(ys[0]), x1, int(ys[-1]) + 1)
            g = Image.fromarray(np.where(ink[box[1]:box[3], box[0]:box[2]], 0, 255).astype(np.uint8))
            hints.append(g.convert("RGB"))
        x = x1 + 1
    return pic, hints


def _radial_canvas(im, size: int = 48) -> np.ndarray:
    """Strokes centred on their centroid and scaled by their max radius (rotation safe)."""
    from PIL import Image

    m = hint_mask(im, 64)
    ys, xs = np.nonzero(m > 0.3)
    if len(xs) < 4:
        return np.zeros((size, size), np.float32)
    cy, cx = float(ys.mean()), float(xs.mean())
    r = float(np.sqrt((ys - cy) ** 2 + (xs - cx) ** 2).max()) + 1.0
    f = (size / 2 - 2) / r
    big = Image.fromarray((m * 255).astype(np.uint8))
    big = big.transform((size, size), Image.AFFINE, (1 / f, 0, cx - size / 2 / f, 0, 1 / f, cy - size / 2 / f),
                        resample=Image.BILINEAR)
    return np.asarray(big, np.float32) / 255.0


def _are_arrows(hints) -> bool:
    """True when all hints are one shape in different rotations (v3 arrow hints)."""
    from PIL import Image

    if len(hints) < 2:
        return False
    cans = [_radial_canvas(h) for h in hints]
    c0 = Image.fromarray((cans[0] * 255).astype(np.uint8))
    rots = []
    for k in range(8):
        r = np.asarray(c0.rotate(45 * k, resample=Image.BILINEAR), np.float32) / 255.0
        rots.append((r - r.mean()) / (r.std() + 1e-6))
    for c in cans[1:]:
        z = (c - c.mean()) / (c.std() + 1e-6)
        if max(float((z * r).mean()) for r in rots) < ARROW_MIN:
            return False
    return True


# ------------------------------------------------------------------------ models
def _load():
    global _models
    if _models is None:
        with _lock:
            if _models is None:
                if not WEIGHTS.is_file():
                    raise UnsupportedGeeTest(f"icon click model weights are missing ({WEIGHTS.name})")
                try:
                    import torch

                    from .icon_model import IconModels
                except ImportError as exc:
                    raise UnsupportedGeeTest(f"icon click needs torch: {exc}") from exc
                env = os.environ.get("AWS_CAPSOLVER_TORCH_THREADS", "").strip()
                torch.set_num_threads(int(env) if env.isdigit() and int(env) > 0 else min(8, os.cpu_count() or 1))
                m = IconModels()
                state = torch.load(WEIGHTS, map_location="cpu", weights_only=True)
                m.load_state_dict(state["model"])
                m.eval()
                _models = (torch, m, threading.Lock())
    return _models


def _peaks(heat: np.ndarray, k: int, radius: int, min_value: float) -> list[tuple[float, float, float]]:
    h = heat.copy()
    out = []
    for _ in range(k):
        i = int(np.argmax(h))
        y, x = divmod(i, h.shape[1])
        v = float(h[y, x])
        if v < min_value:
            break
        # sub-pixel centre: weighted mean in a 3x3 window
        y0, y1, x0, x1 = max(0, y - 1), min(h.shape[0], y + 2), max(0, x - 1), min(h.shape[1], x + 2)
        win = np.clip(heat[y0:y1, x0:x1], 0, None)
        yy, xx = np.mgrid[y0:y1, x0:x1]
        s = float(win.sum()) or 1.0
        out.append((float((xx * win).sum() / s), float((yy * win).sum() / s), v))
        h[max(0, y - radius):y + radius + 1, max(0, x - radius):x + radius + 1] = -np.inf
    return out


def detect(clean: np.ndarray) -> list[tuple[float, float, float]]:
    torch, m, lock = _load()
    x = torch.from_numpy(clean.astype(np.float32) / 255.0).permute(2, 0, 1)[None]
    with lock, torch.inference_mode():
        heat = torch.sigmoid(m.detector(x))[0, 0].numpy()
    return _peaks(heat, MAX_CANDIDATES, NMS_RADIUS, 0.05)


def similarity(clean: np.ndarray, cands, masks: list[np.ndarray]) -> np.ndarray:
    torch, m, lock = _load()
    crops = np.stack([crop_at(clean, cx, cy) for cx, cy, _ in cands])
    with lock, torch.inference_mode():
        ce = m.icon(torch.from_numpy(crops).permute(0, 3, 1, 2))
        he = m.hint(torch.from_numpy(np.stack(masks))[:, None])
    return (he @ ce.T).numpy()


def assign(sim: np.ndarray) -> list[int]:
    """Best one-to-one hint -> candidate assignment (exhaustive; at most 8 choose 4)."""
    n, k = sim.shape
    if k < n:
        raise UnsupportedGeeTest(f"found {k} icons for {n} hints")
    best, best_s = None, -np.inf
    for perm in itertools.permutations(range(k), n):
        s = sum(float(sim[i, j]) for i, j in enumerate(perm))
        if s > best_s:
            best, best_s = perm, s
    return list(best)


def click_icons(image, hints) -> list[tuple[int, int]]:
    im = load_image(image, "image", "RGB")
    hint_ims = []
    if hints:
        for i, h in enumerate(hints):
            if isinstance(h, str) and len(h) < 16:
                raise InvalidGeeTestInput(f"hints[{i}]: icon click hints must be images")
            hint_ims.append(load_image(h, f"hints[{i}]", "RGBA"))
        pic = im
    else:
        pic, hint_ims = split_v3_strip(im)
        if not hint_ims:
            raise InvalidGeeTestInput("no hint icons given and none found below the picture")
        if _are_arrows(hint_ims):
            # v3 "direction" variant: click the icons that FACE the arrow directions
            raise UnsupportedGeeTest("GeeTest v3 direction click (arrow hints) is not supported")
    if len(hint_ims) > 6:
        raise InvalidGeeTestInput("too many hint icons")
    clean, sx, sy, ox, oy = clean_picture(pic)
    cands = detect(clean)
    if len(cands) < len(hint_ims):
        raise UnsupportedGeeTest(f"found {len(cands)} icons for {len(hint_ims)} hints")
    sim = similarity(clean, cands, [hint_mask(h) for h in hint_ims])
    order = assign(sim)
    out = []
    for j in order:
        cx, cy, _ = cands[j]
        x = int(round(cx * sx + ox))
        y = int(round(cy * sy + oy))
        out.append((min(max(x, 0), pic.width - 1), min(max(y, 0), pic.height - 1)))
    return out


def warm() -> None:
    try:
        _load()
    except UnsupportedGeeTest as exc:
        log.info("icon click model not loaded: %s", exc)
        return
    clean = np.full((100, WORK_WIDTH, 3), 128, np.uint8)
    detect(clean)
