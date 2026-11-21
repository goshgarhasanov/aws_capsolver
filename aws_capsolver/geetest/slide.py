"""GeeTest slide puzzle (v3 "slide3" and v4 "slide"): find where the puzzle piece goes.

Methods, in order of preference
-------------------------------
1. ``full_background`` given (v3 ships the clean picture as ``fullbg``): the gap is the
   set of pixels where the two pictures differ. With a piece, the piece's alpha shape is
   aligned on that difference mask (exact); without one the difference's bounding box is
   used.
2. ``piece`` given (v3 / v4 ``slice`` PNG with alpha): GeeTest draws the gap by darkening
   the picture inside the piece outline (v3: x0.54 on average). Along the true outline the
   log-brightness therefore steps UP when leaving the shape. We correlate the gradient of
   log-brightness with the outward normals of the piece outline ("oriented shadow edge"):
   it is texture independent, sign aware (a dark object is not a shadow of the piece
   shape) and shape specific, which rejects the decoy gaps v4 draws with other shapes.
3. Background only: the same oriented-edge score with a generic square jigsaw body.

Raw v3 pictures (312x160, served as 52 shuffled strips) are reassembled automatically.

The returned x is where the LEFT EDGE OF THE PIECE IMAGE must end up (the piece image
starts at x=0 in both v3 and v4 widgets, so this equals the drag distance in background
pixels). Without a piece image it is the left edge of the visible gap. Piece and
background must have the same scale (both as served, or both captured at one zoom).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from ._io import load_image
from .errors import InvalidGeeTestInput, UnsupportedGeeTest

log = logging.getLogger(__name__)

# v3 shuffles the background into 2 rows x 26 strips of 10x80 px (12 px pitch in the raw image).
V3_RAW_SIZE = (312, 160)
V3_SIZE = (260, 160)
_V3_ORDER = (39, 38, 48, 49, 41, 40, 46, 47, 35, 34, 50, 51, 33, 32, 28, 29, 27, 26, 36, 37, 31, 30,
             44, 45, 43, 42, 12, 13, 23, 22, 14, 15, 21, 20, 8, 9, 25, 24, 6, 7, 3, 2, 0, 1, 11, 10,
             4, 5, 19, 18, 16, 17)
_PAD = 4            # template padding around the piece outline (px)
_Y_TOLERANCE = 1    # rows searched around a known ``y``


@dataclass(frozen=True)
class SlideResult:
    x: int            # left edge of the piece image when aligned (== gap_x without a piece)
    gap_x: int        # left edge of the visible gap (opaque part of the piece)
    y: int            # top edge of the piece image when aligned (gap top without a piece)
    score: float      # method-specific confidence (higher is better)
    method: str       # "diff" | "match" | "gap"


def unscramble_v3(im):
    """Reassemble a raw 312x160 v3 picture (shuffled strips) into the shown 260x160 one."""
    from PIL import Image

    out = Image.new(im.mode, V3_SIZE)
    for i, c in enumerate(_V3_ORDER):
        sx, sy = c % 26 * 12 + 1, 80 if c > 25 else 0
        out.paste(im.crop((sx, sy, sx + 10, sy + 80)), (i % 26 * 10, 0 if i < 26 else 80))
    return out


def _prepare_bg(value, what: str):
    im = load_image(value, what, "RGB")
    if im.size == V3_RAW_SIZE:
        im = unscramble_v3(im)
    return im


def _piece_mask(piece_rgba: np.ndarray) -> tuple[np.ndarray, int, int]:
    """Opaque part of the piece cropped to its bounding box -> (mask uint8, ox, oy)."""
    alpha = piece_rgba[:, :, 3]
    ys, xs = np.where(alpha > 128)
    if len(xs) < 50:
        raise InvalidGeeTestInput("piece: the image has no opaque puzzle piece (needs an alpha channel)")
    ox, oy = int(xs.min()), int(ys.min())
    mask = (alpha[oy:int(ys.max()) + 1, ox:int(xs.max()) + 1] > 128).astype(np.uint8)
    return mask, ox, oy


def _restrict_rows(scores: np.ndarray, top: int | None) -> np.ndarray:
    """Keep only the rows around ``top`` (a known y of the template's top-left corner)."""
    if top is None:
        return scores
    lo, hi = max(0, top - _Y_TOLERANCE), min(scores.shape[0], top + _Y_TOLERANCE + 1)
    if lo >= hi:
        return scores
    out = np.full_like(scores, -np.inf)
    out[lo:hi] = scores[lo:hi]
    return out


def _argmax2(scores: np.ndarray) -> tuple[int, int, float]:
    yy, xx = np.unravel_index(int(np.argmax(scores)), scores.shape)
    return int(yy), int(xx), float(scores[yy, xx])


# ---------------------------------------------------------------------- method 1: diff
def _diff_mask(bg: np.ndarray, full: np.ndarray) -> np.ndarray:
    import cv2

    diff = np.abs(bg.astype(np.int16) - full.astype(np.int16)).max(axis=2)
    mask = (diff > 24).astype(np.uint8)
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))


def _refine_on_full(full: np.ndarray, piece_rgba: np.ndarray, mask: np.ndarray, ox: int, oy: int,
                    gx: int, gy: int, radius: int = 4) -> int:
    """Refine the gap x by aligning the piece's own pixels on the clean picture.

    The difference mask is blurred by JPEG noise (about 1-2 px); the piece content matches
    the clean picture exactly when aligned. Kept only when the content match is clear."""
    import cv2

    h, w = mask.shape
    core = cv2.erode((piece_rgba[oy:oy + h, ox:ox + w, 3] > 250).astype(np.uint8), np.ones((7, 7), np.uint8))
    if core.sum() < 50 or gy < 0 or gy + h > full.shape[0]:
        return gx
    lo, hi = max(0, gx - radius), min(full.shape[1], gx + w + radius)
    win = full[gy:gy + h, lo:hi].astype(np.float32)
    if win.shape[1] < w:
        return gx
    tmpl = piece_rgba[oy:oy + h, ox:ox + w, :3].astype(np.float32)
    s = cv2.matchTemplate(win, np.ascontiguousarray(tmpl), cv2.TM_CCOEFF_NORMED,
                          mask=np.repeat(core[:, :, None], 3, 2).astype(np.float32))[0]
    s = np.nan_to_num(s, nan=-1.0, posinf=-1.0, neginf=-1.0)
    if float(s.max()) < 0.8:
        return gx
    return lo + int(np.argmax(s))


def _diff_solve(bg: np.ndarray, full: np.ndarray, piece_rgba: np.ndarray | None, y: int | None) -> SlideResult:
    import cv2

    D = _diff_mask(bg, full)
    if int(D.sum()) < 30:
        raise UnsupportedGeeTest("background and full background do not differ: no gap found")
    if piece_rgba is not None:
        mask, ox, oy = _piece_mask(piece_rgba)
        h, w = mask.shape
        if h <= D.shape[0] and w <= D.shape[1]:
            # +1 for gap pixels covered by the shape, -1 for changed pixels outside it
            t = mask.astype(np.float32) * 2 - 1
            s = cv2.matchTemplate(D.astype(np.float32), t, cv2.TM_CCORR) / float(mask.sum())
            s = _restrict_rows(s, None if y is None else y + oy)
            yy, xx, sc = _argmax2(s)
            xx = _refine_on_full(full, piece_rgba, mask, ox, oy, xx, yy)
            return SlideResult(x=xx - ox, gap_x=xx, y=yy - oy, score=sc, method="diff")
    n, _labels, stats, _ = cv2.connectedComponentsWithStats(D, connectivity=8)
    best = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    x0, y0, w0, h0, area = (int(v) for v in stats[best])
    x1, y1 = x0 + w0, y0 + h0
    for i in range(1, n):  # merge fragments of the same gap (flat areas barely change)
        xi, yi, wi, hi, ai = (int(v) for v in stats[i])
        if i == best or ai < 15:
            continue
        if xi < x1 + 15 and xi + wi > x0 - 15 and yi < y1 + 15 and yi + hi > y0 - 15:
            nx0, ny0, nx1, ny1 = min(x0, xi), min(y0, yi), max(x1, xi + wi), max(y1, yi + hi)
            if nx1 - nx0 <= 90 and ny1 - ny0 <= 90:
                x0, y0, x1, y1, area = nx0, ny0, nx1, ny1, area + ai
    return SlideResult(x=x0, gap_x=x0, y=y0, score=float(area), method="diff")


# --------------------------------------------------------------- method 2/3: shadow edge
def _log_gradients(bg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    import cv2

    gray = cv2.cvtColor(bg, cv2.COLOR_RGB2GRAY).astype(np.float32)
    L = np.log(cv2.GaussianBlur(gray, (0, 0), 0.8) + 10.0)
    return cv2.Sobel(L, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(L, cv2.CV_32F, 0, 1, ksize=3)


def _shadow_scores(lx: np.ndarray, ly: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Mean outward log-brightness step along the shape outline, per padded top-left position."""
    import cv2

    m = np.pad(cv2.GaussianBlur(mask.astype(np.float32), (0, 0), 1.2), _PAD)
    tx = -cv2.Sobel(m, cv2.CV_32F, 1, 0, ksize=3)
    ty = -cv2.Sobel(m, cv2.CV_32F, 0, 1, ksize=3)
    norm = float(np.sqrt(tx ** 2 + ty ** 2).sum()) or 1.0
    if lx.shape[0] < m.shape[0] or lx.shape[1] < m.shape[1]:
        raise InvalidGeeTestInput("piece is larger than the background")
    return (cv2.matchTemplate(lx, tx, cv2.TM_CCORR) + cv2.matchTemplate(ly, ty, cv2.TM_CCORR)) / norm


def _peaks(s: np.ndarray, k: int, radius: int) -> list[tuple[int, int, float]]:
    """Top-k local maxima of a score map with non-maximum suppression."""
    s = s.copy()
    out = []
    for _ in range(k):
        yy, xx, v = _argmax2(s)
        if not np.isfinite(v):
            break
        out.append((yy, xx, v))
        s[max(0, yy - radius):yy + radius + 1, max(0, xx - radius):xx + radius + 1] = -np.inf
    return out


def _highpass(a: np.ndarray) -> np.ndarray:
    import cv2

    a = a.astype(np.float32)
    return a - cv2.GaussianBlur(a, (0, 0), 2.0)


def _content_scores(bg: np.ndarray, bg_hp: np.ndarray, tmpl: np.ndarray, tmpl_hp: np.ndarray,
                    core3: np.ndarray, gx: int, gy: int, r: int) -> tuple[float, int, int]:
    """Best content similarity of the piece around gap position (gx, gy) +- r.

    The gap shows the piece's own pixels darkened, so zero-mean NCC (gain invariant) of
    the plain and of the high-pass (texture) image is high only at the real gap."""
    import cv2

    h, w = core3.shape[:2]
    lx, ly = max(0, gx - r), max(0, gy - r)
    hx, hy = min(bg.shape[1], gx + w + r), min(bg.shape[0], gy + h + r)
    if hx - lx < w or hy - ly < h:
        return -1.0, gx, gy
    total = None
    for img, t in ((bg, tmpl), (bg_hp, tmpl_hp)):
        win = np.ascontiguousarray(img[ly:hy, lx:hx], dtype=np.float32)
        sc = cv2.matchTemplate(win, t, cv2.TM_CCOEFF_NORMED, mask=core3)
        sc = np.clip(np.nan_to_num(sc, nan=-1.0, posinf=-1.0, neginf=-1.0), -1.0, 1.0)
        total = sc if total is None else total + sc
    yy, xx, v = _argmax2(total / 2.0)
    return v, lx + xx, ly + yy


def _content_map(bg: np.ndarray, bg_hp: np.ndarray, tmpl: np.ndarray, tmpl_hp: np.ndarray,
                 core3: np.ndarray) -> np.ndarray:
    """Content similarity (mean of plain and high-pass masked NCC) at every gap position."""
    import cv2

    total = None
    for img, t in ((bg, tmpl), (bg_hp, tmpl_hp)):
        sc = cv2.matchTemplate(np.ascontiguousarray(img, dtype=np.float32), t, cv2.TM_CCOEFF_NORMED, mask=core3)
        sc = np.clip(np.nan_to_num(sc, nan=-1.0, posinf=-1.0, neginf=-1.0), -1.0, 1.0)
        total = sc if total is None else total + sc
    return total / 2.0


def _match(bg: np.ndarray, piece_rgba: np.ndarray, y: int | None) -> SlideResult:
    import cv2

    mask, ox, oy = _piece_mask(piece_rgba)
    h, w = mask.shape
    lx, ly = _log_gradients(bg)
    s = _shadow_scores(lx, ly, mask)
    # score index p is the padded template's top-left; the opaque shape starts at p + _PAD
    s = _restrict_rows(s, None if y is None else y + oy - _PAD)
    cands = _peaks(s, 8, max(6, min(h, w) // 4))
    top = cands[0][2]
    cands = [c for c in cands if c[2] >= 0.35 * top] if top > 0 else cands[:1]

    core = cv2.erode((piece_rgba[oy:oy + h, ox:ox + w, 3] > 250).astype(np.uint8), np.ones((9, 9), np.uint8))
    tmpl = np.ascontiguousarray(piece_rgba[oy:oy + h, ox:ox + w, :3], dtype=np.float32)
    textured = core.sum() >= 50 and float(tmpl[core > 0].std()) >= 3.0
    if not textured or len(cands) == 1:
        yy, xx, sc = cands[0]
        gap_x, gap_y = xx + _PAD, yy + _PAD
        return SlideResult(x=gap_x - ox, gap_x=gap_x, y=gap_y - oy, score=sc, method="match")
    # Several gap-shaped shadows (v4 draws decoys, sometimes in the same shape and row):
    # pick the one whose content is the piece's own picture content.
    core3 = np.repeat(core[:, :, None], 3, 2).astype(np.float32)
    bg_hp, tmpl_hp = _highpass(bg), np.ascontiguousarray(_highpass(tmpl))
    # A gap in a dark area has a weak outline: add the best content matches as candidates.
    content = _content_map(bg, bg_hp, tmpl, tmpl_hp, core3)
    content = _restrict_rows(content, None if y is None else y + oy)
    seen = {(yy, xx) for yy, xx, _ in cands}
    for gy, gx, _v in _peaks(content, 4, max(6, min(h, w) // 4)):
        p = (gy - _PAD, gx - _PAD)
        if 0 <= p[0] < s.shape[0] and 0 <= p[1] < s.shape[1] and p not in seen:
            cands.append((p[0], p[1], float(max(s[p], 0.0))))
    best = None
    for yy, xx, sc in cands:
        c, cx, cy = _content_scores(bg, bg_hp, tmpl, tmpl_hp, core3, xx + _PAD, yy + _PAD, 3)
        total = c + 0.3 * sc / top
        if best is None or total > best[0]:
            best = (total, cx, cy)
    _total, gap_x, gap_y = best
    return SlideResult(x=gap_x - ox, gap_x=gap_x, y=gap_y - oy, score=float(_total), method="match")


_GAP_SIDES = (0.27, 0.32, 0.37)  # square gap body sizes tried, as a fraction of the height


def _gap_only(bg: np.ndarray, y: int | None) -> SlideResult:
    """No piece: best square, darker-inside outline over a few typical body sizes."""
    lx, ly = _log_gradients(bg)
    best: SlideResult | None = None
    for frac in _GAP_SIDES:
        side = max(16, int(round(bg.shape[0] * frac)))
        if side + 2 * _PAD > min(bg.shape[:2]):
            continue
        s = _shadow_scores(lx, ly, np.ones((side, side), np.uint8))
        s = _restrict_rows(s, None if y is None else y - _PAD)
        s[:, : max(1, bg.shape[1] // 6)] = -np.inf  # the gap never sits under the start position
        yy, xx, sc = _argmax2(s)
        if np.isfinite(sc) and (best is None or sc > best.score):
            best = SlideResult(x=xx + _PAD, gap_x=xx + _PAD, y=yy + _PAD, score=sc, method="gap")
    if best is None:
        raise InvalidGeeTestInput("background is too small to hold a gap")
    return best


def solve(background, piece=None, full_background=None, *, y: int | None = None) -> SlideResult:
    """Locate the gap; see the module docstring. ``y`` (optional) = known top of the piece
    image (``ypos``), which restricts the search to that row band."""
    bg_im = _prepare_bg(background, "background")
    bg = np.asarray(bg_im)
    piece_rgba = np.asarray(load_image(piece, "piece", "RGBA")) if piece is not None else None
    if piece_rgba is not None and (piece_rgba.shape[1] >= bg.shape[1] or piece_rgba.shape[0] > bg.shape[0]):
        raise InvalidGeeTestInput(f"piece {piece_rgba.shape[1]}x{piece_rgba.shape[0]} does not fit into "
                                  f"background {bg.shape[1]}x{bg.shape[0]}")
    if full_background is not None:
        full = np.asarray(_prepare_bg(full_background, "full_background"))
        if full.shape != bg.shape:
            raise InvalidGeeTestInput(f"full_background is {full.shape[1]}x{full.shape[0]}, "
                                      f"background is {bg.shape[1]}x{bg.shape[0]}")
        return _diff_solve(bg, full, piece_rgba, y)
    if piece_rgba is not None:
        return _match(bg, piece_rgba, y)
    return _gap_only(bg, y)
