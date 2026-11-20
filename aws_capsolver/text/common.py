"""
The "common" ImageToTextTask module: a general text-captcha reader.

Backend: ddddocr's bundled ONNX CTC model (https://github.com/sml2h3/ddddocr), run through
its onnxruntime session directly so the alphabet can be restricted at decoding time
(ASCII letters + digits by default, digits or letters only for `numeric` 1 / 2) and every
character gets a probability. If ddddocr's internals ever change shape, the public
`classification()` call is used instead (no alphabet mask, vote-share confidence).

Accuracy trick (from the CaptchaBypasser project): an OCR model is deterministic, so reading
one image N times is not an ensemble. The image is rendered several genuinely different
ways (plain, contrast-enhanced, autocontrast, Otsu binary); each rendering is read once and
the readings vote, weighted by their confidence. A rendering artefact then only affects
the reading that saw it, and agreement across renderings is real signal.
"""

import hashlib
import io
import logging
import threading
from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

from . import NUMERIC_DIGITS, NUMERIC_LETTERS, Reading
from .errors import InvalidImage, ModelUnavailable

logger = logging.getLogger(__name__)

MODEL_HEIGHT = 64            # ddddocr's default model reads 64 px high grayscale strips
MAX_SIDE = 2000              # larger inputs are not text captchas
MAX_MODEL_WIDTH = 4096       # after scaling to MODEL_HEIGHT (absurd aspect ratios)
VARIANTS = ("plain", "enhanced", "moderate", "binary")

_lock = threading.Lock()
_engine: Optional["_Engine"] = None
_failed: Optional[str] = None   # why loading failed (kept so every request does not retry)


class _Engine:
    """ddddocr's model plus the charset masks used for constrained CTC decoding."""

    def __init__(self):
        try:
            import ddddocr
        except ImportError as exc:
            raise ModelUnavailable("ddddocr is not installed (pip install ddddocr)") from exc
        import numpy as np
        self._np = np
        try:
            self._ocr = ddddocr.DdddOcr(show_ad=False)
        except Exception as exc:  # broken onnxruntime / missing model file
            raise ModelUnavailable(f"ddddocr failed to load: {exc.__class__.__name__}: "
                                   f"{exc}") from exc
        self.session = None
        try:
            session = self._ocr.ocr_engine.session
            charset = list(self._ocr.get_charset())
            self.input_name = session.get_inputs()[0].name
        except (AttributeError, IndexError, TypeError) as exc:
            logger.warning("ddddocr internals changed (%s); common OCR runs without an "
                           "alphabet mask", exc)
            return
        self.session, self.charset = session, charset
        size = len(charset)

        def mask(pred) -> "np.ndarray":
            out = np.zeros(size, dtype=bool)
            out[[i for i, ch in enumerate(charset) if ch and pred(ch)]] = True
            return out

        ascii_alnum = mask(lambda ch: all(c.isascii() and c.isalnum() for c in ch))
        self.masks = {
            0: ascii_alnum,
            NUMERIC_DIGITS: mask(lambda ch: all(c.isascii() and c.isdigit() for c in ch)),
            NUMERIC_LETTERS: mask(lambda ch: all(c.isascii() and c.isalpha() for c in ch)),
        }
        for m in self.masks.values():
            m[0] = True  # CTC blank

    def read(self, gray, numeric: int) -> Tuple[str, float]:
        """One rendering (PIL "L", MODEL_HEIGHT high) -> (text, confidence)."""
        if self.session is None:
            buf = io.BytesIO()
            gray.save(buf, format="PNG")
            text = self._ocr.classification(buf.getvalue())
            text = "".join(c for c in (text or "") if c.isascii() and c.isalnum())
            return text, 1.0
        np = self._np
        x = (np.asarray(gray, dtype=np.float32) / 255.0)[None, None, :, :]
        logits = np.asarray(self.session.run(None, {self.input_name: x})[0])
        if logits.ndim == 3:  # (T, 1, C) or (1, T, C)
            logits = logits[:, 0, :] if logits.shape[1] == 1 else logits[0]
        logits = logits.astype(np.float64)
        probs = np.exp(logits - logits.max(axis=1, keepdims=True))
        probs /= probs.sum(axis=1, keepdims=True)
        allowed = self.masks.get(numeric, self.masks[0])
        best = np.where(allowed, probs, -1.0).argmax(axis=1)
        chars: List[str] = []
        scores: List[float] = []
        prev = -1
        for t, idx in enumerate(best.tolist()):
            if idx != prev and idx != 0:
                chars.append(self.charset[idx])
                scores.append(float(probs[t, idx]))
            elif idx == prev and idx != 0:
                scores[-1] = max(scores[-1], float(probs[t, idx]))
            prev = idx
        if not chars:
            return "", 0.0
        # the weakest character bounds how much the whole string can be trusted
        confidence = 0.5 * (min(scores) + sum(scores) / len(scores))
        return "".join(chars), confidence


def _get_engine() -> _Engine:
    global _engine, _failed
    with _lock:
        if _engine is None:
            if _failed is not None:
                raise ModelUnavailable(_failed)
            try:
                _engine = _Engine()
            except ModelUnavailable as exc:
                _failed = str(exc)
                logger.warning("common OCR unavailable: %s", exc)
                raise
        return _engine


def warmup() -> bool:
    _get_engine()
    return True


# preprocessing ----------------------------------------------------------------
def _load_gray(image: bytes):
    """Decode, flatten transparency onto white and convert to grayscale."""
    from PIL import Image, UnidentifiedImageError
    try:
        with Image.open(io.BytesIO(image)) as img:
            if max(img.size) > MAX_SIDE:
                raise InvalidImage(f"image is larger than {MAX_SIDE} px")
            img.load()
            rgba = img.convert("RGBA")
    except InvalidImage:
        raise
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError) as exc:
        raise InvalidImage("body is not a readable image") from exc
    # transparent captchas often hide black RGB under alpha 0: flatten onto white first
    background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
    gray = Image.alpha_composite(background, rgba).convert("L")
    width, height = gray.size
    if width < 2 or height < 2:
        raise InvalidImage("image is too small")
    new_width = max(1, round(width * MODEL_HEIGHT / height))
    if new_width > MAX_MODEL_WIDTH:
        raise InvalidImage("image aspect ratio is too wide for a text captcha")
    return gray.resize((new_width, MODEL_HEIGHT), Image.LANCZOS)


def _otsu(gray):
    import numpy as np
    from PIL import Image
    arr = np.asarray(gray)
    hist = np.bincount(arr.ravel(), minlength=256).astype(np.float64)
    total = hist.sum()
    weight = np.cumsum(hist)
    mean = np.cumsum(hist * np.arange(256))
    with np.errstate(divide="ignore", invalid="ignore"):
        between = (mean[-1] * weight - mean * total) ** 2 / (weight * (total - weight))
    threshold = int(np.nanargmax(between))
    return Image.fromarray(np.where(arr > threshold, 255, 0).astype(np.uint8))


def render_variants(gray, names: Sequence[str] = VARIANTS) -> List[Tuple[str, object]]:
    """The renderings to vote over; byte-identical renderings are kept once."""
    from PIL import ImageEnhance, ImageOps
    renderers = {
        "plain": lambda g: g,
        "enhanced": lambda g: ImageEnhance.Sharpness(ImageEnhance.Contrast(
            ImageOps.autocontrast(g, cutoff=2)).enhance(1.8)).enhance(2.0),
        "moderate": lambda g: ImageOps.autocontrast(g, cutoff=2),
        "binary": _otsu,
    }
    out, seen = [], set()
    for name in names:
        try:
            img = renderers[name](gray)
        except (ValueError, OSError, ZeroDivisionError) as exc:
            logger.debug("variant %s failed: %s", name, exc)
            continue
        digest = hashlib.blake2b(img.tobytes(), digest_size=16).digest()
        if digest not in seen:  # duplicates would inflate the agreement
            seen.add(digest)
            out.append((name, img))
    return out


def vote(reads: List[Tuple[str, float]], total: int) -> Reading:
    """
    Confidence-weighted vote over (text, confidence) reads of `total` renderings.

    The winner's confidence is its summed confidence over all renderings, so a single
    confident reading out of four scores lower than four agreeing ones.
    """
    score: Dict[str, float] = defaultdict(float)
    order: Dict[str, int] = {}
    for i, (text, conf) in enumerate(reads):
        if text:
            score[text] += conf
            order.setdefault(text, i)
    if not score:
        return Reading("", 0.0, [])
    total = max(1, total)
    ranked = sorted(score, key=lambda t: (-score[t], order[t]))
    candidates = [(t, min(1.0, score[t] / total)) for t in ranked]
    return Reading(candidates[0][0], candidates[0][1], candidates)


def recognize(image: bytes, numeric: int = 0) -> Reading:
    """Read `image`; InvalidImage for unusable input, ModelUnavailable without ddddocr."""
    engine = _get_engine()
    gray = _load_gray(image)
    lo, hi = gray.getextrema()
    if lo == hi:  # a flat image has no text at all
        return Reading("", 0.0, [])
    variants = render_variants(gray)
    reads = [engine.read(img, numeric) for _, img in variants]
    return vote(reads, len(variants))
