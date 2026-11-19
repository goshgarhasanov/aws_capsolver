"""
Lazy access to the vision package (built separately in aws_capsolver/vision).

Until that package exists, a fallback raises UnsupportedProblem for every call so
the protocol layer and the API keep working and report ERROR_CAPTCHA_UNSOLVABLE.
"""

import importlib
import logging
import threading
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)
_lock = threading.Lock()
_module = None


class FallbackUnsupported(Exception):
    """Raised when the vision package is missing or incomplete."""


def _load():
    global _module
    with _lock:
        if _module is None:
            try:
                _module = importlib.import_module("aws_capsolver.vision")
                for name in ("classify_grid", "solve_toycar", "transcribe", "UnsupportedProblem"):
                    getattr(_module, name)
            except Exception as exc:  # ImportError, AttributeError, broken native deps
                logger.warning("vision package unavailable (%s); captcha solving disabled", exc)
                _module = False
        return _module or None


def unsupported_types() -> tuple:
    """Exceptions meaning "this problem cannot be solved" (any vision failure)."""
    mod = _load()
    if not mod:
        return (FallbackUnsupported,)
    return (FallbackUnsupported, getattr(mod, "VisionError", mod.UnsupportedProblem),
            mod.UnsupportedProblem)


def invalid_input_types() -> tuple:
    """Exceptions meaning the client sent undecodable images/audio."""
    mod = _load()
    invalid = getattr(mod, "InvalidInput", None) if mod else None
    return (invalid,) if invalid else ()


def available() -> bool:
    return _load() is not None


def classify_grid(images: List[bytes], target: str, count: Optional[int] = None) -> List[int]:
    """Tile indices showing `target`; `count` is the number of tiles the problem expects."""
    mod = _load()
    if not mod:
        raise FallbackUnsupported("vision package is not installed")
    if count is None:
        return list(mod.classify_grid(images, target))
    try:
        return list(mod.classify_grid(images, target, count=count))
    except TypeError as exc:  # an older vision build without the `count` keyword
        if "count" not in str(exc):
            raise
        return list(mod.classify_grid(images, target))


def solve_toycar(image: bytes) -> Tuple[int, int]:
    mod = _load()
    if not mod:
        raise FallbackUnsupported("vision package is not installed")
    x, y = mod.solve_toycar(image)
    return int(x), int(y)


def transcribe(audio_b64: str) -> str:
    mod = _load()
    if not mod:
        raise FallbackUnsupported("vision package is not installed")
    return str(mod.transcribe(audio_b64))


def warmup() -> None:
    """Load the grid model once so the first real request does not pay for it."""
    mod = _load()
    if not mod:
        return
    import io
    from PIL import Image
    buf = io.BytesIO()
    Image.new("RGB", (100, 100), (127, 127, 127)).save(buf, format="JPEG")
    try:
        mod.classify_grid([buf.getvalue()] * 9, "bed")
        logger.info("vision warm-up done")
    except Exception as exc:  # warm-up is best effort; real calls report their own errors
        logger.warning("vision warm-up failed: %s: %s", exc.__class__.__name__, exc)
