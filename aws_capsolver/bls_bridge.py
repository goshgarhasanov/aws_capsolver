"""
Lazy access to bls_capsolver, the offline CNN for the BLS grid captcha
("select all boxes with number NNN", e.g. turkey.blsspainglobal.com).

bls_capsolver is a separate project (private repo goshgarhasanov/bls_capsolver)
installed into this environment, e.g. `pip install --no-deps -e ../bls_capsolver`.
When it is missing, every call raises BlsUnavailable and the API answers
ERROR_TASK_NOT_SUPPORTED instead of crashing.
"""

import importlib
import io
import logging
import os
import threading
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)

# Production threshold used by the BlsSpainTurkey bot (bls_capsolver reads it at import).
DEFAULT_THRESHOLD = "0.3"

_lock = threading.Lock()
_infer_lock = threading.Lock()
_module = None


class BlsUnavailable(Exception):
    """bls_capsolver is not installed or its weights cannot be loaded."""


class BlsInvalidInput(ValueError):
    """A tile could not be decoded as an image."""


def _load():
    global _module
    with _lock:
        if _module is None:
            os.environ.setdefault("BLS_CAPSOLVER_THRESHOLD", DEFAULT_THRESHOLD)
            try:
                _module = importlib.import_module("bls_capsolver")
                getattr(_module, "match_images")
            except Exception as exc:  # ImportError, AttributeError, broken torch install
                logger.warning("bls_capsolver unavailable (%s); BLS tasks disabled", exc)
                _module = False
        return _module or None


def available() -> bool:
    return _load() is not None


def _decode(data: bytes, index: int):
    from PIL import Image, UnidentifiedImageError
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise BlsInvalidInput(f"images[{index}] is not a readable image") from exc
    if max(img.size) > 1024:
        raise BlsInvalidInput(f"images[{index}] is larger than 1024 px")
    return img.convert("RGB")


def _get_model(mod):
    """Load the CNN once without letting it change torch's process-wide thread count."""
    import torch
    threads = torch.get_num_threads()
    try:
        return mod.predict.get_model()
    finally:
        torch.set_num_threads(threads)


def match(images: List[bytes], target: str) -> Tuple[List[int], List[str]]:
    """
    Tiles showing the 3-digit `target` -> (indices to click, per-tile readings).

    Raises BlsInvalidInput for undecodable tiles and BlsUnavailable when the model
    cannot be used.
    """
    mod = _load()
    if not mod:
        raise BlsUnavailable("bls_capsolver is not installed")
    items = [{"id": str(i), "img": _decode(data, i)} for i, data in enumerate(images)]
    with _infer_lock:
        if _get_model(mod) is None:
            raise BlsUnavailable(f"bls_capsolver weights missing at {mod.MODEL_PATH}")
        try:
            ids, info = mod.match_images(items, target)
        except mod.predict.CaptchaModelError as exc:
            raise BlsUnavailable(str(exc)) from exc
    readings = [info[it["id"]][0] if it["id"] in info else "" for it in items]
    return sorted(int(i) for i in ids), readings


def warmup() -> Optional[bool]:
    """Load the CNN up front; best effort."""
    mod = _load()
    if not mod:
        return None
    with _infer_lock:
        try:
            return _get_model(mod) is not None
        except Exception as exc:  # warm-up must never take the service down
            logger.warning("bls warm-up failed: %s: %s", exc.__class__.__name__, exc)
            return False
