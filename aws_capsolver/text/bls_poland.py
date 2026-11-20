"""Dedicated reader for the BLS Poland login captcha (appointment.blspolandvisa.com).

The captcha is CodeIgniter 3 ``create_captcha`` output: 160x55 JPEG, 5 black Open Sans
digits at random baselines over a red line spiral. A small CRNN+CTC (model.py) trained
on a faithful synthetic generator (tools/text_synth.py) and fine-tuned on real samples
reads it; decoding is constrained to exactly 5 digits.

    from aws_capsolver.text.bls_poland import solve
    text, confidence = solve(jpeg_bytes)

The model is loaded lazily (thread-safe singleton) from ``weights/bls_poland_ocr.pt``
(override with AWS_CAPSOLVER_BLS_POLAND_WEIGHTS) with ``torch.load(weights_only=True)``.
Inference is deterministic (eval mode, no sampling).

Environment variables
---------------------
AWS_CAPSOLVER_BLS_POLAND_WEIGHTS   weights path (default weights/bls_poland_ocr.pt)
AWS_CAPSOLVER_TEXT_THREADS         torch intra-op threads used for inference (default 2)
"""

from __future__ import annotations

import io
import logging
import os
import threading
import warnings
from pathlib import Path
from typing import List, Sequence, Tuple

from .errors import InvalidImage, ModelUnavailable

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_WEIGHTS = ROOT / "weights" / "bls_poland_ocr.pt"
LENGTH = 5
MAX_BYTES = 2 * 1024 * 1024
MAX_SIDE = 1024
MAX_PIXELS = 1024 * 1024

_lock = threading.Lock()
_model = None
_infer_lock = threading.Lock()


def weights_path() -> Path:
    return Path(os.environ.get("AWS_CAPSOLVER_BLS_POLAND_WEIGHTS") or DEFAULT_WEIGHTS)


def _load():
    """Load the model once (double-checked locking)."""
    global _model
    if _model is not None:
        return _model
    with _lock:
        if _model is not None:
            return _model
        path = weights_path()
        if not path.is_file():
            raise ModelUnavailable(f"BLS Poland OCR weights not found: {path}")
        try:
            import torch

            from .model import ARCH, CRNN
        except ImportError as exc:
            raise ModelUnavailable(f"torch is not installed ({exc})") from exc
        try:
            ckpt = torch.load(path, map_location="cpu", weights_only=True)
            if ckpt.get("arch") != ARCH:
                raise ModelUnavailable(f"{path} has arch {ckpt.get('arch')!r}, expected {ARCH!r}")
            model = CRNN()
            model.load_state_dict(ckpt["state_dict"])
        except ModelUnavailable:
            raise
        except Exception as exc:  # corrupt / incompatible file
            raise ModelUnavailable(f"cannot load {path}: {exc.__class__.__name__}: {exc}") from exc
        model.eval()
        threads = int(os.environ.get("AWS_CAPSOLVER_TEXT_THREADS", "2") or 2)
        _model = (model, max(1, threads))
        log.info("BLS Poland OCR loaded from %s", path)
        return _model


def warmup() -> bool:
    """Load the model up front (raises ModelUnavailable when the weights are missing)."""
    _load()
    return True


def _decode(image) -> "object":
    """bytes -> HxWx3 uint8 array at the model's input size; InvalidImage when unusable."""
    from PIL import Image, UnidentifiedImageError

    from .model import load_rgb

    if isinstance(image, (bytearray, memoryview)):
        image = bytes(image)
    if not isinstance(image, bytes) or not image:
        raise InvalidImage("image must be non-empty bytes")
    if len(image) > MAX_BYTES:
        raise InvalidImage(f"image is too large ({len(image)} bytes > {MAX_BYTES})")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            im = Image.open(io.BytesIO(image))
            w, h = im.size
            if w < 8 or h < 8 or w > MAX_SIDE or h > MAX_SIDE or w * h > MAX_PIXELS:
                raise InvalidImage(f"unexpected image size {w}x{h}")
            im.load()
            return load_rgb(im)
    except InvalidImage:
        raise
    except (UnidentifiedImageError, OSError, ValueError, SyntaxError,
            Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        raise InvalidImage(f"cannot decode image: {exc.__class__.__name__}: {exc}") from exc


def solve_many(images: Sequence[bytes]) -> List[Tuple[str, float]]:
    """Read several captchas in one batch: [(text, confidence)] in input order.

    Raises InvalidImage if any input is unusable, ModelUnavailable without weights.
    """
    if not images:
        return []
    arrays = [_decode(img) for img in images]
    model, threads = _load()
    import torch

    from .model import constrained_decode, to_tensor

    x = torch.stack([to_tensor(a) for a in arrays])
    with _infer_lock:  # torch thread settings are process-global
        prev = torch.get_num_threads()
        try:
            torch.set_num_threads(threads)
            with torch.inference_mode():
                logp = model(x).numpy()
        finally:
            torch.set_num_threads(prev)
    out = []
    for lp in logp:
        text, conf, _ = constrained_decode(lp, LENGTH)
        out.append((text, max(0.0, min(1.0, conf))))
    return out


def solve(image: bytes) -> Tuple[str, float]:
    """Read one BLS Poland captcha: (5-digit text, confidence 0..1)."""
    return solve_many([image])[0]


__all__ = ["solve", "solve_many", "warmup", "weights_path", "InvalidImage", "ModelUnavailable"]
