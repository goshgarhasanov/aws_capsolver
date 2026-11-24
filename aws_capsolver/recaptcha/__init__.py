"""reCAPTCHA v2 image classification (CapSolver ``ReCaptchaV2Classification`` style).

Public interface:
    classify(image, question, size=None) -> dict
    normalize_question(question) -> str
    warm(background=False)
    RecaptchaError, UnsupportedQuestion, InvalidRecaptchaInput

``image`` is the challenge picture as served (raw bytes or base64): a 3x3 grid (9 photos,
~300x300), a 4x4 grid (one photo split into 16 cells, ~450x450) or a single ~100x100
replacement tile. ``question`` is a Freebase MID ("/m/0k4j") or a text target in many
languages ("cars", "Select all images with traffic lights", "arabalar", "светофоры").

Results:
    size 3 / 4 -> {"type": "multi", "objects": [0-based row-major indexes], "size": 3|4}
    size 1     -> {"type": "single", "hasObject": bool, "size": 1}

Heavy dependencies (torch / open_clip) are imported only on the first classification,
so importing this package is cheap. Thread-safe.
"""
from __future__ import annotations

import logging
import threading

from .errors import InvalidRecaptchaInput, RecaptchaError, UnsupportedQuestion
from .questions import CLASSES, MIDS, normalize_question

log = logging.getLogger(__name__)


def _check_size(size: object) -> int | None:
    if size is None:
        return None
    if isinstance(size, bool) or not isinstance(size, int):
        if isinstance(size, str) and size.strip().isdigit():
            size = int(size.strip())
        else:
            raise InvalidRecaptchaInput(f"size must be 1, 3 or 4, got {size!r}")
    if size not in (1, 3, 4):
        raise InvalidRecaptchaInput(f"size must be 1, 3 or 4, got {size!r}")
    return size


def classify_details(image: bytes | str, question: str, size: int | None = None) -> dict:
    """Like :func:`classify` but also returns per-cell probabilities, threshold and method."""
    target = normalize_question(question)  # validate before the (slow) model load
    size = _check_size(size)
    from . import features
    from .model import get_engine

    img = features.decode(image)
    if size is None:
        size = features.infer_size(img)
    res = get_engine().classify(img, target, size)
    res.update(target=target, size=size)
    return res


def classify(image: bytes | str, question: str, size: int | None = None) -> dict:
    """Solve one reCAPTCHA v2 image challenge (see module docstring for the shapes).

    Raises InvalidRecaptchaInput (bad image / size), UnsupportedQuestion (unknown MID or
    unintelligible text) or RecaptchaError (model could not be loaded).
    """
    res = classify_details(image, question, size)
    if res["size"] == 1:
        return {"type": "single", "hasObject": bool(res["selected"]), "size": 1}
    return {"type": "multi", "objects": res["selected"], "size": res["size"]}


def warm(background: bool = False) -> threading.Thread | None:
    """Load backbones + heads + zero-shot text vocabulary and run dummy 1/3/4 inferences.

    Call it at service start (``background=True`` returns at once; requests arriving
    meanwhile wait for the load) so the first real task does not pay ~20 s of loading.
    """
    def _run() -> None:
        import io
        import time

        from PIL import Image

        from .model import get_engine

        try:
            t0 = time.time()
            eng = get_engine()
            for side, q, size in ((300, "/m/0k4j", 3), (450, "/m/04_sv", 4), (100, "/m/015qff", 1),
                                  (300, "Select all images with guitars", 3)):
                buf = io.BytesIO()
                Image.new("RGB", (side, side), (120, 130, 140)).save(buf, "JPEG")
                classify(buf.getvalue(), q, size)
            eng.warm_text()
            log.info("recaptcha warm-up done in %.1fs", time.time() - t0)
        except Exception:  # noqa: BLE001 - a failed warm-up must not kill the thread silently
            log.exception("recaptcha warm-up failed")

    if not background:
        _run()
        return None
    th = threading.Thread(target=_run, name="recaptcha-warmup", daemon=True)
    th.start()
    return th


warmup = warm

__all__ = ["classify", "classify_details", "normalize_question", "warm", "warmup", "CLASSES", "MIDS",
           "RecaptchaError", "UnsupportedQuestion", "InvalidRecaptchaInput"]
