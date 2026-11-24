"""
Lazy access to the reCAPTCHA v2 image model (package aws_capsolver.recaptcha, built separately).

The package exposes one classifier and typed errors:

* ``classify(image, question, size=None) -> dict``
  - grid picture (size 3 or 4): ``{"type": "multi", "objects": [0-based, row-major], "size": 3|4}``
  - one replaced tile (size 1): ``{"type": "single", "hasObject": bool, "size": 1}``
  ``question`` is a Freebase MID (``/m/0k4j``) or the instruction text in any language.
* ``normalize_question(question)`` (not used here)
* ``RecaptchaError`` > ``UnsupportedQuestion`` / ``InvalidRecaptchaInput``

This bridge never imports the package at service start. Every call is translated into one
of the bridge's own exceptions, so the task layer does not depend on the model package:

* ``RecaptchaUnavailable``  - package or classify() missing, model weights missing
* ``RecaptchaInvalidInput`` - the client sent an image the model cannot read
* ``RecaptchaUnsupported``  - the model does not know the asked object (a subclass of
  ``RecaptchaUnsolvable``; the client should load another challenge)
* ``RecaptchaUnsolvable``   - the model could not answer (or answered nonsense)
"""

import importlib
import logging
import sys
import threading
import time
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

PACKAGE = "aws_capsolver.recaptcha"
SIZES = (1, 3, 4)
RETRY_SECONDS = 300.0   # a failed import is retried after this long (the package may land later)

_lock = threading.Lock()
_module: Any = None      # module, False (failed), or None (not tried yet)
_failed_at = 0.0


class RecaptchaUnavailable(Exception):
    """The reCAPTCHA model package (or its classify()) is not installed or cannot load."""


class RecaptchaInvalidInput(ValueError):
    """The image could not be decoded / is not a reCAPTCHA picture of a known layout."""


class RecaptchaUnsolvable(Exception):
    """The model could not produce an answer."""


class RecaptchaUnsupported(RecaptchaUnsolvable):
    """The asked object is not one the model knows."""


def _load():
    global _module, _failed_at
    with _lock:
        if _module is False and time.monotonic() - _failed_at >= RETRY_SECONDS:
            _module = None
        if _module is None:
            try:
                _module = importlib.import_module(PACKAGE)
                if not callable(getattr(_module, "classify", None)):
                    # a half-written package: drop it from the import cache so the retry re-reads it
                    for name in [n for n in sys.modules if n == PACKAGE or n.startswith(PACKAGE + ".")]:
                        sys.modules.pop(name, None)
                    raise ImportError("the package has no classify()")
            except Exception as exc:  # ImportError, SyntaxError while the package is being written, native deps
                logger.warning("recaptcha package unavailable (%s: %s); reCAPTCHA tasks disabled",
                               exc.__class__.__name__, exc)
                _module = False
                _failed_at = time.monotonic()
        return _module or None


def available() -> bool:
    """True when the package imports and offers classify() (weights are only checked on use)."""
    mod = _load()
    return bool(mod is not None and callable(getattr(mod, "classify", None)))


def _error_types(mod) -> Tuple[tuple, tuple, tuple]:
    """(invalid input, unsupported question, any reCAPTCHA error) exception classes of the package."""
    names = ("InvalidRecaptchaInput", "UnsupportedQuestion", "RecaptchaError")
    errors = mod
    if not all(hasattr(mod, n) for n in names):  # not re-exported: read the errors module
        errors = getattr(mod, "errors", None)
        if errors is None:
            try:
                errors = importlib.import_module(PACKAGE + ".errors")
            except Exception:  # a build without an errors module
                errors = None

    def pick(name):
        cls = getattr(mod, name, None) or (getattr(errors, name, None) if errors is not None else None)
        return (cls,) if isinstance(cls, type) and issubclass(cls, BaseException) else ()

    return pick("InvalidRecaptchaInput"), pick("UnsupportedQuestion"), pick("RecaptchaError")


def _call(image: bytes, question: str, size: Optional[int]):
    mod = _load()
    if mod is None:
        raise RecaptchaUnavailable("the reCAPTCHA model package is not installed")
    fn = getattr(mod, "classify", None)
    if not callable(fn):
        raise RecaptchaUnavailable("the reCAPTCHA model package has no classify()")
    invalid, unsupported, base = _error_types(mod)
    try:
        from PIL import UnidentifiedImageError
    except ImportError:  # Pillow is a hard dependency; this only keeps the bridge importable
        UnidentifiedImageError = ()  # type: ignore[assignment]
    try:
        return fn(image, question, size=size)
    except invalid as exc:
        raise RecaptchaInvalidInput(str(exc) or "invalid reCAPTCHA image") from exc
    except unsupported as exc:
        raise RecaptchaUnsupported(str(exc) or f"unsupported question {question!r}") from exc
    except base as exc:
        raise RecaptchaUnsolvable(str(exc) or exc.__class__.__name__) from exc
    except UnidentifiedImageError as exc:  # a model that does not wrap Pillow's error
        raise RecaptchaInvalidInput("the image is not readable") from exc
    except (ImportError, FileNotFoundError) as exc:  # weights or a native dependency missing
        raise RecaptchaUnavailable(f"classify: {exc}") from exc


def _int(value: Any, what: str) -> int:
    """A whole number from model output (numpy integers allowed, floats and booleans not)."""
    if isinstance(value, bool):
        raise RecaptchaUnsolvable(f"bad model output: {what} is a boolean")
    try:
        as_int = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise RecaptchaUnsolvable(f"bad model output: {what} is not an integer") from exc
    try:
        if as_int != value:  # 1.5, nan
            raise RecaptchaUnsolvable(f"bad model output: {what} is not an integer")
    except TypeError as exc:  # an exotic type without comparison
        raise RecaptchaUnsolvable(f"bad model output: {what} is not an integer") from exc
    return as_int


def check_output(raw: Any, size_hint: Optional[int] = None) -> Dict[str, Any]:
    """
    Validate and normalize a classify() result into the CapSolver solution shape.
    Duplicated cells are folded and the cells sorted; anything else unexpected
    (wrong type, size or cell out of range, a size that contradicts the hint) is unsolvable.
    """
    if not isinstance(raw, dict):
        raise RecaptchaUnsolvable("bad model output: not an object")
    size = _int(raw.get("size"), "size")
    if size not in SIZES:
        raise RecaptchaUnsolvable(f"bad model output: size {size}")
    if size_hint is not None and size != size_hint:
        raise RecaptchaUnsolvable(f"bad model output: size {size}, the task says {size_hint}")
    kind = raw.get("type")
    if size == 1:
        has = raw.get("hasObject")
        # numpy.bool_ is not a bool subclass; accept it, but never 0 / 1 / "yes"
        numpy_bool = type(has).__module__ == "numpy" and type(has).__name__ in ("bool_", "bool")
        if kind != "single" or not (isinstance(has, bool) or numpy_bool):
            raise RecaptchaUnsolvable("bad model output: a 1x1 answer needs type single and hasObject")
        return {"type": "single", "hasObject": bool(has), "size": 1}
    if kind != "multi":
        raise RecaptchaUnsolvable(f"bad model output: a {size}x{size} answer needs type multi")
    objects = raw.get("objects")
    if isinstance(objects, (str, bytes, dict)) or objects is None:
        raise RecaptchaUnsolvable("bad model output: objects is not a list")
    try:
        items = list(objects)
    except TypeError as exc:
        raise RecaptchaUnsolvable("bad model output: objects is not a list") from exc
    cells = sorted({_int(i, "cell") for i in items})
    if any(not 0 <= i < size * size for i in cells):
        raise RecaptchaUnsolvable(f"bad model output: cells {cells} outside the {size}x{size} grid")
    return {"type": "multi", "objects": cells, "size": size}


def classify(image: bytes, question: str, size: Optional[int] = None) -> Dict[str, Any]:
    """The checked answer for one reCAPTCHA v2 picture (see check_output for the shape)."""
    if size is not None and size not in SIZES:
        raise RecaptchaInvalidInput(f"size must be one of {SIZES}")
    return check_output(_call(image, question, size), size)


def warmup() -> Optional[bool]:
    """Let the package load its models up front, when it offers a warmup(); best effort."""
    mod = _load()
    if mod is None:
        return None
    fn = getattr(mod, "warmup", None)
    if not callable(fn):
        return None
    try:
        fn()
        return True
    except Exception as exc:  # warm-up must never take the service down
        logger.warning("recaptcha warm-up failed: %s: %s", exc.__class__.__name__, exc)
        return False
