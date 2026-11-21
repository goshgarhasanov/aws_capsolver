"""
Lazy access to the GeeTest models (package aws_capsolver.geetest, built separately).

The package exposes four solvers and three typed errors:

* ``slide_offset(background, piece=None, full_background=None, y=None) -> int``
* ``click_points(image, kind, hints=None) -> [(x, y), ...]`` (kind icon | word | space)
* ``nine_grid(image, target, count=None) -> [index, ...]`` (0-based, row-major)
* ``gobang(board) -> ((row, col), (row, col))``
* ``GeeTestError`` > ``UnsupportedGeeTest`` / ``InvalidGeeTestInput``

This bridge never imports the package at service start. Every call is translated into one
of the bridge's own exceptions, so the task layer does not depend on the model package:

* ``GeeTestUnavailable``  - package or solver function missing, model weights missing
* ``GeeTestInvalidInput`` - the client sent an image the model cannot read
* ``GeeTestUnsolvable``   - the model could not answer (or answered nonsense)
"""

import importlib
import logging
import threading
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

logger = logging.getLogger(__name__)

PACKAGE = "aws_capsolver.geetest"
FUNCTIONS = ("slide_offset", "click_points", "nine_grid", "gobang")
CLICK_KINDS = ("icon", "word", "space")
RETRY_SECONDS = 300.0   # a failed import is retried after this long (the package may land later)
MAX_POINTS = 16         # click captchas never ask for more clicks than this
MAX_BOARD = 16          # gobang board side upper bound for sanity checks

_lock = threading.Lock()
_module: Any = None      # module, False (failed), or None (not tried yet)
_failed_at = 0.0


class GeeTestUnavailable(Exception):
    """The GeeTest model package (or the needed solver) is not installed or cannot load."""


class GeeTestInvalidInput(ValueError):
    """An image could not be decoded / is not a GeeTest picture of the expected shape."""


class GeeTestUnsolvable(Exception):
    """The model could not produce an answer."""


def _load():
    global _module, _failed_at
    with _lock:
        if _module is False and time.monotonic() - _failed_at >= RETRY_SECONDS:
            _module = None
        if _module is None:
            try:
                _module = importlib.import_module(PACKAGE)
            except Exception as exc:  # ImportError, SyntaxError while the package is being written, native deps
                logger.warning("geetest package unavailable (%s: %s); GeeTest tasks disabled",
                               exc.__class__.__name__, exc)
                _module = False
                _failed_at = time.monotonic()
        return _module or None


def available() -> bool:
    return _load() is not None


def solvers() -> Dict[str, bool]:
    """Which solver functions the installed package provides (for /health)."""
    mod = _load()
    return {name: bool(mod is not None and callable(getattr(mod, name, None))) for name in FUNCTIONS}


def _error_types(mod) -> Tuple[tuple, tuple, tuple]:
    """(invalid input, unsupported, any GeeTest error) exception classes of the package."""
    names = ("InvalidGeeTestInput", "UnsupportedGeeTest", "GeeTestError")
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

    return pick("InvalidGeeTestInput"), pick("UnsupportedGeeTest"), pick("GeeTestError")


def _call(name: str, *args, **kwargs):
    mod = _load()
    if mod is None:
        raise GeeTestUnavailable("the GeeTest model package is not installed")
    fn = getattr(mod, name, None)
    if not callable(fn):
        raise GeeTestUnavailable(f"the GeeTest model package has no {name}()")
    invalid, unsupported, base = _error_types(mod)
    try:
        from PIL import UnidentifiedImageError
    except ImportError:  # Pillow is a hard dependency; this only keeps the bridge importable
        UnidentifiedImageError = ()  # type: ignore[assignment]
    try:
        return fn(*args, **kwargs)
    except invalid as exc:
        raise GeeTestInvalidInput(str(exc) or "invalid GeeTest image") from exc
    except unsupported as exc:
        raise GeeTestUnsolvable(str(exc) or "unsupported GeeTest challenge") from exc
    except base as exc:
        raise GeeTestUnsolvable(str(exc) or exc.__class__.__name__) from exc
    except UnidentifiedImageError as exc:  # a model that does not wrap Pillow's error
        raise GeeTestInvalidInput("an image is not readable") from exc
    except (ImportError, FileNotFoundError) as exc:  # weights or a native dependency missing
        raise GeeTestUnavailable(f"{name}: {exc}") from exc


def _int(value: Any, what: str) -> int:
    """A whole number from model output (numpy scalars and integral floats allowed)."""
    if isinstance(value, bool):
        raise GeeTestUnsolvable(f"bad model output: {what} is a boolean")
    try:
        f = float(value)
    except (TypeError, ValueError) as exc:
        raise GeeTestUnsolvable(f"bad model output: {what} is not a number") from exc
    if f != f or f in (float("inf"), float("-inf")):
        raise GeeTestUnsolvable(f"bad model output: {what} is not finite")
    return int(round(f))


def _pair(value: Any, what: str) -> Tuple[int, int]:
    try:
        a, b = value
    except (TypeError, ValueError) as exc:
        raise GeeTestUnsolvable(f"bad model output: {what} is not a pair") from exc
    return _int(a, what), _int(b, what)


# ---------------------------------------------------------------------- solvers

def slide_offset(background: bytes, piece: Optional[bytes] = None,
                 full_background: Optional[bytes] = None, y: Optional[int] = None) -> int:
    """
    x (background pixels) where the piece image's left edge must end (the drag distance);
    without a piece, the left edge of the gap. `y` = top of the piece image in background
    pixels (rules out decoy gaps); a model build without the parameter is called without it.
    """
    if y is not None:
        try:
            return _int(_call("slide_offset", background, piece=piece, full_background=full_background,
                              y=y), "x")
        except TypeError as exc:
            if "'y'" not in str(exc):
                raise
            logger.info("geetest slide_offset() has no y parameter; solving without it")
    return _int(_call("slide_offset", background, piece=piece, full_background=full_background), "x")


def click_points(image: bytes, kind: str, hints: Optional[Sequence[Union[str, bytes]]] = None
                 ) -> List[Tuple[int, int]]:
    """Points to click in order (image pixels)."""
    if kind not in CLICK_KINDS:
        raise GeeTestInvalidInput(f"unknown click kind {kind!r}")
    raw = _call("click_points", image, kind, hints=list(hints) if hints else None)
    try:
        items = list(raw)
    except TypeError as exc:
        raise GeeTestUnsolvable("bad model output: points is not a list") from exc
    if len(items) > MAX_POINTS:
        raise GeeTestUnsolvable(f"bad model output: {len(items)} points")
    return [_pair(p, "point") for p in items]


def nine_grid(image: bytes, target: Union[str, bytes], count: Optional[int] = None) -> List[int]:
    """Cells (0-based, row-major) that show the target; `count` = how many to pick, if known."""
    raw = _call("nine_grid", image, target) if count is None else _call("nine_grid", image, target, count=count)
    try:
        items = list(raw)
    except TypeError as exc:
        raise GeeTestUnsolvable("bad model output: cells is not a list") from exc
    return [_int(i, "cell") for i in items]


def gobang(board: bytes) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    """The move that completes five in a row: ((row, col) from, (row, col) to)."""
    raw = _call("gobang", board)
    try:
        a, b = raw
    except (TypeError, ValueError) as exc:
        raise GeeTestUnsolvable("bad model output: move is not a (from, to) pair") from exc
    src, dst = _pair(a, "move from"), _pair(b, "move to")
    if not all(0 <= v < MAX_BOARD for v in (*src, *dst)):
        raise GeeTestUnsolvable(f"bad model output: move {src} -> {dst} is off the board")
    return src, dst


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
        logger.warning("geetest warm-up failed: %s: %s", exc.__class__.__name__, exc)
        return False
