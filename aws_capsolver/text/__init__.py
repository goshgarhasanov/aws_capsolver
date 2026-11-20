"""
Text captcha recognition for ImageToTextTask: module routing and post-filters.

Modules:
  bls_poland  the dedicated BLS Poland model (aws_capsolver.text.bls_poland), loaded lazily;
              aliases "blspoland" and "bls-poland"
  common      every other text captcha (alias "general"): the broad general model
              (aws_capsolver.text.general) when it is installed and has its weights, else
              the ddddocr fallback
  ddddocr     force the fallback (common.py: ddddocr's ONNX model, multi-preprocessing vote)

Nothing heavy is imported here: a module's backend is imported on first use, so the
service starts (and the other task types work) when a backend is missing. A missing
backend raises ModelUnavailable, an unusable image InvalidImage (see errors.py).
"""

import importlib
import importlib.util
import logging
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from .errors import InvalidImage, ModelUnavailable

logger = logging.getLogger(__name__)

MODULE_BLS_POLAND = "bls_poland"
MODULE_COMMON = "common"
MODULE_DDDDOCR = "ddddocr"
MODULES = (MODULE_BLS_POLAND, MODULE_COMMON, MODULE_DDDDOCR)
DEFAULT_MODULE = MODULE_COMMON
_ALIASES = {
    "bls_poland": MODULE_BLS_POLAND,
    "blspoland": MODULE_BLS_POLAND,
    "bls-poland": MODULE_BLS_POLAND,
    "common": MODULE_COMMON,
    "general": MODULE_COMMON,
    "ddddocr": MODULE_DDDDOCR,
}
_BLS_POLAND_PATH = __name__ + ".bls_poland"
_GENERAL_PATH = __name__ + ".general"

# `numeric` (anti-captcha / 2captcha semantics, accepted by CapSolver clients)
NUMERIC_ANY = 0       # no restriction
NUMERIC_DIGITS = 1    # digits only
NUMERIC_LETTERS = 2   # letters only
NUMERIC_EITHER = 3    # only digits or only letters (prefer such a reading, strip nothing)
NUMERIC_BOTH = 4      # digits and letters (prefer such a reading, strip nothing)
NUMERIC_VALUES = (NUMERIC_ANY, NUMERIC_DIGITS, NUMERIC_LETTERS, NUMERIC_EITHER, NUMERIC_BOTH)

__all__ = ["MODULES", "DEFAULT_MODULE", "Constraints", "Reading", "UnknownModule",
           "TextUnsolvable", "InvalidImage", "ModelUnavailable", "resolve_module", "recognize",
           "apply_filters", "available", "warmup"]


class UnknownModule(ValueError):
    """`module` names no known recogniser."""


class TextUnsolvable(Exception):
    """No reading satisfies the requested filters (empty text, length out of range)."""


@dataclass
class Reading:
    """A recogniser's answer: the best text, its confidence (0..1) and ranked candidates."""
    text: str
    confidence: float
    candidates: List[Tuple[str, float]] = field(default_factory=list)


def resolve_module(name: Optional[str], default: str = DEFAULT_MODULE) -> str:
    """Canonical module name for `name` (None / "" -> `default`); UnknownModule otherwise."""
    if name is None or not str(name).strip():
        name = default
    key = str(name).strip().lower()
    if key not in _ALIASES:
        raise UnknownModule(f"unknown module {name!r}; valid modules: "
                            f"{', '.join(sorted(_ALIASES))}")
    return _ALIASES[key]


# backends ----------------------------------------------------------------------
@dataclass(frozen=True)
class Constraints:
    """What the task asks for; recognisers that can decode under constraints use them."""
    numeric: int = NUMERIC_ANY
    case: Optional[bool] = None        # True = case-sensitive, False = lower-case, None = as read
    min_length: int = 0                # 0 = no limit
    max_length: int = 0

    @property
    def charset(self) -> Optional[str]:
        return {NUMERIC_DIGITS: "digits", NUMERIC_LETTERS: "letters"}.get(self.numeric)


def _import(path: str, what: str):
    """Import a model module lazily; ModelUnavailable when it is missing or broken."""
    try:
        return importlib.import_module(path)
    except ImportError as exc:  # not built yet, or a dependency (torch, ...) is missing
        raise ModelUnavailable(f"the {what} model is not installed ({exc})") from exc


def _solver(mod):
    solve = getattr(mod, "solve", None)
    if not callable(solve):
        raise ModelUnavailable(f"{mod.__name__} has no solve()")
    return solve


def _reading(text, confidence, what: str) -> Reading:
    if not isinstance(text, str):
        raise ValueError(f"{what} returned {type(text).__name__} instead of text")
    confidence = _unit(confidence)
    return Reading(text, confidence, [(text, confidence)])


def _recognize_bls_poland(image: bytes, c: Constraints) -> Reading:
    solve = _solver(_import(_BLS_POLAND_PATH, "bls_poland"))
    return _reading(*solve(image), "bls_poland")


def _recognize_general(image: bytes, c: Constraints) -> Reading:
    solve = _solver(_import(_GENERAL_PATH, "general"))
    text, confidence = solve(image, charset=c.charset, min_len=c.min_length or None,
                             max_len=c.max_length or None,
                             # omitted `case` keeps the text as read
                             case_sensitive=c.case is not False)
    return _reading(text, confidence, "general")


def _recognize_ddddocr(image: bytes, c: Constraints) -> Reading:
    from . import common
    return common.recognize(image, numeric=c.numeric)


def _recognize_common(image: bytes, c: Constraints) -> Reading:
    """The general model when it can run, else the ddddocr fallback."""
    try:
        return _recognize_general(image, c)
    except ModelUnavailable as exc:
        logger.debug("general text model unavailable (%s); using ddddocr", exc)
    return _recognize_ddddocr(image, c)


_BACKENDS: Dict[str, Callable[[bytes, Constraints], Reading]] = {
    MODULE_BLS_POLAND: _recognize_bls_poland,
    MODULE_COMMON: _recognize_common,
    MODULE_DDDDOCR: _recognize_ddddocr,
}


def _unit(value) -> float:
    """A confidence clamped to 0..1 (TypeError / ValueError for non-numbers)."""
    value = float(value)
    if value != value:  # NaN
        raise ValueError("confidence is NaN")
    return min(1.0, max(0.0, value))


def recognize(module: str, image: bytes, constraints: Optional[Constraints] = None) -> Reading:
    """
    Read the text in `image` with `module` (a canonical name or alias).

    The constraints are passed to recognisers that decode under them (general, ddddocr);
    apply_filters() still enforces them on every result as a safety net.
    """
    return _BACKENDS[resolve_module(module)](image, constraints or Constraints())


def _has(path: str) -> bool:
    try:
        return importlib.util.find_spec(path) is not None
    except (ImportError, ValueError):
        return False


def available(module: str) -> bool:
    """Cheap check (no model load, weights not checked): could `module` run here?"""
    module = resolve_module(module)
    if module == MODULE_BLS_POLAND:
        return _has(_BLS_POLAND_PATH)
    if module == MODULE_COMMON:
        return _has(_GENERAL_PATH) or _has("ddddocr")
    return _has("ddddocr")


def _warm_model(path: str, what: str) -> Optional[bool]:
    """Call a model module's warmup() / load_model() / get_model() if it defines one."""
    if not _has(path):
        return None
    mod = _import(path, what)
    loader = next((getattr(mod, n) for n in ("warmup", "load_model", "get_model")
                   if callable(getattr(mod, n, None))), None)
    if loader is None:
        return None  # loads on the first request
    result = loader()
    return result is not False


def warmup() -> Dict[str, Optional[bool]]:
    """
    Load every available recogniser up front; best effort, never raises.

    Returns {backend: loaded?}, None = not installed here (or no loader to call).
    """
    jobs = (("bls_poland", lambda: _warm_model(_BLS_POLAND_PATH, "bls_poland")),
            ("general", lambda: _warm_model(_GENERAL_PATH, "general")),
            ("ddddocr", lambda: _has("ddddocr") and _warm_ddddocr() or None))
    result: Dict[str, Optional[bool]] = {}
    for name, job in jobs:
        try:
            result[name] = job()
        except Exception as exc:  # warm-up must never take the service down
            logger.warning("text warm-up of %s failed: %s: %s", name,
                           exc.__class__.__name__, exc)
            result[name] = False
    return result


def _warm_ddddocr() -> bool:
    from . import common
    return common.warmup()


# post-filters -----------------------------------------------------------------
def _kind_ok(text: str, numeric: int) -> bool:
    if numeric == NUMERIC_DIGITS:
        return all(c.isdigit() for c in text)
    if numeric == NUMERIC_LETTERS:
        return all(c.isalpha() for c in text)
    if numeric == NUMERIC_EITHER:
        return all(c.isdigit() for c in text) or all(c.isalpha() for c in text)
    if numeric == NUMERIC_BOTH:
        return any(c.isdigit() for c in text) and any(c.isalpha() for c in text)
    return True


def _strip_kind(text: str, numeric: int) -> str:
    if numeric == NUMERIC_DIGITS:
        return "".join(c for c in text if c.isdigit())
    if numeric == NUMERIC_LETTERS:
        return "".join(c for c in text if c.isalpha())
    return text


def apply_filters(reading: Reading, numeric: int = NUMERIC_ANY, case: Optional[bool] = None,
                  min_length: int = 0, max_length: int = 0) -> Tuple[str, float]:
    """
    Final (text, confidence) after the task's filters.

    The first ranked candidate that already satisfies `numeric` and the length range wins.
    Without one, the best reading has the other kind of character removed (numeric 1/2)
    and must then fit the length range, else TextUnsolvable. `case=False` lower-cases.
    A length of 0 means "no limit".
    """
    def fits(text: str) -> bool:
        return (bool(text) and (not min_length or len(text) >= min_length)
                and (not max_length or len(text) <= max_length))

    candidates = reading.candidates or [(reading.text, reading.confidence)]
    chosen = next(((t, c) for t, c in candidates if t and _kind_ok(t, numeric) and fits(t)),
                  None)
    if chosen is None:
        text = _strip_kind(reading.text, numeric)
        if not text:
            raise TextUnsolvable("no text recognised" if not reading.text else
                                 f"no characters left after numeric={numeric}")
        if not fits(text):
            raise TextUnsolvable(f"recognised {len(text)} characters, expected "
                                 f"{min_length or 1}..{max_length or 'any'}")
        chosen = (text, reading.confidence)
    text, confidence = chosen
    if case is False:
        text = text.lower()
    return text, confidence
