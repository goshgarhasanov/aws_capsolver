"""Task validation, the bounded worker pool and the in-memory task registry."""

import base64
import binascii
import io
import hmac
import logging
import threading
import time
import traceback
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from dataclasses import dataclass, field, replace
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Tuple
from urllib.parse import urlparse

from . import bls_bridge, geetest_bridge, recaptcha_bridge, text, vision_bridge
from .billing import BillingError, InsufficientTokens
from .config import Settings, UsageStore
from .waf.errors import (ALL_CODES, ERROR_CAPTCHA_UNSOLVABLE, ERROR_INVALID_TASK_DATA,
                         ERROR_NO_SLOT_AVAILABLE, ERROR_SERVICE_UNAVAILABLE, ERROR_ZERO_BALANCE,
                         ERROR_TASK_NOT_SUPPORTED, ERROR_TASK_TIMEOUT, ERROR_TASKID_INVALID,
                         SolverError)
from .waf.http import parse_proxy, redact
from .waf.solver import SolveRequest, SolverSettings, solve_website

logger = logging.getLogger(__name__)

TASK_WAF = "AntiAwsWafTask"
TASK_WAF_PROXYLESS = "AntiAwsWafTaskProxyLess"
TASK_CLASSIFY = "AwsWafClassification"
TASK_BLS = "BlsClassification"
TASK_TEXT = "ImageToTextTask"
TASK_GEETEST = "GeeTestClassification"
TASK_RECAPTCHA = "ReCaptchaV2Classification"
# synchronous task types: answered in the createTask reply, run on the bounded
# classification executor (shared runners, queue and admission control)
CLASSIFY_TYPES = (TASK_CLASSIFY, TASK_BLS, TASK_TEXT, TASK_GEETEST, TASK_RECAPTCHA)
SUPPORTED_TYPES = (TASK_WAF, TASK_WAF_PROXYLESS, TASK_CLASSIFY, TASK_BLS, TASK_TEXT, TASK_GEETEST,
                   TASK_RECAPTCHA)
BLS_GRID_TILES = 9
MAX_IMAGES = 16
MAX_IMAGE_BYTES = 2 * 1024 * 1024
MAX_URL_LENGTH = 4096
MAX_FIELD_LENGTH = 16 * 1024      # gokuProps context and similar opaque blobs
MAX_USER_AGENT_LENGTH = 512
MAX_TEXT_LENGTH = 64              # ImageToTextTask minLength / maxLength upper bound
WATCHDOG_GRACE = 15.0             # seconds past the deadline before a task is force-failed


@dataclass
class Task:
    id: str
    client_key: str
    type: str
    created: float = field(default_factory=time.monotonic)
    status: str = "idle"            # idle -> processing -> ready | failed
    solution: Optional[Dict[str, Any]] = None
    error_code: Optional[str] = None
    error_description: Optional[str] = None
    finished: Optional[float] = None


def _opt_str(task: Dict[str, Any], name: str, max_length: int = MAX_FIELD_LENGTH) -> Optional[str]:
    value = task.get(name)
    if value is None or value == "":
        return None
    if not isinstance(value, str):
        raise SolverError(ERROR_INVALID_TASK_DATA, f"{name} must be a string")
    value = value.strip()
    if len(value) > max_length:
        raise SolverError(ERROR_INVALID_TASK_DATA, f"{name} is too long")
    if any(ord(c) < 32 or ord(c) == 127 for c in value):
        raise SolverError(ERROR_INVALID_TASK_DATA, f"{name} contains control characters")
    return value or None


def _http_url(task: Dict[str, Any], name: str, required: bool = False) -> Optional[str]:
    value = _opt_str(task, name, MAX_URL_LENGTH)
    if value is None:
        if required:
            raise SolverError(ERROR_INVALID_TASK_DATA, f"{name} is required")
        return None
    try:
        parsed = urlparse(value)
        parsed.port  # raises ValueError on a malformed port
    except ValueError as exc:
        raise SolverError(ERROR_INVALID_TASK_DATA, f"{name} is not a valid URL") from exc
    if parsed.scheme.lower() not in ("http", "https") or not parsed.hostname:
        raise SolverError(ERROR_INVALID_TASK_DATA, f"{name} must be an absolute http(s) URL")
    return value


def build_solve_request(task: Dict[str, Any]) -> SolveRequest:
    """Validate an AntiAwsWafTask[ProxyLess] payload."""
    task_type = task.get("type")
    website_url = _http_url(task, "websiteURL", required=True)
    proxy_url = None
    if task_type == TASK_WAF:
        raw = task.get("proxy") or ({k: task[k] for k in ("proxyType", "proxyAddress", "proxyPort",
                                                          "proxyLogin", "proxyPassword")
                                     if k in task} or None)
        if not raw:
            raise SolverError(ERROR_INVALID_TASK_DATA, f"{TASK_WAF} requires a proxy")
        proxy_url = parse_proxy(raw)
    return SolveRequest(
        website_url=website_url,
        proxy_url=proxy_url,
        user_agent=_opt_str(task, "userAgent", MAX_USER_AGENT_LENGTH),
        aws_key=_opt_str(task, "awsKey"),
        aws_iv=_opt_str(task, "awsIv"),
        aws_context=_opt_str(task, "awsContext"),
        challenge_js=_http_url(task, "awsChallengeJS"),
        api_js=_http_url(task, "awsApiJs"),
        problem_url=_http_url(task, "awsProblemUrl"),
        api_key=_opt_str(task, "awsApiKey"),
        locale=_opt_str(task, "locale", 16) or "en-us",
    )


def _decode_image(value: Any, name: str) -> bytes:
    """Base64 (or a data: URL) -> bytes, at most MAX_IMAGE_BYTES; `name` labels errors."""
    if not isinstance(value, str) or not value:
        raise SolverError(ERROR_INVALID_TASK_DATA, f"{name} must be a base64 string")
    if value.startswith("data:"):
        value = value.split(",", 1)[-1]
    if len(value) > MAX_IMAGE_BYTES * 4 // 3 + 8:
        raise SolverError(ERROR_INVALID_TASK_DATA, f"{name} has a bad size")
    try:
        data = base64.b64decode(value, validate=False)
    except (binascii.Error, ValueError) as exc:
        raise SolverError(ERROR_INVALID_TASK_DATA, f"{name} is not base64") from exc
    if not data or len(data) > MAX_IMAGE_BYTES:
        raise SolverError(ERROR_INVALID_TASK_DATA, f"{name} has a bad size")
    return data


def parse_bls(task: Dict[str, Any]) -> Tuple[str, str, List[bytes]]:
    """
    Validate a BlsClassification payload -> ("bls", target, decoded images).

    The target comes from `question` ("bls:grid:748") or `target` ("748"); `images`
    are the 9 grid tiles in page order (base64 or data: URLs).
    """
    question = _opt_str(task, "question")
    target = _opt_str(task, "target", 16)
    if question:
        parts = question.split(":", 2)
        if len(parts) != 3 or parts[0] != "bls" or parts[1] != "grid":
            raise SolverError(ERROR_INVALID_TASK_DATA, "question must look like bls:grid:<NNN>")
        target = parts[2]
    target = (target or "").strip()
    if len(target) != 3 or not target.isdigit():
        raise SolverError(ERROR_INVALID_TASK_DATA, "the BLS target must be a 3-digit number")
    images = task.get("images")
    if not isinstance(images, list) or len(images) != BLS_GRID_TILES:
        raise SolverError(ERROR_INVALID_TASK_DATA,
                          f"images[] must hold the {BLS_GRID_TILES} grid tiles")
    return "bls", target, [_decode_image(v, f"images[{i}]") for i, v in enumerate(images)]


def parse_classification(task: Dict[str, Any]) -> Tuple[str, str, List[bytes]]:
    """Validate a classification payload -> (family, target, decoded images)."""
    if task.get("type") == TASK_BLS:
        return parse_bls(task)
    question = _opt_str(task, "question")
    images = task.get("images")
    if not question or not isinstance(images, list) or not images:
        raise SolverError(ERROR_INVALID_TASK_DATA, "question and images[] are required")
    if len(images) > MAX_IMAGES:
        raise SolverError(ERROR_INVALID_TASK_DATA, f"at most {MAX_IMAGES} images")
    parts = question.split(":", 2)
    if len(parts) != 3 or parts[0] != "aws":
        raise SolverError(ERROR_INVALID_TASK_DATA, "question must look like aws:grid:<target>")
    _, family, target = parts
    if family not in ("grid", "toycarcity"):
        raise SolverError(ERROR_TASK_NOT_SUPPORTED, f"question family {family!r} is not supported")
    if family == "grid" and not target:
        raise SolverError(ERROR_INVALID_TASK_DATA, "grid question without a target")
    return family, target, [_decode_image(v, f"images[{i}]") for i, v in enumerate(images)]


def classify(task: Dict[str, Any]) -> Dict[str, Any]:
    """AwsWafClassification: answer one already-fetched problem."""
    return run_classification(*parse_classification(task))


def run_classification(family: str, target: str, decoded: List[bytes]) -> Dict[str, Any]:
    """The vision call for a validated classification."""
    if family == "bls":
        return _run_bls(target, decoded)
    try:
        if family == "grid":
            objects = sorted({int(i) for i in vision_bridge.classify_grid(decoded, target)})
            if not objects or any(i < 0 or i >= len(decoded) for i in objects):
                raise SolverError(ERROR_CAPTCHA_UNSOLVABLE, f"classifier returned {objects}")
            return {"objects": objects}
        if family == "toycarcity":
            x, y = vision_bridge.solve_toycar(decoded[0])
            return {"box": [x, y]}
    except vision_bridge.invalid_input_types() as exc:
        raise SolverError(ERROR_INVALID_TASK_DATA, str(exc)) from exc
    except vision_bridge.unsupported_types() as exc:
        raise SolverError(ERROR_CAPTCHA_UNSOLVABLE, str(exc)) from exc
    except (TypeError, ValueError) as exc:  # malformed solver output
        raise SolverError(ERROR_CAPTCHA_UNSOLVABLE,
                          f"bad solver output ({exc.__class__.__name__})") from exc
    raise SolverError(ERROR_TASK_NOT_SUPPORTED, f"question family {family!r} is not supported")


def _run_bls(target: str, decoded: List[bytes]) -> Dict[str, Any]:
    """BLS grid: indices to click plus what the model read on every tile."""
    try:
        objects, readings = bls_bridge.match(decoded, target)
    except bls_bridge.BlsInvalidInput as exc:
        raise SolverError(ERROR_INVALID_TASK_DATA, str(exc)) from exc
    except bls_bridge.BlsUnavailable as exc:
        raise SolverError(ERROR_TASK_NOT_SUPPORTED, str(exc)) from exc
    if not objects:
        raise SolverError(ERROR_CAPTCHA_UNSOLVABLE, f"no tile shows {target}")
    return {"objects": objects, "readings": readings}


@dataclass(frozen=True)
class TextJob:
    """A validated ImageToTextTask."""
    module: str
    image: bytes
    numeric: int = text.NUMERIC_ANY
    case: Optional[bool] = None
    min_length: int = 0
    max_length: int = 0


def _opt_int(task: Dict[str, Any], name: str, low: int, high: int) -> int:
    value = task.get(name)
    if value is None:
        return 0
    if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
        raise SolverError(ERROR_INVALID_TASK_DATA,
                          f"{name} must be an integer from {low} to {high}")
    return value


def parse_text(task: Dict[str, Any], default_module: str = text.DEFAULT_MODULE) -> TextJob:
    """
    Validate an ImageToTextTask (CapSolver shape):
    {body: base64 | data: URL, module?, case?: bool, numeric?: 0-4, minLength?, maxLength?}.
    """
    try:
        module = text.resolve_module(_opt_str(task, "module", 64), default_module)
    except text.UnknownModule as exc:
        raise SolverError(ERROR_INVALID_TASK_DATA, str(exc)) from exc
    case = task.get("case")
    if case is not None and not isinstance(case, bool):
        raise SolverError(ERROR_INVALID_TASK_DATA, "case must be a boolean")
    numeric = _opt_int(task, "numeric", 0, max(text.NUMERIC_VALUES))
    min_length = _opt_int(task, "minLength", 0, MAX_TEXT_LENGTH)
    max_length = _opt_int(task, "maxLength", 0, MAX_TEXT_LENGTH)
    if min_length and max_length and min_length > max_length:
        raise SolverError(ERROR_INVALID_TASK_DATA, "minLength is greater than maxLength")
    if "body" not in task:
        raise SolverError(ERROR_INVALID_TASK_DATA, "body is required")
    return TextJob(module=module, image=_decode_image(task.get("body"), "body"),
                   numeric=numeric, case=case, min_length=min_length, max_length=max_length)


def run_text(job: TextJob) -> Dict[str, Any]:
    """The OCR call for a validated ImageToTextTask -> {"text", "confidence"}."""
    constraints = text.Constraints(numeric=job.numeric, case=job.case,
                                   min_length=job.min_length, max_length=job.max_length)
    try:
        reading = text.recognize(job.module, job.image, constraints)
        answer, confidence = text.apply_filters(reading, job.numeric, job.case,
                                                job.min_length, job.max_length)
    except text.InvalidImage as exc:
        raise SolverError(ERROR_INVALID_TASK_DATA, str(exc) or "body is not a readable image") \
            from exc
    except text.ModelUnavailable as exc:
        raise SolverError(ERROR_TASK_NOT_SUPPORTED, str(exc)) from exc
    except text.TextUnsolvable as exc:
        raise SolverError(ERROR_CAPTCHA_UNSOLVABLE, str(exc)) from exc
    except (TypeError, ValueError) as exc:  # malformed recogniser output
        raise SolverError(ERROR_CAPTCHA_UNSOLVABLE,
                          f"bad solver output ({exc.__class__.__name__})") from exc
    return {"text": answer, "confidence": round(confidence, 4)}


# ---------------------------------------------------------------------- GeeTest

GEETEST_FAMILIES = ("slide", "icon", "word", "space", "nine", "gobang")
GEETEST_MAX_HINTS = 8
GEETEST_MAX_HINT_TEXT = 64
GEETEST_MAX_TARGET = 64
GEETEST_MAX_SIDE = 2000          # px; GeeTest pictures are a few hundred pixels wide
NINE_CELLS = 9


@dataclass(frozen=True)
class GeeTestJob:
    """A validated GeeTestClassification; `module` is the family (billing may price it)."""
    module: str
    image: bytes
    piece: Optional[bytes] = None
    full_background: Optional[bytes] = None
    hints: Tuple[Any, ...] = ()
    target: Any = None           # nine: str (a word) or bytes (a picture)
    count: Optional[int] = None  # nine: how many cells to pick, when the widget says so
    y: Optional[int] = None      # slide: top of the piece image in background pixels


def _geetest_question(task: Dict[str, Any]) -> Tuple[str, Optional[str]]:
    question = _opt_str(task, "question", 16 + GEETEST_MAX_TARGET)
    if not question:
        raise SolverError(ERROR_INVALID_TASK_DATA, "question is required (geetest:<family>)")
    parts = question.split(":", 2)
    if len(parts) < 2 or parts[0] != "geetest":
        raise SolverError(ERROR_INVALID_TASK_DATA,
                          "question must look like geetest:slide, geetest:icon, geetest:word, "
                          "geetest:space, geetest:nine:<target> or geetest:gobang")
    family = parts[1]
    if family not in GEETEST_FAMILIES:
        raise SolverError(ERROR_TASK_NOT_SUPPORTED, f"GeeTest family {family!r} is not supported")
    target = parts[2].strip() if len(parts) == 3 else None
    if target and family != "nine":
        raise SolverError(ERROR_INVALID_TASK_DATA, f"geetest:{family} takes no target")
    return family, target or None


def _geetest_hints(task: Dict[str, Any], family: str) -> Tuple[Any, ...]:
    hints = task.get("hints")
    if hints is None or hints == []:
        return ()
    if family not in ("icon", "word", "space"):
        raise SolverError(ERROR_INVALID_TASK_DATA, f"geetest:{family} takes no hints")
    if not isinstance(hints, list) or len(hints) > GEETEST_MAX_HINTS:
        raise SolverError(ERROR_INVALID_TASK_DATA,
                          f"hints must be a list of at most {GEETEST_MAX_HINTS} items")
    if family == "icon":  # the small icons to click, in order (pictures)
        return tuple(_decode_image(h, f"hints[{i}]") for i, h in enumerate(hints))
    out = []
    for i, h in enumerate(hints):  # word / space: the characters or the question text
        if not isinstance(h, str) or not h.strip() or len(h) > GEETEST_MAX_HINT_TEXT:
            raise SolverError(ERROR_INVALID_TASK_DATA,
                              f"hints[{i}] must be a non-empty string of at most "
                              f"{GEETEST_MAX_HINT_TEXT} characters")
        if any(ord(c) < 32 or ord(c) == 127 for c in h):
            raise SolverError(ERROR_INVALID_TASK_DATA, f"hints[{i}] contains control characters")
        out.append(h.strip())
    return tuple(out)


def parse_geetest(task: Dict[str, Any]) -> GeeTestJob:
    """
    Validate a GeeTestClassification payload:
    {question: geetest:<family>[:<target>], images: [one base64 picture], piece?,
     fullBackground?, hints?, targetImage?}.
    """
    family, target = _geetest_question(task)
    images = task.get("images")
    if not isinstance(images, list) or len(images) != 1:
        raise SolverError(ERROR_INVALID_TASK_DATA,
                          "images[] must hold exactly one picture (the challenge image)")
    image = _decode_image(images[0], "images[0]")
    piece = full = None
    for name in ("piece", "fullBackground"):
        if task.get(name) not in (None, "") and family != "slide":
            raise SolverError(ERROR_INVALID_TASK_DATA, f"{name} is only for geetest:slide")
    if family == "slide":
        if task.get("piece") not in (None, ""):
            piece = _decode_image(task.get("piece"), "piece")
        if task.get("fullBackground") not in (None, ""):
            full = _decode_image(task.get("fullBackground"), "fullBackground")
    hints = _geetest_hints(task, family)
    target_image = task.get("targetImage")
    if target_image not in (None, ""):
        if family != "nine":
            raise SolverError(ERROR_INVALID_TASK_DATA, "targetImage is only for geetest:nine")
        if target:
            raise SolverError(ERROR_INVALID_TASK_DATA, "give the nine target as text or as targetImage, not both")
        target = _decode_image(target_image, "targetImage")
    if family == "nine" and not target:
        raise SolverError(ERROR_INVALID_TASK_DATA,
                          "geetest:nine needs a target (geetest:nine:<word> or targetImage)")
    y = None
    if task.get("y") is not None:
        if family != "slide":
            raise SolverError(ERROR_INVALID_TASK_DATA, "y is only for geetest:slide")
        y = _opt_int(task, "y", 0, GEETEST_MAX_SIDE)
    count = None
    if task.get("count") is not None:
        if family != "nine":
            raise SolverError(ERROR_INVALID_TASK_DATA, "count is only for geetest:nine")
        count = _opt_int(task, "count", 1, NINE_CELLS)
    return GeeTestJob(module=family, image=image, piece=piece, full_background=full,
                      hints=hints, target=target, count=count, y=y)


def _image_size(data: bytes, name: str) -> Tuple[int, int]:
    """Width/height from the image header (no full decode)."""
    from PIL import Image, UnidentifiedImageError
    try:
        with Image.open(io.BytesIO(data)) as img:
            width, height = img.size
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError) as exc:
        raise SolverError(ERROR_INVALID_TASK_DATA, f"{name} is not a readable image") from exc
    if not (1 <= width <= GEETEST_MAX_SIDE and 1 <= height <= GEETEST_MAX_SIDE):
        raise SolverError(ERROR_INVALID_TASK_DATA,
                          f"{name} must be at most {GEETEST_MAX_SIDE} px per side")
    return width, height


def run_geetest(job: GeeTestJob) -> Dict[str, Any]:
    """The model call for a validated GeeTestClassification."""
    width, height = _image_size(job.image, "images[0]")
    for data, name in ((job.piece, "piece"), (job.full_background, "fullBackground")):
        if data is not None:
            _image_size(data, name)
    try:
        if job.module == "slide":
            if job.y is not None and job.y >= height:
                raise SolverError(ERROR_INVALID_TASK_DATA, f"y={job.y} is below the {height} px background")
            x = geetest_bridge.slide_offset(job.image, job.piece, job.full_background, job.y)
            if not 0 <= x <= width:
                raise SolverError(ERROR_CAPTCHA_UNSOLVABLE, f"model returned x={x} outside the image")
            return {"x": x}
        if job.module in geetest_bridge.CLICK_KINDS:
            points = geetest_bridge.click_points(job.image, job.module, list(job.hints) or None)
            if not points:
                raise SolverError(ERROR_CAPTCHA_UNSOLVABLE, "model found nothing to click")
            if any(not (0 <= x <= width and 0 <= y <= height) for x, y in points):
                raise SolverError(ERROR_CAPTCHA_UNSOLVABLE, "model returned a point outside the image")
            return {"points": [[x, y] for x, y in points]}
        if job.module == "nine":
            cells = sorted(set(geetest_bridge.nine_grid(job.image, job.target, job.count)))
            if not cells or any(not 0 <= i < NINE_CELLS for i in cells):
                raise SolverError(ERROR_CAPTCHA_UNSOLVABLE, f"model returned cells {cells}")
            return {"objects": cells}
        if job.module == "gobang":
            src, dst = geetest_bridge.gobang(job.image)
            if src == dst:
                raise SolverError(ERROR_CAPTCHA_UNSOLVABLE, "model returned an empty move")
            return {"move": [list(src), list(dst)]}
    except geetest_bridge.GeeTestInvalidInput as exc:
        raise SolverError(ERROR_INVALID_TASK_DATA, str(exc)) from exc
    except geetest_bridge.GeeTestUnavailable as exc:
        raise SolverError(ERROR_TASK_NOT_SUPPORTED, str(exc)) from exc
    except geetest_bridge.GeeTestUnsolvable as exc:
        raise SolverError(ERROR_CAPTCHA_UNSOLVABLE, str(exc)) from exc
    raise SolverError(ERROR_TASK_NOT_SUPPORTED, f"GeeTest family {job.module!r} is not supported")


# ---------------------------------------------------------------------- reCAPTCHA v2

RECAPTCHA_MAX_QUESTION = 128
RECAPTCHA_MAX_SITE_KEY = 256
RECAPTCHA_MAX_SIDE = 1000        # px; reCAPTCHA pictures are 300 (3x3), 450 (4x4) or 100 (one tile)
RECAPTCHA_MIN_SIDE = 16


@dataclass(frozen=True)
class RecaptchaJob:
    """A validated ReCaptchaV2Classification."""
    image: bytes
    question: str
    size: Optional[int] = None        # 1 / 3 / 4 when the client knows the layout
    website_url: Optional[str] = None
    website_key: Optional[str] = None


def parse_recaptcha(task: Dict[str, Any]) -> RecaptchaJob:
    """
    Validate a ReCaptchaV2Classification payload (CapSolver shape):
    {image: base64 | data: URL, question: "/m/0k4j" | instruction text, size?: 1|3|4,
     websiteURL?, websiteKey?}.
    """
    if task.get("image") in (None, ""):
        raise SolverError(ERROR_INVALID_TASK_DATA, "image is required (the challenge picture, base64)")
    image = _decode_image(task.get("image"), "image")
    question = _opt_str(task, "question", RECAPTCHA_MAX_QUESTION)
    if not question:
        raise SolverError(ERROR_INVALID_TASK_DATA,
                          "question is required (a Freebase id such as /m/0k4j, or the object name)")
    size = None
    if task.get("size") is not None:
        size = task.get("size")
        if isinstance(size, bool) or not isinstance(size, int) or size not in recaptcha_bridge.SIZES:
            raise SolverError(ERROR_INVALID_TASK_DATA, "size must be 1, 3 or 4")
    return RecaptchaJob(image=image, question=question, size=size,
                        website_url=_http_url(task, "websiteURL"),
                        website_key=_opt_str(task, "websiteKey", RECAPTCHA_MAX_SITE_KEY))


def run_recaptcha(job: RecaptchaJob) -> Dict[str, Any]:
    """The model call for a validated ReCaptchaV2Classification."""
    from PIL import Image, UnidentifiedImageError
    try:
        with Image.open(io.BytesIO(job.image)) as img:
            width, height = img.size
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError) as exc:
        raise SolverError(ERROR_INVALID_TASK_DATA, "image is not a readable picture") from exc
    if not (RECAPTCHA_MIN_SIDE <= width <= RECAPTCHA_MAX_SIDE
            and RECAPTCHA_MIN_SIDE <= height <= RECAPTCHA_MAX_SIDE):
        raise SolverError(ERROR_INVALID_TASK_DATA,
                          f"image must be {RECAPTCHA_MIN_SIDE}-{RECAPTCHA_MAX_SIDE} px per side")
    try:
        return recaptcha_bridge.classify(job.image, job.question, job.size)
    except recaptcha_bridge.RecaptchaInvalidInput as exc:
        raise SolverError(ERROR_INVALID_TASK_DATA, str(exc)) from exc
    except recaptcha_bridge.RecaptchaUnavailable as exc:
        raise SolverError(ERROR_TASK_NOT_SUPPORTED, str(exc)) from exc
    except recaptcha_bridge.RecaptchaUnsupported as exc:
        # the client should load another challenge (reCAPTCHA's reload button)
        raise SolverError(ERROR_CAPTCHA_UNSOLVABLE, f"unsupported question: {exc}") from exc
    except recaptcha_bridge.RecaptchaUnsolvable as exc:
        raise SolverError(ERROR_CAPTCHA_UNSOLVABLE, str(exc)) from exc


def parse_sync_job(task: Dict[str, Any], default_text_module: str = text.DEFAULT_MODULE
                   ) -> Tuple[str, Callable[[], Dict[str, Any]]]:
    """Validate any synchronous task -> (task type, zero-argument call producing the solution)."""
    task_type = task.get("type")
    if task_type == TASK_TEXT:
        return TASK_TEXT, partial(run_text, parse_text(task, default_text_module))
    if task_type == TASK_GEETEST:
        return TASK_GEETEST, partial(run_geetest, parse_geetest(task))
    if task_type == TASK_RECAPTCHA:
        return TASK_RECAPTCHA, partial(run_recaptcha, parse_recaptcha(task))
    job = parse_classification(task)
    return (TASK_BLS if task_type == TASK_BLS else TASK_CLASSIFY), partial(run_classification,
                                                                         *job)


class TaskManager:
    """
    Owns the worker pools and the task registry.

    Every task carries an absolute monotonic deadline (created + task_timeout): queue
    time counts against it, a task that is still queued at its deadline is failed without
    running, and the solver receives the same deadline. The registry is guarded by one
    lock; status transitions happen only under it, and a finished task never changes.
    """

    def __init__(self, settings: Settings, usage: UsageStore,
                 solver: Callable = solve_website, billing=None):
        self.settings = settings
        self.usage = usage
        self.billing = billing  # aws_capsolver.billing.Billing or None (no token accounting)
        self._solver = solver
        self._tasks: Dict[str, Task] = {}
        self._lock = threading.Lock()
        self._cancel = threading.Event()
        self._pool = ThreadPoolExecutor(max_workers=settings.workers,
                                        thread_name_prefix="solver")
        # classification is synchronous for the client; it gets its own bounded executor
        # (vision_concurrency runners) and an admission cap on runners + waiting jobs, so
        # an overload is answered at once with ERROR_NO_SLOT_AVAILABLE instead of timing out
        self._classify_pool = ThreadPoolExecutor(max_workers=settings.vision_concurrency,
                                                 thread_name_prefix="classify")
        self._classify_waiting = 0
        self._classify_running = 0
        self._solver_settings = SolverSettings(
            http_timeout=settings.http_timeout, pow_processes=settings.pow_processes,
            captcha_rounds=settings.captcha_rounds, verify_token=settings.verify_token,
            allow_private=settings.allow_private, cancel=self._cancel)

    @property
    def closed(self) -> bool:
        return self._cancel.is_set()

    # registry ---------------------------------------------------------------
    def _register(self, client_key: str, task_type: str, module: Optional[str] = None) -> Task:
        task = Task(id=str(uuid.uuid4()), client_key=client_key, type=task_type)
        if self.billing is not None:
            # reserve the price up front: concurrent tasks can never overdraw a key
            try:
                self.billing.charge(client_key, task_type, task.id, module)
            except InsufficientTokens as exc:
                raise SolverError(ERROR_ZERO_BALANCE, str(exc)) from exc
            except BillingError as exc:
                logger.error("billing failed for %s: %s", task_type, exc)
                raise SolverError(ERROR_SERVICE_UNAVAILABLE, "billing is unavailable") from exc
        with self._lock:
            closed = self.closed
            timed_out = [] if closed else self._cleanup_locked()
            active = sum(1 for t in self._tasks.values() if t.status in ("idle", "processing"))
            accepted = not closed and active < self.settings.max_pending
            if accepted:
                self._tasks[task.id] = task
        self._count_failed(timed_out)
        if not accepted:
            self._refund(task)
            if closed:
                raise SolverError(ERROR_SERVICE_UNAVAILABLE, "service is shutting down")
            raise SolverError(ERROR_NO_SLOT_AVAILABLE, "too many tasks in progress")
        self.usage.bump(client_key, task_type, "created")
        return task

    def _count_failed(self, tasks) -> None:
        for task in tasks:
            self.usage.bump(task.client_key, task.type, "failed")
            self._refund(task)

    def _refund(self, task: Task) -> None:
        """Failed tasks cost nothing: give the reserved tokens back (idempotent)."""
        if self.billing is None:
            return
        try:
            self.billing.refund(task.id)
        except BillingError as exc:
            logger.error("refund of task %s failed: %s", task.id, exc)

    def _deadline(self, task: Task) -> float:
        return task.created + self.settings.task_timeout

    def _cleanup_locked(self) -> list:
        """Run the watchdog over every task and forget expired ones; returns timed-out tasks."""
        now = time.monotonic()
        timed_out = [t for t in self._tasks.values() if self._watchdog_locked(t, now)]
        expired = [tid for tid, t in self._tasks.items()
                   if t.finished is not None and now - t.finished > self.settings.task_ttl]
        for tid in expired:
            del self._tasks[tid]
        return timed_out

    def _watchdog_locked(self, task: Task, now: float) -> bool:
        """Fail a task stuck past its deadline (+ grace); its late result is then ignored."""
        if task.status in ("idle", "processing") and now > self._deadline(task) + WATCHDOG_GRACE:
            self._fail_locked(task, ERROR_TASK_TIMEOUT, "task exceeded the time limit", now)
            return True
        return False

    @staticmethod
    def _fail_locked(task: Task, code: str, description: str, now: float) -> None:
        task.error_code, task.error_description, task.finished = code, description, now
        task.status = "failed"

    def _snapshot(self, task: Task) -> Task:
        """A consistent copy for the API layer (the live object keeps changing)."""
        with self._lock:
            return replace(task)

    def cleanup(self) -> None:
        with self._lock:
            timed_out = self._cleanup_locked()
        self._count_failed(timed_out)

    def _start(self, task: Task) -> bool:
        """idle -> processing; False when the task already finished (watchdog, shutdown)."""
        with self._lock:
            if task.status != "idle":
                return False
            task.status = "processing"
            return True

    def _finish(self, task: Task, solution: Optional[dict] = None,
                error: Optional[SolverError] = None) -> bool:
        with self._lock:
            if task.status in ("ready", "failed"):
                return False  # already failed by the watchdog or the shutdown
            task.finished = time.monotonic()
            if error is None:
                task.solution = solution
                task.status = "ready"
            else:
                task.error_code, task.error_description = error.code, error.description
                task.status = "failed"
        self.usage.bump(task.client_key, task.type, "solved" if error is None else "failed")
        if error is not None:
            self._refund(task)
        return True

    # task types -------------------------------------------------------------
    def create(self, client_key: str, payload: Any) -> Task:
        if self.closed:
            raise SolverError(ERROR_SERVICE_UNAVAILABLE, "service is shutting down")
        if not isinstance(payload, dict):
            raise SolverError(ERROR_INVALID_TASK_DATA, "task must be an object")
        task_type = payload.get("type")
        if not isinstance(task_type, str) or task_type not in SUPPORTED_TYPES:
            shown = task_type if isinstance(task_type, str) else type(task_type).__name__
            raise SolverError(ERROR_TASK_NOT_SUPPORTED, f"task type {shown!r} is not supported")

        if task_type in CLASSIFY_TYPES:
            return self._create_classification(client_key, payload)

        request = build_solve_request(payload)  # validate before queueing
        task = self._register(client_key, task_type)
        try:
            self._pool.submit(self._run_waf, task, request)
        except RuntimeError as exc:  # the pool was shut down after the closed check
            self._finish(task, error=SolverError(ERROR_SERVICE_UNAVAILABLE,
                                                 "service is shutting down"))
            raise SolverError(ERROR_SERVICE_UNAVAILABLE, "service is shutting down") from exc
        return self._snapshot(task)

    def _create_classification(self, client_key: str, payload: Dict[str, Any]) -> Task:
        """Blocking variant (CLI, tests); the API awaits the future without a thread."""
        task, future = self.submit_classification(client_key, payload)
        if future is not None:
            try:
                future.result(timeout=self.settings.classify_timeout)
            except FutureTimeout:
                pass
            except BaseException:  # reported by finish_classification
                pass
        return self.finish_classification(task, future)

    def submit_classification(self, client_key: str, payload: Any
                              ) -> Tuple[Task, Optional[Future]]:
        """
        Admit and queue one synchronous task (classification or ImageToTextTask). Raises
        ERROR_NO_SLOT_AVAILABLE when the runners and the queue are full. Returns the task
        and the future of the model call (None when it could not be queued; the task is then already finished).
        """
        if self.closed:
            raise SolverError(ERROR_SERVICE_UNAVAILABLE, "service is shutting down")
        if not isinstance(payload, dict):
            raise SolverError(ERROR_INVALID_TASK_DATA, "task must be an object")
        # bad input never takes a queue slot
        task_type, job = parse_sync_job(payload, self.settings.text_default_module)
        capacity = self.settings.vision_concurrency + self.settings.classify_queue_limit
        with self._lock:
            if self._classify_waiting + self._classify_running >= capacity:
                raise SolverError(ERROR_NO_SLOT_AVAILABLE, "classification queue is full")
            self._classify_waiting += 1  # reserve the slot before registering
        module = getattr(getattr(job, "args", (None,))[0], "module", None) \
            if getattr(job, "args", None) else None
        try:
            task = self._register(client_key, task_type, module if isinstance(module, str) else None)
        except BaseException:
            with self._lock:
                self._classify_waiting -= 1
            raise
        self._start(task)
        deadline = task.created + self.settings.classify_timeout
        try:
            future = self._classify_pool.submit(self._run_classification, task, job,
                                                deadline)
        except RuntimeError:
            with self._lock:
                self._classify_waiting -= 1
            self._finish(task, error=SolverError(ERROR_SERVICE_UNAVAILABLE,
                                                 "service is shutting down"))
            return self._snapshot(task), None
        future.add_done_callback(self._release_if_cancelled)
        return task, future

    def _release_if_cancelled(self, future: Future) -> None:
        if future.cancelled():  # never started: give its queue slot back
            with self._lock:
                self._classify_waiting -= 1

    def _run_classification(self, task: Task, job: Callable[[], Dict[str, Any]],
                            deadline: float) -> Dict[str, Any]:
        with self._lock:
            self._classify_waiting -= 1
            stale = time.monotonic() >= deadline or task.status != "processing"
            if not stale:
                self._classify_running += 1
        if stale:  # the client already got its timeout: do not spend CPU on it
            raise SolverError(ERROR_TASK_TIMEOUT, "classification waited too long in the queue")
        try:
            return job()
        finally:
            with self._lock:
                self._classify_running -= 1

    def finish_classification(self, task: Task, future: Optional[Future]) -> Task:
        """Record the outcome of `future` (timed out when it is not done yet)."""
        if future is None:
            return self._snapshot(task)
        if not future.done():
            future.cancel()  # drop it if it never started; a running call cannot be stopped
            self._finish(task, error=SolverError(ERROR_TASK_TIMEOUT, "classification timed out"))
            return self._snapshot(task)
        if future.cancelled():
            self._finish(task, error=SolverError(ERROR_SERVICE_UNAVAILABLE,
                                                 "classification was cancelled"))
            return self._snapshot(task)
        exc = future.exception()
        if exc is None:
            self._finish(task, solution=future.result())
        elif isinstance(exc, SolverError):
            self._finish(task, error=exc)
        else:  # a vision bug must not leave the task processing forever
            logger.error("classification task %s crashed: %s", task.id,
                         redact("".join(traceback.format_exception(exc))))
            self._finish(task, error=SolverError(
                ERROR_CAPTCHA_UNSOLVABLE, f"internal error: {exc.__class__.__name__}"))
        return self._snapshot(task)

    def stats(self) -> Dict[str, int]:
        """Load figures for /health."""
        with self._lock:
            active = [t for t in self._tasks.values() if t.status in ("idle", "processing")]
            return {
                "tasks_active": len(active),
                "waf_queued": sum(1 for t in active if t.type not in CLASSIFY_TYPES
                                  and t.status == "idle"),
                "waf_running": sum(1 for t in active if t.type not in CLASSIFY_TYPES
                                   and t.status == "processing"),
                "classify_queued": self._classify_waiting,
                "classify_running": self._classify_running,
                "classify_capacity": (self.settings.vision_concurrency
                                      + self.settings.classify_queue_limit),
            }

    def _run_waf(self, task: Task, request: SolveRequest) -> None:
        deadline = self._deadline(task)
        if time.monotonic() >= deadline:
            self._finish(task, error=SolverError(ERROR_TASK_TIMEOUT,
                                                 "task waited too long in the queue"))
            return
        if not self._start(task):
            return
        started = time.monotonic()
        try:
            outcome = self._solver(request, self._solver_settings, deadline=deadline)
        except SolverError as exc:
            code = exc.code if exc.code in ALL_CODES else ERROR_CAPTCHA_UNSOLVABLE
            logger.info("task %s failed in %.1fs: %s %s", task.id, time.monotonic() - started,
                        code, redact(exc.description))
            self._finish(task, error=SolverError(code, exc.description))
            return
        except Exception as exc:  # last-resort guard so a worker bug never hangs a task
            logger.error("task %s crashed: %s", task.id,
                         redact("".join(traceback.format_exception(exc))))
            self._finish(task, error=SolverError(ERROR_CAPTCHA_UNSOLVABLE,
                                                 f"internal error: {exc.__class__.__name__}"))
            return
        if outcome is None or not (outcome.token or outcome.voucher):
            self._finish(task, error=SolverError(ERROR_CAPTCHA_UNSOLVABLE, "no token produced"))
            return
        solution: Dict[str, Any] = {"cookie": outcome.token or "",
                                    "userAgent": outcome.user_agent}
        if outcome.voucher:
            solution["captchaVoucher"] = outcome.voucher
        logger.info("task %s ready in %.1fs (captcha=%s rounds=%d)", task.id,
                    time.monotonic() - started, outcome.used_captcha, outcome.captcha_rounds)
        self._finish(task, solution=solution)

    def get(self, client_key: str, task_id: Any) -> Task:
        if not isinstance(task_id, str) or not task_id:
            raise SolverError(ERROR_TASKID_INVALID, "taskId must be a non-empty string")
        with self._lock:
            task = self._tasks.get(task_id)
            # same answer for "unknown" and "owned by another key": nothing leaks
            if task is None or not hmac.compare_digest(task.client_key.encode(),
                                                       client_key.encode()):
                raise SolverError(ERROR_TASKID_INVALID, "task not found or expired")
            timed_out = self._watchdog_locked(task, time.monotonic())
            snapshot = replace(task)
        if timed_out:
            self._count_failed([task])
        return snapshot

    def shutdown(self, wait: bool = False) -> None:
        """Stop accepting tasks, abort running solves at their next check and fail the rest."""
        failed = []
        with self._lock:
            self._cancel.set()
            now = time.monotonic()
            for task in self._tasks.values():
                if task.status in ("idle", "processing"):
                    self._fail_locked(task, ERROR_SERVICE_UNAVAILABLE,
                                      "service is shutting down", now)
                    failed.append(task)
        for task in failed:
            self._refund(task)
        self._pool.shutdown(wait=wait, cancel_futures=True)
        self._classify_pool.shutdown(wait=wait, cancel_futures=True)
