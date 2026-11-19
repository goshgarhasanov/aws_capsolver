"""Task validation, the bounded worker pool and the in-memory task registry."""

import base64
import binascii
import hmac
import logging
import threading
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Dict, Optional
from urllib.parse import urlparse

from . import vision_bridge
from .config import Settings, UsageStore
from .waf.errors import (ALL_CODES, ERROR_CAPTCHA_UNSOLVABLE, ERROR_INVALID_TASK_DATA,
                         ERROR_NO_SLOT_AVAILABLE, ERROR_SERVICE_UNAVAILABLE,
                         ERROR_TASK_NOT_SUPPORTED, ERROR_TASK_TIMEOUT, ERROR_TASKID_INVALID,
                         SolverError)
from .waf.http import parse_proxy, redact
from .waf.solver import SolveRequest, SolverSettings, solve_website

logger = logging.getLogger(__name__)

TASK_WAF = "AntiAwsWafTask"
TASK_WAF_PROXYLESS = "AntiAwsWafTaskProxyLess"
TASK_CLASSIFY = "AwsWafClassification"
SUPPORTED_TYPES = (TASK_WAF, TASK_WAF_PROXYLESS, TASK_CLASSIFY)
MAX_IMAGES = 16
MAX_IMAGE_BYTES = 2 * 1024 * 1024
MAX_URL_LENGTH = 4096
MAX_FIELD_LENGTH = 16 * 1024      # gokuProps context and similar opaque blobs
MAX_USER_AGENT_LENGTH = 512
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


def _decode_image(value: Any, index: int) -> bytes:
    if not isinstance(value, str) or not value:
        raise SolverError(ERROR_INVALID_TASK_DATA, f"images[{index}] must be a base64 string")
    if value.startswith("data:"):
        value = value.split(",", 1)[-1]
    if len(value) > MAX_IMAGE_BYTES * 4 // 3 + 8:
        raise SolverError(ERROR_INVALID_TASK_DATA, f"images[{index}] has a bad size")
    try:
        data = base64.b64decode(value, validate=False)
    except (binascii.Error, ValueError) as exc:
        raise SolverError(ERROR_INVALID_TASK_DATA, f"images[{index}] is not base64") from exc
    if not data or len(data) > MAX_IMAGE_BYTES:
        raise SolverError(ERROR_INVALID_TASK_DATA, f"images[{index}] has a bad size")
    return data


def classify(task: Dict[str, Any]) -> Dict[str, Any]:
    """AwsWafClassification: answer one already-fetched problem."""
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
    decoded = [_decode_image(v, i) for i, v in enumerate(images)]
    try:
        if family == "grid":
            if not target:
                raise SolverError(ERROR_INVALID_TASK_DATA, "grid question without a target")
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


class TaskManager:
    """
    Owns the worker pools and the task registry.

    Every task carries an absolute monotonic deadline (created + task_timeout): queue
    time counts against it, a task that is still queued at its deadline is failed without
    running, and the solver receives the same deadline. The registry is guarded by one
    lock; status transitions happen only under it, and a finished task never changes.
    """

    def __init__(self, settings: Settings, usage: UsageStore,
                 solver: Callable = solve_website):
        self.settings = settings
        self.usage = usage
        self._solver = solver
        self._tasks: Dict[str, Task] = {}
        self._lock = threading.Lock()
        self._cancel = threading.Event()
        self._pool = ThreadPoolExecutor(max_workers=settings.workers,
                                        thread_name_prefix="solver")
        # classification is synchronous for the client; keep it off the WAF queue
        self._classify_pool = ThreadPoolExecutor(max_workers=max(2, settings.workers // 2),
                                                 thread_name_prefix="classify")
        self._solver_settings = SolverSettings(
            http_timeout=settings.http_timeout, pow_processes=settings.pow_processes,
            captcha_rounds=settings.captcha_rounds, verify_token=settings.verify_token,
            allow_private=settings.allow_private, cancel=self._cancel)

    @property
    def closed(self) -> bool:
        return self._cancel.is_set()

    # registry ---------------------------------------------------------------
    def _register(self, client_key: str, task_type: str) -> Task:
        task = Task(id=str(uuid.uuid4()), client_key=client_key, type=task_type)
        with self._lock:
            if self.closed:
                raise SolverError(ERROR_SERVICE_UNAVAILABLE, "service is shutting down")
            timed_out = self._cleanup_locked()
            active = sum(1 for t in self._tasks.values() if t.status in ("idle", "processing"))
            accepted = active < self.settings.max_pending
            if accepted:
                self._tasks[task.id] = task
        self._count_failed(timed_out)
        if not accepted:
            raise SolverError(ERROR_NO_SLOT_AVAILABLE, "too many tasks in progress")
        self.usage.bump(client_key, task_type, "created")
        return task

    def _count_failed(self, tasks) -> None:
        for task in tasks:
            self.usage.bump(task.client_key, task.type, "failed")

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

        if task_type == TASK_CLASSIFY:
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
        task = self._register(client_key, TASK_CLASSIFY)
        self._start(task)
        try:
            future = self._classify_pool.submit(classify, payload)
        except RuntimeError:
            self._finish(task, error=SolverError(ERROR_SERVICE_UNAVAILABLE,
                                                 "service is shutting down"))
            return self._snapshot(task)
        try:
            solution = future.result(timeout=self.settings.classify_timeout)
        except SolverError as exc:
            self._finish(task, error=exc)
        except FutureTimeout:
            future.cancel()  # drop it if it never started; a running call cannot be stopped
            self._finish(task, error=SolverError(ERROR_TASK_TIMEOUT, "classification timed out"))
        except Exception as exc:  # a vision bug must not leave the task processing forever
            logger.error("classification task %s crashed: %s", task.id,
                         redact("".join(traceback.format_exception(exc))))
            self._finish(task, error=SolverError(
                ERROR_CAPTCHA_UNSOLVABLE, f"internal error: {exc.__class__.__name__}"))
        else:
            self._finish(task, solution=solution)
        return self._snapshot(task)

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
        with self._lock:
            self._cancel.set()
            now = time.monotonic()
            for task in self._tasks.values():
                if task.status in ("idle", "processing"):
                    self._fail_locked(task, ERROR_SERVICE_UNAVAILABLE,
                                      "service is shutting down", now)
        self._pool.shutdown(wait=wait, cancel_futures=True)
        self._classify_pool.shutdown(wait=wait, cancel_futures=True)
