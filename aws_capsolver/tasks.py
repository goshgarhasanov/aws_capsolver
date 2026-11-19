"""Task validation, the bounded worker pool and the in-memory task registry."""

import base64
import binascii
import logging
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional

from . import vision_bridge
from .config import Settings, UsageStore
from .waf.errors import (ERROR_CAPTCHA_UNSOLVABLE, ERROR_INVALID_TASK_DATA,
                         ERROR_NO_SLOT_AVAILABLE, ERROR_SERVICE_UNAVAILABLE,
                         ERROR_TASK_NOT_SUPPORTED, ERROR_TASK_TIMEOUT, ERROR_TASKID_INVALID,
                         SolverError)
from .waf.http import parse_proxy
from .waf.solver import SolveRequest, SolverSettings, solve_website

logger = logging.getLogger(__name__)

TASK_WAF = "AntiAwsWafTask"
TASK_WAF_PROXYLESS = "AntiAwsWafTaskProxyLess"
TASK_CLASSIFY = "AwsWafClassification"
SUPPORTED_TYPES = (TASK_WAF, TASK_WAF_PROXYLESS, TASK_CLASSIFY)
MAX_IMAGES = 16
MAX_IMAGE_BYTES = 2 * 1024 * 1024


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


def _opt_str(task: Dict[str, Any], name: str) -> Optional[str]:
    value = task.get(name)
    if value is None or value == "":
        return None
    if not isinstance(value, str):
        raise SolverError(ERROR_INVALID_TASK_DATA, f"{name} must be a string")
    return value


def build_solve_request(task: Dict[str, Any]) -> SolveRequest:
    """Validate an AntiAwsWafTask[ProxyLess] payload."""
    task_type = task.get("type")
    website_url = _opt_str(task, "websiteURL")
    if not website_url:
        raise SolverError(ERROR_INVALID_TASK_DATA, "websiteURL is required")
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
        user_agent=_opt_str(task, "userAgent"),
        aws_key=_opt_str(task, "awsKey"),
        aws_iv=_opt_str(task, "awsIv"),
        aws_context=_opt_str(task, "awsContext"),
        challenge_js=_opt_str(task, "awsChallengeJS"),
        api_js=_opt_str(task, "awsApiJs"),
        problem_url=_opt_str(task, "awsProblemUrl"),
        api_key=_opt_str(task, "awsApiKey"),
        locale=_opt_str(task, "locale") or "en-us",
    )


def _decode_image(value: Any, index: int) -> bytes:
    if not isinstance(value, str) or not value:
        raise SolverError(ERROR_INVALID_TASK_DATA, f"images[{index}] must be a base64 string")
    if value.startswith("data:"):
        value = value.split(",", 1)[-1]
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
    decoded = [_decode_image(v, i) for i, v in enumerate(images)]
    try:
        if family == "grid":
            if not target:
                raise SolverError(ERROR_INVALID_TASK_DATA, "grid question without a target")
            objects = sorted({int(i) for i in vision_bridge.classify_grid(decoded, target)})
            return {"objects": objects}
        if family == "toycarcity":
            x, y = vision_bridge.solve_toycar(decoded[0])
            return {"box": [x, y]}
    except vision_bridge.invalid_input_types() as exc:
        raise SolverError(ERROR_INVALID_TASK_DATA, str(exc)) from exc
    except vision_bridge.unsupported_types() as exc:
        raise SolverError(ERROR_CAPTCHA_UNSOLVABLE, str(exc)) from exc
    raise SolverError(ERROR_TASK_NOT_SUPPORTED, f"question family {family!r} is not supported")


class TaskManager:
    """Owns the worker pool and the task registry."""

    def __init__(self, settings: Settings, usage: UsageStore,
                 solver: Callable = solve_website):
        self.settings = settings
        self.usage = usage
        self._solver = solver
        self._tasks: Dict[str, Task] = {}
        self._lock = threading.Lock()
        self._pool = ThreadPoolExecutor(max_workers=settings.workers,
                                        thread_name_prefix="solver")
        # classification is synchronous for the client; keep it off the WAF queue
        self._classify_pool = ThreadPoolExecutor(max_workers=max(2, settings.workers // 2),
                                                 thread_name_prefix="classify")
        self._solver_settings = SolverSettings(
            http_timeout=settings.http_timeout, pow_processes=settings.pow_processes,
            captcha_rounds=settings.captcha_rounds, verify_token=settings.verify_token)
        self._closed = False

    # registry ---------------------------------------------------------------
    def _register(self, client_key: str, task_type: str) -> Task:
        task = Task(id=str(uuid.uuid4()), client_key=client_key, type=task_type)
        with self._lock:
            self._cleanup_locked()
            active = sum(1 for t in self._tasks.values() if t.status in ("idle", "processing"))
            if active >= self.settings.max_pending:
                raise SolverError(ERROR_NO_SLOT_AVAILABLE, "too many tasks in progress")
            self._tasks[task.id] = task
        self.usage.bump(client_key, task_type, "created")
        return task

    def _cleanup_locked(self) -> None:
        now = time.monotonic()
        expired = [tid for tid, t in self._tasks.items()
                   if (t.finished is not None and now - t.finished > self.settings.task_ttl)
                   or now - t.created > self.settings.task_ttl + self.settings.task_timeout]
        for tid in expired:
            del self._tasks[tid]

    def cleanup(self) -> None:
        with self._lock:
            self._cleanup_locked()

    def _finish(self, task: Task, solution: Optional[dict] = None,
                error: Optional[SolverError] = None) -> None:
        with self._lock:
            if task.status in ("ready", "failed"):
                return  # already timed out by the watchdog
            if error is None:
                task.status, task.solution = "ready", solution
            else:
                task.status, task.error_code, task.error_description = (
                    "failed", error.code, error.description)
            task.finished = time.monotonic()
        self.usage.bump(task.client_key, task.type, "solved" if error is None else "failed")

    # task types -------------------------------------------------------------
    def create(self, client_key: str, payload: Any) -> Task:
        if self._closed:
            raise SolverError(ERROR_SERVICE_UNAVAILABLE, "service is shutting down")
        if not isinstance(payload, dict):
            raise SolverError(ERROR_INVALID_TASK_DATA, "task must be an object")
        task_type = payload.get("type")
        if task_type not in SUPPORTED_TYPES:
            raise SolverError(ERROR_TASK_NOT_SUPPORTED, f"task type {task_type!r} is not supported")

        if task_type == TASK_CLASSIFY:
            task = self._register(client_key, task_type)
            task.status = "processing"
            future = self._classify_pool.submit(classify, payload)
            try:
                solution = future.result(timeout=self.settings.classify_timeout)
            except SolverError as exc:
                self._finish(task, error=exc)
                return task
            except FutureTimeout:
                self._finish(task, error=SolverError(ERROR_TASK_TIMEOUT, "classification timed out"))
                return task
            self._finish(task, solution=solution)
            return task

        request = build_solve_request(payload)  # validate before queueing
        task = self._register(client_key, task_type)
        self._pool.submit(self._run_waf, task, request)
        return task

    def _run_waf(self, task: Task, request: SolveRequest) -> None:
        task.status = "processing"
        started = time.monotonic()
        deadline = started + self.settings.task_timeout
        try:
            outcome = self._solver(request, self._solver_settings, deadline=deadline)
        except SolverError as exc:
            code = exc.code if exc.code.startswith("ERROR_") else ERROR_CAPTCHA_UNSOLVABLE
            logger.info("task %s failed in %.1fs: %s %s", task.id, time.monotonic() - started,
                        code, exc.description)
            self._finish(task, error=SolverError(code, exc.description))
            return
        except Exception as exc:  # last-resort guard so a worker bug never hangs a task
            logger.exception("task %s crashed", task.id)
            self._finish(task, error=SolverError(ERROR_CAPTCHA_UNSOLVABLE,
                                                 f"internal error: {exc.__class__.__name__}"))
            return
        solution = {"cookie": outcome.token, "userAgent": outcome.user_agent}
        if outcome.voucher:
            solution["captchaVoucher"] = outcome.voucher
        logger.info("task %s ready in %.1fs (captcha=%s rounds=%d)", task.id,
                    time.monotonic() - started, outcome.used_captcha, outcome.captcha_rounds)
        self._finish(task, solution=solution)

    def get(self, client_key: str, task_id: Any) -> Task:
        if not isinstance(task_id, str):
            raise SolverError(ERROR_TASKID_INVALID, "taskId must be a string")
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None or task.client_key != client_key:
                raise SolverError(ERROR_TASKID_INVALID, "task not found or expired")
            # watchdog: a task stuck beyond the hard limit is reported as timed out
            if task.status in ("idle", "processing") and \
                    time.monotonic() - task.created > self.settings.task_timeout + 15:
                task.status, task.error_code = "failed", ERROR_TASK_TIMEOUT
                task.error_description = "task exceeded the time limit"
                task.finished = time.monotonic()
            return task

    def shutdown(self) -> None:
        self._closed = True
        self._pool.shutdown(wait=False, cancel_futures=True)
        self._classify_pool.shutdown(wait=False, cancel_futures=True)
