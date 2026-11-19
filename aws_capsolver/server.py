"""CapSolver-compatible HTTP API (createTask / getTaskResult / getBalance)."""

import json
import logging
import threading
import time
from contextlib import asynccontextmanager
from typing import Any, Dict, Optional

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

from . import __version__, vision_bridge
from .config import KeyStore, Settings, UsageStore
from .tasks import Task, TaskManager
from .waf import pow as powlib
from .waf.errors import (ERROR_INVALID_TASK_DATA, ERROR_KEY_DENIED_ACCESS,
                         ERROR_SERVICE_UNAVAILABLE, SolverError)

logger = logging.getLogger(__name__)

JANITOR_INTERVAL = 30.0


def _error(code: str, description: str, status_code: int = 200) -> JSONResponse:
    # CapSolver answers errors with HTTP 200 and errorId=1
    return JSONResponse({"errorId": 1, "errorCode": code, "errorDescription": description},
                        status_code=status_code)


async def _read_body(request: Request, limit: int) -> bytes:
    """The raw body, refusing more than `limit` bytes without buffering the excess."""
    declared = request.headers.get("content-length")
    if declared is not None:
        try:
            if int(declared) > limit:
                raise SolverError(ERROR_INVALID_TASK_DATA, f"body larger than {limit} bytes")
        except ValueError as exc:
            raise SolverError(ERROR_INVALID_TASK_DATA, "bad Content-Length") from exc
    chunks, size = [], 0
    async for chunk in request.stream():
        size += len(chunk)
        if size > limit:
            raise SolverError(ERROR_INVALID_TASK_DATA, f"body larger than {limit} bytes")
        chunks.append(chunk)
    return b"".join(chunks)


async def _body(request: Request, limit: int) -> Dict[str, Any]:
    raw = await _read_body(request, limit)
    try:
        data = json.loads(raw)
    except (ValueError, RecursionError) as exc:  # bad JSON / bad UTF-8 / absurd nesting
        raise SolverError(ERROR_INVALID_TASK_DATA, "body must be JSON") from exc
    if not isinstance(data, dict):
        raise SolverError(ERROR_INVALID_TASK_DATA, "body must be a JSON object")
    return data


def _task_json(task: Task) -> Dict[str, Any]:
    if task.status == "failed":
        return {"errorId": 1, "errorCode": task.error_code,
                "errorDescription": task.error_description, "taskId": task.id,
                "status": "failed"}
    # CapSolver clients poll until ready/failed; "idle" (queued) is reported as processing
    status = "processing" if task.status == "idle" else task.status
    out: Dict[str, Any] = {"errorId": 0, "taskId": task.id, "status": status}
    if task.status == "ready":
        out["solution"] = task.solution
    return out


def create_app(settings: Optional[Settings] = None, manager: Optional[TaskManager] = None,
               keys: Optional[KeyStore] = None, usage: Optional[UsageStore] = None) -> FastAPI:
    settings = settings or Settings.from_env()
    owns_usage = usage is None
    keys = keys or KeyStore(settings)
    usage = usage or UsageStore(settings.usage_db)
    manager = manager or TaskManager(settings, usage)
    stop = threading.Event()

    def janitor() -> None:
        while not stop.wait(JANITOR_INTERVAL):
            try:
                manager.cleanup()
            except Exception:  # the janitor must survive anything
                logger.exception("task cleanup failed")

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        janitor_thread = threading.Thread(target=janitor, name="task-janitor", daemon=True)
        janitor_thread.start()
        if settings.warmup:
            threading.Thread(target=vision_bridge.warmup, name="vision-warmup",
                             daemon=True).start()
        logger.info("aws_capsolver %s ready (workers=%d, vision=%s, ssrf_guard=%s)",
                    __version__, settings.workers, vision_bridge.available(),
                    not settings.allow_private)
        try:
            yield
        finally:
            stop.set()
            manager.shutdown()
            powlib.shutdown_pool(wait=False)
            janitor_thread.join(timeout=1)
            if owns_usage:
                usage.close()

    app = FastAPI(title="aws_capsolver", version=__version__, lifespan=lifespan,
                  docs_url=None, redoc_url=None, openapi_url=None)
    app.state.manager, app.state.keys, app.state.usage = manager, keys, usage

    @app.exception_handler(SolverError)
    async def _solver_error(_: Request, exc: SolverError) -> JSONResponse:
        return _error(exc.code, exc.description)

    @app.exception_handler(Exception)
    async def _unexpected(_: Request, exc: Exception) -> JSONResponse:
        logger.error("unhandled %s in request", exc.__class__.__name__, exc_info=exc)
        return _error(ERROR_SERVICE_UNAVAILABLE, "internal error", status_code=500)

    async def _authed_body(request: Request):
        body = await _body(request, settings.max_body_bytes)
        client_key = body.get("clientKey")
        if client_key is None or client_key == "":
            raise SolverError(ERROR_KEY_DENIED_ACCESS, "clientKey is required")
        # inline on purpose: lookup is cheap, and the threadpool may be busy with
        # synchronous classifications - authentication must never queue behind them
        if keys.lookup(client_key) is None:
            raise SolverError(ERROR_KEY_DENIED_ACCESS, "invalid clientKey")
        return body, client_key

    @app.post("/createTask")
    async def create_task(request: Request) -> JSONResponse:
        body, client_key = await _authed_body(request)
        if "task" not in body:
            raise SolverError(ERROR_INVALID_TASK_DATA, "task is required")
        # task creation may block (synchronous classification) - keep it off the event loop
        task = await run_in_threadpool(manager.create, client_key, body.get("task"))
        if task.status in ("ready", "failed"):
            return JSONResponse(_task_json(task))
        return JSONResponse({"errorId": 0, "taskId": task.id})

    @app.post("/getTaskResult")
    async def get_task_result(request: Request) -> JSONResponse:
        body, client_key = await _authed_body(request)
        task = manager.get(client_key, body.get("taskId"))
        return JSONResponse(_task_json(task))

    @app.post("/getBalance")
    async def get_balance(request: Request) -> JSONResponse:
        body, client_key = await _authed_body(request)
        balance = keys.balance(client_key)
        stats = await run_in_threadpool(usage.stats, client_key)
        return JSONResponse({"errorId": 0, "balance": balance, "packages": [], "usage": stats})

    @app.get("/health")
    async def health() -> Dict[str, Any]:
        return {"ok": not manager.closed, "version": __version__,
                "vision": vision_bridge.available(), "time": int(time.time())}

    return app
