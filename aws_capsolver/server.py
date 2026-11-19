"""CapSolver-compatible HTTP API (createTask / getTaskResult / getBalance)."""

import logging
import threading
import time
from typing import Any, Dict, Optional

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from starlette.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse

from . import __version__, vision_bridge
from .config import KeyStore, Settings, UsageStore
from .tasks import TaskManager
from .waf.errors import ERROR_INVALID_TASK_DATA, ERROR_KEY_DENIED_ACCESS, SolverError

logger = logging.getLogger(__name__)


def _error(code: str, description: str) -> JSONResponse:
    # CapSolver answers errors with HTTP 200 and errorId=1
    return JSONResponse({"errorId": 1, "errorCode": code, "errorDescription": description})


async def _body(request: Request) -> Dict[str, Any]:
    try:
        data = await request.json()
    except ValueError as exc:
        raise SolverError(ERROR_INVALID_TASK_DATA, "body must be JSON") from exc
    if not isinstance(data, dict):
        raise SolverError(ERROR_INVALID_TASK_DATA, "body must be a JSON object")
    return data


def create_app(settings: Optional[Settings] = None, manager: Optional[TaskManager] = None,
               keys: Optional[KeyStore] = None, usage: Optional[UsageStore] = None) -> FastAPI:
    settings = settings or Settings.from_env()
    keys = keys or KeyStore(settings)
    usage = usage or UsageStore(settings.usage_db)
    manager = manager or TaskManager(settings, usage)
    stop = threading.Event()

    def janitor() -> None:
        while not stop.wait(30):
            manager.cleanup()

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        threading.Thread(target=janitor, name="task-janitor", daemon=True).start()
        if settings.warmup:
            threading.Thread(target=vision_bridge.warmup, name="vision-warmup",
                             daemon=True).start()
        logger.info("aws_capsolver %s ready (workers=%d, vision=%s)", __version__,
                    settings.workers, vision_bridge.available())
        try:
            yield
        finally:
            stop.set()
            manager.shutdown()

    app = FastAPI(title="aws_capsolver", version=__version__, lifespan=lifespan)
    app.state.manager, app.state.keys, app.state.usage = manager, keys, usage

    @app.exception_handler(SolverError)
    async def _solver_error(_: Request, exc: SolverError) -> JSONResponse:
        return _error(exc.code, exc.description)

    def _auth(body: Dict[str, Any]) -> str:
        client_key = body.get("clientKey")
        if keys.lookup(client_key) is None:
            raise SolverError(ERROR_KEY_DENIED_ACCESS, "invalid clientKey")
        return client_key

    def _task_json(task) -> Dict[str, Any]:
        if task.status == "failed":
            return {"errorId": 1, "errorCode": task.error_code,
                    "errorDescription": task.error_description, "taskId": task.id,
                    "status": "failed"}
        out: Dict[str, Any] = {"errorId": 0, "taskId": task.id, "status": task.status}
        if task.status == "ready":
            out["solution"] = task.solution
        return out

    @app.post("/createTask")
    async def create_task(request: Request) -> JSONResponse:
        body = await _body(request)
        client_key = _auth(body)
        # task creation may block (synchronous classification) - keep it off the event loop
        task = await run_in_threadpool(manager.create, client_key, body.get("task"))
        if task.status in ("ready", "failed"):
            return JSONResponse(_task_json(task))
        return JSONResponse({"errorId": 0, "taskId": task.id})

    @app.post("/getTaskResult")
    async def get_task_result(request: Request) -> JSONResponse:
        body = await _body(request)
        client_key = _auth(body)
        task = manager.get(client_key, body.get("taskId"))
        out = _task_json(task)
        if out.get("status") == "idle":
            out["status"] = "processing"  # CapSolver clients poll until ready/failed
        return JSONResponse(out)

    @app.post("/getBalance")
    async def get_balance(request: Request) -> JSONResponse:
        body = await _body(request)
        client_key = _auth(body)
        return JSONResponse({"errorId": 0, "balance": keys.balance(client_key),
                             "packages": [], "usage": usage.stats(client_key)})

    @app.get("/health")
    async def health() -> Dict[str, Any]:
        return {"ok": True, "version": __version__, "vision": vision_bridge.available(),
                "time": int(time.time())}

    return app
