"""Minimal typed client for the aws_capsolver HTTP API (CapSolver-compatible).

Only depends on ``requests``. Copy this single file into your project.

    from aws_capsolver_client import AwsCapsolverClient, CapsolverError

    client = AwsCapsolverClient("http://127.0.0.1:8790", "MY-KEY")
    solution = client.solve_aws_waf("https://protected.example/",
                                    proxy="http://user:pass@1.2.3.4:8080")
    print(solution["cookie"])          # value of the aws-waf-token cookie

    print(client.classify("aws:grid:bed", [b64_tile_0, ..., b64_tile_8]))   # {"objects": [...]}
    print(client.classify_recaptcha_v2(grid_png, "/m/0k4j"))  # {"type": "multi", "objects": [...], "size": 3}

Every API failure is raised as ``CapsolverError`` with the CapSolver ``errorCode``.
Transport failures (connection refused, HTTP 5xx, invalid JSON) are raised as
``CapsolverError`` with the client-side code ``ERROR_CLIENT_TRANSPORT``, and a polling
timeout as ``ERROR_CLIENT_TIMEOUT``.
"""

from __future__ import annotations

import base64
import time
from typing import Any, Dict, List, Optional, Sequence, TypedDict, Union

import requests

__all__ = ["AwsCapsolverClient", "CapsolverError", "WafSolution", "GridSolution",
           "ToycarSolution", "RecaptchaV2Solution", "ERROR_CLIENT_TIMEOUT", "ERROR_CLIENT_TRANSPORT"]

ERROR_CLIENT_TIMEOUT = "ERROR_CLIENT_TIMEOUT"
ERROR_CLIENT_TRANSPORT = "ERROR_CLIENT_TRANSPORT"


class WafSolution(TypedDict, total=False):
    cookie: str            # aws-waf-token value; "" for a voucher-only outcome
    userAgent: str         # send it with the cookie
    captchaVoucher: str    # present when a CAPTCHA was solved


class GridSolution(TypedDict):
    objects: List[int]     # 0-based, row-major tile indices


class ToycarSolution(TypedDict):
    box: List[int]         # [x, y]


class RecaptchaV2Solution(TypedDict, total=False):
    type: str              # "multi" (3x3 / 4x4 grid) or "single" (one 1x1 tile)
    objects: List[int]     # multi: 0-based, row-major cells to click (may be empty = skip)
    hasObject: bool        # single: the tile shows the object
    size: int              # 3, 4 or 1


class CapsolverError(Exception):
    """An error reply from the service (errorId=1) or a client-side failure."""

    def __init__(self, code: str, description: str = "", task_id: Optional[str] = None):
        super().__init__(f"{code}: {description}" if description else code)
        self.code = code
        self.description = description
        self.task_id = task_id


ImageInput = Union[str, bytes, bytearray]


def _b64(image: ImageInput) -> str:
    if isinstance(image, (bytes, bytearray)):
        return base64.b64encode(bytes(image)).decode("ascii")
    if isinstance(image, str):
        return image
    raise TypeError("images must be base64 strings or raw bytes")


class AwsCapsolverClient:
    """Synchronous client; one instance per thread is the simplest safe usage."""

    def __init__(self, base_url: str = "http://127.0.0.1:8790", client_key: str = "",
                 request_timeout: float = 90.0, session: Optional[requests.Session] = None):
        """
        request_timeout bounds every single HTTP call. It must exceed the server's
        AWSCS_CLASSIFY_TIMEOUT because AwsWafClassification answers synchronously.
        """
        self.base_url = base_url.rstrip("/")
        self.client_key = client_key
        self.request_timeout = request_timeout
        self.session = session or requests.Session()

    # transport ---------------------------------------------------------------
    def _post(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        body = dict(payload, clientKey=self.client_key)
        try:
            resp = self.session.post(self.base_url + path, json=body,
                                     timeout=self.request_timeout)
        except requests.RequestException as exc:
            raise CapsolverError(ERROR_CLIENT_TRANSPORT, f"{path}: {exc}") from exc
        try:
            data = resp.json()
        except ValueError:
            data = None
        if not isinstance(data, dict) or (resp.status_code != 200 and not data.get("errorId")):
            # a proxy error page or similar; the service itself always answers a JSON object
            raise CapsolverError(ERROR_CLIENT_TRANSPORT,
                                 f"{path}: HTTP {resp.status_code} {resp.text[:200]}")
        if data.get("errorId"):  # also for HTTP 500 replies that carry an errorCode
            raise CapsolverError(str(data.get("errorCode") or "ERROR_UNKNOWN"),
                                 str(data.get("errorDescription") or ""), data.get("taskId"))
        return data

    # raw API -----------------------------------------------------------------
    def create_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """POST /createTask. Returns the raw reply: {"taskId"} or, for synchronous
        tasks, {"taskId", "status": "ready", "solution"}."""
        return self._post("/createTask", {"task": task})

    def get_task_result(self, task_id: str, timeout: float = 150.0,
                        poll_interval: float = 2.0) -> Dict[str, Any]:
        """Poll /getTaskResult until the task is ready; return its solution.

        Raises CapsolverError with the task's errorCode when it failed, or
        ERROR_CLIENT_TIMEOUT when it is still processing after ``timeout`` seconds.
        """
        deadline = time.monotonic() + timeout
        while True:
            data = self._post("/getTaskResult", {"taskId": task_id})
            if data.get("status") == "ready":
                return data.get("solution") or {}
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise CapsolverError(ERROR_CLIENT_TIMEOUT,
                                     f"task {task_id} not ready after {timeout:.0f}s", task_id)
            time.sleep(min(poll_interval, remaining))

    def get_balance(self) -> Dict[str, Any]:
        return self._post("/getBalance", {})

    def health(self) -> Dict[str, Any]:
        try:
            resp = self.session.get(self.base_url + "/health", timeout=self.request_timeout)
            resp.raise_for_status()
            return resp.json()
        except (requests.RequestException, ValueError) as exc:
            raise CapsolverError(ERROR_CLIENT_TRANSPORT, f"/health: {exc}") from exc

    # helpers -----------------------------------------------------------------
    def solve_aws_waf(self, website_url: str, proxy: Optional[str] = None,
                      user_agent: Optional[str] = None, timeout: float = 150.0,
                      poll_interval: float = 2.0, **extra: Any) -> WafSolution:
        """Solve the AWS WAF of ``website_url``; returns {"cookie", "userAgent", ...}.

        Uses AntiAwsWafTask when ``proxy`` is given, otherwise AntiAwsWafTaskProxyLess.
        ``extra`` is passed through (awsKey, awsIv, awsContext, awsChallengeJS,
        awsProblemUrl, awsApiJs, awsApiKey, locale, proxyType/proxyAddress/...).
        """
        task: Dict[str, Any] = {"type": "AntiAwsWafTask" if proxy else "AntiAwsWafTaskProxyLess",
                                "websiteURL": website_url}
        if proxy:
            task["proxy"] = proxy
        if user_agent:
            task["userAgent"] = user_agent
        task.update({k: v for k, v in extra.items() if v is not None})
        created = self.create_task(task)
        if created.get("status") == "ready":
            return created.get("solution") or {}  # type: ignore[return-value]
        task_id = created.get("taskId")
        if not task_id:
            raise CapsolverError(ERROR_CLIENT_TRANSPORT, "createTask returned no taskId")
        return self.get_task_result(task_id, timeout=timeout,  # type: ignore[return-value]
                                    poll_interval=poll_interval)

    def classify(self, question: str, images: Sequence[ImageInput]) -> Dict[str, Any]:
        """AwsWafClassification (synchronous).

        question: "aws:grid:<target>" -> {"objects": [...]}, or
                  "aws:toycarcity:carcity" -> {"box": [x, y]}.
        images: base64 strings (data: URLs accepted) or raw bytes.
        """
        created = self.create_task({"type": "AwsWafClassification", "question": question,
                                    "images": [_b64(i) for i in images]})
        if created.get("status") == "ready":
            return created.get("solution") or {}
        # not expected from this server, but CapSolver-style async replies are handled too
        return self.get_task_result(created["taskId"])


    def classify_recaptcha_v2(self, image: ImageInput, question: str,
                              size: Optional[int] = None) -> RecaptchaV2Solution:
        """ReCaptchaV2Classification (synchronous).

        image: the challenge picture (the whole 3x3 / 4x4 grid, or one new 1x1 tile), base64
               (data: URL accepted) or raw bytes.
        question: a Freebase id such as "/m/0k4j" or the object name in any language ("cars").
        size: 3, 4 or 1 when known (otherwise the model infers it from the picture).
        """
        task: Dict[str, Any] = {"type": "ReCaptchaV2Classification", "image": _b64(image),
                                "question": question}
        if size is not None:
            task["size"] = size
        created = self.create_task(task)
        if created.get("status") == "ready":
            return created.get("solution") or {}  # type: ignore[return-value]
        return self.get_task_result(created["taskId"])  # type: ignore[return-value]


if __name__ == "__main__":  # tiny smoke test: python aws_capsolver_client.py URL KEY
    import json
    import sys

    c = AwsCapsolverClient(sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8790",
                           sys.argv[2] if len(sys.argv) > 2 else "")
    print(json.dumps({"health": c.health(), "balance": c.get_balance()}, indent=2))
