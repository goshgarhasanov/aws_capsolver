"""
The captcha.js exchange on <captcha_base>, followed by the voucher exchange on <token_base>:

    GET  /problem?kind=visual|audio&domain=&locale=[&api_key=]
         -> {problem_type, state, key, hmac_tag, assets, localized_assets, region, ...}
    POST /verify[?region=]  text/plain JSON {state, key, hmac_tag, client_solution,
                             metrics:{solve_time_millis}, goku_props, [api_key], locale}
         -> success=true + captcha_voucher            done
            success=true + problem (more required)    solve the next problem
            success=false + problem                   wrong answer, retry with it
    POST <token_base>/voucher[?region=] {captcha_voucher, existing_token} -> {token}

client_solution per problem family (from captcha.js):
    gridcaptcha     sorted list of tile indices (row-major)
    toycarcity      {"x": (g/c - 16)/32 | 0, "y": h/c/32 | 0}; g/c equals 2 * image px
    noisyneighbour  transcribed text, lower-cased, trimmed, NFC-normalised
"""

import base64
import json
import logging
import time
import unicodedata
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from urllib.parse import quote

import requests

from .. import vision_bridge
from .errors import (ERROR_CAPTCHA_UNSOLVABLE, ERROR_TASK_TIMEOUT, SolverError)
from .http import request_json

logger = logging.getLogger(__name__)


def as_list(value: Any) -> List[Any]:
    """assets.images / assets.target arrive as JSON-encoded strings; normalise to a list."""
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        text = value.strip()
        if text.startswith("["):
            try:
                parsed = json.loads(text)
            except ValueError:
                return []
            return parsed if isinstance(parsed, list) else []
        return [text] if text else []
    return []


def problem_family(problem: Dict[str, Any]) -> str:
    return str(problem.get("problem_type") or "").split("-")[0]


def toycar_solution(x: int, y: int) -> Dict[str, int]:
    """Map a point in natural image pixels onto captcha.js' toycarcity solution cell."""
    return {"x": int((2 * x - 16) // 32), "y": int((2 * y) // 32)}


def normalise_audio_answer(text: str) -> str:
    return unicodedata.normalize("NFC", (text or "").lower().strip())


def solve_problem(problem: Dict[str, Any]) -> Any:
    """Compute client_solution for one problem via the vision package."""
    family = problem_family(problem)
    assets = problem.get("assets") or {}
    try:
        if family == "gridcaptcha":
            images = [base64.b64decode(i) for i in as_list(assets.get("images"))]
            targets = as_list(assets.get("target"))
            if not images or not targets:
                raise SolverError(ERROR_CAPTCHA_UNSOLVABLE, "grid problem without images/target")
            cells = sorted({int(i) for i in vision_bridge.classify_grid(images, str(targets[0]))})
            if any(i < 0 or i >= len(images) for i in cells):
                raise SolverError(ERROR_CAPTCHA_UNSOLVABLE, f"vision returned bad tiles {cells}")
            return cells
        if family == "toycarcity":
            image = assets.get("image")
            if not image:
                raise SolverError(ERROR_CAPTCHA_UNSOLVABLE, "toycarcity problem without image")
            return toycar_solution(*vision_bridge.solve_toycar(base64.b64decode(image)))
        if family == "noisyneighbour":
            audio = assets.get("audio")
            if not audio:
                raise SolverError(ERROR_CAPTCHA_UNSOLVABLE, "audio problem without audio")
            return normalise_audio_answer(vision_bridge.transcribe(audio))
    except vision_bridge.unsupported_types() as exc:
        raise SolverError(ERROR_CAPTCHA_UNSOLVABLE, f"{family}: {exc}") from exc
    raise SolverError(ERROR_CAPTCHA_UNSOLVABLE, f"unsupported problem type {family!r}")


@dataclass
class CaptchaContext:
    captcha_base: str
    token_base: Optional[str]
    domain: str
    goku_props: Optional[Dict[str, str]]
    api_key: Optional[str] = None
    locale: str = "en-us"


def _with_region(url: str, region: Optional[str]) -> str:
    return f"{url}?region={quote(region, safe='')}" if region else url


def fetch_problem(session: requests.Session, ctx: CaptchaContext, kind: str,
                  timeout: float) -> Dict[str, Any]:
    params = {"kind": kind, "domain": ctx.domain, "locale": ctx.locale}
    if ctx.api_key:
        params["api_key"] = ctx.api_key
    headers = {"accept": "*/*", "origin": f"https://{ctx.domain}",
               "referer": f"https://{ctx.domain}/"}
    problem = request_json(session, "GET", f"{ctx.captcha_base}/problem", "captcha /problem",
                           timeout, params=params, headers=headers)
    if not problem.get("problem_type") or not problem.get("state"):
        raise SolverError(ERROR_CAPTCHA_UNSOLVABLE, f"unexpected /problem response "
                          f"(keys={sorted(problem)})")
    return problem


def submit_solution(session: requests.Session, ctx: CaptchaContext, problem: Dict[str, Any],
                    solution: Any, solve_millis: int, timeout: float) -> Dict[str, Any]:
    body: Dict[str, Any] = {
        "state": problem.get("state"),
        "key": problem.get("key"),
        "hmac_tag": problem.get("hmac_tag"),
        "client_solution": solution,
        "metrics": {"solve_time_millis": int(solve_millis)},
        "goku_props": ctx.goku_props,
        "locale": ctx.locale,
    }
    if ctx.api_key:
        body["api_key"] = ctx.api_key
    headers = {"accept": "*/*", "content-type": "text/plain;charset=UTF-8",
               "origin": f"https://{ctx.domain}", "referer": f"https://{ctx.domain}/"}
    return request_json(session, "POST",
                        _with_region(f"{ctx.captcha_base}/verify", problem.get("region")),
                        "captcha /verify", timeout, data=json.dumps(body), headers=headers)


def exchange_voucher(session: requests.Session, ctx: CaptchaContext, voucher: str,
                     region: Optional[str], existing_token: Optional[str],
                     timeout: float) -> str:
    if not ctx.token_base:
        raise SolverError(ERROR_CAPTCHA_UNSOLVABLE, "no token endpoint to redeem the voucher")
    headers = {"accept": "*/*", "origin": f"https://{ctx.domain}",
               "referer": f"https://{ctx.domain}/"}
    result = request_json(session, "POST", _with_region(f"{ctx.token_base}/voucher", region),
                          "token /voucher", timeout,
                          json={"captcha_voucher": voucher, "existing_token": existing_token},
                          headers=headers)
    token = result.get("token")
    if not token:
        raise SolverError(ERROR_CAPTCHA_UNSOLVABLE, f"/voucher returned no token "
                          f"(keys={sorted(result)})")
    return token


@dataclass
class CaptchaResult:
    voucher: str
    token: Optional[str]
    rounds: int


def solve_captcha(session: requests.Session, ctx: CaptchaContext, *,
                  existing_token: Optional[str] = None, kinds: tuple = ("visual", "audio"),
                  max_rounds: int = 5, timeout: float = 20.0,
                  deadline: Optional[float] = None, redeem: bool = True) -> CaptchaResult:
    """Solve problems until a voucher is issued, then redeem it for an aws-waf-token."""
    last_error: Optional[SolverError] = None
    rounds = 0
    for kind in kinds:
        try:
            problem: Optional[Dict[str, Any]] = fetch_problem(session, ctx, kind, timeout)
        except SolverError as exc:
            last_error = exc
            continue
        while problem is not None and rounds < max_rounds:
            if deadline is not None and time.monotonic() > deadline:
                raise SolverError(ERROR_TASK_TIMEOUT, "captcha deadline exceeded")
            rounds += 1
            t0 = time.monotonic()
            try:
                solution = solve_problem(problem)
            except SolverError as exc:
                logger.info("captcha kind=%s type=%s unsolvable: %s", kind,
                            problem.get("problem_type"), exc.description)
                last_error = exc
                break  # try the next kind (e.g. audio when the grid model lacks a class)
            elapsed_ms = int((time.monotonic() - t0) * 1000)
            # a human never answers in a few ms; report a plausible solve time
            verdict = submit_solution(session, ctx, problem, solution,
                                      max(elapsed_ms, 3500 + 400 * rounds), timeout)
            voucher = verdict.get("captcha_voucher")
            logger.info("captcha round=%d type=%s success=%s provided=%s required=%s voucher=%s",
                        rounds, problem.get("problem_type"), verdict.get("success"),
                        verdict.get("num_solutions_provided"),
                        verdict.get("num_solutions_required"), bool(voucher))
            if voucher and verdict.get("success") is not False:
                token = None
                if redeem:
                    token = exchange_voucher(session, ctx, voucher, verdict.get("region") or
                                             problem.get("region"), existing_token, timeout)
                return CaptchaResult(voucher=voucher, token=token, rounds=rounds)
            if verdict.get("success") is True:
                # correct, but the site requires more solved problems
                last_error = SolverError(ERROR_CAPTCHA_UNSOLVABLE, "ran out of rounds")
            else:
                last_error = SolverError(ERROR_CAPTCHA_UNSOLVABLE,
                                         f"answer rejected: {verdict.get('reason') or 'wrong'}")
            problem = verdict.get("problem") or None
        if rounds >= max_rounds:
            break
    raise last_error or SolverError(ERROR_CAPTCHA_UNSOLVABLE, "captcha not solved")
