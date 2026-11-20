"""
The challenge.js proof-of-work exchange on <token_base>:

    GET  /inputs?client=browser -> {challenge:{input, hmac?, region}, challenge_type, difficulty}
    solve according to challenge_type
    POST /verify (hashcash, JSON) or /mp_verify (bandwidth, multipart) -> {token}

challenge.js dispatches on the hashed `challenge_type`, and the hash -> solver mapping
has changed between script versions (e.g. ha9faaff... was scrypt in older builds and is
the NetworkBandwidth solver in the current one). Each hash therefore has an ordered
list of candidate solvers; a 400 from verify moves on to the next candidate with a
fresh /inputs.
"""

import json
import logging
import threading
import time
from typing import Any, Callable, Dict, List, Optional

import requests

from . import pow as powlib
from .errors import (ERROR_CAPTCHA_UNSOLVABLE, ERROR_SERVICE_UNAVAILABLE, ERROR_TASK_TIMEOUT,
                     SolverError)
from .http import request_json

logger = logging.getLogger(__name__)

HASH_SHA256 = "h7b0c470f0cfe3a80a9e26526ad185f484f6817d0832712a4a37a908786a6a67f"
HASH_BANDWIDTH_OLD = "h72f957df656e80ba55f5d8ce2e8c7ccb59687dba3bfb273d54b08a261b2f3002"
HASH_A9F = "ha9faaffd31b4d5ede2a2e19d2d7fd525f66fee61911511960dcbb52d3c48ce25"

ALL_KINDS = [powlib.KIND_BANDWIDTH, powlib.KIND_SCRYPT, powlib.KIND_SHA256]
CANDIDATES: Dict[str, List[str]] = {
    # current challenge.js: ha9faaff... is the NetworkBandwidth solver
    HASH_A9F: [powlib.KIND_BANDWIDTH, powlib.KIND_SCRYPT, powlib.KIND_SHA256],
    HASH_SHA256: [powlib.KIND_SHA256, powlib.KIND_SCRYPT, powlib.KIND_BANDWIDTH],
    HASH_BANDWIDTH_OLD: [powlib.KIND_BANDWIDTH, powlib.KIND_SHA256, powlib.KIND_SCRYPT],
}


# challenge_type -> the kind /verify last accepted. Tried first next time, so a mapping that
# differs from CANDIDATES costs one rejected /verify per process instead of one per solve
# (measured on a live endpoint, Nov 2026: ha9faaff... wants scrypt, the bandwidth try wasted
# ~0.5 s on every solve).
_LEARNED: Dict[str, str] = {}
_LEARNED_LOCK = threading.Lock()


def candidate_kinds(challenge_type: str) -> List[str]:
    kinds = list(CANDIDATES.get(challenge_type, ALL_KINDS))
    with _LEARNED_LOCK:
        learned = _LEARNED.get(challenge_type)
    if learned in kinds:
        kinds.remove(learned)
        kinds.insert(0, learned)
    return kinds


def remember_kind(challenge_type: str, kind: str) -> None:
    with _LEARNED_LOCK:
        if _LEARNED.get(challenge_type) != kind:
            logger.info("pow type=%s.. -> kind=%s learned (tried first from now on)",
                        challenge_type[:10], kind)
        _LEARNED[challenge_type] = kind


def forget_kind(challenge_type: str, kind: str) -> None:
    with _LEARNED_LOCK:
        if _LEARNED.get(challenge_type) == kind:
            del _LEARNED[challenge_type]


def check_stop(deadline: Optional[float], cancel: Optional[threading.Event],
               what: str) -> None:
    """Raise when the task was cancelled (service shutdown) or ran out of time."""
    if cancel is not None and cancel.is_set():
        raise SolverError(ERROR_SERVICE_UNAVAILABLE, f"{what}: service is shutting down")
    if deadline is not None and time.monotonic() > deadline:
        raise SolverError(ERROR_TASK_TIMEOUT, f"{what}: deadline exceeded")


def compute_solution(kind: str, challenge: dict, difficulty: int, memory: int,
                     processes: Optional[int], deadline: Optional[float] = None,
                     cancel: Optional[threading.Event] = None) -> str:
    if kind == powlib.KIND_BANDWIDTH:
        return powlib.solve_bandwidth(difficulty)
    solution = powlib.solve_hashcash(kind, str(challenge.get("input") or ""),
                                     str(challenge.get("hmac") or ""), difficulty,
                                     memory=memory, processes=processes, deadline=deadline,
                                     cancel=cancel)
    if solution is None:
        check_stop(deadline, cancel, f"{kind} hashcash")
        raise SolverError(ERROR_CAPTCHA_UNSOLVABLE, f"{kind} hashcash found no solution")
    return solution


def _as_difficulty(value: Any) -> Optional[int]:
    """The difficulty as an int; numeric strings are accepted, bools are not."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    return None


def _verify_body(challenge: dict, solution: Optional[str], domain: str,
                 existing_token: Optional[str]) -> dict:
    return {
        "challenge": challenge,
        "solution": solution,
        "signals": [],
        "checksum": "",
        "existing_token": existing_token,
        "client": "Browser",
        "domain": domain,
        "metrics": [],
    }


def solve_challenge(session: requests.Session, token_base: str, domain: str, *,
                    existing_token: Optional[str] = None, timeout: float = 20.0,
                    scrypt_memory: int = powlib.DEFAULT_SCRYPT_MEMORY,
                    processes: Optional[int] = None,
                    deadline: Optional[float] = None,
                    cancel: Optional[threading.Event] = None,
                    sleep: Callable[[float], None] = time.sleep) -> str:
    """Run the proof-of-work exchange and return a fresh aws-waf-token."""
    headers = {"accept": "*/*", "origin": f"https://{domain}", "referer": f"https://{domain}/"}
    started = time.monotonic()
    tried: List[str] = []
    last_error: Optional[SolverError] = None
    kinds: Optional[List[str]] = None

    while True:
        check_stop(deadline, cancel, "challenge")
        if kinds is not None and all(k in tried for k in kinds):
            raise last_error or SolverError(ERROR_CAPTCHA_UNSOLVABLE, "no PoW solver accepted")
        inputs = request_json(session, "GET", f"{token_base}/inputs", "challenge /inputs",
                              timeout, deadline, params={"client": "browser"}, headers=headers)
        challenge = inputs.get("challenge")
        challenge_type = inputs.get("challenge_type")
        difficulty = _as_difficulty(inputs.get("difficulty"))
        if not isinstance(challenge, dict) or not challenge.get("input") \
                or not isinstance(challenge_type, str) or not challenge_type \
                or difficulty is None:
            raise SolverError(ERROR_CAPTCHA_UNSOLVABLE,
                              f"unexpected /inputs response (keys={sorted(inputs)})")
        if kinds is None:
            kinds = candidate_kinds(challenge_type)
        remaining = [k for k in kinds if k not in tried]
        if not remaining:
            raise last_error or SolverError(ERROR_CAPTCHA_UNSOLVABLE, "no PoW solver accepted")
        kind = remaining[0]
        tried.append(kind)

        t0 = time.monotonic()
        try:
            solution = compute_solution(kind, challenge, difficulty, scrypt_memory, processes,
                                        deadline, cancel)
        except ValueError as exc:  # e.g. bandwidth difficulty outside the known table
            last_error = SolverError(ERROR_CAPTCHA_UNSOLVABLE, str(exc))
            continue
        logger.info("pow type=%s.. kind=%s difficulty=%s solved in %.2fs",
                    challenge_type[:10], kind, difficulty, time.monotonic() - t0)

        body = _verify_body(challenge, solution, domain, existing_token)
        try:
            if kind == powlib.KIND_BANDWIDTH:
                multipart = dict(body, solution=None)
                result = request_json(
                    session, "POST", f"{token_base}/mp_verify", "challenge /mp_verify",
                    timeout * 2, deadline, headers=headers,
                    files={"challenge": (None, json.dumps(multipart)),
                           "solution_data": (None, solution)})
            else:
                result = request_json(session, "POST", f"{token_base}/verify",
                                      "challenge /verify", timeout, deadline, headers=headers,
                                      json=body)
        except SolverError as exc:
            if exc.http_status == 400:
                logger.warning("pow kind=%s rejected (400), trying next candidate", kind)
                forget_kind(challenge_type, kind)
                last_error = SolverError(ERROR_CAPTCHA_UNSOLVABLE, exc.description)
                sleep(0.2)
                continue
            raise
        token = result.get("token")
        if not isinstance(token, str) or not token:
            raise SolverError(ERROR_CAPTCHA_UNSOLVABLE, f"verify returned no token "
                              f"(keys={sorted(result)})")
        remember_kind(challenge_type, kind)
        logger.info("pow token acquired in %.2fs (kind=%s)", time.monotonic() - started, kind)
        return token
