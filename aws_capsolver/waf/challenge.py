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
import time
from typing import Callable, Dict, List, Optional

import requests

from . import pow as powlib
from .errors import ERROR_CAPTCHA_UNSOLVABLE, ERROR_TASK_TIMEOUT, SolverError
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


def candidate_kinds(challenge_type: str) -> List[str]:
    return CANDIDATES.get(challenge_type, ALL_KINDS)


def compute_solution(kind: str, challenge: dict, difficulty: int, memory: int,
                     processes: Optional[int], deadline: Optional[float] = None) -> str:
    if kind == powlib.KIND_BANDWIDTH:
        return powlib.solve_bandwidth(difficulty)
    solution = powlib.solve_hashcash(kind, challenge.get("input", ""), challenge.get("hmac") or "",
                                     difficulty, memory=memory, processes=processes,
                                     deadline=deadline)
    if solution is None:
        raise SolverError(ERROR_CAPTCHA_UNSOLVABLE, f"{kind} hashcash found no solution")
    return solution


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
                    sleep: Callable[[float], None] = time.sleep) -> str:
    """Run the proof-of-work exchange and return a fresh aws-waf-token."""
    headers = {"accept": "*/*", "origin": f"https://{domain}", "referer": f"https://{domain}/"}
    started = time.monotonic()
    tried: List[str] = []
    last_error: Optional[SolverError] = None
    kinds: Optional[List[str]] = None

    while True:
        if deadline is not None and time.monotonic() > deadline:
            raise SolverError(ERROR_TASK_TIMEOUT, "challenge deadline exceeded")
        inputs = request_json(session, "GET", f"{token_base}/inputs", "challenge /inputs",
                              timeout, params={"client": "browser"}, headers=headers)
        challenge = inputs.get("challenge")
        challenge_type = inputs.get("challenge_type")
        difficulty = inputs.get("difficulty")
        if not isinstance(challenge, dict) or not challenge.get("input") or not challenge_type \
                or not isinstance(difficulty, int):
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
                                        deadline)
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
                    timeout * 2, headers=headers,
                    files={"challenge": (None, json.dumps(multipart)),
                           "solution_data": (None, solution)})
            else:
                result = request_json(session, "POST", f"{token_base}/verify",
                                      "challenge /verify", timeout, headers=headers,
                                      json=body)
        except SolverError as exc:
            if exc.code == "HTTP_400":
                logger.warning("pow kind=%s rejected (400), trying next candidate", kind)
                last_error = SolverError(ERROR_CAPTCHA_UNSOLVABLE, exc.description)
                sleep(0.2)
                continue
            raise
        token = result.get("token")
        if not token:
            raise SolverError(ERROR_CAPTCHA_UNSOLVABLE, f"verify returned no token "
                              f"(keys={sorted(result)})")
        logger.info("pow token acquired in %.2fs (kind=%s)", time.monotonic() - started, kind)
        return token
