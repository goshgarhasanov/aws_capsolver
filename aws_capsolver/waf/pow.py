"""
Proof-of-work solvers used by challenge.js.

Three kinds exist:
  * sha256  - find a decimal counter i so sha256(input + hmac + i) has `difficulty`
              leading zero bits.
  * scrypt  - same search with scrypt(N=memory, r=8, p=1, dkLen=16, salt=hmac).
  * bandwidth - the "solution" is N zero bytes (base64), N given by the difficulty.

Hashcash searches run across processes in fixed-size chunks; the smallest valid
counter of the first successful batch is returned, as a sequential search would.
"""

import base64
import hashlib
import logging
import os
import time
from concurrent.futures import ProcessPoolExecutor
from typing import Optional

logger = logging.getLogger(__name__)

KIND_SHA256 = "sha256"
KIND_SCRYPT = "scrypt"
KIND_BANDWIDTH = "bandwidth"

BANDWIDTH_SIZES = {1: 0x400, 2: 0x2800, 3: 0x19000, 4: 0x100000, 5: 0xA00000}
DEFAULT_SCRYPT_MEMORY = 128
MAX_ITERATIONS = 50_000_000
CHUNK = {KIND_SHA256: 200_000, KIND_SCRYPT: 2_000}
# below this expected work the process pool costs more than it saves
PARALLEL_MIN_DIFFICULTY = {KIND_SHA256: 16, KIND_SCRYPT: 8}


def leading_zero_bits_ok(digest: bytes, bits: int) -> bool:
    """True when `digest` starts with at least `bits` zero bits."""
    full, rest = divmod(bits, 8)
    if len(digest) < full + (1 if rest else 0):
        return False
    if any(digest[:full]):
        return False
    return not rest or (digest[full] >> (8 - rest)) == 0


def _digest(kind: str, data: bytes, salt: bytes, memory: int) -> bytes:
    if kind == KIND_SHA256:
        return hashlib.sha256(data).digest()
    return hashlib.scrypt(data, salt=salt, n=memory, r=8, p=1, dklen=16)


def _search(kind: str, base: str, salt: str, difficulty: int, memory: int,
            start: int, stop: int) -> Optional[int]:
    """Sequential search over [start, stop); also the pure-python fallback."""
    prefix, salt_b = base.encode(), salt.encode()
    for counter in range(start, stop):
        if leading_zero_bits_ok(_digest(kind, prefix + str(counter).encode(), salt_b, memory),
                                difficulty):
            return counter
    return None


def solve_hashcash(kind: str, challenge_input: str, hmac: str, difficulty: int,
                   memory: int = DEFAULT_SCRYPT_MEMORY, processes: Optional[int] = None,
                   max_iterations: int = MAX_ITERATIONS,
                   deadline: Optional[float] = None) -> Optional[str]:
    """
    Return the counter string solving the hashcash, or None when the iteration budget
    or the monotonic `deadline` runs out (the deadline is checked between batches).
    """
    if kind not in (KIND_SHA256, KIND_SCRYPT):
        raise ValueError(f"not a hashcash kind: {kind}")
    if not isinstance(difficulty, int) or difficulty < 0 or difficulty > 64:
        raise ValueError(f"bad difficulty: {difficulty!r}")
    base, salt = challenge_input + (hmac or ""), hmac or ""
    processes = processes or os.cpu_count() or 1
    if processes <= 1 or difficulty < PARALLEL_MIN_DIFFICULTY[kind]:
        found = _search(kind, base, salt, difficulty, memory, 0, max_iterations)
        return None if found is None else str(found)

    chunk = CHUNK[kind]
    try:
        pool = ProcessPoolExecutor(max_workers=processes)
    except (OSError, NotImplementedError) as exc:  # no multiprocessing in this environment
        logger.warning("process pool unavailable (%s), searching sequentially", exc)
        found = _search(kind, base, salt, difficulty, memory, 0, max_iterations)
        return None if found is None else str(found)
    with pool:
        start = 0
        while start < max_iterations:
            if deadline is not None and time.monotonic() > deadline:
                logger.error("%s hashcash: deadline reached after %d iterations", kind, start)
                return None
            futures = []
            for _ in range(processes):
                stop = min(start + chunk, max_iterations)
                if start >= stop:
                    break
                futures.append(pool.submit(_search, kind, base, salt, difficulty, memory,
                                           start, stop))
                start = stop
            results = [f.result() for f in futures]
            hits = [r for r in results if r is not None]
            if hits:
                return str(min(hits))
    logger.error("%s hashcash: no solution within %d iterations", kind, max_iterations)
    return None


def solve_bandwidth(difficulty: int) -> str:
    size = BANDWIDTH_SIZES.get(difficulty)
    if size is None:
        raise ValueError(f"unknown NetworkBandwidth difficulty: {difficulty!r}")
    return base64.b64encode(b"\x00" * size).decode()
