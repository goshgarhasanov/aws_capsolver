"""
Proof-of-work solvers used by challenge.js.

Three kinds exist:
  * sha256  - find a decimal counter i so sha256(input + hmac + i) has `difficulty`
              leading zero bits.
  * scrypt  - same search with scrypt(N=memory, r=8, p=1, dkLen=16, salt=hmac).
  * bandwidth - the "solution" is N zero bytes (base64), N given by the difficulty.

Hashcash searches run on a process pool shared by every task of this OS process, in
fixed-size chunks; the smallest valid counter of the first successful batch is
returned, exactly as a sequential search would.

The pool uses the "forkserver" start method (or "spawn"), never "fork": forking a
multi-threaded server (uvicorn, the worker threads) can deadlock the child on locks
held by other threads. Workers exit on their own when the parent process dies.

Every search honours a monotonic `deadline` and an optional `cancel` event, checked
between chunks; a chunk is short (well under a second), so both are respected promptly
and no work is left running after the call returns except at most one chunk per worker.
"""

import base64
import hashlib
import logging
import multiprocessing
import os
import threading
import time
from concurrent.futures import FIRST_EXCEPTION, ProcessPoolExecutor, wait
from concurrent.futures.process import BrokenProcessPool
from typing import Optional

logger = logging.getLogger(__name__)

KIND_SHA256 = "sha256"
KIND_SCRYPT = "scrypt"
KIND_BANDWIDTH = "bandwidth"

BANDWIDTH_SIZES = {1: 0x400, 2: 0x2800, 3: 0x19000, 4: 0x100000, 5: 0xA00000}
DEFAULT_SCRYPT_MEMORY = 128
MAX_ITERATIONS = 50_000_000
MAX_DIFFICULTY = 64
CHUNK = {KIND_SHA256: 200_000, KIND_SCRYPT: 2_000}
# below this expected work the process pool costs more than it saves
PARALLEL_MIN_DIFFICULTY = {KIND_SHA256: 16, KIND_SCRYPT: 8}
_WAIT_SLICE = 0.25  # how often a waiting caller re-checks its deadline / cancel event

_pool_lock = threading.Lock()
_pool: Optional[ProcessPoolExecutor] = None
_pool_pid: Optional[int] = None


class _PoolUnavailable(Exception):
    """The process pool cannot be used; the caller falls back to a sequential search."""


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
    """Sequential search over [start, stop); runs in the pool workers too."""
    prefix, salt_b = base.encode(), salt.encode()
    for counter in range(start, stop):
        if leading_zero_bits_ok(_digest(kind, prefix + str(counter).encode(), salt_b, memory),
                                difficulty):
            return counter
    return None


def _stopped(deadline: Optional[float], cancel: Optional[threading.Event]) -> bool:
    return (cancel is not None and cancel.is_set()) or \
        (deadline is not None and time.monotonic() > deadline)


def _sequential(kind: str, base: str, salt: str, difficulty: int, memory: int, start: int,
                stop: int, deadline: Optional[float],
                cancel: Optional[threading.Event]) -> Optional[int]:
    """In-process search in chunks so the deadline and the cancel event are honoured."""
    chunk = CHUNK[kind]
    while start < stop:
        if _stopped(deadline, cancel):
            logger.warning("%s hashcash stopped (deadline/cancel) after %d iterations",
                           kind, start)
            return None
        end = min(start + chunk, stop)
        found = _search(kind, base, salt, difficulty, memory, start, end)
        if found is not None:
            return found
        start = end
    return None


def _mp_context():
    methods = multiprocessing.get_all_start_methods()
    return multiprocessing.get_context("forkserver" if "forkserver" in methods else "spawn")


def _get_pool(processes: int) -> ProcessPoolExecutor:
    """The process-wide pool, (re)created when missing, broken or inherited via fork."""
    global _pool, _pool_pid
    with _pool_lock:
        pid = os.getpid()
        if _pool is not None and _pool_pid == pid and not getattr(_pool, "_broken", False):
            return _pool
        if _pool is not None and _pool_pid == pid:
            _pool.shutdown(wait=False, cancel_futures=True)
        _pool = None
        try:
            _pool = ProcessPoolExecutor(max_workers=processes, mp_context=_mp_context())
        except (OSError, ValueError, NotImplementedError) as exc:
            raise _PoolUnavailable(str(exc)) from exc
        _pool_pid = pid
        return _pool


def _discard_pool(pool: ProcessPoolExecutor) -> None:
    global _pool
    with _pool_lock:
        if _pool is pool:
            _pool = None
    try:
        pool.shutdown(wait=False, cancel_futures=True)
    except Exception:  # best effort on an already broken pool
        pass


def shutdown_pool(wait: bool = True) -> None:
    """Stop the shared worker processes (service shutdown, tests)."""
    global _pool
    with _pool_lock:
        pool, _pool = (_pool if _pool_pid == os.getpid() else None), None
    if pool is not None:
        pool.shutdown(wait=wait, cancel_futures=True)


def _parallel(kind: str, base: str, salt: str, difficulty: int, memory: int, processes: int,
              start: int, max_iterations: int, deadline: Optional[float],
              cancel: Optional[threading.Event]) -> Optional[int]:
    """Batched search on the pool; raises _PoolUnavailable (with the resume point)."""
    chunk = CHUNK[kind]
    pool = _get_pool(processes)
    while start < max_iterations:
        if _stopped(deadline, cancel):
            logger.warning("%s hashcash stopped (deadline/cancel) after %d iterations",
                           kind, start)
            return None
        futures, batch_start = [], start
        try:
            for _ in range(processes):
                stop = min(start + chunk, max_iterations)
                if start >= stop:
                    break
                futures.append(pool.submit(_search, kind, base, salt, difficulty, memory,
                                           start, stop))
                start = stop
        except (RuntimeError, BrokenProcessPool, AssertionError, OSError) as exc:
            for f in futures:
                f.cancel()
            _discard_pool(pool)
            raise _PoolUnavailable(f"{exc.__class__.__name__}: {exc}", batch_start) from exc
        pending = set(futures)
        while pending:
            if _stopped(deadline, cancel):
                for f in pending:
                    f.cancel()
                logger.warning("%s hashcash stopped (deadline/cancel) after %d iterations",
                               kind, batch_start)
                return None
            _, pending = wait(pending, timeout=_WAIT_SLICE, return_when=FIRST_EXCEPTION)
            if any(f.done() and not f.cancelled() and f.exception() is not None
                   for f in futures):
                break
        try:
            results = [f.result() for f in futures]
        except BaseException as exc:  # BrokenProcessPool, CancelledError, worker failure
            for f in futures:
                f.cancel()
            _discard_pool(pool)
            raise _PoolUnavailable(f"{exc.__class__.__name__}: {exc}", batch_start) from exc
        hits = [r for r in results if r is not None]
        if hits:
            return min(hits)
    return None


def solve_hashcash(kind: str, challenge_input: str, hmac: str, difficulty: int,
                   memory: int = DEFAULT_SCRYPT_MEMORY, processes: Optional[int] = None,
                   max_iterations: int = MAX_ITERATIONS,
                   deadline: Optional[float] = None,
                   cancel: Optional[threading.Event] = None) -> Optional[str]:
    """
    Return the counter string solving the hashcash, or None when the iteration budget
    runs out, the monotonic `deadline` passes or `cancel` is set.
    """
    if kind not in (KIND_SHA256, KIND_SCRYPT):
        raise ValueError(f"not a hashcash kind: {kind}")
    if isinstance(difficulty, bool) or not isinstance(difficulty, int) \
            or not 0 <= difficulty <= MAX_DIFFICULTY:
        raise ValueError(f"bad difficulty: {difficulty!r}")
    if kind == KIND_SCRYPT and (memory < 2 or memory & (memory - 1)):
        raise ValueError(f"scrypt memory must be a power of two: {memory!r}")
    base, salt = (challenge_input or "") + (hmac or ""), hmac or ""
    processes = max(1, processes or os.cpu_count() or 1)
    found: Optional[int]
    if processes <= 1 or difficulty < PARALLEL_MIN_DIFFICULTY[kind]:
        found = _sequential(kind, base, salt, difficulty, memory, 0, max_iterations,
                            deadline, cancel)
    else:
        try:
            found = _parallel(kind, base, salt, difficulty, memory, processes, 0,
                              max_iterations, deadline, cancel)
        except _PoolUnavailable as exc:
            resume = exc.args[1] if len(exc.args) > 1 else 0
            logger.warning("process pool unavailable (%s); searching sequentially", exc.args[0])
            found = _sequential(kind, base, salt, difficulty, memory, resume, max_iterations,
                                deadline, cancel)
    if found is None and not _stopped(deadline, cancel):
        logger.error("%s hashcash: no solution within %d iterations", kind, max_iterations)
    return None if found is None else str(found)


def solve_bandwidth(difficulty: int) -> str:
    size = BANDWIDTH_SIZES.get(difficulty) if not isinstance(difficulty, bool) else None
    if size is None:
        raise ValueError(f"unknown NetworkBandwidth difficulty: {difficulty!r}")
    return base64.b64encode(b"\x00" * size).decode()
