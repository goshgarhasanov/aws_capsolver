import base64
import hashlib

import pytest

from aws_capsolver.waf import pow as powlib


def _reference(kind, base, salt, bits, memory=128):
    """Independent, bit-string based reference search."""
    for i in range(10_000_000):
        data = f"{base}{i}".encode()
        if kind == "sha256":
            d = hashlib.sha256(data).digest()
        else:
            d = hashlib.scrypt(data, salt=salt.encode(), n=memory, r=8, p=1, dklen=16)
        if "".join(f"{b:08b}" for b in d).startswith("0" * bits):
            return str(i)


@pytest.mark.parametrize("digest,bits,ok", [
    (bytes([0x00, 0x0F]), 12, True),
    (bytes([0x00, 0x1F]), 12, False),
    (bytes([0x00, 0x00, 0x80]), 16, True),
    (bytes([0x80]), 0, True),
    (bytes([0x7F]), 1, True),
    (bytes([0x7F]), 2, False),
])
def test_leading_zero_bits(digest, bits, ok):
    assert powlib.leading_zero_bits_ok(digest, bits) is ok


def test_sha256_known_vector():
    # sha256("abc" + "hm" + "39") style vectors: compare against the reference search
    expected = _reference("sha256", "inputhmac", "hmac", 12)
    assert powlib.solve_hashcash("sha256", "input", "hmac", 12, processes=1) == expected
    digest = hashlib.sha256(f"inputhmac{expected}".encode()).digest()
    assert digest[0] == 0 and digest[1] >> 4 == 0


def test_sha256_parallel_matches_sequential():
    seq = powlib.solve_hashcash("sha256", "challenge-x", "", 17, processes=1)
    par = powlib.solve_hashcash("sha256", "challenge-x", "", 17, processes=4)
    assert seq == par == _reference("sha256", "challenge-x", "", 17)


def test_scrypt_vector_and_parallel():
    expected = _reference("scrypt", "in" + "salt", "salt", 6)
    assert powlib.solve_hashcash("scrypt", "in", "salt", 6, processes=1) == expected
    assert powlib.solve_hashcash("scrypt", "in", "salt", 9, processes=4) == \
        _reference("scrypt", "insalt", "salt", 9)


def test_hashcash_budget_exhausted():
    assert powlib.solve_hashcash("sha256", "x", "", 40, processes=1, max_iterations=50) is None


def test_bandwidth_sizes():
    assert len(base64.b64decode(powlib.solve_bandwidth(1))) == 1024
    assert len(base64.b64decode(powlib.solve_bandwidth(3))) == 0x19000
    with pytest.raises(ValueError):
        powlib.solve_bandwidth(9)


def test_bad_difficulty():
    with pytest.raises(ValueError):
        powlib.solve_hashcash("sha256", "x", "", -1)


def test_hashcash_deadline():
    import time
    assert powlib.solve_hashcash("sha256", "x", "", 60, processes=2,
                                 deadline=time.monotonic() - 1) is None


# --- cancellation, deadlines, pool robustness --------------------------------------------
import multiprocessing  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402
from concurrent.futures.process import BrokenProcessPool  # noqa: E402

from aws_capsolver.waf.challenge import compute_solution  # noqa: E402
from aws_capsolver.waf.errors import SolverError  # noqa: E402


@pytest.mark.parametrize("processes", [1, 2])
def test_deadline_interrupts_a_running_search(processes):
    t0 = time.monotonic()
    found = powlib.solve_hashcash("sha256", "x", "", 60, processes=processes,
                                  deadline=t0 + 0.4)
    assert found is None
    assert time.monotonic() - t0 < 3.0


@pytest.mark.parametrize("processes", [1, 2])
def test_cancel_event_interrupts_a_running_search(processes):
    cancel = threading.Event()
    threading.Timer(0.3, cancel.set).start()
    t0 = time.monotonic()
    assert powlib.solve_hashcash("sha256", "x", "", 60, processes=processes,
                                 cancel=cancel) is None
    assert time.monotonic() - t0 < 3.0


def test_compute_solution_reports_why_it_stopped():
    with pytest.raises(SolverError) as err:
        compute_solution("sha256", {"input": "x"}, 60, 128, 1, deadline=time.monotonic() - 1)
    assert err.value.code == "ERROR_TASK_TIMEOUT"
    cancel = threading.Event()
    cancel.set()
    with pytest.raises(SolverError) as err:
        compute_solution("sha256", {"input": "x"}, 60, 128, 1, cancel=cancel)
    assert err.value.code == "ERROR_SERVICE_UNAVAILABLE"


def test_pool_unavailable_falls_back_to_sequential(monkeypatch):
    def broken(_):
        raise powlib._PoolUnavailable("no processes here")

    monkeypatch.setattr(powlib, "_get_pool", broken)
    assert powlib.solve_hashcash("sha256", "challenge-x", "", 17, processes=4) == \
        _reference("sha256", "challenge-x", "", 17)


def test_pool_breaking_mid_search_resumes_sequentially(monkeypatch):
    class FlakyPool:
        """Accepts the first batch (no hit), then breaks like a crashed worker."""
        calls = 0
        _broken = False

        def submit(self, fn, *args):
            FlakyPool.calls += 1
            if FlakyPool.calls > 2:
                raise BrokenProcessPool("worker died")
            from concurrent.futures import Future
            f = Future()
            f.set_result(fn(*args))
            return f

        def shutdown(self, **kw):
            pass

    monkeypatch.setitem(powlib.CHUNK, "sha256", 1000)
    monkeypatch.setattr(powlib, "_get_pool", lambda n: FlakyPool())
    expected = _reference("sha256", "challenge-x", "", 17)
    assert int(expected) > 2000  # the answer lies beyond the batch the pool handled
    assert powlib.solve_hashcash("sha256", "challenge-x", "", 17, processes=2) == expected


def test_pool_is_shared_and_shuts_down_cleanly():
    powlib.solve_hashcash("sha256", "challenge-x", "", 17, processes=2)
    first = powlib._pool
    powlib.solve_hashcash("sha256", "challenge-y", "", 16, processes=2)
    assert powlib._pool is first and first is not None
    powlib.shutdown_pool(wait=True)
    assert powlib._pool is None
    assert multiprocessing.active_children() == []  # no zombie workers left behind


def test_pool_never_uses_fork():
    assert powlib._mp_context().get_start_method() in ("forkserver", "spawn")


def test_concurrent_searches_share_the_pool():
    results, errors = [], []

    def run(seed):
        try:
            results.append((seed, powlib.solve_hashcash("sha256", seed, "", 16, processes=2)))
        except Exception as exc:  # pragma: no cover - reported below
            errors.append(exc)

    threads = [threading.Thread(target=run, args=(f"seed-{i}",)) for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(60)
    assert not errors
    for seed, found in results:
        assert found == _reference("sha256", seed, "", 16)
    powlib.shutdown_pool(wait=True)


@pytest.mark.parametrize("difficulty", [True, 65, "8", None, 1.5])
def test_bad_difficulty_types(difficulty):
    with pytest.raises(ValueError):
        powlib.solve_hashcash("sha256", "x", "", difficulty)


def test_bad_scrypt_memory_and_kind():
    with pytest.raises(ValueError):
        powlib.solve_hashcash("scrypt", "x", "", 4, memory=100)
    with pytest.raises(ValueError):
        powlib.solve_hashcash("md5", "x", "", 4)
    with pytest.raises(ValueError):
        powlib.solve_bandwidth(True)
