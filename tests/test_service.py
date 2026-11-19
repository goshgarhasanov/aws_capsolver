"""Service hardening: API contract edge cases, task lifecycle races, key and usage stores."""

import json
import logging
import threading
import time

import pytest
from fastapi.testclient import TestClient

from aws_capsolver import tasks as tasksmod
from aws_capsolver import vision_bridge
from aws_capsolver.config import KeyStore, Settings, UsageStore
from aws_capsolver.server import create_app
from aws_capsolver.tasks import TaskManager
from aws_capsolver.waf.errors import SolverError
from aws_capsolver.waf.solver import SolveOutcome

from .conftest import jpeg_b64

KEY = "test-key-123"
OTHER = "other-key-456"
WAF = {"type": "AntiAwsWafTaskProxyLess", "websiteURL": "https://shop.example/"}


@pytest.fixture
def make_client(tmp_path):
    made = []

    def factory(solver=None, raise_server_exceptions=True, **overrides):
        keys_file = tmp_path / "keys.json"
        keys_file.write_text(json.dumps({KEY: {"balance": 42}, OTHER: {}}))
        opts = dict(keys_file=keys_file, usage_db=tmp_path / "usage.db", workers=2,
                    task_timeout=5)
        opts.update(overrides)
        settings = Settings(**opts)
        usage = UsageStore(settings.usage_db)
        manager = TaskManager(settings, usage, solver=solver) if solver else \
            TaskManager(settings, usage)
        client = TestClient(create_app(settings, manager=manager, keys=KeyStore(settings),
                                       usage=usage),
                            raise_server_exceptions=raise_server_exceptions)
        client.__enter__()
        made.append(client)
        client.manager, client.usage = manager, usage
        return client

    yield factory
    for c in made:
        c.__exit__(None, None, None)


def _create(client, task, key=KEY):
    return client.post("/createTask", json={"clientKey": key, "task": task}).json()


def _result(client, task_id, key=KEY):
    return client.post("/getTaskResult", json={"clientKey": key, "taskId": task_id}).json()


def _poll(client, task_id, key=KEY, limit=10.0):
    end = time.time() + limit
    while time.time() < end:
        res = _result(client, task_id, key)
        if res.get("status") in ("ready", "failed") or res.get("errorId"):
            return res
        time.sleep(0.03)
    raise AssertionError("task did not finish")


# --- request parsing -------------------------------------------------------------------
@pytest.mark.parametrize("content", [
    b"", b"not json", b"[1, 2]", b'"str"', b"null", b"\xff\xfe\xfa", b"{" * 100_000,
    b'{"clientKey": "test-key-123", "task": ',
])
def test_malformed_bodies(make_client, content):
    res = make_client().post("/createTask", content=content,
                             headers={"content-type": "application/json"}).json()
    assert res["errorId"] == 1 and res["errorCode"] == "ERROR_INVALID_TASK_DATA"


def test_oversized_body_is_refused(make_client):
    client = make_client(max_body_bytes=2048)
    big = {"clientKey": KEY, "task": dict(WAF, pad="x" * 5000)}
    res = client.post("/createTask", json=big).json()
    assert res["errorCode"] == "ERROR_INVALID_TASK_DATA" and "body larger" in \
        res["errorDescription"]

    def chunks():  # no Content-Length: the streamed size is enforced too
        for _ in range(10):
            yield b" " * 1024

    res = client.post("/createTask", content=chunks()).json()
    assert res["errorCode"] == "ERROR_INVALID_TASK_DATA"
    assert client.post("/createTask", content=b"{}",
                       headers={"content-length": "abc"}).status_code in (200, 400)


@pytest.mark.parametrize("body,description", [
    ({}, "clientKey is required"),
    ({"clientKey": ""}, "clientKey is required"),
    ({"clientKey": 123}, "invalid clientKey"),
    ({"clientKey": ["k"]}, "invalid clientKey"),
    ({"clientKey": "x" * 10_000}, "invalid clientKey"),
    ({"clientKey": KEY[:-1]}, "invalid clientKey"),
    ({"clientKey": KEY + " "}, "invalid clientKey"),
])
def test_client_key_errors(make_client, body, description):
    client = make_client()
    for path in ("/createTask", "/getTaskResult", "/getBalance"):
        res = client.post(path, json=body).json()
        assert res == {"errorId": 1, "errorCode": "ERROR_KEY_DENIED_ACCESS",
                       "errorDescription": description}


@pytest.mark.parametrize("task,code", [
    (None, "ERROR_INVALID_TASK_DATA"),
    ("AntiAwsWafTask", "ERROR_INVALID_TASK_DATA"),
    ({}, "ERROR_TASK_NOT_SUPPORTED"),
    ({"type": ["AntiAwsWafTask"]}, "ERROR_TASK_NOT_SUPPORTED"),
    ({"type": "antiawswaftask"}, "ERROR_TASK_NOT_SUPPORTED"),
    ({"type": "ReCaptchaV2Task"}, "ERROR_TASK_NOT_SUPPORTED"),
    (dict(WAF, websiteURL="ftp://shop.example/"), "ERROR_INVALID_TASK_DATA"),
    (dict(WAF, websiteURL="shop.example"), "ERROR_INVALID_TASK_DATA"),
    (dict(WAF, websiteURL="https://shop.example:99999/"), "ERROR_INVALID_TASK_DATA"),
    (dict(WAF, websiteURL="https://" + "a" * 5000 + ".example/"), "ERROR_INVALID_TASK_DATA"),
    (dict(WAF, websiteURL=["https://shop.example/"]), "ERROR_INVALID_TASK_DATA"),
    (dict(WAF, userAgent="UA\r\nX-Injected: 1"), "ERROR_INVALID_TASK_DATA"),
    (dict(WAF, userAgent=5), "ERROR_INVALID_TASK_DATA"),
    (dict(WAF, awsChallengeJS="javascript:alert(1)"), "ERROR_INVALID_TASK_DATA"),
    (dict(WAF, type="AntiAwsWafTask", proxy="h:99999"), "ERROR_INVALID_TASK_DATA"),
    (dict(WAF, type="AntiAwsWafTask", proxyAddress="h"), "ERROR_INVALID_TASK_DATA"),
])
def test_task_validation(make_client, task, code):
    called = []
    client = make_client(lambda *a, **k: called.append(1))
    body = {"clientKey": KEY}
    if task is not None:
        body["task"] = task
    res = client.post("/createTask", json=body).json()
    assert res["errorId"] == 1 and res["errorCode"] == code, res
    assert "taskId" not in res
    assert called == []


def test_capsolver_proxy_fields(make_client):
    seen = {}

    def solver(req, settings, deadline=None):
        seen["proxy"] = req.proxy_url
        return SolveOutcome("T", "UA")

    client = make_client(solver)
    res = _create(client, {"type": "AntiAwsWafTask", "websiteURL": "https://shop.example/",
                           "proxyType": "socks5", "proxyAddress": "5.6.7.8",
                           "proxyPort": 1080, "proxyLogin": "u", "proxyPassword": "p:w"})
    assert _poll(client, res["taskId"])["status"] == "ready"
    assert seen["proxy"] == "socks5://u:p%3Aw@5.6.7.8:1080"


# --- task ownership / ids --------------------------------------------------------------
def test_task_of_another_key_does_not_leak(make_client):
    client = make_client(lambda req, s, deadline=None: SolveOutcome("SECRET-TOKEN", "UA"))
    res = _create(client, WAF)
    assert _poll(client, res["taskId"])["solution"]["cookie"] == "SECRET-TOKEN"
    foreign = _result(client, res["taskId"], key=OTHER)
    missing = _result(client, "00000000-0000-0000-0000-000000000000", key=OTHER)
    assert foreign == missing == {"errorId": 1, "errorCode": "ERROR_TASKID_INVALID",
                                  "errorDescription": "task not found or expired"}


@pytest.mark.parametrize("task_id", [None, "", 5, ["x"], {"a": 1}])
def test_bad_task_ids(make_client, task_id):
    res = _result(make_client(), task_id)
    assert res["errorCode"] == "ERROR_TASKID_INVALID"


# --- solver outcomes -------------------------------------------------------------------
def test_non_capsolver_codes_are_normalised(make_client):
    def solver(req, settings, deadline=None):
        raise SolverError("HTTP_403", "upstream said no")

    client = make_client(solver)
    done = _poll(client, _create(client, WAF)["taskId"])
    assert done["errorCode"] == "ERROR_CAPTCHA_UNSOLVABLE"


def test_crash_is_reported_and_logs_are_redacted(make_client, caplog):
    def solver(req, settings, deadline=None):
        raise RuntimeError(f"boom via {req.proxy_url}")

    from aws_capsolver.waf.http import install_log_redaction
    install_log_redaction([caplog.handler])
    client = make_client(solver)
    with caplog.at_level(logging.INFO):
        res = _create(client, dict(WAF, type="AntiAwsWafTask",
                                   proxy="http://user:hunter2@1.2.3.4:8080"))
        done = _poll(client, res["taskId"])
    assert done["errorCode"] == "ERROR_CAPTCHA_UNSOLVABLE"
    assert done["errorDescription"] == "internal error: RuntimeError"
    assert "boom via" in caplog.text and "hunter2" not in caplog.text


def test_empty_outcome_is_a_failure(make_client):
    client = make_client(lambda req, s, deadline=None: SolveOutcome(None, "UA"))
    assert _poll(client, _create(client, WAF)["taskId"])["status"] == "failed"


def test_voucher_only_outcome(make_client):
    client = make_client(lambda req, s, deadline=None: SolveOutcome(None, "UA", voucher="V"))
    done = _poll(client, _create(client, WAF)["taskId"])
    assert done["solution"] == {"cookie": "", "userAgent": "UA", "captchaVoucher": "V"}


def test_deadline_is_passed_and_counts_queue_time(make_client):
    gate, seen = threading.Event(), []

    def solver(req, settings, deadline=None):
        seen.append(deadline)
        gate.wait(5)
        return SolveOutcome("T", "UA")

    client = make_client(solver, workers=1, task_timeout=0.5)
    first = _create(client, WAF)
    second = _create(client, WAF)  # queued behind the first one
    time.sleep(0.7)
    gate.set()
    assert _poll(client, first["taskId"])["status"] == "ready"
    late = _poll(client, second["taskId"])
    assert late["errorCode"] == "ERROR_TASK_TIMEOUT"
    assert len(seen) == 1  # the queued task never reached the solver
    assert seen[0] - time.monotonic() < 0.5


def test_watchdog_fails_a_stuck_task_and_ignores_its_late_result(make_client, monkeypatch):
    monkeypatch.setattr(tasksmod, "WATCHDOG_GRACE", 0.0)
    gate = threading.Event()

    def solver(req, settings, deadline=None):
        gate.wait(5)
        return SolveOutcome("LATE", "UA")

    client = make_client(solver, task_timeout=0.3)
    task_id = _create(client, WAF)["taskId"]
    time.sleep(0.5)
    res = _result(client, task_id)
    assert res["errorCode"] == "ERROR_TASK_TIMEOUT"
    gate.set()
    time.sleep(0.2)
    assert _result(client, task_id)["errorCode"] == "ERROR_TASK_TIMEOUT"
    usage = client.post("/getBalance", json={"clientKey": KEY}).json()["usage"]
    assert usage[WAF["type"]] == {"created": 1, "solved": 0, "failed": 1}


def test_ttl_cleanup_forgets_finished_tasks(make_client):
    client = make_client(lambda req, s, deadline=None: SolveOutcome("T", "UA"), task_ttl=0.2)
    task_id = _create(client, WAF)["taskId"]
    assert _poll(client, task_id)["status"] == "ready"
    time.sleep(0.3)
    client.manager.cleanup()
    assert _result(client, task_id)["errorCode"] == "ERROR_TASKID_INVALID"


def test_slot_limit_is_exact_under_concurrency(make_client):
    gate = threading.Event()
    client = make_client(lambda req, s, deadline=None: gate.wait(5) and SolveOutcome("T", "U"),
                         max_pending=5, workers=8)
    results, errors = [], []

    def safe_create():
        try:
            results.append(client.manager.create(KEY, WAF))
        except SolverError as exc:
            errors.append(exc.code)

    threads = [threading.Thread(target=safe_create) for _ in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(results) == 5 and errors == ["ERROR_NO_SLOT_AVAILABLE"] * 15
    gate.set()


def test_snapshots_are_consistent(make_client):
    client = make_client(lambda req, s, deadline=None: SolveOutcome("T", "UA"))
    task = client.manager.create(KEY, WAF)
    live = client.manager._tasks[task.id]
    assert task is not live
    for _ in range(200):
        snap = client.manager.get(KEY, task.id)
        if snap.status == "ready":
            assert snap.solution == {"cookie": "T", "userAgent": "UA"}
            break
        time.sleep(0.01)
    else:
        raise AssertionError("never became ready")


# --- shutdown --------------------------------------------------------------------------
def test_shutdown_cancels_running_and_refuses_new(make_client):
    started, seen = threading.Event(), {}

    def solver(req, settings, deadline=None):
        seen["cancel"] = settings.cancel
        started.set()
        settings.cancel.wait(5)
        raise SolverError("ERROR_SERVICE_UNAVAILABLE", "aborted")

    client = make_client(solver)
    task_id = _create(client, WAF)["taskId"]
    assert started.wait(2)
    client.manager.shutdown()
    assert seen["cancel"].is_set()
    res = _result(client, task_id)
    assert res["errorCode"] == "ERROR_SERVICE_UNAVAILABLE"
    assert _create(client, WAF)["errorCode"] == "ERROR_SERVICE_UNAVAILABLE"
    assert client.get("/health").json()["ok"] is False


# --- classification --------------------------------------------------------------------
def test_classification_crash_and_timeout(make_client, monkeypatch):
    images = [jpeg_b64() for _ in range(9)]
    task = {"type": "AwsWafClassification", "images": images, "question": "aws:grid:bed"}

    def crash(images, target, count=None):
        raise RuntimeError("cuda exploded")

    monkeypatch.setattr(vision_bridge, "classify_grid", crash)
    client = make_client(classify_timeout=0.3)
    res = _create(client, task)
    assert res["errorId"] == 1 and res["status"] == "failed"
    assert res["errorCode"] == "ERROR_CAPTCHA_UNSOLVABLE"
    assert _result(client, res["taskId"])["status"] == "failed"

    monkeypatch.setattr(vision_bridge, "classify_grid",
                        lambda images, target, count=None: time.sleep(1) or [1])
    res = _create(client, task)
    assert res["errorCode"] == "ERROR_TASK_TIMEOUT"


@pytest.mark.parametrize("task", [
    {"images": [jpeg_b64()], "question": "aws:grid"},
    {"images": [jpeg_b64()], "question": "aws:grid:"},
    {"images": [], "question": "aws:grid:bed"},
    {"images": "abc", "question": "aws:grid:bed"},
    {"images": [jpeg_b64()] * 17, "question": "aws:grid:bed"},
    {"images": [123], "question": "aws:grid:bed"},
    {"images": ["A" * (3 * 1024 * 1024)], "question": "aws:grid:bed"},
    {"images": [jpeg_b64()], "question": 5},
])
def test_classification_validation(make_client, task, monkeypatch):
    monkeypatch.setattr(vision_bridge, "classify_grid", lambda *a, **k: [0])
    res = _create(make_client(max_body_bytes=8 * 1024 * 1024),
                  dict(task, type="AwsWafClassification"))
    assert res["errorCode"] == "ERROR_INVALID_TASK_DATA"


def test_classification_out_of_range_tiles(make_client, monkeypatch):
    monkeypatch.setattr(vision_bridge, "classify_grid", lambda *a, **k: [0, 9])
    res = _create(make_client(), {"type": "AwsWafClassification",
                                  "images": [jpeg_b64()] * 9, "question": "aws:grid:bed"})
    assert res["errorCode"] == "ERROR_CAPTCHA_UNSOLVABLE"


# --- unexpected errors -----------------------------------------------------------------
def test_unexpected_exception_returns_json(make_client, monkeypatch):
    client = make_client(raise_server_exceptions=False)

    def explode(*a, **k):
        raise ZeroDivisionError("bug")

    monkeypatch.setattr(client.manager, "get", explode)
    res = client.post("/getTaskResult", json={"clientKey": KEY, "taskId": "x"})
    assert res.status_code == 500
    assert res.json() == {"errorId": 1, "errorCode": "ERROR_SERVICE_UNAVAILABLE",
                          "errorDescription": "internal error"}


def test_ssrf_guard_end_to_end(make_client):
    """The real solver refuses an internal websiteURL before any network access."""
    client = make_client()
    res = _create(client, dict(WAF, websiteURL="http://127.0.0.1:8790/getBalance"))
    done = _poll(client, res["taskId"])
    assert done["errorCode"] == "ERROR_INVALID_TASK_DATA"
    assert "AWSCS_ALLOW_PRIVATE" in done["errorDescription"]


# --- key store -------------------------------------------------------------------------
def test_keystore_reload_and_half_written_file(tmp_path):
    path = tmp_path / "keys.json"
    path.write_text(json.dumps(["k1"]))
    store = KeyStore(Settings(keys_file=path))
    assert store.lookup("k1") == {} and store.lookup("k2") is None
    path.write_text('{"k2": {"bal')  # caught mid-write: keep the old keys
    assert store.lookup("k1") == {}
    path.write_text(json.dumps({"k2": {"balance": "7"}, "k3": "junk"}))
    assert store.lookup("k1") is None and store.lookup("k3") == {}
    assert store.balance("k2") == 7.0
    path.unlink()
    assert store.lookup("k2") is None


def test_keystore_env_keys_open_access_and_bad_values(tmp_path, monkeypatch):
    monkeypatch.setenv("AWSCS_KEYS", " a , ,b ")
    monkeypatch.setenv("AWSCS_KEYS_FILE", str(tmp_path / "none.json"))
    settings = Settings.from_env()
    store = KeyStore(settings)
    assert store.lookup("a") == store.lookup("b") == {"name": "env"}
    assert store.lookup("") is None and store.lookup("c") is None
    assert store.lookup(None) is None and store.lookup(1) is None
    assert store.lookup("\ud800") is None  # lone surrogate from a JSON escape
    settings.open_access = True
    assert store.lookup("anything") == {} and store.lookup("") is None
    (tmp_path / "k.json").write_text(json.dumps({"x": {"balance": "nan"}, "y": {"balance": []}}))
    s2 = KeyStore(Settings(keys_file=tmp_path / "k.json", default_balance=5))
    assert s2.balance("x") == 5 and s2.balance("y") == 5


# --- usage store -----------------------------------------------------------------------
def test_usage_store_is_thread_safe(tmp_path):
    store = UsageStore(tmp_path / "u.db")

    def work():
        for _ in range(50):
            store.bump("k", "T", "created")
            store.bump("k", "T", "solved")
            store.stats("k")

    threads = [threading.Thread(target=work) for _ in range(16)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert store.stats("k") == {"T": {"created": 800, "solved": 800, "failed": 0}}
    with pytest.raises(ValueError):
        store.bump("k", "T", "created; DROP TABLE usage")
    store.close()
    store.bump("k", "T", "created")  # after close: ignored, never raises
    assert store.stats("k") == {}
    store.close()


# --- settings --------------------------------------------------------------------------
def test_settings_from_env(monkeypatch):
    monkeypatch.setenv("AWSCS_ALLOW_PRIVATE", "yes")
    monkeypatch.setenv("AWSCS_TASK_TIMEOUT", "-5")
    monkeypatch.setenv("AWSCS_HTTP_TIMEOUT", "nan")
    monkeypatch.setenv("AWSCS_WORKERS", "zero")
    monkeypatch.setenv("AWSCS_POW_PROCESSES", "-3")
    monkeypatch.setenv("AWSCS_MAX_BODY_BYTES", "10")
    monkeypatch.setenv("AWSCS_VERIFY_TOKEN", "off")
    s = Settings.from_env()
    assert s.allow_private is True
    assert s.task_timeout == 120.0 and s.http_timeout == 20.0
    assert s.workers == 8 and s.pow_processes is None
    assert s.max_body_bytes == 1024 and s.verify_token is False
    monkeypatch.delenv("AWSCS_ALLOW_PRIVATE")
    assert Settings.from_env().allow_private is False


# --- classification admission control -----------------------------------------------------
def _cls_task():
    return {"type": "AwsWafClassification", "images": [jpeg_b64()] * 9,
            "question": "aws:grid:bed"}


def _blocking_vision(monkeypatch, gate, calls):
    def classify_grid(images, target, count=None):
        calls.append(time.monotonic())
        gate.wait(10)
        return [1]

    monkeypatch.setattr(vision_bridge, "classify_grid", classify_grid)


def _wait_for(predicate, limit=5.0):
    end = time.monotonic() + limit
    while time.monotonic() < end:
        if predicate():
            return True
        time.sleep(0.01)
    return False


def test_classify_queue_full_answers_no_slot_immediately(make_client, monkeypatch):
    gate, calls = threading.Event(), []
    _blocking_vision(monkeypatch, gate, calls)
    client = make_client(vision_concurrency=1, max_classify_queue=1, classify_timeout=10)
    results = []
    threads = [threading.Thread(target=lambda: results.append(_create(client, _cls_task())))
               for _ in range(2)]
    for t in threads:
        t.start()
    assert _wait_for(lambda: client.manager.stats()["classify_queued"] == 1
                     and client.manager.stats()["classify_running"] == 1)
    t0 = time.monotonic()
    full = _create(client, _cls_task())
    assert time.monotonic() - t0 < 1.0
    assert full == {"errorId": 1, "errorCode": "ERROR_NO_SLOT_AVAILABLE",
                    "errorDescription": "classification queue is full"}
    load = client.get("/health").json()["load"]
    assert load["classify_running"] == 1 and load["classify_queued"] == 1
    assert load["classify_capacity"] == 2
    gate.set()
    for t in threads:
        t.join(10)
    assert [r["solution"] for r in results] == [{"objects": [1]}] * 2
    stats = client.manager.stats()
    assert stats["classify_queued"] == stats["classify_running"] == 0


def test_stale_queued_classification_never_reaches_vision(make_client, monkeypatch):
    calls = []

    def slow(images, target, count=None):
        calls.append(1)
        time.sleep(0.6)
        return [1]

    monkeypatch.setattr(vision_bridge, "classify_grid", slow)
    client = make_client(vision_concurrency=1, classify_timeout=0.3)
    results = []
    threads = [threading.Thread(target=lambda: results.append(_create(client, _cls_task())))
               for _ in range(3)]
    for t in threads:
        t.start()
        time.sleep(0.02)
    for t in threads:
        t.join(10)
    assert [r["errorCode"] for r in results] == ["ERROR_TASK_TIMEOUT"] * 3
    assert _wait_for(lambda: client.manager.stats()["classify_running"] == 0, 3)
    time.sleep(0.1)
    assert len(calls) == 1  # the two queued ones expired while waiting and were skipped
    assert client.manager.stats()["classify_queued"] == 0


def test_api_stays_responsive_under_classification_backlog(make_client, monkeypatch):
    gate, calls = threading.Event(), []
    _blocking_vision(monkeypatch, gate, calls)
    # more blocked classifications than Starlette's default threadpool (40 threads)
    client = make_client(vision_concurrency=1, max_classify_queue=60, classify_timeout=10)
    waf = make_client(lambda req, s, deadline=None: SolveOutcome("T", "UA"))
    threads = [threading.Thread(target=_create, args=(client, _cls_task())) for _ in range(45)]
    for t in threads:
        t.start()
    try:
        assert _wait_for(lambda: client.manager.stats()["classify_queued"] == 44)
        t0 = time.monotonic()
        assert client.post("/getBalance", json={"clientKey": KEY}).json()["errorId"] == 0
        assert _result(client, "missing")["errorCode"] == "ERROR_TASKID_INVALID"
        assert client.get("/health").json()["load"]["classify_queued"] == 44
        assert time.monotonic() - t0 < 2.0
        # WAF task creation is not stuck behind vision (fake solver: no network)
        assert _poll(waf, _create(waf, WAF)["taskId"])["status"] == "ready"
    finally:
        gate.set()
        for t in threads:
            t.join(15)
    assert len(calls) == 45


def test_classification_validation_does_not_take_a_slot(make_client, monkeypatch):
    gate, calls = threading.Event(), []
    _blocking_vision(monkeypatch, gate, calls)
    client = make_client(vision_concurrency=1, max_classify_queue=0)
    res = _create(client, {"type": "AwsWafClassification", "images": ["%%%"],
                           "question": "aws:grid:bed"})
    assert res["errorCode"] == "ERROR_INVALID_TASK_DATA" and "taskId" not in res
    assert client.manager.stats()["classify_queued"] == 0
    gate.set()


def test_classify_settings_from_env(monkeypatch):
    monkeypatch.setenv("AWSCS_VISION_CONCURRENCY", "2")
    s = Settings.from_env()
    assert s.vision_concurrency == 2 and s.classify_queue_limit == 16
    monkeypatch.setenv("AWSCS_MAX_CLASSIFY_QUEUE", "3")
    assert Settings.from_env().classify_queue_limit == 3
    monkeypatch.setenv("AWSCS_MAX_CLASSIFY_QUEUE", "0")
    assert Settings.from_env().classify_queue_limit == 0
    monkeypatch.setenv("AWSCS_VISION_CONCURRENCY", "0")
    monkeypatch.setenv("AWSCS_MAX_CLASSIFY_QUEUE", "junk")
    s = Settings.from_env()
    assert s.vision_concurrency == 1 and s.classify_queue_limit == 8
