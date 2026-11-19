import json
import threading
import time

import pytest
from fastapi.testclient import TestClient

from aws_capsolver import vision_bridge
from aws_capsolver.config import KeyStore, Settings, UsageStore
from aws_capsolver.server import create_app
from aws_capsolver.tasks import TaskManager
from aws_capsolver.waf.errors import SolverError
from aws_capsolver.waf.http import parse_proxy
from aws_capsolver.waf.solver import SolveOutcome

from .conftest import jpeg_b64

KEY = "test-key-123"


@pytest.fixture
def make_client(tmp_path):
    clients = []

    def factory(solver=None, **overrides):
        keys_file = tmp_path / "keys.json"
        keys_file.write_text(json.dumps({KEY: {"name": "t", "balance": 42}}))
        settings = Settings(keys_file=keys_file, usage_db=tmp_path / "usage.db", workers=2,
                            task_timeout=5, **overrides)
        usage = UsageStore(settings.usage_db)
        manager = TaskManager(settings, usage, solver=solver) if solver else TaskManager(settings, usage)
        client = TestClient(create_app(settings, manager=manager, keys=KeyStore(settings),
                                       usage=usage))
        client.__enter__()
        clients.append(client)
        return client

    yield factory
    for c in clients:
        c.__exit__(None, None, None)


def _poll(client, task_id, key=KEY, limit=5.0):
    end = time.time() + limit
    while time.time() < end:
        res = client.post("/getTaskResult", json={"clientKey": key, "taskId": task_id}).json()
        if res.get("status") in ("ready", "failed") or res.get("errorId"):
            return res
        time.sleep(0.05)
    raise AssertionError("task did not finish")


def test_key_denied(make_client):
    client = make_client()
    res = client.post("/getBalance", json={"clientKey": "nope"}).json()
    assert res == {"errorId": 1, "errorCode": "ERROR_KEY_DENIED_ACCESS",
                   "errorDescription": "invalid clientKey"}


def test_balance(make_client):
    res = make_client().post("/getBalance", json={"clientKey": KEY}).json()
    assert res["errorId"] == 0 and res["balance"] == 42


def test_non_json_body(make_client):
    res = make_client().post("/createTask", content=b"not json").json()
    assert res["errorCode"] == "ERROR_INVALID_TASK_DATA"


def test_unknown_task_type(make_client):
    res = make_client().post("/createTask", json={"clientKey": KEY, "task": {"type": "X"}}).json()
    assert res["errorCode"] == "ERROR_TASK_NOT_SUPPORTED"


def test_waf_task_lifecycle(make_client):
    seen = {}
    gate = threading.Event()

    def solver(req, settings, deadline=None):
        seen["req"] = req
        gate.wait(2)
        return SolveOutcome(token="TOKEN", user_agent=req.user_agent or "UA")

    client = make_client(solver)
    res = client.post("/createTask", json={"clientKey": KEY, "task": {
        "type": "AntiAwsWafTask", "websiteURL": "https://shop.example/",
        "proxy": "http:1.2.3.4:8080:user:p@ss", "userAgent": "MyUA",
        "awsKey": "k", "awsIv": "i", "awsContext": "c"}}).json()
    assert res["errorId"] == 0 and res["taskId"]
    pending = client.post("/getTaskResult", json={"clientKey": KEY, "taskId": res["taskId"]}).json()
    assert pending == {"errorId": 0, "taskId": res["taskId"], "status": "processing"}
    gate.set()
    done = _poll(client, res["taskId"])
    assert done["status"] == "ready"
    assert done["solution"] == {"cookie": "TOKEN", "userAgent": "MyUA"}
    assert seen["req"].proxy_url == "http://user:p%40ss@1.2.3.4:8080"
    assert seen["req"].aws_key == "k"
    usage = client.post("/getBalance", json={"clientKey": KEY}).json()["usage"]
    assert usage["AntiAwsWafTask"] == {"created": 1, "solved": 1, "failed": 0}


def test_waf_task_failure_code(make_client):
    def solver(req, settings, deadline=None):
        raise SolverError("ERROR_PROXY_CONNECT_REFUSED", "proxy down")

    client = make_client(solver)
    res = client.post("/createTask", json={"clientKey": KEY, "task": {
        "type": "AntiAwsWafTaskProxyLess", "websiteURL": "https://shop.example/"}}).json()
    done = _poll(client, res["taskId"])
    assert done["status"] == "failed" and done["errorCode"] == "ERROR_PROXY_CONNECT_REFUSED"


def test_waf_task_requires_proxy_and_url(make_client):
    client = make_client(lambda *a, **k: None)
    res = client.post("/createTask", json={"clientKey": KEY, "task": {
        "type": "AntiAwsWafTask", "websiteURL": "https://shop.example/"}}).json()
    assert res["errorCode"] == "ERROR_INVALID_TASK_DATA"
    res = client.post("/createTask", json={"clientKey": KEY, "task": {
        "type": "AntiAwsWafTaskProxyLess"}}).json()
    assert res["errorCode"] == "ERROR_INVALID_TASK_DATA"


def test_task_id_invalid_and_isolated_per_key(make_client, tmp_path):
    client = make_client(lambda req, s, deadline=None: SolveOutcome("T", "UA"))
    res = client.post("/getTaskResult", json={"clientKey": KEY, "taskId": "missing"}).json()
    assert res["errorCode"] == "ERROR_TASKID_INVALID"
    res = client.post("/getTaskResult", json={"clientKey": KEY, "taskId": 5}).json()
    assert res["errorCode"] == "ERROR_TASKID_INVALID"


def test_classification_sync(make_client, monkeypatch):
    monkeypatch.setattr(vision_bridge, "classify_grid", lambda images, target: [3, 1, 3])
    monkeypatch.setattr(vision_bridge, "solve_toycar", lambda image: (120, 64))
    client = make_client()
    images = [jpeg_b64() for _ in range(9)]
    res = client.post("/createTask", json={"clientKey": KEY, "task": {
        "type": "AwsWafClassification", "images": images, "question": "aws:grid:bed"}}).json()
    assert res["errorId"] == 0 and res["status"] == "ready"
    assert res["solution"] == {"objects": [1, 3]}
    again = client.post("/getTaskResult", json={"clientKey": KEY, "taskId": res["taskId"]}).json()
    assert again["solution"] == {"objects": [1, 3]}
    res = client.post("/createTask", json={"clientKey": KEY, "task": {
        "type": "AwsWafClassification", "images": images[:1],
        "question": "aws:toycarcity:carcity"}}).json()
    assert res["solution"] == {"box": [120, 64]}


def test_classification_errors(make_client, monkeypatch):
    monkeypatch.setattr(vision_bridge, "_module", False)
    client = make_client()
    base = {"clientKey": KEY}
    bad = client.post("/createTask", json=dict(base, task={
        "type": "AwsWafClassification", "images": ["%%%"], "question": "aws:grid:bed"})).json()
    assert bad["errorCode"] in ("ERROR_INVALID_TASK_DATA", "ERROR_CAPTCHA_UNSOLVABLE")
    res = client.post("/createTask", json=dict(base, task={
        "type": "AwsWafClassification", "images": [jpeg_b64()], "question": "grid"})).json()
    assert res["errorCode"] == "ERROR_INVALID_TASK_DATA"
    res = client.post("/createTask", json=dict(base, task={
        "type": "AwsWafClassification", "images": [jpeg_b64()],
        "question": "aws:grid:bed"})).json()
    assert res["errorCode"] == "ERROR_CAPTCHA_UNSOLVABLE" and res["status"] == "failed"
    res = client.post("/createTask", json=dict(base, task={
        "type": "AwsWafClassification", "images": [jpeg_b64()],
        "question": "aws:other:x"})).json()
    assert res["errorCode"] == "ERROR_TASK_NOT_SUPPORTED"


def test_open_access_and_slot_limit(make_client):
    gate = threading.Event()
    client = make_client(lambda req, s, deadline=None: gate.wait(2) and SolveOutcome("T", "UA"),
                         open_access=True, max_pending=1)
    task = {"type": "AntiAwsWafTaskProxyLess", "websiteURL": "https://a.example/"}
    first = client.post("/createTask", json={"clientKey": "anything", "task": task}).json()
    assert first["errorId"] == 0
    second = client.post("/createTask", json={"clientKey": "anything", "task": task}).json()
    assert second["errorCode"] == "ERROR_NO_SLOT_AVAILABLE"
    gate.set()
    assert _poll(client, first["taskId"], key="anything")["status"] == "ready"


@pytest.mark.parametrize("raw,expected", [
    ("http://u:p@h:1", "http://u:p@h:1"),
    ("socks5://h:1080", "socks5://h:1080"),
    ("h:3128", "http://h:3128"),
    ("h:3128:u:p", "http://u:p@h:3128"),
    ("socks5:h:1080:u:p", "socks5://u:p@h:1080"),
    ({"proxyType": "http", "proxyAddress": "h", "proxyPort": 8, "proxyLogin": "u",
      "proxyPassword": "p"}, "http://u:p@h:8"),
])
def test_parse_proxy(raw, expected):
    assert parse_proxy(raw) == expected


@pytest.mark.parametrize("raw", ["h", "ftp://h:1", "h:port", "a:b:c"])
def test_parse_proxy_rejects(raw):
    with pytest.raises(SolverError):
        parse_proxy(raw)


def test_parse_proxy_password_with_colon():
    assert parse_proxy("h:1:u:pa:ss") == "http://u:pa%3Ass@h:1"
