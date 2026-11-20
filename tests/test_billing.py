"""Token billing: ledger semantics, pricing file, task manager / API integration, CLI."""
import json
import threading
import time

import pytest
from fastapi.testclient import TestClient

from aws_capsolver import vision_bridge
from aws_capsolver.__main__ import main as cli_main
from aws_capsolver.billing import (DEFAULT_PRICES, Billing, InsufficientTokens,
                                   load_pricing)
from aws_capsolver.config import KeyStore, Settings, UsageStore
from aws_capsolver.server import create_app
from aws_capsolver.tasks import TaskManager
from aws_capsolver.waf.errors import SolverError
from aws_capsolver.waf.solver import SolveOutcome

from .conftest import jpeg_b64

KEY = "bill-key-1"


@pytest.fixture
def bill(tmp_path):
    b = Billing(tmp_path / "b.db", opening_grant=lambda key: 100)
    yield b
    b.close()


# ------------------------------------------------------------------ ledger
def test_opening_grant_charge_refund(bill):
    assert bill.balance("k") == 100
    price = bill.charge("k", "AwsWafClassification", "t1")
    assert price == DEFAULT_PRICES["AwsWafClassification"]
    assert bill.balance("k") == 100 - price
    assert bill.refund("t1") == price
    assert bill.refund("t1") == 0          # idempotent
    assert bill.refund("unknown") == 0
    assert bill.balance("k") == 100
    kinds = [h["kind"] for h in bill.history("k")]
    assert kinds == ["refund", "charge", "grant"]


def test_insufficient_tokens_changes_nothing(bill):
    bill.topup("k", tokens=-95)            # correction down to 5
    with pytest.raises(InsufficientTokens) as exc:
        bill.charge("k", "AntiAwsWafTask", "t2")
    assert exc.value.balance == 5 and exc.value.price == DEFAULT_PRICES["AntiAwsWafTask"]
    assert bill.balance("k") == 5
    assert all(h["kind"] != "charge" for h in bill.history("k"))


def test_topup_usd_and_validation(bill):
    assert bill.topup("k", usd=10) == 100 + 10 * bill.tokens_per_usd
    assert bill.history("k", 1)[0]["note"] == "$10 package"
    for bad in ({"tokens": 0}, {"usd": -1}, {"usd": float("nan")}, {}, {"tokens": 5, "usd": 1}):
        with pytest.raises(ValueError):
            bill.topup("k", **bad)
    with pytest.raises(InsufficientTokens):
        bill.topup("k", tokens=-10 ** 9)


def test_module_price_and_free_types(tmp_path):
    b = Billing(tmp_path / "m.db", prices={"ImageToTextTask": 2, "ImageToTextTask:bls_poland": 1},
                opening_grant=lambda key: 10)
    assert b.charge("k", "ImageToTextTask", "a", module="bls_poland") == 1
    assert b.charge("k", "ImageToTextTask", "b", module="common") == 2
    assert b.charge("k", "UnpricedType", "c") == 0
    assert b.balance("k") == 7
    b.close()


def test_concurrent_charges_never_overdraw(bill):
    ok, failed = [], []

    def worker(i):
        try:
            bill.charge("k", "AwsWafClassification", f"c{i}")
            ok.append(i)
        except InsufficientTokens:
            failed.append(i)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(40)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    price = DEFAULT_PRICES["AwsWafClassification"]
    assert len(ok) == 100 // price and len(failed) == 40 - 100 // price
    assert bill.balance("k") == 100 - len(ok) * price


def test_persistent_across_instances(tmp_path):
    a = Billing(tmp_path / "p.db", opening_grant=lambda key: 50)
    a.topup("k", tokens=25)
    a.close()
    b = Billing(tmp_path / "p.db", opening_grant=lambda key: 999)  # grant only for new keys
    assert b.balance("k") == 75
    b.close()


def test_pricing_file(tmp_path):
    f = tmp_path / "pricing.json"
    f.write_text(json.dumps({"prices": {"AwsWafClassification": 7, "Bad": -1, "Also": "x"},
                             "tokens_per_usd": 500}))
    prices, per_usd = load_pricing(f)
    assert prices["AwsWafClassification"] == 7 and "Bad" not in prices and "Also" not in prices
    assert prices["AntiAwsWafTask"] == DEFAULT_PRICES["AntiAwsWafTask"] and per_usd == 500
    f.write_text("{broken")
    assert load_pricing(f) == (DEFAULT_PRICES, 1000)
    assert load_pricing(tmp_path / "missing.json") == (DEFAULT_PRICES, 1000)


# ------------------------------------------------------------------ service
@pytest.fixture
def make_client(tmp_path):
    clients = []

    def factory(solver=None, grant=40):
        keys_file = tmp_path / "keys.json"
        keys_file.write_text(json.dumps({KEY: {"name": "t", "balance": grant}}))
        settings = Settings(keys_file=keys_file, usage_db=tmp_path / "usage.db", workers=2,
                            task_timeout=5)
        keys = KeyStore(settings)
        usage = UsageStore(settings.usage_db)
        billing = Billing(tmp_path / "usage.db", opening_grant=lambda k: int(keys.balance(k)))
        kwargs = {"solver": solver} if solver else {}
        manager = TaskManager(settings, usage, billing=billing, **kwargs)
        client = TestClient(create_app(settings, manager=manager, keys=keys, usage=usage))
        client.__enter__()
        client.billing = billing
        clients.append(client)
        return client

    yield factory
    for c in clients:
        c.__exit__(None, None, None)
        c.billing.close()


def _poll(client, task_id, limit=5.0):
    end = time.time() + limit
    while time.time() < end:
        res = client.post("/getTaskResult", json={"clientKey": KEY, "taskId": task_id}).json()
        if res.get("status") in ("ready", "failed") or res.get("errorId"):
            return res
        time.sleep(0.05)
    raise AssertionError("task did not finish")


def _grid(client):
    return client.post("/createTask", json={"clientKey": KEY, "task": {
        "type": "AwsWafClassification", "images": [jpeg_b64() for _ in range(9)],
        "question": "aws:grid:bed"}}).json()


def test_solved_task_is_charged_failed_is_refunded(make_client, monkeypatch):
    calls = {"n": 0}

    def classify(images, target):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("model crashed")
        return [1, 3]

    monkeypatch.setattr(vision_bridge, "classify_grid", classify)
    client = make_client(grant=40)
    price = DEFAULT_PRICES["AwsWafClassification"]
    assert _grid(client)["status"] == "ready"
    assert client.post("/getBalance", json={"clientKey": KEY}).json()["tokens"] == 40 - price
    assert _grid(client)["status"] == "failed"             # crash -> refunded
    bal = client.post("/getBalance", json={"clientKey": KEY}).json()
    assert bal["tokens"] == bal["balance"] == 40 - price
    assert bal["prices"]["AwsWafClassification"] == price and bal["tokensPerUsd"] == 1000


def test_zero_balance_rejects_before_work(make_client, monkeypatch):
    called = []
    monkeypatch.setattr(vision_bridge, "classify_grid", lambda i, t: called.append(1) or [0])
    client = make_client(grant=DEFAULT_PRICES["AwsWafClassification"] - 1)
    res = _grid(client)
    assert res["errorId"] == 1 and res["errorCode"] == "ERROR_ZERO_BALANCE"
    assert not called
    usage = client.post("/getBalance", json={"clientKey": KEY}).json()["usage"]
    assert "AwsWafClassification" not in usage  # never counted as created


def test_async_waf_task_refund_on_failure(make_client):
    def solver(req, settings, deadline=None):
        raise SolverError("ERROR_PROXY_CONNECT_REFUSED", "proxy down")

    client = make_client(solver, grant=100)
    res = client.post("/createTask", json={"clientKey": KEY, "task": {
        "type": "AntiAwsWafTaskProxyLess", "websiteURL": "https://shop.example/"}}).json()
    # a fast failure may already be reported by createTask itself
    done = res if res.get("status") == "failed" else _poll(client, res["taskId"])
    assert done["status"] == "failed" and done["errorCode"] == "ERROR_PROXY_CONNECT_REFUSED"
    assert client.post("/getBalance", json={"clientKey": KEY}).json()["tokens"] == 100


def test_async_waf_task_charged_on_success(make_client):
    client = make_client(lambda req, s, deadline=None: SolveOutcome("T", "UA"), grant=100)
    res = client.post("/createTask", json={"clientKey": KEY, "task": {
        "type": "AntiAwsWafTaskProxyLess", "websiteURL": "https://shop.example/"}}).json()
    done = res if res.get("status") == "ready" else _poll(client, res["taskId"])
    assert done["status"] == "ready"
    assert client.post("/getBalance", json={"clientKey": KEY}).json()["tokens"] == \
        100 - DEFAULT_PRICES["AntiAwsWafTaskProxyLess"]


def test_cli_topup_and_balance(tmp_path, monkeypatch, capsys):
    keys_file = tmp_path / "keys.json"
    keys_file.write_text(json.dumps({KEY: {"balance": 0}}))
    monkeypatch.setenv("AWSCS_KEYS_FILE", str(keys_file))
    monkeypatch.setenv("AWSCS_USAGE_DB", str(tmp_path / "u.db"))
    monkeypatch.setenv("AWSCS_PRICING_FILE", str(tmp_path / "none.json"))
    assert cli_main(["billing", "topup", "--key", KEY, "--usd", "10", "--note", "pay-1"]) == 0
    assert json.loads(capsys.readouterr().out.strip().splitlines()[-1])["tokens"] == 10000
    assert cli_main(["billing", "balance", "--key", KEY]) == 0
    assert json.loads(capsys.readouterr().out.strip().splitlines()[-1]) == {"key": KEY, "tokens": 10000, "usd": 10.0}
    assert cli_main(["billing", "topup", "--key", "nope", "--usd", "1"]) == 2
    assert cli_main(["billing", "topup", "--key", KEY]) == 2
