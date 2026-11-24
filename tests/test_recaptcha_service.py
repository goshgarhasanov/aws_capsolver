"""ReCaptchaV2Classification: validation, model bridge error mapping, output checks, billing.

The real model package (aws_capsolver.recaptcha) is replaced by a fake module everywhere, so
these tests are fast and do not depend on the model being installed.
"""

import base64
import io
import json
import types

import pytest
from fastapi.testclient import TestClient

from aws_capsolver import recaptcha_bridge
from aws_capsolver.billing import DEFAULT_PRICES, Billing
from aws_capsolver.config import KeyStore, Settings, UsageStore
from aws_capsolver.server import create_app
from aws_capsolver.tasks import (CLASSIFY_TYPES, SUPPORTED_TYPES, TaskManager, parse_recaptcha,
                                 run_recaptcha)
from aws_capsolver.waf.errors import SolverError

KEY = "rc-key-1"
TASK_TYPE = "ReCaptchaV2Classification"
PRICE = DEFAULT_PRICES[TASK_TYPE]


def png_b64(width=300, height=300, color=(90, 140, 200)) -> str:
    from PIL import Image
    buf = io.BytesIO()
    Image.new("RGB", (width, height), color).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


GRID33 = png_b64()
GRID44 = png_b64(450, 450)
TILE = png_b64(100, 100)


# ------------------------------------------------------------------ a fake model package

class RecaptchaError(Exception):
    pass


class UnsupportedQuestion(RecaptchaError):
    pass


class InvalidRecaptchaInput(RecaptchaError):
    pass


def fake_module(**overrides):
    calls = []

    def classify(image, question, size=None):
        from PIL import Image
        with Image.open(io.BytesIO(image)) as img:
            width = img.size[0]
        guessed = size or {300: 3, 450: 4, 100: 1}.get(width, 3)
        calls.append((question, size, width))
        if guessed == 1:
            return {"type": "single", "hasObject": True, "size": 1}
        return {"type": "multi", "objects": [4, 0, 8, 4], "size": guessed}

    def normalize_question(question):
        return question

    mod = types.SimpleNamespace(classify=classify, normalize_question=normalize_question,
                                RecaptchaError=RecaptchaError, UnsupportedQuestion=UnsupportedQuestion,
                                InvalidRecaptchaInput=InvalidRecaptchaInput, calls=calls)
    for k, v in overrides.items():
        setattr(mod, k, v)
    return mod


@pytest.fixture
def model(monkeypatch):
    mod = fake_module()
    monkeypatch.setattr(recaptcha_bridge, "_module", mod)
    return mod


def _task(question="/m/0k4j", image=GRID33, **fields):
    task = {"type": TASK_TYPE, "question": question, "image": image}
    task.update(fields)
    return task


# ------------------------------------------------------------------ validation

def test_registered_as_synchronous_type():
    assert TASK_TYPE in SUPPORTED_TYPES and TASK_TYPE in CLASSIFY_TYPES


def test_parse_accepts_capsolver_payloads():
    job = parse_recaptcha(_task())
    assert job.question == "/m/0k4j" and job.size is None and job.image.startswith(b"\x89PNG")
    job = parse_recaptcha(_task("  светофоры ", "data:image/png;base64," + GRID33, size=3,
                                websiteURL="https://example.com/login", websiteKey="6Le-abc"))
    assert job.question == "светофоры" and job.size == 3
    assert job.website_url == "https://example.com/login" and job.website_key == "6Le-abc"
    assert parse_recaptcha(_task(image=TILE, size=1)).size == 1
    assert parse_recaptcha(_task(size=None)).size is None


@pytest.mark.parametrize("fields", [
    {"image": None},
    {"image": ""},
    {"image": 5},
    {"image": "%%% not base64"},
    {"image": "A" * (3 * 1024 * 1024)},
    {"question": None},
    {"question": ""},
    {"question": "   "},
    {"question": 7},
    {"question": "x" * 200},
    {"question": "cars\x01"},
    {"size": 2},
    {"size": 9},
    {"size": "3"},
    {"size": True},
    {"size": 3.0},
    {"websiteURL": "ftp://example.com"},
    {"websiteURL": 12},
    {"websiteKey": "k" * 300},
])
def test_parse_rejects_bad_payloads(fields):
    task = _task()
    task.update(fields)
    with pytest.raises(SolverError) as err:
        parse_recaptcha(task)
    assert err.value.code == "ERROR_INVALID_TASK_DATA"


# ------------------------------------------------------------------ model calls

def test_run_grids_and_tile(model):
    assert run_recaptcha(parse_recaptcha(_task())) == {"type": "multi", "objects": [0, 4, 8], "size": 3}
    assert model.calls[-1] == ("/m/0k4j", None, 300)
    assert run_recaptcha(parse_recaptcha(_task("motorcycles", GRID44, size=4))) == {
        "type": "multi", "objects": [0, 4, 8], "size": 4}
    assert model.calls[-1] == ("motorcycles", 4, 450)
    assert run_recaptcha(parse_recaptcha(_task("cars", TILE, size=1))) == {
        "type": "single", "hasObject": True, "size": 1}


def test_empty_selection_is_a_valid_answer(monkeypatch):
    """'If there are none, click skip': no cell is a legitimate answer."""
    monkeypatch.setattr(recaptcha_bridge, "_module", fake_module(
        classify=lambda *a, **k: {"type": "multi", "objects": [], "size": 3}))
    assert run_recaptcha(parse_recaptcha(_task())) == {"type": "multi", "objects": [], "size": 3}


def test_numpy_like_output_is_normalized(monkeypatch):
    np = pytest.importorskip("numpy")
    monkeypatch.setattr(recaptcha_bridge, "_module", fake_module(
        classify=lambda *a, **k: {"type": "multi", "objects": np.array([3, 1]), "size": np.int64(3)}))
    assert run_recaptcha(parse_recaptcha(_task())) == {"type": "multi", "objects": [1, 3], "size": 3}
    monkeypatch.setattr(recaptcha_bridge, "_module", fake_module(
        classify=lambda *a, **k: {"type": "single", "hasObject": np.bool_(False), "size": 1}))
    assert run_recaptcha(parse_recaptcha(_task(image=TILE))) == {"type": "single", "hasObject": False, "size": 1}


@pytest.mark.parametrize("raised, code, text", [
    (InvalidRecaptchaInput("not a grid"), "ERROR_INVALID_TASK_DATA", "not a grid"),
    (UnsupportedQuestion("zebras"), "ERROR_CAPTCHA_UNSOLVABLE", "unsupported question"),
    (RecaptchaError("low confidence"), "ERROR_CAPTCHA_UNSOLVABLE", "low confidence"),
    (ImportError("torch"), "ERROR_TASK_NOT_SUPPORTED", "torch"),
    (FileNotFoundError("weights.pt"), "ERROR_TASK_NOT_SUPPORTED", "weights"),
])
def test_model_errors_are_mapped(monkeypatch, raised, code, text):
    def boom(*a, **k):
        raise raised
    monkeypatch.setattr(recaptcha_bridge, "_module", fake_module(classify=boom))
    with pytest.raises(SolverError) as err:
        run_recaptcha(parse_recaptcha(_task()))
    assert err.value.code == code and text in err.value.description


def test_errors_from_the_errors_submodule(monkeypatch):
    mod = fake_module()
    for name in ("RecaptchaError", "UnsupportedQuestion", "InvalidRecaptchaInput"):
        delattr(mod, name)
    mod.errors = types.SimpleNamespace(RecaptchaError=RecaptchaError, UnsupportedQuestion=UnsupportedQuestion,
                                       InvalidRecaptchaInput=InvalidRecaptchaInput)

    def unsupported(*a, **k):
        raise UnsupportedQuestion("zebras")
    mod.classify = unsupported
    monkeypatch.setattr(recaptcha_bridge, "_module", mod)
    with pytest.raises(SolverError) as err:
        run_recaptcha(parse_recaptcha(_task()))
    assert err.value.code == "ERROR_CAPTCHA_UNSOLVABLE" and "unsupported" in err.value.description


@pytest.mark.parametrize("output, size", [
    (None, None),
    ([1, 2], None),
    ({"type": "multi", "objects": [9], "size": 3}, None),          # outside the 3x3 grid
    ({"type": "multi", "objects": [-1], "size": 3}, None),
    ({"type": "multi", "objects": [16], "size": 4}, None),
    ({"type": "multi", "objects": [1.5], "size": 3}, None),
    ({"type": "multi", "objects": [True], "size": 3}, None),
    ({"type": "multi", "objects": ["1"], "size": 3}, None),
    ({"type": "multi", "objects": "012", "size": 3}, None),
    ({"type": "multi", "objects": None, "size": 3}, None),
    ({"type": "multi", "objects": 5, "size": 3}, None),
    ({"type": "multi", "objects": [1], "size": 5}, None),          # unknown size
    ({"type": "multi", "objects": [1], "size": "3"}, None),
    ({"type": "multi", "objects": [1]}, None),                     # no size
    ({"type": "single", "objects": [1], "size": 3}, None),         # type does not fit the size
    ({"type": "multi", "objects": [1], "size": 1}, None),
    ({"type": "single", "hasObject": 1, "size": 1}, None),
    ({"type": "single", "size": 1}, None),
    ({"type": "multi", "objects": [1], "size": 4}, 3),             # contradicts the size hint
    ({"type": "single", "hasObject": True, "size": 1}, 3),
])
def test_bad_model_output_is_unsolvable(monkeypatch, output, size):
    monkeypatch.setattr(recaptcha_bridge, "_module", fake_module(classify=lambda *a, **k: output))
    with pytest.raises(SolverError) as err:
        run_recaptcha(parse_recaptcha(_task(size=size)))
    assert err.value.code == "ERROR_CAPTCHA_UNSOLVABLE"


def test_unreadable_or_oversized_image_is_invalid_before_the_model(model):
    garbage = base64.b64encode(b"definitely not an image").decode()
    for image in (garbage, png_b64(1200, 300), png_b64(8, 8)):
        with pytest.raises(SolverError) as err:
            run_recaptcha(parse_recaptcha(_task(image=image)))
        assert err.value.code == "ERROR_INVALID_TASK_DATA"
    assert model.calls == []


def test_missing_package_and_missing_function(monkeypatch):
    monkeypatch.setattr(recaptcha_bridge, "_module", False)
    monkeypatch.setattr(recaptcha_bridge, "_failed_at", float("inf"))  # no retry during the test
    assert not recaptcha_bridge.available()
    with pytest.raises(SolverError) as err:
        run_recaptcha(parse_recaptcha(_task()))
    assert err.value.code == "ERROR_TASK_NOT_SUPPORTED"
    mod = fake_module()
    del mod.classify
    monkeypatch.setattr(recaptcha_bridge, "_module", mod)
    assert not recaptcha_bridge.available()
    with pytest.raises(SolverError) as err:
        run_recaptcha(parse_recaptcha(_task()))
    assert err.value.code == "ERROR_TASK_NOT_SUPPORTED"


def test_failed_import_is_retried_later(monkeypatch):
    attempts = []

    def importer(name):
        attempts.append(name)
        if len(attempts) == 1:
            raise SyntaxError("half written")
        return fake_module()

    monkeypatch.setattr(recaptcha_bridge, "_module", None)
    monkeypatch.setattr(recaptcha_bridge.importlib, "import_module", importer)
    clock = [1000.0]
    monkeypatch.setattr(recaptcha_bridge.time, "monotonic", lambda: clock[0])
    assert not recaptcha_bridge.available()
    assert not recaptcha_bridge.available()            # cached failure, no new import
    assert len(attempts) == 1
    clock[0] += recaptcha_bridge.RETRY_SECONDS + 1
    assert recaptcha_bridge.available()
    assert attempts == [recaptcha_bridge.PACKAGE] * 2


def test_warmup_is_best_effort(monkeypatch):
    monkeypatch.setattr(recaptcha_bridge, "_module", fake_module())
    assert recaptcha_bridge.warmup() is None           # no warmup() offered

    def broken():
        raise RuntimeError("cuda")
    monkeypatch.setattr(recaptcha_bridge, "_module", fake_module(warmup=broken))
    assert recaptcha_bridge.warmup() is False
    monkeypatch.setattr(recaptcha_bridge, "_module", fake_module(warmup=lambda: None))
    assert recaptcha_bridge.warmup() is True


# ------------------------------------------------------------------ API + billing

@pytest.fixture
def make_client(tmp_path):
    clients = []

    def factory(grant=20, prices=None):
        keys_file = tmp_path / "keys.json"
        keys_file.write_text(json.dumps({KEY: {"name": "t", "balance": grant}}))
        settings = Settings(keys_file=keys_file, usage_db=tmp_path / "usage.db", workers=2,
                            task_timeout=5)
        keys = KeyStore(settings)
        usage = UsageStore(settings.usage_db)
        billing = Billing(tmp_path / "usage.db", prices=prices,
                          opening_grant=lambda k: int(keys.balance(k)))
        manager = TaskManager(settings, usage, billing=billing)
        client = TestClient(create_app(settings, manager=manager, keys=keys, usage=usage))
        client.__enter__()
        client.billing = billing
        clients.append(client)
        return client

    yield factory
    for c in clients:
        c.__exit__(None, None, None)
        c.billing.close()


def _create(client, task):
    return client.post("/createTask", json={"clientKey": KEY, "task": task}).json()


def _tokens(client):
    return client.post("/getBalance", json={"clientKey": KEY}).json()["tokens"]


def test_api_solves_synchronously_and_charges(make_client, model):
    client = make_client(grant=20)
    res = _create(client, _task("/m/0k4j", size=3, websiteURL="https://example.com/",
                                websiteKey="6Le-site-key"))
    assert res["errorId"] == 0 and res["status"] == "ready"
    assert res["solution"] == {"type": "multi", "objects": [0, 4, 8], "size": 3}
    assert _tokens(client) == 20 - PRICE
    again = client.post("/getTaskResult", json={"clientKey": KEY, "taskId": res["taskId"]}).json()
    assert again["solution"] == res["solution"]
    res = _create(client, _task("traffic lights", TILE, size=1))
    assert res["solution"] == {"type": "single", "hasObject": True, "size": 1}
    assert _tokens(client) == 20 - 2 * PRICE
    bal = client.post("/getBalance", json={"clientKey": KEY}).json()
    assert bal["prices"][TASK_TYPE] == PRICE == 3
    assert bal["usage"][TASK_TYPE]["solved"] == 2
    assert client.get("/health").json()["recaptcha"] is True


def test_api_refunds_unsolvable_and_bad_output(make_client, monkeypatch):
    def unsupported(*a, **k):
        raise UnsupportedQuestion("zebras")
    monkeypatch.setattr(recaptcha_bridge, "_module", fake_module(classify=unsupported))
    client = make_client(grant=20)
    res = _create(client, _task("zebras"))
    assert res["errorId"] == 1 and res["errorCode"] == "ERROR_CAPTCHA_UNSOLVABLE"
    assert res["status"] == "failed" and "unsupported question" in res["errorDescription"]
    assert _tokens(client) == 20
    monkeypatch.setattr(recaptcha_bridge, "_module", fake_module(
        classify=lambda *a, **k: {"type": "multi", "objects": [42], "size": 3}))
    res = _create(client, _task())
    assert res["errorCode"] == "ERROR_CAPTCHA_UNSOLVABLE"
    assert _tokens(client) == 20


def test_api_model_missing_is_refunded(make_client, monkeypatch):
    monkeypatch.setattr(recaptcha_bridge, "_module", False)
    monkeypatch.setattr(recaptcha_bridge, "_failed_at", float("inf"))
    client = make_client(grant=20)
    res = _create(client, _task())
    assert res["errorCode"] == "ERROR_TASK_NOT_SUPPORTED"
    assert _tokens(client) == 20
    assert client.get("/health").json()["recaptcha"] is False


def test_api_model_crash_is_refunded(make_client, monkeypatch):
    def crash(*a, **k):
        raise RuntimeError("bug in the model")
    monkeypatch.setattr(recaptcha_bridge, "_module", fake_module(classify=crash))
    client = make_client(grant=20)
    res = _create(client, _task())
    assert res["errorId"] == 1 and res["status"] == "failed"
    assert _tokens(client) == 20


def test_api_invalid_task_costs_nothing(make_client, model):
    client = make_client(grant=20)
    for bad in (_task(image=None), _task(question=""), _task(size=2)):
        res = _create(client, bad)
        assert res["errorCode"] == "ERROR_INVALID_TASK_DATA" and "taskId" not in res
    assert _tokens(client) == 20 and model.calls == []


def test_api_zero_balance(make_client, model):
    client = make_client(grant=PRICE - 1)
    res = _create(client, _task())
    assert res["errorCode"] == "ERROR_ZERO_BALANCE"
    assert model.calls == []


def test_custom_price(make_client, model):
    client = make_client(grant=100, prices=dict(DEFAULT_PRICES, **{TASK_TYPE: 7}))
    _create(client, _task())
    assert _tokens(client) == 93
