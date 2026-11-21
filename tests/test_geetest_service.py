"""GeeTestClassification: validation, model bridge error mapping, billing, admission control.

The real model package (aws_capsolver.geetest) is replaced by a fake module everywhere, so
these tests are fast and do not depend on the models being installed.
"""

import base64
import io
import json
import threading
import types

import pytest
from fastapi.testclient import TestClient

from aws_capsolver import geetest_bridge
from aws_capsolver.billing import DEFAULT_PRICES, Billing
from aws_capsolver.config import KeyStore, Settings, UsageStore
from aws_capsolver.server import create_app
from aws_capsolver.tasks import TaskManager, parse_geetest, run_geetest
from aws_capsolver.waf.errors import SolverError

KEY = "gt-key-1"
PRICE = DEFAULT_PRICES["GeeTestClassification"]


def png_b64(width=260, height=160, color=(90, 140, 200)) -> str:
    from PIL import Image
    buf = io.BytesIO()
    Image.new("RGB", (width, height), color).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


BG = png_b64()
PIECE = png_b64(50, 50, (250, 250, 250))
ICON = png_b64(30, 30, (10, 10, 10))


# ------------------------------------------------------------------ a fake model package

class GeeTestError(Exception):
    pass


class UnsupportedGeeTest(GeeTestError):
    pass


class InvalidGeeTestInput(GeeTestError):
    pass


def fake_module(**overrides):
    calls = []

    def slide_offset(background, piece=None, full_background=None, y=None):
        calls.append(("slide", len(background), piece is not None, full_background is not None, y))
        return 172

    def click_points(image, kind, hints=None):
        calls.append(("click", kind, hints))
        return [(60, 70), (200.0, 100), (140, 150)]

    def nine_grid(image, target, count=None):
        calls.append(("nine", target) if count is None else ("nine", target, count))
        return [5, 1, 5]

    def gobang(board):
        calls.append(("gobang",))
        return ((0, 1), (2, 3))

    mod = types.SimpleNamespace(
        slide_offset=slide_offset, click_points=click_points, nine_grid=nine_grid, gobang=gobang,
        GeeTestError=GeeTestError, UnsupportedGeeTest=UnsupportedGeeTest,
        InvalidGeeTestInput=InvalidGeeTestInput, calls=calls)
    for k, v in overrides.items():
        setattr(mod, k, v)
    return mod


@pytest.fixture
def model(monkeypatch):
    mod = fake_module()
    monkeypatch.setattr(geetest_bridge, "_module", mod)
    return mod


def _task(question, **fields):
    task = {"type": "GeeTestClassification", "question": question, "images": [BG]}
    task.update(fields)
    return task


# ------------------------------------------------------------------ validation

def test_parse_families():
    job = parse_geetest(_task("geetest:slide", piece=PIECE, fullBackground=BG))
    assert job.module == "slide" and job.piece and job.full_background
    job = parse_geetest(_task("geetest:icon", hints=[ICON, ICON]))
    assert job.module == "icon" and all(isinstance(h, bytes) for h in job.hints)
    job = parse_geetest(_task("geetest:word", hints=["天", " 地 "]))
    assert job.hints == ("天", "地")
    assert parse_geetest(_task("geetest:space", hints=["click the red cube"])).module == "space"
    assert parse_geetest(_task("geetest:nine:car")).target == "car"
    assert isinstance(parse_geetest(_task("geetest:nine", targetImage=ICON)).target, bytes)
    assert parse_geetest(_task("geetest:gobang")).module == "gobang"
    data_url = parse_geetest(_task("geetest:slide", images=["data:image/png;base64," + BG]))
    assert data_url.image.startswith(b"\x89PNG")


@pytest.mark.parametrize("fields", [
    {"question": "aws:grid:car"},
    {"question": "geetest"},
    {"question": ""},
    {"question": 7},
    {"images": []},
    {"images": [BG, BG]},
    {"images": "nope"},
    {"images": ["%%%"]},
    {"question": "geetest:icon", "piece": PIECE},
    {"question": "geetest:slide", "hints": ["a"]},
    {"question": "geetest:word", "hints": ["a"] * 9},
    {"question": "geetest:word", "hints": [""]},
    {"question": "geetest:word", "hints": [5]},
    {"question": "geetest:word", "hints": ["a\x01"]},
    {"question": "geetest:word", "hints": "abc"},
    {"question": "geetest:icon", "hints": ["not base64 %%%"]},
    {"question": "geetest:nine"},
    {"question": "geetest:nine:car", "targetImage": ICON},
    {"question": "geetest:slide:car"},
    {"question": "geetest:icon", "targetImage": ICON},
    {"question": "geetest:nine:car", "count": 0},
    {"question": "geetest:nine:car", "count": "3"},
    {"question": "geetest:slide", "count": 3},
    {"question": "geetest:slide", "y": -1},
    {"question": "geetest:slide", "y": True},
    {"question": "geetest:slide", "y": 5.5},
    {"question": "geetest:slide", "y": 5000},
    {"question": "geetest:icon", "y": 10},
])
def test_parse_rejects_bad_payloads(fields):
    task = _task("geetest:slide")
    task.update(fields)
    with pytest.raises(SolverError) as err:
        parse_geetest(task)
    assert err.value.code == "ERROR_INVALID_TASK_DATA"


def test_unknown_family_is_not_supported():
    with pytest.raises(SolverError) as err:
        parse_geetest(_task("geetest:match"))
    assert err.value.code == "ERROR_TASK_NOT_SUPPORTED"


# ------------------------------------------------------------------ model calls

def test_run_each_family(model):
    assert run_geetest(parse_geetest(_task("geetest:slide", piece=PIECE))) == {"x": 172}
    assert model.calls[-1] == ("slide", len(base64.b64decode(BG)), True, False, None)
    assert run_geetest(parse_geetest(_task("geetest:slide", piece=PIECE, y=57))) == {"x": 172}
    assert model.calls[-1][-1] == 57
    assert run_geetest(parse_geetest(_task("geetest:word", hints=["天"]))) == {
        "points": [[60, 70], [200, 100], [140, 150]]}
    assert model.calls[-1] == ("click", "word", ["天"])
    run_geetest(parse_geetest(_task("geetest:icon")))
    assert model.calls[-1] == ("click", "icon", None)
    assert run_geetest(parse_geetest(_task("geetest:nine:car"))) == {"objects": [1, 5]}
    assert model.calls[-1] == ("nine", "car")
    run_geetest(parse_geetest(_task("geetest:nine:car", count=2)))
    assert model.calls[-1] == ("nine", "car", 2)
    assert run_geetest(parse_geetest(_task("geetest:gobang"))) == {"move": [[0, 1], [2, 3]]}


@pytest.mark.parametrize("raised, code", [
    (InvalidGeeTestInput("not a geetest picture"), "ERROR_INVALID_TASK_DATA"),
    (UnsupportedGeeTest("unknown layout"), "ERROR_CAPTCHA_UNSOLVABLE"),
    (GeeTestError("no gap found"), "ERROR_CAPTCHA_UNSOLVABLE"),
    (ImportError("torch"), "ERROR_TASK_NOT_SUPPORTED"),
    (FileNotFoundError("weights.onnx"), "ERROR_TASK_NOT_SUPPORTED"),
])
def test_model_errors_are_mapped(monkeypatch, raised, code):
    def boom(*a, **k):
        raise raised
    monkeypatch.setattr(geetest_bridge, "_module", fake_module(slide_offset=boom))
    with pytest.raises(SolverError) as err:
        run_geetest(parse_geetest(_task("geetest:slide")))
    assert err.value.code == code


def test_errors_from_the_errors_submodule(monkeypatch):
    """The typed errors are found in mod.errors when the package does not re-export them."""
    mod = fake_module()
    for name in ("GeeTestError", "UnsupportedGeeTest", "InvalidGeeTestInput"):
        delattr(mod, name)
    mod.errors = types.SimpleNamespace(GeeTestError=GeeTestError, UnsupportedGeeTest=UnsupportedGeeTest,
                                       InvalidGeeTestInput=InvalidGeeTestInput)

    def bad(*a, **k):
        raise InvalidGeeTestInput("bad")
    mod.nine_grid = bad
    monkeypatch.setattr(geetest_bridge, "_module", mod)
    with pytest.raises(SolverError) as err:
        run_geetest(parse_geetest(_task("geetest:nine:car")))
    assert err.value.code == "ERROR_INVALID_TASK_DATA"


@pytest.mark.parametrize("question, override", [
    ("geetest:slide", {"slide_offset": lambda *a, **k: 999}),          # beyond the 260 px image
    ("geetest:slide", {"slide_offset": lambda *a, **k: -3}),
    ("geetest:slide", {"slide_offset": lambda *a, **k: "far"}),
    ("geetest:slide", {"slide_offset": lambda *a, **k: float("nan")}),
    ("geetest:icon", {"click_points": lambda *a, **k: []}),
    ("geetest:icon", {"click_points": lambda *a, **k: [(10, 500)]}),
    ("geetest:icon", {"click_points": lambda *a, **k: [(1, 2, 3)]}),
    ("geetest:icon", {"click_points": lambda *a, **k: 5}),
    ("geetest:icon", {"click_points": lambda *a, **k: [(1, 1)] * 40}),
    ("geetest:nine:car", {"nine_grid": lambda *a, **k: [9]}),
    ("geetest:nine:car", {"nine_grid": lambda *a, **k: []}),
    ("geetest:gobang", {"gobang": lambda *a, **k: ((0, 0), (0, 0))}),
    ("geetest:gobang", {"gobang": lambda *a, **k: ((0, 0), (99, 0))}),
    ("geetest:gobang", {"gobang": lambda *a, **k: (0, 1)}),
])
def test_bad_model_output_is_unsolvable(monkeypatch, question, override):
    monkeypatch.setattr(geetest_bridge, "_module", fake_module(**override))
    with pytest.raises(SolverError) as err:
        run_geetest(parse_geetest(_task(question)))
    assert err.value.code == "ERROR_CAPTCHA_UNSOLVABLE"


def test_y_below_the_background_is_invalid(model):
    with pytest.raises(SolverError) as err:
        run_geetest(parse_geetest(_task("geetest:slide", y=160)))   # BG is 160 px high
    assert err.value.code == "ERROR_INVALID_TASK_DATA"
    assert model.calls == []


def test_model_without_y_parameter_still_solves(monkeypatch):
    seen = []

    def old_slide(background, piece=None, full_background=None):  # a build before y existed
        seen.append(piece is not None)
        return 99
    monkeypatch.setattr(geetest_bridge, "_module", fake_module(slide_offset=old_slide))
    assert run_geetest(parse_geetest(_task("geetest:slide", piece=PIECE, y=40))) == {"x": 99}
    assert seen == [True]


def test_type_error_inside_the_model_is_not_retried(monkeypatch):
    def broken(background, piece=None, full_background=None, y=None):
        raise TypeError("unsupported operand type(s)")
    monkeypatch.setattr(geetest_bridge, "_module", fake_module(slide_offset=broken))
    with pytest.raises(TypeError):
        geetest_bridge.slide_offset(b"x", None, None, 3)


def test_undecodable_image_is_invalid_before_the_model(model):
    garbage = base64.b64encode(b"definitely not an image").decode()
    with pytest.raises(SolverError) as err:
        run_geetest(parse_geetest(_task("geetest:slide", images=[garbage])))
    assert err.value.code == "ERROR_INVALID_TASK_DATA"
    with pytest.raises(SolverError) as err:
        run_geetest(parse_geetest(_task("geetest:slide", piece=garbage)))
    assert err.value.code == "ERROR_INVALID_TASK_DATA"
    huge = png_b64(2400, 20)
    with pytest.raises(SolverError) as err:
        run_geetest(parse_geetest(_task("geetest:gobang", images=[huge])))
    assert err.value.code == "ERROR_INVALID_TASK_DATA"
    assert model.calls == []


def test_missing_package_and_missing_function(monkeypatch):
    monkeypatch.setattr(geetest_bridge, "_module", False)
    monkeypatch.setattr(geetest_bridge, "_failed_at", float("inf"))  # no retry during the test
    assert not geetest_bridge.available()
    assert geetest_bridge.solvers() == dict.fromkeys(geetest_bridge.FUNCTIONS, False)
    with pytest.raises(SolverError) as err:
        run_geetest(parse_geetest(_task("geetest:slide")))
    assert err.value.code == "ERROR_TASK_NOT_SUPPORTED"
    mod = fake_module()
    del mod.gobang
    monkeypatch.setattr(geetest_bridge, "_module", mod)
    assert geetest_bridge.solvers()["gobang"] is False
    with pytest.raises(SolverError) as err:
        run_geetest(parse_geetest(_task("geetest:gobang")))
    assert err.value.code == "ERROR_TASK_NOT_SUPPORTED"


def test_failed_import_is_retried_later(monkeypatch):
    attempts = []

    def importer(name):
        attempts.append(name)
        if len(attempts) == 1:
            raise ImportError("not written yet")
        return fake_module()

    monkeypatch.setattr(geetest_bridge, "_module", None)
    monkeypatch.setattr(geetest_bridge.importlib, "import_module", importer)
    clock = [1000.0]
    monkeypatch.setattr(geetest_bridge.time, "monotonic", lambda: clock[0])
    assert not geetest_bridge.available()
    assert not geetest_bridge.available()            # cached failure, no new import
    assert len(attempts) == 1
    clock[0] += geetest_bridge.RETRY_SECONDS + 1
    assert geetest_bridge.available()
    assert attempts == [geetest_bridge.PACKAGE] * 2


# ------------------------------------------------------------------ API + billing

@pytest.fixture
def make_client(tmp_path):
    clients = []

    def factory(grant=40, prices=None):
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


def test_api_solves_and_charges(make_client, model):
    client = make_client(grant=40)
    res = _create(client, _task("geetest:slide", piece=PIECE))
    assert res["errorId"] == 0 and res["status"] == "ready" and res["solution"] == {"x": 172}
    assert _tokens(client) == 40 - PRICE
    again = client.post("/getTaskResult", json={"clientKey": KEY, "taskId": res["taskId"]}).json()
    assert again["solution"] == {"x": 172}
    res = _create(client, _task("geetest:gobang"))
    assert res["solution"] == {"move": [[0, 1], [2, 3]]}
    assert _tokens(client) == 40 - 2 * PRICE
    bal = client.post("/getBalance", json={"clientKey": KEY}).json()
    assert bal["prices"]["GeeTestClassification"] == PRICE
    assert bal["usage"]["GeeTestClassification"]["solved"] == 2


def test_api_refunds_failures(make_client, monkeypatch):
    def unsolvable(*a, **k):
        raise UnsupportedGeeTest("no gap")
    monkeypatch.setattr(geetest_bridge, "_module", fake_module(slide_offset=unsolvable))
    client = make_client(grant=40)
    res = _create(client, _task("geetest:slide"))
    assert res["errorId"] == 1 and res["errorCode"] == "ERROR_CAPTCHA_UNSOLVABLE"
    assert res["status"] == "failed"
    assert _tokens(client) == 40


def test_api_model_missing_is_refunded(make_client, monkeypatch):
    monkeypatch.setattr(geetest_bridge, "_module", False)
    monkeypatch.setattr(geetest_bridge, "_failed_at", float("inf"))
    client = make_client(grant=40)
    res = _create(client, _task("geetest:icon"))
    assert res["errorCode"] == "ERROR_TASK_NOT_SUPPORTED"
    assert _tokens(client) == 40
    assert client.get("/health").json()["geetest"] == dict.fromkeys(geetest_bridge.FUNCTIONS, False)


def test_api_invalid_task_costs_nothing(make_client, model):
    client = make_client(grant=40)
    res = _create(client, _task("geetest:slide", images=[BG, BG]))
    assert res["errorCode"] == "ERROR_INVALID_TASK_DATA" and "taskId" not in res
    assert _tokens(client) == 40 and model.calls == []


def test_api_zero_balance(make_client, model):
    client = make_client(grant=PRICE - 1)
    res = _create(client, _task("geetest:slide"))
    assert res["errorCode"] == "ERROR_ZERO_BALANCE"
    assert model.calls == []


def test_per_family_price(make_client, model):
    prices = dict(DEFAULT_PRICES, **{"GeeTestClassification:gobang": 20})
    client = make_client(grant=100, prices=prices)
    _create(client, _task("geetest:gobang"))
    assert _tokens(client) == 80
    _create(client, _task("geetest:slide"))
    assert _tokens(client) == 80 - PRICE


def test_admission_control_shares_the_classification_queue(tmp_path, monkeypatch):
    release = threading.Event()
    started = threading.Event()

    def slow(*a, **k):
        started.set()
        release.wait(10)
        return 10

    monkeypatch.setattr(geetest_bridge, "_module", fake_module(slide_offset=slow))
    settings = Settings(keys_file=tmp_path / "keys.json", usage_db=tmp_path / "u.db", workers=1,
                        task_timeout=30)
    usage = UsageStore(settings.usage_db)
    manager = TaskManager(settings, usage)
    try:
        capacity = settings.vision_concurrency + settings.classify_queue_limit
        futures = [manager.submit_classification(KEY, _task("geetest:slide"))[1] for _ in range(capacity)]
        assert started.wait(5)
        with pytest.raises(SolverError) as err:
            manager.submit_classification(KEY, _task("geetest:slide"))
        assert err.value.code == "ERROR_NO_SLOT_AVAILABLE"
        release.set()
        assert all(f.result(10) == {"x": 10} for f in futures)
    finally:
        release.set()
        manager.shutdown()
        usage.close()
