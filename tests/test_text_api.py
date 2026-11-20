"""ImageToTextTask: module routing, post-filters, validation, error mapping, queueing."""

import base64
import io
import os
import threading
import types

import pytest

from aws_capsolver import text
from aws_capsolver.config import Settings
from aws_capsolver.tasks import MAX_IMAGE_BYTES, parse_text, run_text
from aws_capsolver.text import Reading, TextUnsolvable, apply_filters, common
from aws_capsolver.text.errors import InvalidImage, ModelUnavailable
from aws_capsolver.waf.errors import SolverError

from .test_api import KEY, make_client  # noqa: F401  (fixture re-export)

SKIP_MODELS = os.environ.get("AWS_CAPSOLVER_SKIP_MODEL_TESTS") == "1"


def png_bytes(label="ab12x", size=(160, 60)) -> bytes:
    from PIL import Image, ImageDraw, ImageFont
    img = Image.new("RGB", size, "white")
    ImageDraw.Draw(img).text((10, 8), label, fill="black", font=ImageFont.load_default(size=36))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def png_b64(label="ab12x") -> str:
    return base64.b64encode(png_bytes(label)).decode()


def _task(**fields):
    task = {"type": "ImageToTextTask", "body": png_b64()}
    task.update(fields)
    return task


@pytest.fixture
def fakes(monkeypatch):
    """Replace the three recognisers; records which one ran and with what."""
    calls = []
    answers = {"bls_poland": ("BLSP42", 0.97), "general": ("Gen7x", 0.88),
               "ddddocr": Reading("dd8q", 0.5, [("dd8q", 0.5)])}

    def bls_solve(image):
        calls.append(("bls_poland", image, {}))
        return answers["bls_poland"]

    def general_solve(image, **kwargs):
        calls.append(("general", image, kwargs))
        result = answers["general"]
        if isinstance(result, Exception):
            raise result
        return result

    modules = {text._BLS_POLAND_PATH: types.SimpleNamespace(__name__="bls", solve=bls_solve),
               text._GENERAL_PATH: types.SimpleNamespace(__name__="general",
                                                         solve=general_solve)}

    def fake_import(path, what):
        if path not in modules:
            raise ModelUnavailable(f"{what} missing")
        return modules[path]

    def fake_common(image, numeric=0):
        calls.append(("ddddocr", image, {"numeric": numeric}))
        result = answers["ddddocr"]
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr(text, "_import", fake_import)
    monkeypatch.setattr(common, "recognize", fake_common)
    return types.SimpleNamespace(calls=calls, answers=answers, modules=modules)


def _create(client, **fields):
    return client.post("/createTask", json={"clientKey": KEY, "task": _task(**fields)}).json()


# routing --------------------------------------------------------------------------
@pytest.mark.parametrize("module, backend, answer", [
    ("bls_poland", "bls_poland", "BLSP42"),
    ("blspoland", "bls_poland", "BLSP42"),
    ("bls-poland", "bls_poland", "BLSP42"),
    (" BLS_Poland ", "bls_poland", "BLSP42"),
    ("common", "general", "Gen7x"),
    ("general", "general", "Gen7x"),
    (None, "general", "Gen7x"),
    ("", "general", "Gen7x"),
    ("ddddocr", "ddddocr", "dd8q"),
])
def test_api_routes_modules_and_aliases(make_client, fakes, module, backend, answer):  # noqa: F811
    fields = {} if module is None else {"module": module}
    res = _create(make_client(), **fields)
    assert res["errorId"] == 0 and res["status"] == "ready", res
    assert res["solution"]["text"] == answer
    assert 0.0 <= res["solution"]["confidence"] <= 1.0
    assert [c[0] for c in fakes.calls] == [backend]
    assert fakes.calls[0][1] == png_bytes()


def test_ready_reply_shape_and_get_task_result(make_client, fakes):  # noqa: F811
    client = make_client()
    res = _create(client, module="bls_poland")
    assert res == {"errorId": 0, "taskId": res["taskId"], "status": "ready",
                   "solution": {"text": "BLSP42", "confidence": 0.97}}
    again = client.post("/getTaskResult", json={"clientKey": KEY,
                                                "taskId": res["taskId"]}).json()
    assert again["solution"] == res["solution"]


def test_common_falls_back_to_ddddocr_when_general_is_unavailable(make_client, fakes):  # noqa: F811
    fakes.answers["general"] = ModelUnavailable("weights missing")
    res = _create(make_client(), module="common")
    assert res["solution"]["text"] == "dd8q"
    assert [c[0] for c in fakes.calls] == ["general", "ddddocr"]

    del fakes.modules[text._GENERAL_PATH]  # not installed at all
    fakes.calls.clear()
    res = _create(make_client(), module="general")
    assert res["solution"]["text"] == "dd8q"
    assert [c[0] for c in fakes.calls] == ["ddddocr"]


def test_constraints_reach_the_general_model(make_client, fakes):  # noqa: F811
    fakes.answers["general"] = ("4821", 0.9)
    res = _create(make_client(), numeric=1, minLength=4, maxLength=6, case=True)
    assert res["solution"]["text"] == "4821"
    assert fakes.calls[0][2] == {"charset": "digits", "min_len": 4, "max_len": 6,
                                 "case_sensitive": True}
    fakes.calls.clear()
    _create(make_client(), numeric=2, case=False)
    assert fakes.calls[0][2] == {"charset": "letters", "min_len": None, "max_len": None,
                                 "case_sensitive": False}
    fakes.calls.clear()
    _create(make_client())  # omitted case keeps the text as read
    assert fakes.calls[0][2]["case_sensitive"] is True
    assert fakes.calls[0][2]["charset"] is None


def test_numeric_reaches_ddddocr(make_client, fakes):  # noqa: F811
    fakes.answers["ddddocr"] = Reading("123", 0.6, [("123", 0.6)])
    _create(make_client(), module="ddddocr", numeric=1)
    assert fakes.calls[0][2] == {"numeric": 1}


def test_default_module_from_settings(make_client, fakes):  # noqa: F811
    res = _create(make_client(text_default_module="bls_poland"))
    assert res["solution"]["text"] == "BLSP42"


def test_default_module_env(monkeypatch, caplog):
    monkeypatch.setenv("AWSCS_TEXT_DEFAULT_MODULE", "BLS-Poland")
    assert Settings.from_env().text_default_module == "bls_poland"
    monkeypatch.setenv("AWSCS_TEXT_DEFAULT_MODULE", "nope")
    assert Settings.from_env().text_default_module == "common"
    monkeypatch.delenv("AWSCS_TEXT_DEFAULT_MODULE")
    assert Settings.from_env().text_default_module == "common"


# filters ----------------------------------------------------------------------------
@pytest.mark.parametrize("reading, kwargs, expected", [
    (Reading("a1b2", 0.9), {"numeric": 1}, "12"),
    (Reading("a1b2", 0.9), {"numeric": 2}, "ab"),
    (Reading("AbC9", 0.9), {"case": False}, "abc9"),
    (Reading("AbC9", 0.9), {"case": True}, "AbC9"),
    (Reading("AbC9", 0.9), {}, "AbC9"),
    # a candidate that already satisfies numeric wins over stripping the best one
    (Reading("a1b2", 0.9, [("a1b2", 0.9), ("4182", 0.4)]), {"numeric": 1}, "4182"),
    (Reading("4182", 0.9, [("4182", 0.9), ("aibz", 0.3)]), {"numeric": 2}, "aibz"),
    # a candidate within the length range wins
    (Reading("12345", 0.9, [("12345", 0.9), ("1234", 0.5)]),
     {"min_length": 4, "max_length": 4}, "1234"),
    (Reading("ab12", 0.9, [("ab12", 0.9), ("abcd", 0.5)]), {"numeric": 3}, "abcd"),
    (Reading("abcd", 0.9, [("abcd", 0.9), ("ab12", 0.5)]), {"numeric": 4}, "ab12"),
    (Reading("abcd", 0.9), {"numeric": 4}, "abcd"),  # 3 / 4 only prefer, never strip
])
def test_filters(reading, kwargs, expected):
    assert apply_filters(reading, **kwargs)[0] == expected


def test_filters_keep_the_chosen_confidence():
    assert apply_filters(Reading("a1", 0.9, [("a1", 0.9), ("71", 0.4)]), numeric=1) == ("71", 0.4)
    assert apply_filters(Reading("a1", 0.9), numeric=1) == ("1", 0.9)


@pytest.mark.parametrize("reading, kwargs", [
    (Reading("", 0.0), {}),
    (Reading("abc", 0.9), {"numeric": 1}),
    (Reading("123", 0.9), {"numeric": 2}),
    (Reading("12345", 0.9), {"max_length": 4}),
    (Reading("12", 0.9), {"min_length": 4}),
    (Reading("a1b2c", 0.9), {"numeric": 1, "min_length": 3}),
])
def test_filters_unsolvable(reading, kwargs):
    with pytest.raises(TextUnsolvable):
        apply_filters(reading, **kwargs)


def test_api_length_out_of_range_is_unsolvable(make_client, fakes):  # noqa: F811
    res = _create(make_client(), module="bls_poland", maxLength=4)
    assert res["errorId"] == 1 and res["errorCode"] == "ERROR_CAPTCHA_UNSOLVABLE"
    assert res["status"] == "failed"


def test_api_applies_filters_as_a_safety_net(make_client, fakes):  # noqa: F811
    res = _create(make_client(), module="bls_poland", numeric=1, case=False)
    assert res["solution"]["text"] == "42"
    fakes.answers["general"] = ("ABcd", 0.8)
    res = _create(make_client(), case=False)
    assert res["solution"]["text"] == "abcd"


# validation -------------------------------------------------------------------------
@pytest.mark.parametrize("fields, fragment", [
    ({"module": "nope"}, "valid modules"),
    ({"module": 5}, "module"),
    ({"body": None}, "body"),
    ({"body": ""}, "body"),
    ({"body": "%%%"}, "body"),
    ({"body": 123}, "body"),
    ({"case": "yes"}, "case"),
    ({"numeric": 5}, "numeric"),
    ({"numeric": True}, "numeric"),
    ({"numeric": "1"}, "numeric"),
    ({"minLength": -1}, "minLength"),
    ({"maxLength": 65}, "maxLength"),
    ({"minLength": 6, "maxLength": 4}, "minLength"),
])
def test_parse_rejects_bad_payloads(fields, fragment):
    with pytest.raises(SolverError) as err:
        parse_text(_task(**fields))
    assert err.value.code == "ERROR_INVALID_TASK_DATA"
    assert fragment in err.value.description


def test_parse_requires_body():
    with pytest.raises(SolverError) as err:
        parse_text({"type": "ImageToTextTask"})
    assert err.value.code == "ERROR_INVALID_TASK_DATA"


def test_unknown_module_lists_the_valid_ones(make_client, fakes):  # noqa: F811
    res = _create(make_client(), module="recaptcha")
    assert res["errorCode"] == "ERROR_INVALID_TASK_DATA"
    for name in ("bls_poland", "common", "general", "ddddocr"):
        assert name in res["errorDescription"]
    assert fakes.calls == []


def test_parse_accepts_data_url_and_defaults():
    job = parse_text(_task(body="data:image/png;base64," + png_b64()))
    assert job.image == png_bytes() and job.module == "common"
    assert (job.numeric, job.case, job.min_length, job.max_length) == (0, None, 0, 0)


def test_body_size_limit():
    big = base64.b64encode(b"\0" * (MAX_IMAGE_BYTES + 1)).decode()
    with pytest.raises(SolverError) as err:
        parse_text(_task(body=big))
    assert err.value.code == "ERROR_INVALID_TASK_DATA"


# error mapping ---------------------------------------------------------------------
@pytest.mark.parametrize("raised, code", [
    (InvalidImage("body is not a readable image"), "ERROR_INVALID_TASK_DATA"),
    (ModelUnavailable("weights missing"), "ERROR_TASK_NOT_SUPPORTED"),
    (RuntimeError("model bug"), "ERROR_CAPTCHA_UNSOLVABLE"),
])
def test_api_maps_recogniser_errors(make_client, fakes, raised, code):  # noqa: F811
    def boom(image):
        raise raised

    fakes.modules[text._BLS_POLAND_PATH].solve = boom
    res = _create(make_client(), module="bls_poland")
    assert res["errorId"] == 1 and res["errorCode"] == code and res["status"] == "failed"


def test_missing_bls_poland_model_is_not_supported(make_client, fakes):  # noqa: F811
    del fakes.modules[text._BLS_POLAND_PATH]
    res = _create(make_client(), module="bls-poland")
    assert res["errorCode"] == "ERROR_TASK_NOT_SUPPORTED"


def test_no_backend_at_all_is_not_supported(make_client, fakes):  # noqa: F811
    fakes.answers["general"] = ModelUnavailable("weights missing")
    fakes.answers["ddddocr"] = ModelUnavailable("ddddocr is not installed")
    res = _create(make_client())
    assert res["errorCode"] == "ERROR_TASK_NOT_SUPPORTED"


@pytest.mark.parametrize("output", [(None, 0.9), ("abc", "high"), ("abc", float("nan"))])
def test_bad_model_output_is_unsolvable(fakes, output):
    fakes.answers["bls_poland"] = output
    with pytest.raises(SolverError) as err:
        run_text(parse_text(_task(module="bls_poland")))
    assert err.value.code == "ERROR_CAPTCHA_UNSOLVABLE"


def test_real_undecodable_body_is_invalid_data():
    """The real ddddocr path decodes the image first: garbage bytes are bad input."""
    with pytest.raises(SolverError) as err:
        run_text(parse_text(_task(module="ddddocr", body=base64.b64encode(
            b"definitely not an image").decode())))
    assert err.value.code in ("ERROR_INVALID_TASK_DATA", "ERROR_TASK_NOT_SUPPORTED")


# queue + usage + health -------------------------------------------------------------
def test_queue_full_answers_no_slot(make_client, fakes):  # noqa: F811
    gate, started = threading.Event(), threading.Event()

    def slow(image):
        started.set()
        gate.wait(5)
        return "SLOW1", 0.9

    fakes.modules[text._BLS_POLAND_PATH].solve = slow
    client = make_client(vision_concurrency=1, max_classify_queue=0)
    manager = client.app.state.manager
    task, future = manager.submit_classification(KEY, _task(module="bls_poland"))
    assert started.wait(5)
    try:
        res = _create(client, module="bls_poland")
        assert res["errorCode"] == "ERROR_NO_SLOT_AVAILABLE"
        assert manager.stats()["classify_running"] == 1
    finally:
        gate.set()
    future.result(timeout=5)
    done = manager.finish_classification(task, future)
    assert done.status == "ready" and done.solution["text"] == "SLOW1"
    assert _create(client, module="bls_poland")["status"] == "ready"  # slot released


def test_text_shares_the_queue_with_classification(make_client, fakes, monkeypatch):  # noqa: F811
    from aws_capsolver import vision_bridge
    from .conftest import jpeg_b64
    gate, started = threading.Event(), threading.Event()

    def slow_grid(images, target):
        started.set()
        gate.wait(5)
        return [0]

    monkeypatch.setattr(vision_bridge, "classify_grid", slow_grid)
    client = make_client(vision_concurrency=1, max_classify_queue=0)
    manager = client.app.state.manager
    task, future = manager.submit_classification(KEY, {
        "type": "AwsWafClassification", "question": "aws:grid:bed",
        "images": [jpeg_b64()]})
    assert started.wait(5)
    try:
        assert _create(client)["errorCode"] == "ERROR_NO_SLOT_AVAILABLE"
    finally:
        gate.set()
    future.result(timeout=5)
    manager.finish_classification(task, future)


def test_usage_counts_the_text_type(make_client, fakes):  # noqa: F811
    client = make_client()
    _create(client, module="bls_poland")
    _create(client, module="bls_poland", maxLength=2)
    usage = client.post("/getBalance", json={"clientKey": KEY}).json()["usage"]
    assert usage["ImageToTextTask"] == {"created": 2, "solved": 1, "failed": 1}


def test_health_lists_text_modules(make_client):  # noqa: F811
    health = make_client().get("/health").json()
    assert set(health["text"]) == set(text.MODULES)
    assert all(isinstance(v, bool) for v in health["text"].values())
    assert "classify_capacity" in health["load"]


def test_warmup_never_raises(monkeypatch):
    def broken():
        raise RuntimeError("no onnxruntime")

    monkeypatch.setattr(common, "warmup", broken)
    monkeypatch.setattr(text, "_has", lambda path: path == "ddddocr")
    assert text.warmup() == {"bls_poland": None, "general": None, "ddddocr": False}


# common.py internals (no model needed) ----------------------------------------------
def test_vote_weights_agreement():
    reading = common.vote([("abcd", 0.6), ("abcd", 0.5), ("abed", 0.9), ("", 0.0)], 4)
    assert reading.text == "abcd"
    assert reading.confidence == pytest.approx(1.1 / 4)
    assert [t for t, _ in reading.candidates] == ["abcd", "abed"]
    assert common.vote([], 4) == Reading("", 0.0, [])


def test_render_variants_drop_duplicates():
    from PIL import Image
    flat = Image.new("L", (100, 64), 0)
    flat.paste(255, (0, 0, 50, 64))  # already pure black/white: most renderings coincide
    names = [name for name, _ in common.render_variants(flat)]
    assert names[0] == "plain" and len(names) == len(set(names)) and len(names) < 4


@pytest.mark.parametrize("data", [b"not an image", b""])
def test_load_gray_rejects_garbage(data):
    with pytest.raises(InvalidImage):
        common._load_gray(data)


def test_load_gray_rejects_huge_and_flattens_alpha():
    from PIL import Image
    buf = io.BytesIO()
    Image.new("L", (common.MAX_SIDE + 1, 10)).save(buf, format="PNG")
    with pytest.raises(InvalidImage):
        common._load_gray(buf.getvalue())
    buf = io.BytesIO()
    Image.new("RGBA", (40, 20), (0, 0, 0, 0)).save(buf, format="PNG")  # black under alpha 0
    gray = common._load_gray(buf.getvalue())
    assert gray.size == (128, common.MODEL_HEIGHT) and gray.getextrema() == (255, 255)


# real models ---------------------------------------------------------------------
def _need(module: str):
    if SKIP_MODELS:
        pytest.skip("AWS_CAPSOLVER_SKIP_MODEL_TESTS=1")
    if not text.available(module):
        pytest.skip(f"{module} backend is not installed")


def test_real_ddddocr_reads_rendered_text():
    _need("ddddocr")
    try:
        solution = run_text(parse_text(_task(module="ddddocr")))
    except SolverError as exc:
        if exc.code == "ERROR_TASK_NOT_SUPPORTED":
            pytest.skip(exc.description)
        raise
    assert solution["text"] == "ab12x" and solution["confidence"] > 0.5
    digits = run_text(parse_text(_task(module="ddddocr", body=png_b64("4821"), numeric=1)))
    assert digits["text"] == "4821"


def test_real_ddddocr_flat_image_is_unsolvable():
    _need("ddddocr")
    from PIL import Image
    buf = io.BytesIO()
    Image.new("RGB", (120, 40), "white").save(buf, format="PNG")
    with pytest.raises(SolverError) as err:
        run_text(parse_text(_task(module="ddddocr",
                                  body=base64.b64encode(buf.getvalue()).decode())))
    assert err.value.code in ("ERROR_CAPTCHA_UNSOLVABLE", "ERROR_TASK_NOT_SUPPORTED")


@pytest.mark.parametrize("module", ["bls_poland", "common"])
def test_real_model_contract(module):
    """The dedicated / general model when present: skips while its weights are missing."""
    _need(module)
    try:
        solution = run_text(parse_text(_task(module=module, body=png_b64("7Kp3m"))))
    except SolverError as exc:
        if exc.code == "ERROR_TASK_NOT_SUPPORTED":
            pytest.skip(f"{module}: {exc.description}")
        if exc.code == "ERROR_CAPTCHA_UNSOLVABLE":
            return  # a model trained on other captchas may read nothing here
        raise
    assert isinstance(solution["text"], str) and solution["text"]
    assert 0.0 <= solution["confidence"] <= 1.0
