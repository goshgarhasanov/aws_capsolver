"""BlsClassification: the BLS grid captcha ("select all boxes with number NNN")."""

import base64
import json
from pathlib import Path

import pytest

from aws_capsolver import bls_bridge
from aws_capsolver.tasks import parse_classification, run_classification
from aws_capsolver.waf.errors import SolverError

from .conftest import jpeg_b64
from .test_api import KEY, make_client  # noqa: F401  (fixture re-export)

BLS_DATA = Path(__file__).resolve().parents[2] / "bls_capsolver" / "data" / "real"


def _task(**fields):
    task = {"type": "BlsClassification", "images": [jpeg_b64() for _ in range(9)]}
    task.update(fields)
    return task


def test_parse_question_and_target_forms():
    family, target, images = parse_classification(_task(question="bls:grid:748"))
    assert (family, target, len(images)) == ("bls", "748", 9)
    assert parse_classification(_task(target=" 042 "))[1] == "042"


@pytest.mark.parametrize("fields", [
    {"question": "bls:grid:74"},
    {"question": "bls:grid:abc"},
    {"question": "aws:grid:748"},
    {"question": "bls:row:748"},
    {"target": "1234"},
    {},
    {"target": "748", "images": [jpeg_b64() for _ in range(8)]},
    {"target": "748", "images": "not-a-list"},
    {"target": "748", "images": ["%%%"] + [jpeg_b64() for _ in range(8)]},
])
def test_parse_rejects_bad_payloads(fields):
    with pytest.raises(SolverError) as err:
        parse_classification(_task(**fields))
    assert err.value.code == "ERROR_INVALID_TASK_DATA"


def test_api_ready_with_objects_and_readings(make_client, monkeypatch):  # noqa: F811
    seen = {}

    def fake_match(images, target):
        seen["n"], seen["target"] = len(images), target
        return [4, 1], ["748"] * 9

    monkeypatch.setattr(bls_bridge, "match", fake_match)
    client = make_client()
    res = client.post("/createTask", json={"clientKey": KEY, "task": _task(
        question="bls:grid:748")}).json()
    assert res["errorId"] == 0 and res["status"] == "ready"
    assert res["solution"] == {"objects": [4, 1], "readings": ["748"] * 9}
    assert seen == {"n": 9, "target": "748"}
    again = client.post("/getTaskResult", json={"clientKey": KEY, "taskId": res["taskId"]}).json()
    assert again["solution"]["objects"] == [4, 1]


@pytest.mark.parametrize("raised, code", [
    (bls_bridge.BlsUnavailable("not installed"), "ERROR_TASK_NOT_SUPPORTED"),
    (bls_bridge.BlsInvalidInput("images[0] is not a readable image"), "ERROR_INVALID_TASK_DATA"),
])
def test_api_maps_bridge_errors(make_client, monkeypatch, raised, code):  # noqa: F811
    def fake_match(images, target):
        raise raised

    monkeypatch.setattr(bls_bridge, "match", fake_match)
    res = make_client().post("/createTask", json={"clientKey": KEY, "task": _task(
        target="748")}).json()
    assert res["errorId"] == 1 and res["errorCode"] == code


def test_no_match_is_unsolvable(monkeypatch):
    monkeypatch.setattr(bls_bridge, "match", lambda images, target: ([], ["111"] * 9))
    with pytest.raises(SolverError) as err:
        run_classification(*parse_classification(_task(target="748")))
    assert err.value.code == "ERROR_CAPTCHA_UNSOLVABLE"


def test_missing_package_reports_unavailable(monkeypatch):
    monkeypatch.setattr(bls_bridge, "_module", False)
    assert not bls_bridge.available()
    with pytest.raises(bls_bridge.BlsUnavailable):
        bls_bridge.match([b"x"] * 9, "748")


def test_undecodable_tile_is_invalid_input():
    if not bls_bridge.available():
        pytest.skip("bls_capsolver is not installed")
    data = [base64.b64decode(jpeg_b64())] * 8 + [b"definitely not an image"]
    with pytest.raises(bls_bridge.BlsInvalidInput):
        bls_bridge.match(data, "748")


def _real_grid():
    grids = BLS_DATA / "grids.jsonl"
    labels = BLS_DATA / "labels.json"
    if not grids.exists() or not labels.exists():
        return None
    tile_labels = json.loads(labels.read_text())
    for line in grids.read_text().splitlines():
        grid = json.loads(line)
        paths = [BLS_DATA / "raw" / name for name in grid["tiles"]]
        if all(name in tile_labels for name in grid["tiles"]) and all(p.exists() for p in paths):
            truth = [i for i, name in enumerate(grid["tiles"])
                     if tile_labels[name] == grid["target"]]
            return grid["target"], [p.read_bytes() for p in paths], truth
    return None


def test_real_model_on_a_labelled_grid():
    """End to end with the real CNN when bls_capsolver and its local data are present."""
    if not bls_bridge.available():
        pytest.skip("bls_capsolver is not installed")
    sample = _real_grid()
    if sample is None:
        pytest.skip("no local BLS grids")
    target, tiles, truth = sample
    images = [base64.b64encode(t).decode() for t in tiles]
    solution = run_classification(*parse_classification(_task(target=target, images=images)))
    assert solution["objects"] == truth
    assert len(solution["readings"]) == 9
