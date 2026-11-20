"""Tests for aws_capsolver.vision.

Fast tests cover input validation and selection logic without loading any model.
Tests marked ``model`` load the real backbones (skipped when they are not cached locally,
or when AWS_CAPSOLVER_SKIP_MODEL_TESTS=1).
"""
from __future__ import annotations

import base64
import io
import json
import os
import threading
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from aws_capsolver import vision
from aws_capsolver.vision import grid
from aws_capsolver.vision.errors import (InvalidInput, ModelUnavailable, UnknownTarget,
                                         UnsupportedProblem, VisionError)

FIXTURES = Path(__file__).parent / "fixtures" / "grids"


def _jpeg(color=(120, 120, 120), size=(100, 100)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, "JPEG")
    return buf.getvalue()


def _fixture_grids() -> list[tuple[str, list[bytes], str, list[int]]]:
    out = []
    for d in sorted(FIXTURES.glob("*")):
        meta = json.loads((d / "meta.json").read_text())
        tiles = [(d / f"tile_{i}.jpg").read_bytes() for i in range(9)]
        out.append((d.name, tiles, meta["target"][0], meta["answer"]))
    return out


# ------------------------------------------------------------------ public interface
def test_public_names():
    for name in ("classify_grid", "solve_toycar", "transcribe", "UnsupportedProblem"):
        assert hasattr(vision, name)
    assert issubclass(UnsupportedProblem, VisionError)
    assert issubclass(UnknownTarget, UnsupportedProblem)
    assert issubclass(InvalidInput, ValueError)


def test_toycar_is_unsupported():
    with pytest.raises(UnsupportedProblem):
        vision.solve_toycar(_jpeg())


def test_transcribe_raises_typed_error_when_unavailable():
    try:
        import aws_capsolver.vision.audio  # noqa: F401
    except ImportError:
        with pytest.raises(UnsupportedProblem):
            vision.transcribe("AAAA")
    else:
        pytest.skip("audio solver installed; covered by its own tests")


@pytest.mark.parametrize("ptype,expected", [
    ("gridcaptcha-5-0.1-0", 5), ("gridcaptcha-3-x", 3), ("gridcaptcha-x-0", None),
    ("toycarcity", None), ("", None), (None, None),
])
def test_count_from_problem_type(ptype, expected):
    assert vision.count_from_problem_type(ptype) == expected


# ------------------------------------------------------------------ target handling
@pytest.mark.parametrize("raw,expected", [
    ("clock", "clock"), ("Clock", "clock"), ("the clocks", "clock"), ('["clock"]', "clock"),
    (["bucket"], "bucket"), ("  the   curtains ", "curtain"), ("hats", "hat"),
    ("beds", "bed"), ("bag", "bag"), ("bags", "bag"), ("teapot", "teapot"),
])
def test_normalize_target(raw, expected):
    assert grid.normalize_target(raw) == expected


@pytest.mark.parametrize("raw", ["", "   ", "[]", '["a","b"]', "[bad json", 42, None,
                                  "<script>", "x" * 80, ["clock", "bed"]])
def test_normalize_target_rejects(raw):
    with pytest.raises(InvalidInput):
        grid.normalize_target(raw)


# ------------------------------------------------------------------ tile decoding
def test_decode_tiles_accepts_bytes_base64_and_data_url():
    j = _jpeg()
    b64 = base64.b64encode(j).decode()
    imgs = grid.decode_tiles([j, bytearray(j), memoryview(j), b64, "data:image/jpeg;base64," + b64] + [j] * 4)
    assert len(imgs) == 9 and all(im.mode == "RGB" for im in imgs)


@pytest.mark.parametrize("tiles", [
    [], [b"x"] * 8, [b"x"] * 10, b"notalist", "notalist",
])
def test_decode_tiles_wrong_count_or_type(tiles):
    with pytest.raises(InvalidInput):
        grid.decode_tiles(tiles)


@pytest.mark.parametrize("bad", [b"", b"not an image", "!!!not-base64!!!", 123, None,
                                 b"\xff\xd8\xff" + b"\x00" * 50])
def test_decode_tiles_bad_tile(bad):
    tiles = [_jpeg()] * 8 + [bad]
    with pytest.raises(InvalidInput):
        grid.decode_tiles(tiles)


def test_decode_tiles_rejects_oversized():
    with pytest.raises(InvalidInput):
        grid.decode_tiles([_jpeg(size=(2000, 50))] * 9)
    with pytest.raises(InvalidInput):
        grid.decode_tiles([b"\xff" * (grid.MAX_TILE_BYTES + 1)] * 9)


# ------------------------------------------------------------------ selection
def test_select_tiles_top_k_is_sorted_and_deterministic():
    s = [0.1, 0.9, 0.3, 0.8, 0.7, 0.2, 0.95, 0.6, 0.0]
    assert grid.select_tiles(s, count=5) == [1, 3, 4, 6, 7]
    # ties resolved by the lower index
    assert grid.select_tiles([0.5] * 9, count=3) == [0, 1, 2]
    assert grid.select_tiles(s, count=99) == list(range(9))


def test_select_tiles_threshold_never_empty():
    assert grid.select_tiles([0.1, 0.2, 0.05], threshold=0.5) == [1]
    assert grid.select_tiles([0.6, 0.2, 0.7], threshold=0.5) == [0, 2]


def test_select_tiles_rejects_empty():
    with pytest.raises(InvalidInput):
        grid.select_tiles([])


@pytest.mark.parametrize("kwargs", [{"threshold": 1.5}, {"threshold": -0.1}, {"threshold": "x"},
                                    {"count": 0}, {"count": 10}, {"count": 2.5}, {"count": True}])
def test_classify_grid_rejects_bad_params_without_loading_model(kwargs, monkeypatch):
    monkeypatch.setattr(grid, "get_engine", lambda: pytest.fail("model must not be loaded"))
    with pytest.raises(InvalidInput):
        vision.classify_grid([_jpeg()] * 9, "clock", **kwargs)


def test_classify_grid_rejects_bad_target_without_loading_model(monkeypatch):
    monkeypatch.setattr(grid, "get_engine", lambda: pytest.fail("model must not be loaded"))
    with pytest.raises(InvalidInput):
        vision.classify_grid([_jpeg()] * 9, "")


def test_engine_load_failure_is_typed(monkeypatch):
    with pytest.raises(ModelUnavailable):
        grid.GridEngine(specs=["NoSuchModel-XYZ:none"], head_path="none")


# ------------------------------------------------------------------ real model
def _models_cached() -> bool:
    if os.environ.get("AWS_CAPSOLVER_SKIP_MODEL_TESTS") == "1":
        return False
    hub = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"
    return hub.is_dir() and any(hub.glob("models--*"))


model = pytest.mark.skipif(not _models_cached(), reason="backbone weights not cached")


@model
def test_fixture_grids_are_solved():
    grids = _fixture_grids()
    assert grids, "fixtures missing"
    for name, tiles, target, answer in grids:
        assert vision.classify_grid(tiles, target) == answer, name


@model
def test_head_is_loaded_for_default_engine():
    eng = grid.get_engine()
    assert eng.head is not None, "weights/aws_capsolver_head.pt missing or incompatible"
    assert set(grid.KNOWN_TARGETS) <= set(eng.head["classes"])


@model
def test_threshold_mode_and_base64_input():
    name, tiles, target, answer = _fixture_grids()[0]
    b64 = [base64.b64encode(t).decode() for t in tiles]
    assert vision.classify_grid(b64, json.dumps([target])) == answer
    sel = vision.classify_grid(tiles, target, threshold=0.5)
    assert sel and all(0 <= i < 9 for i in sel)


@model
def test_unknown_target_falls_back_or_raises_in_strict_mode(monkeypatch):
    _, tiles, _, _ = _fixture_grids()[0]
    sel = vision.classify_grid(tiles, "teapot")
    assert len(sel) == 5
    monkeypatch.setenv("AWS_CAPSOLVER_STRICT_TARGETS", "1")
    with pytest.raises(UnknownTarget):
        vision.classify_grid(tiles, "teapot")


@model
def test_concurrent_calls_are_consistent():
    grids = _fixture_grids()
    expected = {n: vision.classify_grid(t, tg) for n, t, tg, _ in grids}
    results, errors = [], []

    def worker(n, t, tg):
        try:
            results.append((n, vision.classify_grid(t, tg)))
        except Exception as exc:  # noqa: BLE001 - collected and asserted below
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=g[:3]) for g in grids * 4]
    for th in threads:
        th.start()
    for th in threads:
        th.join(timeout=120)
    assert not errors
    assert len(results) == len(threads)
    assert all(expected[n] == sel for n, sel in results)


@model
def test_scores_are_probabilities():
    _, tiles, target, _ = _fixture_grids()[0]
    s, method = grid.get_engine().scores(tiles, target)
    assert method in ("head", "zeroshot")
    assert s.shape == (9,) and np.all((s >= 0) & (s <= 1))


# ------------------------------------------------------------------ open vocabulary
@pytest.mark.parametrize("kwargs", [{"top_k": 0}, {"top_k": 21}, {"top_k": "3"}, {"top_k": True}])
def test_describe_tiles_rejects_bad_top_k(kwargs, monkeypatch):
    monkeypatch.setattr(grid, "get_engine", lambda: pytest.fail("model must not be loaded"))
    with pytest.raises(InvalidInput):
        vision.describe_tiles([_jpeg()], **kwargs)


@pytest.mark.parametrize("tiles", [[], [b"x"] * 17, "notalist"])
def test_describe_tiles_rejects_bad_tiles(tiles, monkeypatch):
    monkeypatch.setattr(grid, "get_engine", lambda: pytest.fail("model must not be loaded"))
    with pytest.raises(InvalidInput):
        vision.describe_tiles(tiles)


def test_open_vocabulary_contains_known_and_everyday_objects():
    names = grid.open_vocabulary()
    assert len(names) > 900 and len(set(names)) == len(names)
    for n in (*grid.KNOWN_TARGETS, "acoustic guitar", "airliner", "goldfinch"):
        assert n in names


@model
def test_describe_tiles_names_the_objects():
    hits = total = 0
    for _, tiles, _, _ in _fixture_grids():
        meta_classes = None
        for d in FIXTURES.glob("*"):
            m = json.loads((d / "meta.json").read_text())
            if [(d / f"tile_{i}.jpg").read_bytes() for i in range(9)] == tiles:
                meta_classes = m["tile_classes"]
        desc = vision.describe_tiles(tiles, top_k=3)
        assert len(desc) == 9 and all(len(d) == 3 for d in desc)
        assert all(0.0 <= p <= 1.0 for d in desc for _, p in d)
        for cls, d in zip(meta_classes, desc):
            words = {cls, *grid.SYNONYMS.get(cls, ())}
            hits += any(any(w in name for w in words) for name, _ in d)
            total += 1
    assert hits / total >= 0.85, f"{hits}/{total}"


@model
def test_never_seen_target_uses_open_vocabulary():
    eng = grid.get_engine()
    for _, tiles, target, answer in _fixture_grids():
        feats = eng.embed(grid.decode_tiles(tiles))
        # leave-class-out: the target is treated as unknown (contrast = open vocabulary)
        p = eng.zero_shot_probs(feats, target, open_vocab=True)
        assert grid.select_tiles(p, count=5) == answer
        # an object that is not in the grid gets low scores everywhere
        assert float(eng.zero_shot_probs(feats, "acoustic guitar").max()) < 0.5
