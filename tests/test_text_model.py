"""BLS Poland text captcha reader: decoding, input validation, model loading, accuracy smoke."""

import io
import math
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from aws_capsolver.text import bls_poland  # noqa: E402
from aws_capsolver.text.errors import InvalidImage, ModelUnavailable  # noqa: E402
from aws_capsolver.text.model import (CRNN, IMG_H, IMG_W, NUM_CLASSES,  # noqa: E402
                                      constrained_decode, greedy_decode)

FIXTURES = Path(__file__).parent / "fixtures" / "bls_poland"
HAS_WEIGHTS = bls_poland.DEFAULT_WEIGHTS.is_file()
needs_weights = pytest.mark.skipif(not HAS_WEIGHTS, reason="weights/bls_poland_ocr.pt missing")


def _logp(seq, T=40, hi=0.9):
    lp = np.full((T, NUM_CLASSES), (1 - hi) / (NUM_CLASSES - 1))
    for t, s in enumerate(seq + [0] * (T - len(seq))):
        lp[t] = (1 - hi) / (NUM_CLASSES - 1)
        lp[t, s] = hi
    return np.log(lp)


def _png(size=(160, 55), color=(255, 255, 255)) -> bytes:
    from PIL import Image
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format="PNG")
    return buf.getvalue()


# decoding ---------------------------------------------------------------------
def test_constrained_decode_simple():
    seq = [0, 2, 2, 0, 3, 0, 4, 4, 0, 5, 0, 6]  # "12345"
    text, conf, probs = constrained_decode(_logp(seq), 5)
    assert text == "12345" and greedy_decode(_logp(seq)) == "12345"
    assert len(probs) == 5 and 0 < conf <= 1
    assert math.isclose(conf, 0.9 ** 5, rel_tol=1e-6)


def test_constrained_decode_repeated_digit_needs_blank():
    seq = [0, 2, 2, 0, 2, 0, 3, 0, 4, 0, 5]  # "11234"
    assert constrained_decode(_logp(seq), 5)[0] == "11234"


def test_constrained_decode_forces_length():
    seq = [0, 2, 0, 3, 0, 4, 0, 5]  # only 4 labels on the greedy path
    text, conf, _ = constrained_decode(_logp(seq), 5)
    assert len(text) == 5 and conf < 0.9 ** 4
    seq6 = [0, 2, 0, 3, 0, 4, 0, 5, 0, 6, 0, 7]
    assert len(constrained_decode(_logp(seq6), 5)[0]) == 5


def test_model_shapes_and_size():
    m = CRNN().eval()
    with torch.no_grad():
        out = m(torch.zeros(2, 3, IMG_H, IMG_W))
    assert out.shape == (2, 40, NUM_CLASSES)
    n_bytes = sum(p.numel() for p in m.parameters()) * 4
    assert n_bytes < 10 * 1024 * 1024


# input validation --------------------------------------------------------------
@pytest.mark.parametrize("bad", [b"", b"not an image", b"\xff\xd8\xff" + b"0" * 50])
def test_invalid_image(bad):
    with pytest.raises(InvalidImage):
        bls_poland.solve(bad)


def test_oversized_image():
    with pytest.raises(InvalidImage):
        bls_poland.solve(b"x" * (bls_poland.MAX_BYTES + 1))
    with pytest.raises(InvalidImage):
        bls_poland.solve(_png(size=(4000, 30)))


def test_invalid_image_is_value_error():
    assert issubclass(InvalidImage, ValueError)
    assert issubclass(ModelUnavailable, RuntimeError)


def test_missing_weights(monkeypatch, tmp_path):
    monkeypatch.setattr(bls_poland, "_model", None)
    monkeypatch.setenv("AWS_CAPSOLVER_BLS_POLAND_WEIGHTS", str(tmp_path / "nope.pt"))
    with pytest.raises(ModelUnavailable):
        bls_poland.solve(_png())


def test_corrupt_weights(monkeypatch, tmp_path):
    bad = tmp_path / "bad.pt"
    bad.write_bytes(b"garbage")
    monkeypatch.setattr(bls_poland, "_model", None)
    monkeypatch.setenv("AWS_CAPSOLVER_BLS_POLAND_WEIGHTS", str(bad))
    with pytest.raises(ModelUnavailable):
        bls_poland.solve(_png())


def test_solve_many_empty():
    assert bls_poland.solve_many([]) == []


# trained model -------------------------------------------------------------------
@needs_weights
def test_weights_small_and_safe_to_load():
    assert bls_poland.DEFAULT_WEIGHTS.stat().st_size < 10 * 1024 * 1024
    ckpt = torch.load(bls_poland.DEFAULT_WEIGHTS, weights_only=True)
    assert ckpt["arch"] == "crnn_v2"


@needs_weights
def test_real_fixtures(monkeypatch):
    monkeypatch.delenv("AWS_CAPSOLVER_BLS_POLAND_WEIGHTS", raising=False)
    monkeypatch.setattr(bls_poland, "_model", None)
    files = sorted(FIXTURES.glob("*.jpg"))
    assert files
    images = [f.read_bytes() for f in files]
    results = bls_poland.solve_many(images)
    for f, (text, conf) in zip(files, results):
        assert text == f.stem
        assert 0.0 <= conf <= 1.0
    # single-image path agrees with the batch path and is deterministic
    for img, res in zip(images, results):
        a, b = bls_poland.solve(img), bls_poland.solve(img)
        assert a == b
        assert a[0] == res[0] and abs(a[1] - res[1]) < 1e-4


@needs_weights
def test_blank_image_gives_five_digits_low_confidence(monkeypatch):
    monkeypatch.delenv("AWS_CAPSOLVER_BLS_POLAND_WEIGHTS", raising=False)
    monkeypatch.setattr(bls_poland, "_model", None)
    text, conf = bls_poland.solve(_png())
    assert len(text) == 5 and text.isdigit()
    assert conf < 0.5


@needs_weights
def test_thread_safe(monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    monkeypatch.delenv("AWS_CAPSOLVER_BLS_POLAND_WEIGHTS", raising=False)
    monkeypatch.setattr(bls_poland, "_model", None)
    files = sorted(FIXTURES.glob("*.jpg"))
    with ThreadPoolExecutor(8) as ex:
        out = list(ex.map(lambda f: bls_poland.solve(f.read_bytes())[0], files * 4))
    assert out == [f.stem for f in files] * 4
