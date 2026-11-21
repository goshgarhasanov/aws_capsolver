"""Tests of the general text-captcha reader (aws_capsolver.text.general)."""

import io
import itertools
import random
import string
import threading

import numpy as np
import pytest

from aws_capsolver.text import general, general_model as gm
from aws_capsolver.text.errors import InvalidImage, ModelUnavailable

HAVE_WEIGHTS = general.DEFAULT_WEIGHTS.exists()
needs_model = pytest.mark.skipif(not HAVE_WEIGHTS, reason="weights/general_ocr.pt missing")


def _render(text: str, size=(200, 60), bg="white", fg="black", fmt="PNG") -> bytes:
    """A plain, clean captcha-like image (DejaVu Sans when available)."""
    from PIL import Image, ImageDraw, ImageFont
    img = Image.new("RGB", size, bg)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", int(size[1] * 0.6))
    except OSError:
        font = ImageFont.load_default()
    d = ImageDraw.Draw(img)
    box = d.textbbox((0, 0), text, font=font)
    d.text(((size[0] - box[2]) / 2, (size[1] - box[3]) / 2 - box[1] / 2), text, font=font,
           fill=fg)
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


# pure decoding helpers (no model needed) ----------------------------------------
def test_ctc_log_likelihood_matches_torch():
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(1)
    for _ in range(10):
        T = int(rng.integers(6, 20))
        lp = gm.log_softmax(rng.normal(0, 2, (T, gm.NUM_CLASSES)))
        label = [int(v) for v in rng.integers(1, gm.NUM_CLASSES, size=int(rng.integers(1, 5)))]
        if len(label) > 1 and rng.random() < 0.5:
            label[1] = label[0]          # repeated character needs a blank in between
        ref = -torch.nn.functional.ctc_loss(
            torch.tensor(lp)[:, None], torch.tensor([label]), torch.tensor([T]),
            torch.tensor([len(label)]), reduction="sum").item()
        assert general.ctc_log_likelihood(lp, label) == pytest.approx(ref, abs=1e-6)


def test_constrained_viterbi_is_exact_on_small_problems():
    rng = np.random.default_rng(2)
    mask = np.zeros(gm.NUM_CLASSES, dtype=bool)
    mask[[0, 1, 2, 3]] = True
    for _ in range(25):
        T = 5
        lp = gm.log_softmax(rng.normal(0, 2, (T, gm.NUM_CLASSES)))
        lo, hi = int(rng.integers(0, 3)), int(rng.integers(2, 4))
        best = None
        for path in itertools.product([0, 1, 2, 3], repeat=T):
            out, prev = [], 0
            for c in path:
                if c and c != prev:
                    out.append(c)
                prev = c
            if lo <= len(out) <= hi:
                s = sum(lp[t, c] for t, c in enumerate(path))
                if best is None or s > best:
                    best = s
        got = general.constrained_viterbi(lp, mask, lo, hi)
        if best is None:
            assert got is None
        else:
            assert lo <= len(got[0]) <= hi
            assert got[1] == pytest.approx(best, abs=1e-9)


def test_beam_search_respects_mask_and_max_len():
    rng = np.random.default_rng(3)
    lp = gm.log_softmax(rng.normal(0, 3, (32, gm.NUM_CLASSES)))
    mask = gm.charset_mask("digits")
    for prefix, _ in gm.ctc_beam_search(lp, mask, beam=6, max_len=4):
        text = gm.classes_to_text(prefix)
        assert len(text) <= 4 and text.isdigit() or text == ""


def test_charset_masks():
    assert gm.charset_mask("digits")[1:11].all() and not gm.charset_mask("digits")[11:].any()
    letters = gm.charset_mask("letters")
    assert not letters[1:11].any() and letters[11:].all()
    explicit = gm.charset_mask("abc123")
    allowed = {gm.CHARSET[i - 1] for i in np.flatnonzero(explicit) if i}
    assert allowed == set("abc123")
    assert all(m[gm.BLANK] for m in (letters, explicit, gm.charset_mask(None)))


def test_fold_case_merges_probability_mass():
    lp = np.full((1, gm.NUM_CLASSES), -50.0)
    lp[0, gm.CHAR_TO_INDEX["a"]] = np.log(0.3)
    lp[0, gm.CHAR_TO_INDEX["A"]] = np.log(0.3)
    lp[0, gm.CHAR_TO_INDEX["b"]] = np.log(0.4)
    folded, fmask = gm.fold_case(lp, gm.charset_mask(None))
    assert np.exp(folded[0, gm.CHAR_TO_INDEX["a"]]) == pytest.approx(0.6, abs=1e-6)
    assert not fmask[gm.CHAR_TO_INDEX["A"]]
    # an upper-case-only charset keeps the upper-case classes
    up, umask = gm.fold_case(lp, gm.charset_mask("upper"))
    assert umask[gm.CHAR_TO_INDEX["A"]] and not umask[gm.CHAR_TO_INDEX["a"]]


# input validation ---------------------------------------------------------------
@pytest.mark.parametrize("bad", [b"", b"not an image", b"\x89PNG\r\n\x1a\n" + b"\0" * 40])
def test_invalid_images(bad):
    with pytest.raises(InvalidImage):
        general.load_image(bad)


def test_too_small_and_too_large_images():
    from PIL import Image
    for size in ((4, 4), (3000, 50)):
        buf = io.BytesIO()
        Image.new("RGB", size, "white").save(buf, format="PNG")
        with pytest.raises(InvalidImage):
            general.load_image(buf.getvalue())


def test_transparent_png_is_flattened_onto_white():
    from PIL import Image
    img = Image.new("RGBA", (40, 20), (0, 0, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    rgb = general.load_image(buf.getvalue())
    assert rgb.mode == "RGB" and rgb.getpixel((5, 5)) == (255, 255, 255)


def test_bad_arguments():
    with pytest.raises(ValueError):
        general.solve(_render("abc"), min_len=6, max_len=4)
    with pytest.raises(ValueError):
        general.solve(_render("abc"), charset="#$%")


def test_missing_weights_without_ddddocr_raises_model_unavailable(monkeypatch, tmp_path):
    monkeypatch.setenv(general.WEIGHTS_ENV, str(tmp_path / "nope.pt"))
    monkeypatch.setattr(general, "_engine", None)
    monkeypatch.setattr(general, "_failed", None)
    monkeypatch.setattr(general, "_Dddd", lambda: (_ for _ in ()).throw(ImportError("x")))
    with pytest.raises(ModelUnavailable):
        general.solve(_render("abc"))
    with pytest.raises(ModelUnavailable):   # the failure is remembered
        general.solve(_render("abc"))


# with the trained model ---------------------------------------------------------
@needs_model
@pytest.mark.parametrize("text", ["7K3PQ", "a8bx2", "50917", "XWKZT"])
def test_reads_clean_text(text):
    out, conf = general.solve(_render(text))
    assert out.lower() == text.lower()
    assert 0.0 <= conf <= 1.0


@needs_model
def test_light_on_dark_and_jpeg():
    out, _ = general.solve(_render("48213", bg=(20, 20, 40), fg=(240, 240, 120), fmt="JPEG"))
    assert out == "48213"


@needs_model
def test_charset_is_a_constrained_decode():
    data = _render("B0O8S")
    text, _ = general.solve(data, charset="digits")
    assert text.isdigit() and len(text) >= 1
    text, _ = general.solve(data, charset="letters")
    assert text.isalpha()
    text, _ = general.solve(data, charset="0123456789ABCDEF")
    assert set(text) <= set("0123456789ABCDEFabcdef")


@needs_model
def test_length_limits_shape_the_answer():
    data = _render("ab12cd")
    for lo, hi in ((3, 3), (8, 9), (6, 6)):
        text, _ = general.solve(data, min_len=lo, max_len=hi)
        assert lo <= len(text) <= hi


@needs_model
def test_case_sensitive_reading():
    text, _ = general.solve(_render("AbGhE"), case_sensitive=True)
    assert text == "AbGhE"   # letters whose two cases differ in shape, not only size


@needs_model
def test_deterministic_and_thread_safe():
    images = [_render("".join(random.Random(i).choice(string.ascii_lowercase + string.digits)
                               for _ in range(5))) for i in range(6)]
    expected = [general.solve(im) for im in images]
    assert expected == [general.solve(im) for im in images]
    results = {}

    def worker(k):
        results[k] = [general.solve(im) for im in images]

    threads = [threading.Thread(target=worker, args=(k,)) for k in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert all(r == expected for r in results.values())


@needs_model
def test_confidence_separates_text_from_noise():
    from PIL import Image
    rng = np.random.default_rng(0)
    noise = Image.fromarray(rng.integers(0, 255, (60, 200, 3), dtype=np.uint8))
    buf = io.BytesIO()
    noise.save(buf, format="PNG")
    _, conf_noise = general.solve(buf.getvalue())
    _, conf_text = general.solve(_render("k4m9p"))
    assert conf_text > conf_noise
