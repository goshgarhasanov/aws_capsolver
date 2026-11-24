"""reCAPTCHA v2 classifier: question parsing, input validation, grid splitting, model smoke tests.

Model tests skip when torch / open_clip or the backbone weights are unavailable; the
accuracy check on real challenges skips when the trained head or the dataset is missing.
"""
import base64
import io
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

import aws_capsolver.recaptcha as R
from aws_capsolver.recaptcha import InvalidRecaptchaInput, RecaptchaError, UnsupportedQuestion, normalize_question

ROOT = Path(__file__).resolve().parents[1]
HEAD = ROOT / "weights" / "recaptcha_head.pt"
REVIEW3 = ROOT / "data" / "recaptcha" / "bx" / "data" / "3x3" / "review"


def test_import_is_cheap_and_errors_are_typed():
    code = ("import sys, aws_capsolver.recaptcha as r; "
            "print(','.join(m for m in ('torch', 'open_clip', 'numpy') if m in sys.modules))")
    out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == ""
    assert issubclass(UnsupportedQuestion, RecaptchaError)
    assert issubclass(InvalidRecaptchaInput, RecaptchaError) and issubclass(InvalidRecaptchaInput, ValueError)
    assert callable(R.classify) and callable(R.warm)


@pytest.mark.parametrize("q,key", [
    ("/m/0k4j", "car"), ("/m/04_sv", "motorcycle"), ("/m/0199g", "bicycle"), ("/m/01bjv", "bus"),
    ("/m/07r04", "truck"), ("/m/015qff", "traffic_light"), ("/m/01pns0", "fire_hydrant"),
    ("/m/014xcs", "crosswalk"), ("/m/015kr", "bridge"), ("/m/01lynh", "stairs"), ("/m/01jk_4", "chimney"),
    ("/m/0pg52", "taxi"), ("/m/015qbp", "parking_meter"), ("/m/0cdl1", "palm_tree"),
    ("/m/09d_r", "mountain"), ("/m/019jd", "boat"), ("/m/013xlm", "tractor"), ("/m/02yvhj", "school_bus"),
    ("cars", "car"), ("Select all images with traffic lights", "traffic_light"),
    ("Select all squares with motorcycles. If there are none, click skip", "motorcycle"),
    ("Select all images with a fire hydrant", "fire_hydrant"), ("school bus", "school_bus"),
    ("arabalar", "car"), ("Tüm trafik ışıklarını seçin", "traffic_light"), ("yaya geçitleri", "crosswalk"),
    ("TAKSİLER", "taxi"), ("dağlar veya tepeler", "mountain"), ("svetoforlar", "traffic_light"),
    ("Bütün piyada keçidlərini seçin", "crosswalk"), ("светофоры", "traffic_light"),
    ("Выберите все изображения, где есть автомобили", "car"), ("велосипеды", "bicycle"),
    ("Selecciona todas las imágenes con autobuses", "bus"), ("Wählen Sie alle Bilder mit Zebrastreifen aus",
                                                              "crosswalk"),
    ("Sélectionnez toutes les images contenant des bouches d'incendie", "fire_hydrant"),
    ("palmiye ağaçları", "palm_tree"), ("Выберите все изображения с лестницами", "stairs"),
])
def test_normalize_question(q, key):
    assert normalize_question(q) == key


def test_open_vocabulary_and_unsupported_questions():
    assert normalize_question("Select all images with guitars") == "guitar"
    for bad in ("/m/zzzzzz", "", "   ", "?!", "x" * 400):
        with pytest.raises(UnsupportedQuestion):
            normalize_question(bad)
    with pytest.raises(InvalidRecaptchaInput):
        normalize_question(None)


def _jpeg(im) -> bytes:
    buf = io.BytesIO()
    im.save(buf, "JPEG")
    return buf.getvalue()


@pytest.mark.parametrize("bad", [b"", "", "!!!not base64!!!", b"not an image", 123, None])
def test_rejects_bad_images(bad):
    with pytest.raises(InvalidRecaptchaInput):
        R.classify(bad, "/m/0k4j")


def test_rejects_bad_sizes_and_shapes():
    Image = pytest.importorskip("PIL.Image")
    img = _jpeg(Image.new("RGB", (300, 300)))
    for size in (0, 2, 5, "x", True, 3.5):
        with pytest.raises(InvalidRecaptchaInput):
            R.classify(img, "/m/0k4j", size)
    with pytest.raises(InvalidRecaptchaInput):
        R.classify(_jpeg(Image.new("RGB", (900, 100))), "/m/0k4j")  # not square
    with pytest.raises(InvalidRecaptchaInput):
        R.classify(_jpeg(Image.new("RGB", (10, 10))), "/m/0k4j")  # too small
    with pytest.raises(UnsupportedQuestion):
        R.classify(img, "/m/unknown")


def test_split_and_size_inference():
    np = pytest.importorskip("numpy")
    Image = pytest.importorskip("PIL.Image")
    from aws_capsolver.recaptcha import features

    rng = np.random.default_rng(0)
    # 3x3: nine flat tiles of different brightness -> strong seams at the thirds
    a = np.zeros((300, 300, 3), np.uint8)
    for k in range(9):
        a[(k // 3) * 100:(k // 3 + 1) * 100, (k % 3) * 100:(k % 3 + 1) * 100] = rng.integers(0, 255, 3)
    g3 = Image.fromarray(a)
    assert features.infer_size(g3) == 3
    assert [c.size for c in features.split_grid(g3, 3)] == [(100, 100)] * 9
    # 4x4: one smooth photo (gradient + noise), 450 px
    y, x = np.mgrid[0:450, 0:450]
    b = np.stack([x * 0.5, y * 0.5, (x + y) * 0.25], -1) + rng.normal(0, 4, (450, 450, 3))
    g4 = Image.fromarray(np.clip(b, 0, 255).astype(np.uint8))
    assert features.infer_size(g4) == 4
    assert len(features.model_crops(g4, 4)) == 17
    assert features.infer_size(Image.new("RGB", (100, 100))) == 1
    # a big 3x3 (e.g. 450 px) is still recognised as 3x3 by its seams
    assert features.infer_size(g3.resize((450, 450), Image.NEAREST)) == 3


def _model_or_skip():
    pytest.importorskip("torch")
    pytest.importorskip("open_clip")
    try:
        from aws_capsolver.recaptcha.model import get_engine

        return get_engine()
    except RecaptchaError as exc:
        pytest.skip(f"model unavailable: {exc}")


def test_classify_shapes_and_base64():
    Image = pytest.importorskip("PIL.Image")
    _model_or_skip()
    g = _jpeg(Image.new("RGB", (300, 300), (90, 120, 150)))
    res = R.classify(base64.b64encode(g).decode(), "cars")
    assert res["type"] == "multi" and res["size"] == 3 and all(0 <= i < 9 for i in res["objects"])
    res = R.classify(_jpeg(Image.new("RGB", (450, 450), (90, 120, 150))), "/m/04_sv")
    assert res["type"] == "multi" and res["size"] == 4 and all(0 <= i < 16 for i in res["objects"])
    res = R.classify(_jpeg(Image.new("RGB", (100, 100), (90, 120, 150))), "/m/0k4j")
    assert res == {"type": "single", "hasObject": res["hasObject"], "size": 1}
    assert isinstance(res["hasObject"], bool)
    # open-vocabulary target falls back to zero-shot instead of failing
    res = R.classify(g, "Select all images with guitars", size=3)
    assert res["type"] == "multi"


def test_missing_head_falls_back_to_zero_shot(tmp_path, caplog):
    pytest.importorskip("torch")
    pytest.importorskip("open_clip")
    from aws_capsolver.recaptcha.model import Engine

    try:
        eng = Engine(tmp_path / "missing.pt")
    except RecaptchaError as exc:
        pytest.skip(f"backbone unavailable: {exc}")
    assert eng.head is None
    Image = pytest.importorskip("PIL.Image")
    out = eng.classify(Image.new("RGB", (300, 300), (40, 40, 40)), "car", 3)
    assert out["method"] == "zeroshot" and len(out["probs"]) == 9


@pytest.mark.skipif(not HEAD.is_file() or not (REVIEW3 / "metadata.jsonl").is_file(),
                    reason="trained head or recaptcha-67k review split not available")
def test_accuracy_on_reviewed_3x3_sample():
    _model_or_skip()
    rows = [json.loads(line) for line in open(REVIEW3 / "metadata.jsonl")][:int(os.environ.get("RECAPTCHA_TEST_N", 40))]
    ok = sum(sorted(R.classify((REVIEW3 / r["file_name"]).read_bytes(), r["target"])["objects"])
             == sorted(r["correct_answers"]) for r in rows)
    assert ok / len(rows) >= 0.8, f"{ok}/{len(rows)}"
