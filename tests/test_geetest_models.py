"""GeeTest model package: input validation, slide / gobang accuracy on small real fixtures."""
import base64
import io
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

import aws_capsolver.geetest as G
from aws_capsolver.geetest import GeeTestError, InvalidGeeTestInput, UnsupportedGeeTest

FIX = Path(__file__).parent / "fixtures" / "geetest"
ROOT = Path(__file__).resolve().parents[1]

np = pytest.importorskip("numpy")
pytest.importorskip("cv2")
Image = pytest.importorskip("PIL.Image")
ImageDraw = pytest.importorskip("PIL.ImageDraw")


def _png(im) -> bytes:
    buf = io.BytesIO()
    im.save(buf, "PNG")
    return buf.getvalue()


def _read(name: str) -> bytes:
    return (FIX / name).read_bytes()


# ------------------------------------------------------------------ package import
def test_import_is_cheap_and_errors_are_typed():
    code = ("import sys, aws_capsolver.geetest as g; "
            "heavy = [m for m in ('torch', 'open_clip', 'cv2', 'numpy') if m in sys.modules]; "
            "print(','.join(heavy))")
    out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == ""
    assert issubclass(UnsupportedGeeTest, GeeTestError)
    assert issubclass(InvalidGeeTestInput, GeeTestError) and issubclass(InvalidGeeTestInput, ValueError)
    for name in ("slide_offset", "click_points", "nine_grid", "gobang", "warmup"):
        assert callable(getattr(G, name))


# ---------------------------------------------------------------------- validation
@pytest.mark.parametrize("bad", [b"", "", "!!!not base64!!!", b"not an image at all", 12345, None, ["x"],
                                 base64.b64encode(b"GIF89a-garbage").decode()])
def test_slide_rejects_bad_background(bad):
    with pytest.raises(InvalidGeeTestInput):
        G.slide_offset(bad)


def test_rejects_oversized_and_tiny_images():
    with pytest.raises(InvalidGeeTestInput):
        G.slide_offset(_png(Image.new("RGB", (4000, 100))))
    with pytest.raises(InvalidGeeTestInput):
        G.slide_offset(_png(Image.new("RGB", (4, 4))))
    with pytest.raises(InvalidGeeTestInput):
        G.slide_offset(b"\x89PNG" + b"0" * (5 * 1024 * 1024))


def test_slide_piece_without_alpha_or_too_big():
    bg = _png(Image.new("RGB", (300, 200), (100, 120, 140)))
    assert isinstance(G.slide_offset(bg, _png(Image.new("RGB", (60, 60), (1, 2, 3)))), int)  # opaque square
    with pytest.raises(InvalidGeeTestInput):
        G.slide_offset(bg, _png(Image.new("RGBA", (60, 60), (0, 0, 0, 0))))  # no opaque part
    with pytest.raises(InvalidGeeTestInput):
        G.slide_offset(bg, _png(Image.new("RGBA", (400, 60), (9, 9, 9, 255))))
    with pytest.raises(InvalidGeeTestInput):
        G.slide_offset(bg, None, _png(Image.new("RGB", (100, 50))))  # full background of another size
    with pytest.raises(InvalidGeeTestInput):
        G.slide_details(bg, y=-3)


def test_click_and_nine_validation():
    img = _png(Image.new("RGB", (300, 200)))
    with pytest.raises(InvalidGeeTestInput):
        G.click_points(img, "slide")
    with pytest.raises(InvalidGeeTestInput):
        G.click_points(img, 3)
    with pytest.raises(InvalidGeeTestInput):
        G.click_points(img, "icon", hints="abc")
    with pytest.raises(InvalidGeeTestInput):
        G.click_points(img, "word", hints=["a"] * 9)
    with pytest.raises(InvalidGeeTestInput):
        G.nine_grid(img, None)
    with pytest.raises(InvalidGeeTestInput):
        G.nine_grid(img, "cat", count=0)
    with pytest.raises(InvalidGeeTestInput):
        G.nine_grid(img, "cat", count=True)


@pytest.mark.parametrize("bad", [[], [[1, 2], [3]], [[1, 2, 3]] * 2, [["a"] * 5] * 5, [[True] * 5] * 5,
                                 [[1] * 12] * 12, [[-1] * 5] * 5])
def test_gobang_rejects_bad_matrix(bad):
    with pytest.raises(InvalidGeeTestInput):
        G.gobang(bad)


# --------------------------------------------------------------------------- slide
def test_v3_unscramble_restores_the_shown_picture():
    from aws_capsolver.geetest.slide import unscramble_v3
    raw = Image.open(FIX / "v3_slide_fullbg.jpg").convert("RGB")
    assert raw.size == (312, 160)
    out = np.asarray(unscramble_v3(raw)).astype(int)
    assert out.shape == (160, 260, 3)
    # neighbouring strips of the reassembled picture continue each other
    seams = np.abs(out[:, 9::10][:, :-1] - out[:, 10::10]).mean()
    inside = np.abs(out[:, 4::10] - out[:, 5::10]).mean()
    assert seams < inside * 3 + 2


def test_v3_slide_real_sample():
    t = json.loads(_read("v3_slide.json"))
    bg, piece, full = _read("v3_slide_bg.jpg"), _read("v3_slide_slice.png"), _read("v3_slide_fullbg.jpg")
    assert abs(G.slide_offset(bg, piece) - t["x"]) <= 4
    assert abs(G.slide_offset(bg, piece, full) - t["x"]) <= 2
    assert abs(G.slide_offset(bg, None, full) - t["gap_x"]) <= 4  # no piece: the gap's left edge
    assert abs(G.slide_offset(bg) - t["gap_x"]) <= 6
    r = G.slide_details(bg, piece, y=t["ypos"])
    assert abs(r.x - t["x"]) <= 4 and abs(r.y - t["ypos"]) <= 1


@pytest.mark.parametrize("name", ["v4_slide_2300b6", "v4_slide_0714a0"])
def test_v4_slide_with_decoys(name):
    # both samples have decoy gaps; in one a decoy has the strongest outline
    t = json.loads(_read(name + ".json"))
    bg, piece = _read(name + "_bg.png"), _read(name + "_slice.png")
    assert abs(G.slide_offset(bg, piece) - t["x"]) <= 4
    # base64 / data URL inputs give the same answer
    b64 = "data:image/png;base64," + base64.b64encode(bg).decode()
    assert G.slide_offset(b64, base64.b64encode(piece).decode()) == G.slide_offset(bg, piece)


def test_slide_flat_piece_synthetic():
    """A texture-less piece (the service's e2e fixture): the outline alone must place it."""
    rnd = np.random.default_rng(1)
    im = Image.new("RGB", (320, 160))
    d = ImageDraw.Draw(im)
    for y in range(160):
        d.line([(0, y), (320, y)], fill=(40 + y * 120 // 160, 100, 160 - y * 60 // 160))
    for _ in range(12):
        x, y, r = (int(v) for v in rnd.integers([0, 0, 6], [320, 160, 30]))
        d.ellipse([x - r, y - r, x + r, y + r], fill=tuple(int(v) for v in rnd.integers(0, 256, 3)))
    d.rectangle([172, 57, 227, 112], fill=(20, 20, 20))
    piece = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
    ImageDraw.Draw(piece).rectangle([4, 4, 59, 59], fill=(240, 240, 240, 255))
    assert abs(G.slide_offset(_png(im), _png(piece)) - 168) <= 2


def test_slide_is_deterministic_and_thread_safe():
    bg, piece = _read("v4_slide_2300b6_bg.png"), _read("v4_slide_2300b6_slice.png")
    first = G.slide_offset(bg, piece)
    with ThreadPoolExecutor(8) as ex:
        assert set(ex.map(lambda _: G.slide_offset(bg, piece), range(16))) == {first}


# -------------------------------------------------------------------------- gobang
def test_gobang_matrix():
    q = [[0, 0, 3, 1, 2], [1, 1, 3, 0, 4], [0, 1, 0, 0, 1], [4, 3, 3, 0, 1], [0, 4, 3, 2, 0]]
    assert G.gobang(q) == ((3, 1), (2, 2))
    with pytest.raises(UnsupportedGeeTest):
        G.gobang([[1, 2, 3, 4, 0]] * 5)


def test_gobang_real_widget_screenshot():
    assert G.gobang(_read("v4_gobang_widget.png")) == ((0, 4), (2, 1))


def test_gobang_canvas_board():
    q = [[1, 2, 1, 1, 2], [3, 4, 3, 2, 4], [1, 3, 0, 3, 4], [4, 2, 4, 1, 4], [2, 1, 3, 1, 4]]
    colors = {1: (245, 247, 254), 2: (90, 135, 250), 3: (240, 200, 100), 4: (60, 66, 72)}
    im = Image.new("RGB", (400, 400), (255, 255, 255))
    d = ImageDraw.Draw(im)
    for r in range(5):
        for c in range(5):
            if q[r][c]:
                d.ellipse([c * 80 + 4, r * 80 + 4, c * 80 + 76, r * 80 + 76], fill=colors[q[r][c]],
                          outline=(200, 200, 210), width=2)
    assert G.gobang(_png(im)) == G.gobang(q) == ((0, 1), (2, 2))


def test_warmup_runs():
    assert G.warmup(background=False) is None
