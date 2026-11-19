"""Tests for the audio CAPTCHA solver (aws_capsolver.vision.audio).

Fast tests use synthetic signals and stubs; tests on the real sample clip need
data/samples/problem_audio.json, and the end-to-end transcription test additionally needs
the faster-whisper model in the local Hugging Face cache (both are skipped otherwise).
"""
from __future__ import annotations

import ast
import base64
import io
import json
import threading
import wave
from pathlib import Path

import numpy as np
import pytest

from aws_capsolver.vision import audio as A
from aws_capsolver.vision.errors import InvalidInput, UnsupportedProblem

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "data" / "samples" / "problem_audio.json"


def _sample_b64() -> str:
    if not SAMPLE.exists():
        pytest.skip("no real audio sample")
    problem = json.loads(SAMPLE.read_text())
    assets = problem["assets"]
    if isinstance(assets, str):  # stored as a Python repr by the sample collector
        assets = ast.literal_eval(assets)
    return assets["audio"]


def _wav_b64(signal: np.ndarray, sr: int = A.SR) -> str:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes((np.clip(signal, -1, 1) * 32767).astype("<i2").tobytes())
    return base64.b64encode(buf.getvalue()).decode()


# ----------------------------------------------------------------------------- errors

def test_decode_error_is_unsupported_and_invalid_input():
    assert issubclass(A.AudioDecodeError, UnsupportedProblem)
    assert issubclass(A.AudioDecodeError, InvalidInput)


@pytest.mark.parametrize("bad", ["", "   ", None, 123, b"abc"])
def test_rejects_non_string_or_empty(bad):
    with pytest.raises(UnsupportedProblem):
        A.decode_b64(bad)


@pytest.mark.parametrize("bad", ["!!!not base64!!!", "QUJD", base64.b64encode(b"\x00" * 4000)
                                 .decode(), base64.b64encode(b"<html>nope</html>").decode()])
def test_rejects_garbage(bad):
    with pytest.raises(UnsupportedProblem):
        A.decode_b64(bad)


def test_rejects_oversized_clip():
    with pytest.raises(UnsupportedProblem):
        A.decode_audio(b"\xff" * (A.MAX_CLIP_BYTES + 1))


def test_rejects_too_short_audio():
    with pytest.raises(UnsupportedProblem):
        A.decode_b64(_wav_b64(np.zeros(800, dtype=np.float32)))


def test_rejects_too_long_audio():
    sig = np.zeros(int((A.MAX_SECONDS + 5) * 8000), dtype=np.float32)
    with pytest.raises(UnsupportedProblem):
        A.decode_b64(_wav_b64(sig, sr=8000))


def test_transcribe_bad_input_raises_typed_error_without_loading_model(monkeypatch):
    def boom():
        raise AssertionError("model must not load for undecodable input")

    monkeypatch.setattr(A, "get_solver", boom)
    with pytest.raises(UnsupportedProblem):
        A.transcribe("this is not audio")


def test_missing_pyav_is_reported_as_unsupported(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "av":
            raise ImportError("No module named 'av'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(UnsupportedProblem, match="PyAV"):
        A.decode_audio(b"\x00" * 100)


# ----------------------------------------------------------------------------- decoding

def test_decodes_wav_and_data_uri():
    t = np.arange(A.SR) / A.SR
    sig = 0.3 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
    b64 = _wav_b64(sig)
    x = A.decode_b64(b64)
    assert x.dtype == np.float32 and abs(len(x) - A.SR) < 400
    y = A.decode_b64("data:audio/wav;base64," + b64)
    assert len(y) == len(x)
    # missing base64 padding is tolerated (some clients strip it)
    assert len(A.decode_b64(b64.rstrip("="))) == len(x)


def test_decodes_real_aac_sample():
    x = A.decode_b64(_sample_b64())
    assert 9.0 < len(x) / A.SR < 11.0
    assert np.abs(x).max() > 0.1


# ----------------------------------------------------------------------------- segmentation

def test_locates_instruction_in_real_sample():
    x = A.decode_b64(_sample_b64())
    end, score = A.locate_instruction(x)
    assert score > 0.8
    assert 3.0 < end / A.SR < 7.5


def test_instruction_not_found_in_silence_or_short_audio():
    assert A.locate_instruction(np.zeros(A.SR, dtype=np.float32)) == (0, 0.0)
    rng = np.random.default_rng(0)
    _, score = A.locate_instruction(0.05 * rng.standard_normal(10 * A.SR).astype(np.float32))
    assert score < A.MIN_TEMPLATE_SCORE


def test_real_sample_has_two_word_segments():
    x = A.decode_b64(_sample_b64())
    end, _ = A.locate_instruction(x)
    segs = A.find_word_segments(x, end, A.reference_level(x, end))
    assert len(segs) == 2
    assert all(s.start >= end and A.DEFAULT_SEG.min_s <= s.seconds <= A.DEFAULT_SEG.max_s for s in segs)


def test_find_word_segments_on_synthetic_bursts():
    rng = np.random.default_rng(1)
    x = 0.01 * rng.standard_normal(10 * A.SR).astype(np.float32)   # quiet neighbour
    t = np.arange(int(0.4 * A.SR)) / A.SR
    burst = (0.5 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    x[int(5.0 * A.SR):int(5.0 * A.SR) + len(burst)] += burst
    x[int(7.5 * A.SR):int(7.5 * A.SR) + len(burst)] += burst
    x[int(1.0 * A.SR):int(1.0 * A.SR) + len(burst)] += burst      # before the instruction
    segs = A.find_word_segments(x, int(4.0 * A.SR), ref_level=0.5 / np.sqrt(2))
    assert [round(s.start / A.SR, 1) for s in segs] == [5.0, 7.5]


def test_cut_segment_pads_and_reports_offset():
    x = np.ones(3 * A.SR, dtype=np.float32) * 0.1
    seg = A.Segment(A.SR, 2 * A.SR, 0.0, 1.0)
    clip, offset = A.cut_segment(x, seg)
    sil = int(A.SIL_S * A.SR)
    assert offset == seg.start - int(A.PAD_S * A.SR) - sil
    assert not clip[:sil].any() and not clip[-sil:].any()
    assert np.abs(clip).max() == pytest.approx(0.5, rel=1e-3)


# ----------------------------------------------------------------------------- text

@pytest.mark.parametrize("raw,expected", [
    (" Theater. ", "theater"),
    ("ACTIONS!", "actions"),
    ("Don’t", "don't"),
    ("e\u0301clair", "\u00e9clair"),
    ("...", ""),
    ("Next year,", "next year"),
    ("Dr.", "doctor"),
    ("Wall St.", "wall street"),
    ("7", "seven"),
    ("42", "forty two"),
    ("1,000", "one thousand"),
    ("1000", "one thousand"),
])
def test_normalise(raw, expected):
    assert A.normalise(raw) == expected


def test_choose_answer_prefers_confident_single_word():
    seg = A.Segment(0, 1, 0.0, 1.0)
    cands = [A.Candidate("not anymore", -0.2, 0.0, seg), A.Candidate("theater", -0.5, 0.0, seg),
             A.Candidate("dimension", -0.4, 0.0, seg)]
    assert A.choose_answer(cands) == "dimension"
    assert A.choose_answer([A.Candidate("beauty you", -0.3, 0.0, seg)]) == "beauty"
    assert A.choose_answer([]) == ""


# ----------------------------------------------------------------------------- singleton

def test_singleton_loads_once_under_concurrency(monkeypatch):
    created = []

    class Dummy:
        def __init__(self, *a, **k):
            created.append(self)

    monkeypatch.setattr(A, "AudioSolver", Dummy)
    monkeypatch.setattr(A, "_solver", None)
    results = []
    threads = [threading.Thread(target=lambda: results.append(A.get_solver()))
               for _ in range(16)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(10)
    assert len(created) == 1 and all(r is created[0] for r in results)


def test_model_load_failure_is_unsupported_and_retried(monkeypatch):
    calls = []

    class Broken:
        def __init__(self, *a, **k):
            calls.append(1)
            raise RuntimeError("model download failed")

    monkeypatch.setattr(A, "AudioSolver", Broken)
    monkeypatch.setattr(A, "_solver", None)
    for _ in range(2):
        with pytest.raises(UnsupportedProblem, match="model"):
            A.get_solver()
    assert len(calls) == 2


def test_transcribe_returns_normalised_answer(monkeypatch):
    class Stub:
        def solve(self, x):
            return "Theater"

    monkeypatch.setattr(A, "get_solver", lambda: Stub())
    t = np.arange(A.SR) / A.SR
    assert A.transcribe(_wav_b64(0.3 * np.sin(2 * np.pi * 300 * t))) == "theater"


def test_transcribe_empty_answer_is_unsupported(monkeypatch):
    class Stub:
        def solve(self, x):
            return ""

    monkeypatch.setattr(A, "get_solver", lambda: Stub())
    t = np.arange(A.SR) / A.SR
    with pytest.raises(UnsupportedProblem):
        A.transcribe(_wav_b64(0.3 * np.sin(2 * np.pi * 300 * t)))


# ----------------------------------------------------------------------------- end to end

def _model_cached() -> bool:
    try:
        from faster_whisper.utils import download_model
        download_model(A.MODEL_NAME, local_files_only=True)
        return True
    except Exception:  # noqa: BLE001 - any failure means "not available offline"
        return False


@pytest.mark.skipif(not _model_cached(), reason="whisper model not in the local cache")
def test_end_to_end_real_sample():
    answer = A.transcribe(_sample_b64())
    assert answer and answer == answer.strip().lower()
    assert len(answer.split()) == 1


@pytest.mark.parametrize("raw,expected", [
    ("theater theater theater", "theater"),
    ("reputation reputation get reputation get", "reputation get"),
    ("a b a b a", "a b"),
    ("a b c", "a b c"),
    ("", ""),
])
def test_collapse_repeats(raw, expected):
    assert A.collapse_repeats(raw) == expected


@pytest.mark.parametrize("n,words", [(0, "zero"), (13, "thirteen"), (40, "forty"),
                                     (99, "ninety nine"), (305, "three hundred five"),
                                     (2026, "two thousand twenty six")])
def test_number_to_words(n, words):
    assert A.number_to_words(n) == words
