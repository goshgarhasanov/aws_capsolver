"""Solver for the AWS WAF audio CAPTCHA ("noisyneighbour-<locale>").

Structure of a clip (10 s, AAC/ADTS, 16 kHz, mono duplicated to stereo):

    <background: a male voice reading Kant at ~ -15 dB>
    "Type one of the two following words spoken by me."   <- fixed recording, loud
    <word A> ... <word B>                                  <- same loud voice, random times
    <background continues underneath and between>

captcha.js sends ``input.value.toLowerCase().trim().normalize("NFC")`` as client_solution;
the user only has to type ONE of the two words. So the solver:

1. decodes the clip (PyAV) to 16 kHz mono float32;
2. locates the fixed instruction by correlating the clip's energy envelope with an
   embedded envelope template (robust: peak >= 0.87 vs. <= 0.5 elsewhere on real clips);
3. finds the target words as runs that are much louder than the background after it
   (the foreground voice is mixed ~12 dB above the neighbour);
4. transcribes every run in isolation (padded with silence, background gated) with
   faster-whisper on CPU and returns the most confident single word.

If the template is not found (a new instruction recording / locale), it falls back to a
whole-clip transcription with word timestamps and loudness filtering.
"""
from __future__ import annotations

import base64
import binascii
import io
import logging
import os
import re
import threading
import time
import unicodedata
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .errors import InvalidInput, UnsupportedProblem

logger = logging.getLogger(__name__)

SR = 16000
HOP = 160                     # 10 ms envelope frames
MAX_CLIP_BYTES = 2_000_000    # real clips are ~55 kB
MAX_SECONDS = 30.0
DECODE_TIMEOUT_S = 5.0

MODEL_NAME = os.environ.get("AWS_CAPSOLVER_AUDIO_MODEL", "small.en")
CPU_THREADS = int(os.environ.get("AWS_CAPSOLVER_AUDIO_THREADS", "8"))

# dB envelope (10 ms frames, relative to its max) of the spoken instruction
# "Type one of the two following words spoken by me." The speech spans frames 3..254.
_INSTRUCTION_DB = (
    -33, -37, -35, -24, -15, -16, -20, -17, -16, -15, -5, -2, -2, -3, -2, -3, -5, -5, -5,
    -5, -5, -5, -14, -24, -23, -27, -31, -40, -44, -51, -43, -22, -31, -30, -26, -20, -7,
    -5, -5, -3, -3, -4, -4, -6, -4, -6, -4, -5, -6, -2, -6, -4, -6, -8, -12, -11, -11,
    -14, -12, -9, -7, -3, -3, -3, -9, -16, -25, -27, -34, -31, -31, -14, -10, -12, -16,
    -18, -18, -15, -4, -1, 0, -1, -1, -1, -2, -3, -5, -9, -14, -20, -27, -32, -34, -34,
    -35, -37, -21, -22, -8, -5, -4, -3, -4, -3, -6, -5, -7, -5, -6, -7, -6, -9, -7, -9,
    -6, -9, -6, -8, -5, -8, -6, -9, -6, -9, -5, -7, -4, -7, -4, -8, -4, -10, -5, -6, -5,
    -7, -6, -8, -6, -7, -6, -7, -7, -6, -8, -6, -6, -5, -5, -4, -6, -5, -6, -4, -6, -5,
    -6, -5, -6, -6, -8, -14, -23, -25, -29, -23, -10, -11, -10, -12, -10, -10, -12, -10,
    -11, -12, -22, -25, -25, -25, -23, -18, -15, -13, -4, -2, -2, -3, -3, -4, -5, -3, -8,
    -5, -15, -16, -22, -26, -29, -20, -24, -15, -3, -3, -4, -4, -3, -3, -4, -6, -7, -13,
    -18, -25, -13, -6, -6, -5, -7, -5, -6, -6, -5, -7, -6, -7, -7, -5, -8, -6, -4, -6,
    -5, -4, -6, -5, -7, -6, -6, -7, -6, -6, -8, -8, -7, -8, -10, -7, -9, -10, -10, -12,
    -12, -13, -20, -26, -27, -32, -29, -30, -36, -33, -37, -36, -43, -40,
)
_INSTRUCTION_END = 254 * HOP  # samples from template start to the end of "me"
MIN_TEMPLATE_SCORE = 0.6

# segmentation of the foreground words (dB relative to the instruction's loud level)
WORD_THRESHOLD_DB = -12.0
WORD_PEAK_DB = -5.0
MERGE_GAP_S = 0.18
MIN_WORD_S = 0.12
MAX_WORD_S = 1.6
PAD_S = 0.08
SIL_S = 0.4


class AudioDecodeError(UnsupportedProblem, InvalidInput):
    """The audio clip could not be decoded (bad base64, not AAC, empty...)."""


# --------------------------------------------------------------------------- decoding

def decode_audio(data: bytes) -> np.ndarray:
    """Decode an ADTS/AAC (or any ffmpeg-readable) clip to mono float32 at 16 kHz."""
    try:
        import av  # PyAV bundles its own ffmpeg libraries; no ffmpeg binary is needed
    except ImportError as exc:
        raise AudioDecodeError(f"PyAV (pip install av) is required to decode audio: {exc}") \
            from exc

    if not data:
        raise AudioDecodeError("empty audio clip")
    if len(data) > MAX_CLIP_BYTES:
        raise AudioDecodeError(f"audio clip too large ({len(data)} bytes)")
    chunks: List[np.ndarray] = []
    try:
        with av.open(io.BytesIO(data), mode="r") as container:
            if not container.streams.audio:
                raise AudioDecodeError("no audio stream in clip")
            resampler = av.AudioResampler(format="flt", layout="mono", rate=SR)
            deadline = time.monotonic() + DECODE_TIMEOUT_S
            total = 0
            for frame in container.decode(audio=0):
                for out in resampler.resample(frame):
                    chunks.append(out.to_ndarray().reshape(-1))
                    total += len(chunks[-1])
                if total > MAX_SECONDS * SR:
                    raise AudioDecodeError("audio clip too long")
                if time.monotonic() > deadline:
                    raise AudioDecodeError(f"audio decoding exceeded {DECODE_TIMEOUT_S}s")
            for out in resampler.resample(None):
                chunks.append(out.to_ndarray().reshape(-1))
    except (av.error.FFmpegError, ValueError, OSError) as exc:
        raise AudioDecodeError(f"cannot decode audio: {exc}") from exc
    if not chunks:
        raise AudioDecodeError("audio clip decodes to nothing")
    audio = np.concatenate(chunks).astype(np.float32)
    if len(audio) < SR // 2 or not np.isfinite(audio).all():
        raise AudioDecodeError("audio clip is too short or corrupt")
    return audio


def decode_b64(audio_b64: str) -> np.ndarray:
    if not isinstance(audio_b64, str) or not audio_b64.strip():
        raise AudioDecodeError("audio must be a non-empty base64 string")
    text = audio_b64.strip()
    if text.startswith("data:"):
        text = text.split(",", 1)[-1]
    try:
        data = base64.b64decode(text + "=" * (-len(text) % 4), validate=False)
    except (binascii.Error, ValueError) as exc:
        raise AudioDecodeError(f"invalid base64 audio: {exc}") from exc
    return decode_audio(data)


# --------------------------------------------------------------------------- segmentation

def envelope(x: np.ndarray, hop: int = HOP) -> np.ndarray:
    n = len(x) // hop
    frames = x[: n * hop].reshape(n, hop)
    return np.sqrt((frames.astype(np.float64) ** 2).mean(axis=1))


def locate_instruction(x: np.ndarray) -> Tuple[int, float]:
    """Return (end sample of the instruction, correlation score) via envelope matching."""
    tmpl = 10.0 ** (np.asarray(_INSTRUCTION_DB, dtype=np.float64) / 20.0)
    env = envelope(x)
    if len(env) < len(tmpl):
        return 0, 0.0
    win = np.lib.stride_tricks.sliding_window_view(env, len(tmpl))
    wz = win - win.mean(axis=1, keepdims=True)
    wz /= wz.std(axis=1, keepdims=True) + 1e-12
    tz = (tmpl - tmpl.mean()) / tmpl.std()
    scores = wz @ tz / len(tmpl)
    k = int(np.argmax(scores))
    return k * HOP + _INSTRUCTION_END, float(scores[k])


@dataclass
class Segment:
    start: int          # samples
    end: int
    peak_db: float      # relative to the instruction level
    energy: float       # sum of linear power over the run

    @property
    def seconds(self) -> float:
        return (self.end - self.start) / SR


def find_word_segments(x: np.ndarray, instr_end: int, ref_level: float) -> List[Segment]:
    """Loud runs after the instruction: the two foreground words (plus rare false hits)."""
    from scipy.ndimage import uniform_filter1d

    env = envelope(x)
    db = 20 * np.log10(env + 1e-9) - 20 * np.log10(ref_level + 1e-9)
    db = uniform_filter1d(db, 5)
    first = int(np.ceil(instr_end / HOP)) + 3
    loud = db > WORD_THRESHOLD_DB
    loud[:first] = False
    runs: List[List[int]] = []
    i = first
    while i < len(loud):
        if loud[i]:
            j = i
            while j < len(loud) and loud[j]:
                j += 1
            if runs and i - runs[-1][1] <= MERGE_GAP_S * SR / HOP:
                runs[-1][1] = j
            else:
                runs.append([i, j])
            i = j
        else:
            i += 1
    segs = []
    for a, b in runs:
        dur = (b - a) * HOP / SR
        peak = float(db[a:b].max())
        if dur < MIN_WORD_S or peak < WORD_PEAK_DB:
            continue
        if dur > MAX_WORD_S:  # foreground merged with loud background: keep the core
            core = np.argmax(db[a:b])
            a = max(a, a + core - int(0.6 * SR / HOP))
            b = min(b, a + int(MAX_WORD_S * SR / HOP))
        segs.append(Segment(a * HOP, b * HOP, peak, float((env[a:b] ** 2).sum())))
    return segs


def reference_level(x: np.ndarray, instr_end: int) -> float:
    """Loud level of the instruction voice (95th percentile of its 10 ms frames)."""
    start = max(0, instr_end - _INSTRUCTION_END)
    env = envelope(x[start:instr_end])
    return float(np.percentile(env, 95)) if len(env) else float(envelope(x).max())


def cut_segment(x: np.ndarray, seg: Segment) -> Tuple[np.ndarray, int]:
    """Extract one word run with short fades, pad with silence.

    Returns the clip and the (possibly negative) original sample index of clip[0].
    """
    pad = int(PAD_S * SR)
    a, b = max(0, seg.start - pad), min(len(x), seg.end + pad)
    clip = x[a:b].astype(np.float32).copy()
    ramp = min(len(clip) // 4, int(0.03 * SR))
    if ramp > 0:
        fade = np.linspace(0.0, 1.0, ramp, dtype=np.float32)
        clip[:ramp] *= fade
        clip[-ramp:] *= fade[::-1]
    peak = float(np.abs(clip).max()) or 1.0
    clip *= 0.5 / peak
    sil = np.zeros(int(SIL_S * SR), dtype=np.float32)
    return np.concatenate([sil, clip, sil]), a - len(sil)


# --------------------------------------------------------------------------- text

_WORD_RE = re.compile(r"[^\W\d_]+(?:'[^\W\d_]+)?")


def normalise(text: str) -> str:
    """What captcha.js would send: lower-cased, trimmed, NFC; punctuation stripped."""
    text = unicodedata.normalize("NFC", (text or "").lower())
    words = _WORD_RE.findall(text.replace("’", "'"))
    return " ".join(words).strip()


# --------------------------------------------------------------------------- recogniser

@dataclass
class Candidate:
    text: str
    logprob: float      # mean token log-probability of the decoded text
    no_speech: float
    segment: Optional[Segment] = None

    @property
    def score(self) -> float:
        penalty = 0.0 if len(self.text.split()) == 1 else 0.7
        return self.logprob - penalty - 0.5 * self.no_speech


class AudioSolver:
    """One faster-whisper model plus the segmentation pipeline (thread-safe)."""

    def __init__(self, model_name: str = MODEL_NAME, cpu_threads: int = CPU_THREADS,
                 compute_type: str = "int8", beam_size: int = 5):
        from faster_whisper import WhisperModel

        self.model_name = model_name
        self.beam_size = beam_size
        self.model = WhisperModel(model_name, device="cpu", compute_type=compute_type,
                                  cpu_threads=cpu_threads)
        self._lock = threading.Lock()  # CTranslate2 model calls are serialised here
        self.english_only = model_name.endswith(".en")

    def _decode(self, clip: np.ndarray) -> Candidate:
        kwargs = dict(beam_size=self.beam_size, without_timestamps=True,
                      condition_on_previous_text=False, vad_filter=False, temperature=0.0)
        if not self.english_only:
            kwargs["language"] = "en"
        with self._lock:
            segments, _ = self.model.transcribe(clip, **kwargs)
            segments = list(segments)
        if not segments:
            return Candidate("", -9.0, 1.0)
        text = normalise(" ".join(s.text for s in segments))
        logprob = float(np.mean([s.avg_logprob for s in segments]))
        no_speech = float(max(s.no_speech_prob for s in segments))
        return Candidate(text, logprob, no_speech)

    def candidates(self, x: np.ndarray) -> Tuple[List[Candidate], float]:
        """Transcribe every foreground word run; returns (candidates, template score)."""
        instr_end, score = locate_instruction(x)
        if score < MIN_TEMPLATE_SCORE:
            return [], score
        ref = reference_level(x, instr_end)
        out = []
        for seg in find_word_segments(x, instr_end, ref):
            cand = self._decode(cut_segment(x, seg)[0])
            cand.segment = seg
            if cand.text:
                out.append(cand)
        return out, score

    def fallback(self, x: np.ndarray) -> str:
        """No template match: transcribe the clip, keep loud words after 'by me'."""
        kwargs = dict(beam_size=self.beam_size, word_timestamps=True,
                      condition_on_previous_text=False)
        if not self.english_only:
            kwargs["language"] = "en"
        with self._lock:
            segments, _ = self.model.transcribe(x, **kwargs)
            words = [w for s in segments for w in (s.words or [])]
        if not words:
            return ""
        env_db = 20 * np.log10(envelope(x) + 1e-9)
        loudness = []
        for w in words:
            a, b = int(w.start * SR / HOP), max(int(w.start * SR / HOP) + 1, int(w.end * SR / HOP))
            loudness.append(float(np.percentile(env_db[a:b], 90)) if b <= len(env_db) else -99.0)
        tokens = [normalise(w.word) for w in words]
        start = 0
        for i in range(1, len(tokens)):
            if tokens[i - 1] == "by" and tokens[i] == "me":
                start = i + 1
        ref = np.percentile(loudness, 90)
        best = [(loudness[i] + 3 * words[i].probability, tokens[i])
                for i in range(start, len(tokens))
                if tokens[i] and loudness[i] > ref - 8]
        return max(best)[1] if best else ""

    def solve(self, x: np.ndarray) -> str:
        cands, score = self.candidates(x)
        answer = choose_answer(cands)
        if answer:
            return answer
        logger.info("audio: template score %.2f, no candidate; using fallback", score)
        return self.fallback(x)


def choose_answer(cands: Sequence[Candidate]) -> str:
    """Pick the one word to submit: the most confident single-word run wins."""
    if not cands:
        return ""
    best = max(cands, key=lambda c: c.score)
    words = best.text.split()
    # several words in one run (background leaked in): the longest is the safest guess
    return max(words, key=len) if len(words) > 1 else best.text


_solver: Optional[AudioSolver] = None
_solver_lock = threading.Lock()


def get_solver() -> AudioSolver:
    global _solver
    if _solver is None:
        with _solver_lock:
            if _solver is None:
                logger.info("audio: loading faster-whisper model %s", MODEL_NAME)
                try:
                    _solver = AudioSolver()
                except Exception as exc:  # ImportError, download/IO errors, CTranslate2
                    raise UnsupportedProblem(
                        f"audio model {MODEL_NAME!r} could not be loaded: {exc}") from exc
    return _solver


def transcribe(audio_b64: str) -> str:
    """Answer for a noisyneighbour clip: one of the two spoken words, normalised."""
    x = decode_b64(audio_b64)
    answer = get_solver().solve(x)
    if not answer:
        raise UnsupportedProblem("no word recognised in the audio clip")
    return unicodedata.normalize("NFC", answer.lower().strip())


__all__ = ["transcribe", "decode_b64", "decode_audio", "locate_instruction",
           "find_word_segments", "normalise", "choose_answer", "AudioSolver",
           "AudioDecodeError"]
