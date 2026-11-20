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

Model download and offline use
------------------------------
The CTranslate2 model is fetched once from the Hugging Face hub (Systran/faster-whisper-*)
into the standard cache (~/.cache/huggingface/hub, or $HF_HOME). Pre-fetch it with

    python -c "from faster_whisper.utils import download_model as d; d('medium.en'); d('small.en')"

and run with HF_HUB_OFFLINE=1 afterwards. AWS_CAPSOLVER_AUDIO_MODEL accepts a model size
name (tiny.en ... large-v3) or a path to a local CTranslate2 model directory;
AWS_CAPSOLVER_AUDIO_THREADS sets the CPU threads. The model is loaded lazily, once, behind
a lock; inference calls are serialised on the model (one CTranslate2 call at a time).
Decoding uses PyAV, which bundles the ffmpeg libraries (no ffmpeg binary or subprocess).
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
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .errors import InvalidInput, UnsupportedProblem

logger = logging.getLogger(__name__)

SR = 16000
HOP = 160                     # 10 ms envelope frames
MAX_CLIP_BYTES = 2_000_000    # real clips are ~55 kB
MAX_SECONDS = 30.0
DECODE_TIMEOUT_S = 5.0

MODEL_NAME = os.environ.get("AWS_CAPSOLVER_AUDIO_MODEL", "medium.en")
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

PAD_S = 0.08
WINDOW_S = float(os.environ.get("AWS_CAPSOLVER_AUDIO_WINDOW", "20"))  # encoder window (s)
MAX_SEGMENTS = 4
MULTI_WORD_PENALTY = 0.7  # a run decoded as several words has background speech in it
TOKEN_PENALTY = 0.3       # per extra BPE token: common words are 1-2 tokens (tuned on dev)
STEM_MARGIN = float(os.environ.get("AWS_CAPSOLVER_AUDIO_STEM_MARGIN", "0.5"))  # tuned on dev
USE_ALTERNATIVES = False  # second opinion may also confirm a non-top beam of the primary
SECOND_MODEL = os.environ.get("AWS_CAPSOLVER_AUDIO_SECOND_MODEL", "small.en")  # "" = off
SECOND_WINDOW_S = 10.0
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
    """Decode the base64 clip of a problem (``assets.audio``; a data: URI is accepted)."""
    if not isinstance(audio_b64, str) or not audio_b64.strip():
        raise AudioDecodeError("audio must be a non-empty base64 string")
    text = audio_b64.strip()
    if text.startswith("data:"):
        text = text.split(",", 1)[-1]
    text = "".join(text.split())  # tolerate line breaks
    if len(text) > MAX_CLIP_BYTES * 4 // 3 + 4:
        raise AudioDecodeError(f"audio clip too large ({len(text)} base64 chars)")
    try:
        data = base64.b64decode(text + "=" * (-len(text) % 4), validate=True)
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


@dataclass(frozen=True)
class SegParams:
    """Thresholds (dB relative to the instruction voice) for finding the foreground words.

    A word is a run of frames above ``core_db`` (runs closer than ``merge_gap_s`` are
    joined), widened by at most ``edge_s`` on each side while above ``edge_db``. The
    neighbour sits around -20..-10 dB, the foreground words peak near 0 dB.
    """
    core_db: float = -9.0
    edge_db: float = -14.0
    edge_s: float = 0.10
    merge_gap_s: float = 0.25
    peak_db: float = -5.0
    min_s: float = 0.12
    max_s: float = 1.6
    duck_below_db: float = -14.0   # inside a cut, frames quieter than this are ...
    duck_gain_db: float = 0.0      # ... attenuated by this much (0 = no ducking)
    edge_end_s: float = -1.0       # widening after the word (< 0: same as edge_s)
    pad_end_s: float = -1.0        # raw audio kept after the run (< 0: PAD_S)


DEFAULT_SEG = SegParams()


def _runs(mask: np.ndarray, start: int) -> List[List[int]]:
    runs: List[List[int]] = []
    i = start
    n = len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j < n and mask[j]:
                j += 1
            runs.append([i, j])
            i = j
        else:
            i += 1
    return runs


def relative_db(x: np.ndarray, ref_level: float) -> np.ndarray:
    """Smoothed 10 ms frame level in dB relative to the instruction voice."""
    from scipy.ndimage import uniform_filter1d

    db = 20 * np.log10(envelope(x) + 1e-9) - 20 * np.log10(ref_level + 1e-9)
    return uniform_filter1d(db, 5)


def find_word_segments(x: np.ndarray, instr_end: int, ref_level: float,
                       params: SegParams = DEFAULT_SEG) -> List[Segment]:
    """Loud runs after the instruction: the two foreground words (plus rare false hits)."""
    env = envelope(x)
    db = relative_db(x, ref_level)
    first = int(np.ceil(instr_end / HOP)) + 3
    if first >= len(db):
        return []
    fps = SR / HOP
    merged: List[List[int]] = []
    for a, b in _runs(db > params.core_db, first):
        if merged and a - merged[-1][1] <= params.merge_gap_s * fps:
            merged[-1][1] = b
        else:
            merged.append([a, b])
    edge = int(round(params.edge_s * fps))
    segs = []
    for a, b in merged:
        peak = float(db[a:b].max())
        if peak < params.peak_db:
            continue
        lo = a
        while lo > max(first, a - edge) and db[lo - 1] > params.edge_db:
            lo -= 1
        hi = b
        edge_end = edge if params.edge_end_s < 0 else int(round(params.edge_end_s * fps))
        while hi < min(len(db), b + edge_end) and db[hi] > params.edge_db:
            hi += 1
        dur = (hi - lo) / fps
        if dur < params.min_s:
            continue
        if dur > params.max_s:  # foreground merged with loud background: keep the core
            centre = lo + int(np.argmax(db[lo:hi]))
            lo = max(lo, centre - int(params.max_s * fps / 2))
            hi = min(hi, lo + int(params.max_s * fps))
        segs.append(Segment(lo * HOP, hi * HOP, peak, float((env[lo:hi] ** 2).sum())))
    return segs


def reference_level(x: np.ndarray, instr_end: int) -> float:
    """Loud level of the instruction voice (95th percentile of its 10 ms frames)."""
    start = max(0, instr_end - _INSTRUCTION_END)
    env = envelope(x[start:instr_end])
    return float(np.percentile(env, 95)) if len(env) else float(envelope(x).max())


def cut_segment(x: np.ndarray, seg: Segment, rel_db: Optional[np.ndarray] = None,
                params: SegParams = DEFAULT_SEG) -> Tuple[np.ndarray, int]:
    """Extract one word run with short fades, pad with silence.

    With ``rel_db`` (see relative_db) and ``params.duck_gain_db`` < 0, the quiet frames
    of the cut (background speech around the word) are attenuated.
    Returns the clip and the (possibly negative) original sample index of clip[0].
    """
    pad = int(PAD_S * SR)
    pad_end = pad if params.pad_end_s < 0 else int(params.pad_end_s * SR)
    a, b = max(0, seg.start - pad), min(len(x), seg.end + pad_end)
    clip = x[a:b].astype(np.float32).copy()
    if rel_db is not None and params.duck_gain_db < 0 and len(clip):
        from scipy.ndimage import uniform_filter1d

        idx = np.clip(np.arange(a, b) // HOP, 0, len(rel_db) - 1)
        frame_gain = np.where(rel_db >= params.duck_below_db, 1.0,
                              10.0 ** (params.duck_gain_db / 20.0))
        frame_gain = uniform_filter1d(frame_gain, 3)
        clip *= frame_gain[idx].astype(np.float32)
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

_TOKEN_RE = re.compile(r"\d+|[^\W\d_]+(?:'[^\W\d_]+)?")

# whisper writes some spoken words in their written form; the captcha expects the word
_ABBREVIATIONS = {
    "dr": "doctor", "mr": "mister", "mrs": "missus", "st": "street", "mt": "mount",
    "ave": "avenue", "vs": "versus", "etc": "etcetera", "ok": "okay",
    "jr": "junior", "sr": "senior", "prof": "professor", "dept": "department",
}
_ONES = ("zero one two three four five six seven eight nine ten eleven twelve thirteen "
         "fourteen fifteen sixteen seventeen eighteen nineteen").split()
_TENS = "_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()


def number_to_words(n: int) -> str:
    """0..9999 -> English words (as a person would type what they hear)."""
    if n < 20:
        return _ONES[n]
    if n < 100:
        return _TENS[n // 10] + ("" if n % 10 == 0 else " " + _ONES[n % 10])
    if n < 1000:
        rest = n % 100
        return _ONES[n // 100] + " hundred" + ("" if rest == 0 else " " + number_to_words(rest))
    if n < 10000:
        rest = n % 1000
        return _ONES[n // 1000] + " thousand" + ("" if rest == 0 else
                                                 " " + number_to_words(rest))
    return str(n)


def normalise(text: str) -> str:
    """Lower-case words only (what captcha.js would send after the user typed them).

    Punctuation is stripped, abbreviations and small numbers are spelled out, NFC applied.
    """
    text = unicodedata.normalize("NFC", (text or "").lower().replace("\u2019", "'"))
    text = re.sub(r"(?<=\d)[,.](?=\d{3}\b)", "", text)  # 1,000 -> 1000
    words = []
    for tok in _TOKEN_RE.findall(text):
        if tok.isdigit():
            words.append(number_to_words(int(tok)))
        else:
            words.append(_ABBREVIATIONS.get(tok, tok))
    return " ".join(words).strip()


def collapse_repeats(text: str) -> str:
    """'theater theater' -> 'theater'; 'a b a b a' -> 'a b'.

    With a short encoder window whisper tends to loop over the same word(s) until
    max_length; the loop period is the actual transcript.
    """
    words = []
    for w in text.split():
        if not words or words[-1] != w:
            words.append(w)
    n = len(words)
    for period in range(1, n // 2 + 1):
        if all(words[i] == words[i % period] for i in range(n)):
            return " ".join(words[:period])
    return " ".join(words)


# --------------------------------------------------------------------------- recogniser

@dataclass
class Candidate:
    text: str
    logprob: float      # mean token log-probability of the decoded text
    no_speech: float
    segment: Optional[Segment] = None
    n_tokens: int = 1   # BPE tokens of the words; mis-heard non-words split into many
    alternatives: List[Tuple[str, float]] = field(default_factory=list)  # other beams

    @property
    def score(self) -> float:
        penalty = 0.0 if len(self.text.split()) == 1 else MULTI_WORD_PENALTY
        return (self.logprob - penalty - 0.5 * self.no_speech
                - TOKEN_PENALTY * max(0, self.n_tokens - 1))


class AudioSolver:
    """One faster-whisper model plus the segmentation pipeline (thread-safe)."""

    def __init__(self, model_name: str = MODEL_NAME, cpu_threads: int = CPU_THREADS,
                 compute_type: str = "int8", beam_size: int = 5,
                 seg: SegParams = DEFAULT_SEG, window_s: Optional[float] = WINDOW_S):
        from faster_whisper import WhisperModel
        from faster_whisper.tokenizer import Tokenizer
        from faster_whisper.transcribe import get_suppressed_tokens

        self.model_name = model_name
        self.beam_size = beam_size
        self.seg = seg
        self.window_s = window_s
        self.model = WhisperModel(model_name, device="cpu", compute_type=compute_type,
                                  cpu_threads=cpu_threads)
        self._lock = threading.Lock()  # CTranslate2 model calls are serialised here
        self.second: Optional["AudioSolver"] = None
        multilingual = self.model.model.is_multilingual
        self.tokenizer = Tokenizer(self.model.hf_tokenizer, multilingual,
                                   task="transcribe" if multilingual else None,
                                   language="en" if multilingual else None)
        self.prompt = list(self.tokenizer.sot_sequence) + [self.tokenizer.no_timestamps]
        self.suppress = list(get_suppressed_tokens(self.tokenizer, [-1]))

    def _decode_batch(self, clips: Sequence[np.ndarray]) -> List[Candidate]:
        """Beam-search decode of word clips in one batch with a short encoder window.

        Whisper is trained on 30 s windows, but the encoder accepts shorter inputs; a few
        seconds of context are enough for one word and make inference ~5-10x cheaper.
        """
        import ctranslate2

        if not clips:
            return []
        longest = max(len(c) for c in clips)
        # whisper's encoder holds at most 30 s (3000 frames); word cuts are <= ~2.5 s
        n = min(max(int((self.window_s or 30.0) * SR), longest), 30 * SR)
        n = int(np.ceil(n / HOP)) * HOP
        clips = [c[:n] for c in clips]
        feats = []
        for c in clips:
            padded = np.zeros(n, dtype=np.float32)
            padded[: len(c)] = c
            feats.append(self.model.feature_extractor(padded)[:, : n // HOP])
        batch = np.ascontiguousarray(np.stack(feats).astype(np.float32))
        with self._lock:
            enc = self.model.model.encode(ctranslate2.StorageView.from_array(batch))
            results = self.model.model.generate(
                enc, [self.prompt] * len(clips), beam_size=self.beam_size, max_length=24,
                return_scores=True, return_no_speech_prob=True, suppress_blank=True,
                suppress_tokens=self.suppress, num_hypotheses=self.beam_size)
        out = []
        for r in results:
            hyps = []
            for ids, score in zip(r.sequences_ids, r.scores):
                tokens = [t for t in ids if t < self.tokenizer.eot]
                text = collapse_repeats(normalise(self.tokenizer.decode(tokens)))
                avg = score * len(tokens) / (len(tokens) + 1) if tokens else -9.0
                hyps.append((text, float(avg)))
            text, avg = hyps[0]
            out.append(Candidate(text, avg, float(r.no_speech_prob),
                                 n_tokens=self._count_tokens(text),
                                 alternatives=hyps[1:]))
        return out

    def _count_tokens(self, text: str) -> int:
        return sum(len(self.tokenizer.encode(" " + w)) for w in text.split()) or 1

    def _decode(self, clip: np.ndarray) -> Candidate:
        """Decode one clip with the stock 30 s faster-whisper pipeline (reference path)."""
        if self.window_s:
            return self._decode_batch([clip])[0]
        with self._lock:
            segments, _ = self.model.transcribe(
                clip, beam_size=self.beam_size, without_timestamps=True,
                condition_on_previous_text=False, vad_filter=False, temperature=0.0,
                language="en" if self.model.model.is_multilingual else None)
            segments = list(segments)
        if not segments:
            return Candidate("", -9.0, 1.0)
        text = normalise(" ".join(s.text for s in segments))
        logprob = float(np.mean([s.avg_logprob for s in segments]))
        no_speech = float(max(s.no_speech_prob for s in segments))
        return Candidate(text, logprob, no_speech, n_tokens=self._count_tokens(text))

    def fallback(self, x: np.ndarray) -> str:
        """No template match: transcribe the clip, keep loud words after 'by me'."""
        kwargs = dict(beam_size=self.beam_size, word_timestamps=True,
                      condition_on_previous_text=False,
                      language="en" if self.model.model.is_multilingual else None)
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
                if len(tokens[i]) >= 3 and " " not in tokens[i] and loudness[i] > ref - 8]
        return max(best)[1] if best else ""

    def decode(self, clips: Sequence[np.ndarray]) -> List[Candidate]:
        """Decode word clips (batched short-window path, or the stock pipeline)."""
        if self.window_s:
            return self._decode_batch(clips)
        return [self._decode(c) for c in clips]

    def is_common(self, word: str) -> bool:
        """A word whisper's BPE stores as one token: a frequent English word."""
        return bool(word) and len(self.tokenizer.encode(" " + word)) == 1

    def solve(self, x: np.ndarray) -> str:
        """Answer for one decoded clip (``self.second``: optional cheaper second opinion)."""
        second = self.second
        instr_end, score = locate_instruction(x)
        cands: List[Candidate] = []
        others: Optional[List[Candidate]] = None
        if score >= MIN_TEMPLATE_SCORE:
            ref = reference_level(x, instr_end)
            segs = find_word_segments(x, instr_end, ref, self.seg)[:MAX_SEGMENTS]
            rel = relative_db(x, ref)
            clips = [cut_segment(x, seg, rel, self.seg)[0] for seg in segs]
            cands = self.decode(clips)
            for seg, cand in zip(segs, cands):
                cand.segment = seg
            cands = [prefer_stem(c, self.is_common) for c in cands]
            if second is not None and clips:
                others = second.decode(clips)
        answer = choose_answer(cands, others)
        if answer:
            return answer
        logger.info("audio: template score %.2f, no candidate; using fallback", score)
        return self.fallback(x)


def _plural_stem(word: str) -> Optional[str]:
    if len(word) < 4 or not word.endswith("s") or word.endswith(("ss", "us", "is")):
        return None
    return word[:-1]


def prefer_stem(cand: Candidate, is_common, margin: float = None) -> Candidate:
    """'feelings' -> 'feeling' when background noise right after the word added an 's'.

    Applies when the stem is a common word (one BPE token) and another beam hypothesis
    spells exactly the stem with a log-prob within ``margin`` of the best one.
    """
    margin = STEM_MARGIN if margin is None else margin
    stem = _plural_stem(cand.text) if " " not in cand.text else None
    if not stem or margin <= 0 or not is_common(stem):
        return cand
    for text, logprob in cand.alternatives:
        if text == stem and logprob >= cand.logprob - margin:
            return Candidate(stem, logprob, cand.no_speech, cand.segment, 1,
                             cand.alternatives)
    return cand


_FUNCTION_WORDS = frozenset("a an the and of to one".split())


def _core(text: str) -> str:
    """'a thousand' -> 'thousand': drop leading function words that leaked into a run."""
    words = text.split()
    while len(words) > 1 and words[0] in _FUNCTION_WORDS:
        words = words[1:]
    return " ".join(words)


def choose_answer(cands: Sequence[Candidate],
                  others: Optional[Sequence[Candidate]] = None,
                  use_alternatives: bool = None) -> str:
    """Pick the one word to submit.

    ``others`` are the same runs decoded by a second model (index-aligned with ``cands``,
    empty texts included). A single word both models agree on wins (optionally also when
    the second model matches one of the primary's other beam hypotheses); otherwise the
    most confident single-word run of the primary model.
    """
    use_alternatives = USE_ALTERNATIVES if use_alternatives is None else use_alternatives
    if others is not None:
        agreed = []
        for c, o in zip(cands, others):
            other = _core(o.text)
            if not other or len(other.split()) != 1:
                continue
            if _core(c.text) == other:
                agreed.append((c.score, other))
            elif use_alternatives:
                for text, logprob in c.alternatives:
                    if _core(text) == other:
                        agreed.append((logprob - 0.5 * c.no_speech - 0.5, other))
                        break
        if agreed:
            return max(agreed)[1]
    cands = [c for c in cands if c.text]
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
                    solver = AudioSolver()
                    if SECOND_MODEL:
                        logger.info("audio: loading second-opinion model %s", SECOND_MODEL)
                        solver.second = AudioSolver(SECOND_MODEL, window_s=SECOND_WINDOW_S)
                    _solver = solver
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
