"""
General-purpose text-captcha reader: one model for every common style of text captcha.

    from aws_capsolver.text.general import solve
    text, confidence = solve(image_bytes, charset="alnum", min_len=4, max_len=6)

Engines (see the "General model" section of README.md in this directory):

  ours     a CRNN/CTC trained on a broad synthetic generator plus the train splits of
           public captcha datasets (general_model.py, weights/general_ocr.pt)
  ddddocr  the bundled ONNX model of https://github.com/sml2h3/ddddocr, if installed

Both engines produce per-frame CTC log-probabilities over the same 62-character alphabet.
`charset` masks the logits before decoding (a constrained decode, not a post-filter), and
`case_sensitive=False` merges the probability of both cases of a letter. Each engine
proposes candidates (prefix beam search, pruned at `max_len`; a length-constrained
Viterbi decode when nothing fits `min_len`..`max_len`); every candidate is then scored by
BOTH engines with the exact CTC likelihood, and the weighted sum picks the answer
(a product of experts). The confidence is a logistic calibration of the winner's scores,
fitted on held-out data and stored in the weights file.

Lazy, thread-safe singleton; deterministic for a given image and arguments.
Errors: InvalidImage (unusable input), ModelUnavailable (no engine can run).
"""

import io
import logging
import math
import os
import threading
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import general_model as gm
from .errors import InvalidImage, ModelUnavailable

logger = logging.getLogger(__name__)

WEIGHTS_ENV = "GENERAL_OCR_WEIGHTS"
DEFAULT_WEIGHTS = Path(__file__).resolve().parents[2] / "weights" / "general_ocr.pt"
MAX_SIDE = 2000          # larger inputs are not text captchas
MAX_BYTES = 8 << 20
MIN_SIDE = 8
DDDD_HEIGHT = 64
DDDD_MAX_WIDTH = 2048
BEAM = 6
FLOOR = -60.0            # log-likelihood used when an engine cannot produce a candidate

# used when the weights file carries no fitted ensemble parameters
DEFAULT_ENSEMBLE = {"w_ours": 1.0, "w_dddd": 0.5,
                    "calib": [2.0, 0.6, 0.15, 1.0, 0.25]}   # bias, ll_ours, ll_dddd, agree, margin


# engines -----------------------------------------------------------------------
class _Ours:
    def __init__(self, path: Path):
        import torch
        state = torch.load(str(path), map_location="cpu", weights_only=True)
        if state.get("charset") != gm.CHARSET:
            raise ModelUnavailable(f"{path} was trained for another alphabet")
        self.model = gm.build_model()
        self.model.load_state_dict(state["state_dict"])
        self.model.eval()
        self.ensemble = {**DEFAULT_ENSEMBLE, **(state.get("ensemble") or {})}
        self.torch = torch

    def logp(self, rgb) -> np.ndarray:
        x = self.torch.from_numpy(gm.to_array(rgb)[None])
        with self.torch.inference_mode():
            out = self.model(x)[0].log_softmax(-1)
        return out.numpy().astype(np.float64)


class _Dddd:
    """ddddocr's ONNX CTC model, mapped onto our alphabet (blank + 62 characters)."""

    def __init__(self):
        import ddddocr
        ocr = ddddocr.DdddOcr(show_ad=False)
        session = ocr.ocr_engine.session
        charset = list(ocr.get_charset())
        index = {ch: i for i, ch in enumerate(charset)}
        missing = [c for c in gm.CHARSET if c not in index]
        if missing:
            raise ModelUnavailable(f"ddddocr charset lacks {''.join(missing)}")
        self.cols = np.array([0] + [index[c] for c in gm.CHARSET], dtype=np.int64)
        # ddddocr's own session uses one spinning thread per core (and pins them), which
        # starves everything else for a ~5 ms model; run the same model file lean instead
        session = self._lean_session(ddddocr, ocr) or session
        self.session = session
        self.input_name = session.get_inputs()[0].name
        self._ocr = ocr   # keeps the session's owner alive

    @staticmethod
    def _lean_session(ddddocr, ocr):
        """Single-threaded session on ddddocr's default model file; None if unsure."""
        try:
            import onnxruntime as ort
            engine = ocr.ocr_engine
            if getattr(engine, "beta", False) or getattr(engine, "use_import_onnx", False):
                return None
            path = os.path.join(os.path.dirname(ddddocr.__file__), "common_old.onnx")
            if not os.path.exists(path):
                return None
            opts = ort.SessionOptions()
            opts.intra_op_num_threads = int(os.environ.get("GENERAL_OCR_ORT_THREADS", "1"))
            opts.inter_op_num_threads = 1
            opts.add_session_config_entry("session.intra_op.allow_spinning", "0")
            return ort.InferenceSession(path, opts, providers=["CPUExecutionProvider"])
        except Exception as exc:  # any surprise: keep ddddocr's own session
            logger.debug("lean ddddocr session unavailable: %s", exc)
            return None

    def logp(self, gray) -> np.ndarray:
        x = (np.asarray(gray, dtype=np.float32) / 255.0)[None, None]
        out = np.asarray(self.session.run(None, {self.input_name: x})[0], dtype=np.float64)
        if out.ndim == 3:
            out = out[:, 0, :] if out.shape[1] == 1 else out[0]
        if out.min() >= 0.0 and np.allclose(out.sum(axis=1), 1.0, atol=1e-3):
            full = np.log(np.clip(out, 1e-30, None))      # already probabilities
        else:
            full = gm.log_softmax(out)
        return full[:, self.cols]


class _Engine:
    def __init__(self):
        self.ours: Optional[_Ours] = None
        self.dddd: Optional[_Dddd] = None
        errors = []
        path = Path(os.environ.get(WEIGHTS_ENV) or DEFAULT_WEIGHTS)
        try:
            if not path.exists():
                raise ModelUnavailable(f"weights not found: {path}")
            self.ours = _Ours(path)
        except ImportError as exc:
            errors.append(f"torch is not installed ({exc})")
        except ModelUnavailable as exc:
            errors.append(str(exc))
        except Exception as exc:   # corrupt checkpoint
            errors.append(f"cannot load {path}: {exc.__class__.__name__}: {exc}")
        try:
            self.dddd = _Dddd()
        except ImportError:
            errors.append("ddddocr is not installed")
        except ModelUnavailable as exc:
            errors.append(str(exc))
        except Exception as exc:
            errors.append(f"ddddocr failed to load: {exc.__class__.__name__}: {exc}")
        if self.ours is None and self.dddd is None:
            raise ModelUnavailable("; ".join(errors))
        for e in errors:
            logger.warning("general OCR: %s", e)
        self.ensemble = self.ours.ensemble if self.ours else dict(DEFAULT_ENSEMBLE)


_lock = threading.Lock()
_engine: Optional[_Engine] = None
_failed: Optional[str] = None


def _get_engine() -> _Engine:
    global _engine, _failed
    if _engine is not None:
        return _engine
    with _lock:
        if _engine is None:
            if _failed is not None:
                raise ModelUnavailable(_failed)
            try:
                _engine = _Engine()
            except ModelUnavailable as exc:
                _failed = str(exc)
                raise
        return _engine


def warmup() -> bool:
    """Load the engines now (ModelUnavailable when none can run)."""
    _get_engine()
    return True


def engines() -> Dict[str, bool]:
    """Which engines are loaded (loads them on first call)."""
    e = _get_engine()
    return {"ours": e.ours is not None, "ddddocr": e.dddd is not None}


# image -------------------------------------------------------------------------
def load_image(image: bytes):
    """Validated RGB PIL image (alpha flattened onto white); InvalidImage otherwise."""
    from PIL import Image, UnidentifiedImageError
    if not isinstance(image, (bytes, bytearray, memoryview)) or not len(image):
        raise InvalidImage("image must be non-empty bytes")
    if len(image) > MAX_BYTES:
        raise InvalidImage("image is too large")
    try:
        with Image.open(io.BytesIO(bytes(image))) as img:
            if max(img.size) > MAX_SIDE:
                raise InvalidImage(f"image is larger than {MAX_SIDE} px")
            if min(img.size) < MIN_SIDE:
                raise InvalidImage("image is too small")
            img.seek(0)
            img.load()
            rgb = gm.flatten_rgb(img)
    except InvalidImage:
        raise
    except (UnidentifiedImageError, OSError, ValueError, SyntaxError,
            Image.DecompressionBombError) as exc:
        raise InvalidImage("body is not a readable image") from exc
    return rgb


def _dddd_input(rgb):
    from PIL import Image
    gray = rgb.convert("L")
    w, h = gray.size
    width = min(DDDD_MAX_WIDTH, max(16, round(w * DDDD_HEIGHT / h)))
    return gray.resize((width, DDDD_HEIGHT), Image.LANCZOS)


# decoding helpers --------------------------------------------------------------
def ctc_log_likelihood(logp: np.ndarray, label: Sequence[int]) -> float:
    """log P(label | frames) under CTC (forward algorithm) for (T, C) log-probs."""
    T = logp.shape[0]
    if not label:
        return float(logp[:, gm.BLANK].sum())
    ext = np.zeros(2 * len(label) + 1, dtype=np.int64)
    ext[1::2] = label
    S = ext.size
    # a transition s-2 -> s is allowed for non-blank symbols different from s-2
    skip = np.zeros(S, dtype=bool)
    skip[3::2] = ext[3::2] != ext[1:-2:2]
    alpha = np.full(S, -np.inf)
    alpha[0] = logp[0, ext[0]]
    alpha[1] = logp[0, ext[1]]
    for t in range(1, T):
        a1 = np.concatenate(([-np.inf], alpha[:-1]))
        a2 = np.concatenate(([-np.inf, -np.inf], alpha[:-2]))
        a2 = np.where(skip, a2, -np.inf)
        alpha = np.logaddexp(np.logaddexp(alpha, a1), a2) + logp[t, ext]
    return float(np.logaddexp(alpha[-1], alpha[-2]))


def constrained_viterbi(logp: np.ndarray, mask: np.ndarray, min_len: int,
                        max_len: int) -> Optional[Tuple[Tuple[int, ...], float]]:
    """Best single CTC path whose collapsed text has min_len..max_len characters."""
    T, C = logp.shape
    lp = np.where(mask[None, :], logp, -np.inf)
    L = max_len
    dp = np.full((L + 1, C), -np.inf)        # dp[n, c]: n chars emitted, current class c
    dp[0, gm.BLANK] = lp[0, gm.BLANK]
    if L >= 1:
        dp[1, 1:] = lp[0, 1:]
    back = np.zeros((T, L + 1, C, 2), dtype=np.int64)   # (prev n, prev class)
    for t in range(1, T):
        new = np.full_like(dp, -np.inf)
        bk = np.zeros((L + 1, C, 2), dtype=np.int64)
        # to blank: from any class at the same count
        best_c = dp.argmax(axis=1)
        new[:, gm.BLANK] = dp[np.arange(L + 1), best_c] + lp[t, gm.BLANK]
        bk[:, gm.BLANK, 0] = np.arange(L + 1)
        bk[:, gm.BLANK, 1] = best_c
        # stay on the same non-blank class (no new character)
        stay = dp[:, 1:] + lp[t, 1:][None, :]
        # emit a new character k from blank or from a different class
        if L >= 1:
            prev = dp[:-1]                               # (L, C)
            top2 = np.argsort(prev, axis=1)[:, ::-1][:, :2]
            first = prev[np.arange(L), top2[:, 0]]
            second = prev[np.arange(L), top2[:, 1]]
            ks = np.arange(1, C)
            # best predecessor class != k
            src = np.where(top2[:, :1] == ks[None, :], top2[:, 1:2], top2[:, :1])
            src_val = np.where(top2[:, :1] == ks[None, :], second[:, None], first[:, None])
            emit = src_val + lp[t, 1:][None, :]
            emit_full = np.full((L + 1, C - 1), -np.inf)
            emit_full[1:] = emit
            src_full = np.zeros((L + 1, C - 1), dtype=np.int64)
            src_full[1:] = src
        else:
            emit_full = np.full((L + 1, C - 1), -np.inf)
            src_full = np.zeros((L + 1, C - 1), dtype=np.int64)
        take_emit = emit_full > stay
        new[:, 1:] = np.where(take_emit, emit_full, stay)
        n_idx = np.arange(L + 1)[:, None].repeat(C - 1, axis=1)
        bk[:, 1:, 0] = np.where(take_emit, n_idx - 1, n_idx)
        bk[:, 1:, 1] = np.where(take_emit, src_full, np.arange(1, C)[None, :])
        dp, back[t] = new, bk
    best = None
    for n in range(max(0, min_len), L + 1):
        c = int(dp[n].argmax())
        if np.isfinite(dp[n, c]) and (best is None or dp[n, c] > best[2]):
            best = (n, c, float(dp[n, c]))
    if best is None:
        return None
    n, c, score = best
    path = []
    for t in range(T - 1, -1, -1):
        path.append(c)
        if t:
            n, c = back[t, n, c]
    path.reverse()
    out, prev = [], gm.BLANK
    for c in path:
        if c != gm.BLANK and c != prev:
            out.append(int(c))
        prev = c
    return tuple(out), score


def _fold_label(label: Sequence[int]) -> Tuple[int, ...]:
    """Class indices -> their lower-case class (digits unchanged)."""
    return tuple(gm.CHAR_TO_INDEX[gm.CHARSET[c - 1].lower()] for c in label)


def _view(logp: np.ndarray, mask: np.ndarray, case_sensitive: bool):
    if case_sensitive:
        return np.where(mask[None, :], logp, -np.inf), mask
    folded, fmask = gm.fold_case(logp, mask)
    return np.where(fmask[None, :], folded, -np.inf), fmask


def _candidates(view: np.ndarray, vmask: np.ndarray, lo: int, hi: int) -> List[Tuple[int, ...]]:
    beams = gm.ctc_beam_search(view, vmask, beam=BEAM, max_len=hi)
    out = [p for p, _ in beams if lo <= len(p) <= hi]
    if not out:
        forced = constrained_viterbi(view, vmask, lo, hi)
        if forced is not None and forced[0]:
            out = [forced[0]]
    return out


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-50.0, min(50.0, x))))


def _resolve_charset(charset: Optional[str]) -> Optional[str]:
    if charset is None:
        return None
    if not isinstance(charset, str) or not charset:
        raise ValueError("charset must be a non-empty string or None")
    if charset in ("digits", "letters", "alnum", "lower", "upper"):
        return charset
    if not any(c in gm.CHAR_TO_INDEX for c in charset):
        raise ValueError("charset contains no supported characters (0-9, a-z, A-Z)")
    return charset


# public API ----------------------------------------------------------------------
def solve(image: bytes, *, charset: Optional[str] = None, min_len: Optional[int] = None,
          max_len: Optional[int] = None, case_sensitive: bool = False,
          details: bool = False):
    """
    Read the text in a captcha image.

    charset         None (any of 0-9a-zA-Z), "digits", "letters", "alnum", "lower", "upper"
                    or an explicit string of allowed characters
    min_len/max_len length limits (None = 1 / 12)
    case_sensitive  False merges the two cases of each letter while decoding; the answer
                    keeps the case the model reads, compare it case-insensitively
    Returns (text, confidence in 0..1); with details=True a third item, a dict of the
    per-engine readings, is added (used by the benchmark).
    """
    charset = _resolve_charset(charset)
    lo = 1 if min_len is None else max(1, int(min_len))
    hi = gm.MAX_LABEL_LEN if max_len is None else min(gm.MAX_LABEL_LEN, int(max_len))
    if lo > hi:
        raise ValueError(f"min_len {min_len} is larger than max_len {max_len}")
    engine = _get_engine()
    rgb = load_image(image)
    mask = gm.charset_mask(charset)
    if case_sensitive is False and charset is not None and charset not in (
            "digits", "letters", "alnum", "lower", "upper"):
        # an explicit charset given for a case-insensitive captcha covers both cases
        mask = gm.charset_mask(charset.lower() + charset.upper())

    views = {}
    if engine.ours is not None:
        lp_ours = engine.ours.logp(rgb)
        views["ours"] = (lp_ours, *_view(lp_ours, mask, case_sensitive))
    if engine.dddd is not None:
        lp_d = engine.dddd.logp(_dddd_input(rgb))
        views["ddddocr"] = (lp_d, *_view(lp_d, mask, case_sensitive))

    cands: Dict[Tuple[int, ...], None] = {}
    tops: Dict[str, Optional[Tuple[int, ...]]] = {}
    for name, (_, view, vmask) in views.items():
        c = _candidates(view, vmask, lo, hi)
        tops[name] = c[0] if c else None
        for p in c:
            cands.setdefault(p, None)
    if not cands:
        return ("", 0.0, {"engines": {}}) if details else ("", 0.0)

    ens = engine.ensemble
    w_dddd = ens.get("w_dddd_cs", ens["w_dddd"]) if case_sensitive else ens["w_dddd"]
    weights = {"ours": float(ens["w_ours"]), "ddddocr": float(w_dddd)}
    scored = []
    for p in cands:
        lls = {}
        for name, (_, view, _) in views.items():
            ll = ctc_log_likelihood(view, p)
            lls[name] = max(FLOOR, ll) if np.isfinite(ll) else FLOOR
        score = sum(weights[n] * v for n, v in lls.items())
        scored.append((score, p, lls))
    scored.sort(key=lambda s: -s[0])
    best_score, best, lls = scored[0]
    margin = best_score - scored[1][0] if len(scored) > 1 else 10.0
    agree = 1.0 if len(views) > 1 and tops.get("ours") == tops.get("ddddocr") == best else 0.0
    b, c1, c2, c3, c4 = [float(v) for v in ens["calib"]]
    confidence = _sigmoid(b + c1 * lls.get("ours", FLOOR / 4) + c2 * lls.get("ddddocr", FLOOR / 4)
                          + c3 * agree + c4 * min(margin, 10.0))

    text = gm.classes_to_text(best)
    if not case_sensitive and "ours" in views:
        # restore the case the model actually reads (same letters, case-sensitive decode)
        lp_ours = views["ours"][0]
        cs_view = np.where(mask[None, :], lp_ours, -np.inf)
        cs_beams = gm.ctc_beam_search(cs_view, mask, beam=BEAM, max_len=hi)
        for p, _ in cs_beams:
            if _fold_label(p) == best:
                text = gm.classes_to_text(p)
                break
    if details:
        info = {"engines": {n: gm.classes_to_text(t) if t else "" for n, t in tops.items()},
                "lls": lls, "margin": margin, "agree": agree}
        return text, confidence, info
    return text, confidence
