"""AWS WAF ``gridcaptcha`` solver: pick the tiles of a 3x3 grid that show the target.

Pipeline
--------
1. Each of the 9 tiles is embedded by one or more frozen open_clip backbones
   (L2-normalised; optional horizontal-flip test-time augmentation).
2. A trained head (``weights/aws_capsolver_head.pt``, multinomial logistic regression over
   the concatenated embeddings) gives P(class | tile) for the classes it knows.
   For a target the head does not know (or when no compatible head is installed) zero-shot
   CLIP is used instead: prompt-ensembled text embeddings of the target contrasted with
   every other known class and a set of distractor objects.
3. Selection: ``gridcaptcha-5-*`` grids contain exactly 5 target tiles (true for all 1000
   labelled sample grids), so by default the 5 most probable tiles are returned. An explicit ``threshold`` switches to
   "every tile with P >= threshold".

Environment variables
---------------------
AWS_CAPSOLVER_CLIP_MODELS      comma separated ``model:pretrained`` backbones
                               (default: see DEFAULT_MODELS)
AWS_CAPSOLVER_TORCH_THREADS    torch intra-op threads (default: min(8, cpu_count))
AWS_CAPSOLVER_HEAD             head path (default weights/aws_capsolver_head.pt; "none" = off)
AWS_CAPSOLVER_TTA              "1" to average the tile and its mirror image (default "0")
AWS_CAPSOLVER_GRID_K           default number of tiles to select for 9-tile grids (5;
                               "0" = use the head's threshold instead)
AWS_CAPSOLVER_STRICT_TARGETS   "1" = raise UnknownTarget for targets outside the known
                               classes instead of falling back to zero-shot

Backbone weights are downloaded by open_clip from the Hugging Face hub on first use and
cached (HF_HOME); set HF_HUB_OFFLINE=1 on machines that already have them.
"""
from __future__ import annotations

import base64
import binascii
import io
import json
import logging
import os
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence, Union

import numpy as np

from .errors import InvalidInput, ModelUnavailable, UnknownTarget

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_HEAD_PATH = ROOT / "weights" / "aws_capsolver_head.pt"
DEFAULT_MODELS = "ViT-B-32:laion2b_s34b_b79k,ViT-B-16:datacomp_xl_s13b_b90k"
DEFAULT_THRESHOLD = 0.5
GRID_TILES = 9
MAX_TILE_BYTES = 2 * 1024 * 1024
MAX_TILE_SIDE = 1024

TileInput = Union[bytes, bytearray, memoryview, str]

# 8999 of 9000 sampled tiles (1000 grids, Nov 2026) show one of these 7 object classes.
KNOWN_TARGETS: tuple[str, ...] = (
    "bag", "bed", "bucket", "chair", "clock", "curtain", "hat",
)
# Extra zero-shot contrast classes (never selected; they absorb unrelated tiles).
DISTRACTORS: tuple[str, ...] = (
    "backpack", "table", "sofa", "lamp", "shoe", "mug", "box", "basket", "pillow",
    "window", "door", "bottle", "car", "dog", "tree", "flower", "book", "empty room",
)
# Alternative names used in the prompt ensemble of a class.
SYNONYMS: dict[str, tuple[str, ...]] = {
    "bag": ("bag", "handbag", "duffel bag", "tote bag"),
    "bed": ("bed", "double bed", "bed with pillows"),
    "bucket": ("bucket", "pail"),
    "chair": ("chair", "armchair"),
    "clock": ("clock", "wall clock", "analog clock"),
    "curtain": ("curtain", "curtains on a window", "drapes"),
    "hat": ("hat", "cap", "beanie"),
}
TEMPLATES: tuple[str, ...] = (
    "a photo of a {}.",
    "a photo of the {}.",
    "a close-up photo of a {}.",
    "a blurry photo of a {}.",
    "a low quality photo of a {}.",
    "a rendering of a {}.",
    "an image of a {}.",
    "a product photo of a {}.",
)
_TARGET_RE = re.compile(r"^[a-z][a-z -]{0,39}$")


def _env_flag(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip().lower() in ("1", "true", "yes", "on")


# ----------------------------------------------------------------------------- inputs
def normalize_target(target: object) -> str:
    """Map a WAF target ('clock', '["clock"]', 'the clocks', ['clock']) to a class name.

    Raises InvalidInput for empty / non-text / implausible targets.
    """
    if isinstance(target, (list, tuple)):
        if len(target) != 1:
            raise InvalidInput(f"expected exactly one target, got {len(target)}")
        target = target[0]
    if not isinstance(target, str):
        raise InvalidInput(f"target must be a string, got {type(target).__name__}")
    t = target.strip()
    if t.startswith("["):
        try:
            parsed = json.loads(t)
        except ValueError as exc:
            raise InvalidInput(f"malformed target list: {target!r}") from exc
        return normalize_target(parsed)
    t = re.sub(r"\s+", " ", t.strip("\"' ").lower())
    if t.startswith("the "):
        t = t[4:]
    if not t:
        raise InvalidInput("empty target")
    if not _TARGET_RE.match(t):
        raise InvalidInput(f"implausible target: {target!r}")
    if t in KNOWN_TARGETS:
        return t
    for cand in (t[:-2] if t.endswith("es") else None, t[:-1] if t.endswith("s") else None):
        if cand and cand in KNOWN_TARGETS:
            return cand
    return t


def _tile_bytes(i: int, tile: object) -> bytes:
    if isinstance(tile, (bytes, bytearray, memoryview)):
        data = bytes(tile)
    elif isinstance(tile, str):  # base64 (optionally a data: URL)
        s = tile.strip()
        if s.startswith("data:"):
            s = s.partition(",")[2]
        try:
            data = base64.b64decode(s, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise InvalidInput(f"tile {i}: invalid base64") from exc
    else:
        raise InvalidInput(f"tile {i}: expected bytes or base64 str, got {type(tile).__name__}")
    if not data:
        raise InvalidInput(f"tile {i}: empty")
    if len(data) > MAX_TILE_BYTES:
        raise InvalidInput(f"tile {i}: {len(data)} bytes exceeds {MAX_TILE_BYTES}")
    return data


def decode_tiles(images: Sequence[TileInput], expected: int | None = GRID_TILES) -> list:
    """Validate and decode tiles into RGB PIL images."""
    from PIL import Image, UnidentifiedImageError

    if isinstance(images, (bytes, bytearray, str)) or not isinstance(images, Sequence):
        raise InvalidInput("images must be a list of tiles")
    if expected is not None and len(images) != expected:
        raise InvalidInput(f"expected {expected} tiles, got {len(images)}")
    if not images:
        raise InvalidInput("no tiles supplied")
    out = []
    for i, tile in enumerate(images):
        data = _tile_bytes(i, tile)
        try:
            img = Image.open(io.BytesIO(data))
            if max(img.size) > MAX_TILE_SIDE or min(img.size) < 8:
                raise InvalidInput(f"tile {i}: implausible size {img.size}")
            img.load()
            out.append(img.convert("RGB"))
        except (UnidentifiedImageError, OSError, Image.DecompressionBombError, ValueError) as exc:
            if isinstance(exc, InvalidInput):
                raise
            raise InvalidInput(f"tile {i}: cannot decode image: {exc}") from exc
    return out


# ----------------------------------------------------------------------------- engine
def _prompts(cls: str) -> list[str]:
    return [tpl.format(n) for n in SYNONYMS.get(cls, (cls,)) for tpl in TEMPLATES]


def _softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


@dataclass
class _Backbone:
    spec: str
    model: object
    preprocess: object
    tokenizer: object
    text_cache: dict[str, np.ndarray] = field(default_factory=dict)


class GridEngine:
    """Loaded backbones + optional head. Use :func:`get_engine` for the shared instance.

    All public methods are thread-safe: model calls are serialised by one lock (torch
    already parallelises each forward pass across ``threads`` cores).
    """

    def __init__(self, specs: Sequence[str] | None = None, head_path: str | Path | None = None,
                 tta: bool | None = None, threads: int | None = None) -> None:
        try:
            import torch
        except ImportError as exc:
            raise ModelUnavailable(f"torch is not installed: {exc}") from exc
        self._torch = torch
        env_threads = os.environ.get("AWS_CAPSOLVER_TORCH_THREADS", "").strip()
        threads = threads or (int(env_threads) if env_threads.isdigit() and int(env_threads) > 0 else 0) \
            or min(8, os.cpu_count() or 1)
        torch.set_num_threads(threads)
        if specs is None:
            specs = os.environ.get("AWS_CAPSOLVER_CLIP_MODELS", DEFAULT_MODELS).split(",")
        specs = [s.strip() for s in specs if s and s.strip()]
        if not specs:
            raise ModelUnavailable("no backbone configured")
        self.tta = _env_flag("AWS_CAPSOLVER_TTA") if tta is None else bool(tta)
        self._lock = threading.RLock()
        self.backbones: list[_Backbone] = [self._load_backbone(s) for s in specs]
        self.specs = [b.spec for b in self.backbones]
        if head_path is None:
            head_path = os.environ.get("AWS_CAPSOLVER_HEAD", str(DEFAULT_HEAD_PATH))
        self.head: dict | None = None
        if str(head_path).strip().lower() not in ("", "none", "0", "off"):
            self.head = self._load_head(Path(head_path))
        log.info("grid engine ready: backbones=%s head=%s tta=%s threads=%d",
                 self.specs, bool(self.head), self.tta, threads)

    # -------------------------------------------------------------------- loading
    def _load_backbone(self, spec: str) -> _Backbone:
        try:
            import open_clip
        except ImportError as exc:
            raise ModelUnavailable(f"open_clip is not installed: {exc}") from exc
        name, _, pretrained = spec.partition(":")
        t0 = time.time()
        try:
            model, _, preprocess = open_clip.create_model_and_transforms(name, pretrained=pretrained or None)
            tokenizer = open_clip.get_tokenizer(name)
        except (RuntimeError, OSError, ValueError, ImportError) as exc:
            raise ModelUnavailable(f"cannot load backbone {spec!r}: {exc}") from exc
        model.eval()
        log.info("loaded backbone %s in %.1fs", spec, time.time() - t0)
        return _Backbone(spec, model, preprocess, tokenizer)

    def _load_head(self, path: Path) -> dict | None:
        if not path.is_file():
            log.info("no head at %s; using zero-shot only", path)
            return None
        try:
            # weights_only=True: the file holds plain tensors / lists / dicts only.
            head = self._torch.load(path, map_location="cpu", weights_only=True)
            classes = [str(c) for c in head["classes"]]
            W = np.asarray(head["W"], dtype=np.float32)
            b = np.asarray(head["b"], dtype=np.float32)
        except (OSError, RuntimeError, ValueError, KeyError, TypeError) as exc:
            log.warning("cannot load head %s: %s; using zero-shot only", path, exc)
            return None
        if list(head.get("backbones", [])) != self.specs or bool(head.get("tta")) != self.tta:
            log.warning("head %s was trained for %s (tta=%s) but the engine runs %s (tta=%s); "
                        "ignoring it", path, head.get("backbones"), head.get("tta"), self.specs, self.tta)
            return None
        dim = sum(self._embed_dim(bb) for bb in self.backbones)
        if W.shape != (len(classes), dim) or b.shape != (len(classes),):
            log.warning("head %s has shape %s, expected (%d, %d); ignoring it", path, W.shape, len(classes), dim)
            return None
        return {"classes": classes, "W": W, "b": b, "trained_on": str(head.get("trained_on", "unknown")),
                "threshold": float(head.get("threshold", DEFAULT_THRESHOLD)),
                "thresholds": {str(k): float(v) for k, v in dict(head.get("thresholds", {})).items()}}

    def _embed_dim(self, bb: _Backbone) -> int:
        with self._torch.inference_mode():
            return int(bb.model.encode_text(bb.tokenizer(["x"])).shape[-1])

    # ------------------------------------------------------------------ embedding
    def _embed_one(self, bb: _Backbone, pil_images: list) -> np.ndarray:
        torch = self._torch
        batch = torch.stack([bb.preprocess(im) for im in pil_images])
        if self.tta:
            batch = torch.cat([batch, torch.flip(batch, dims=[3])])
        with torch.inference_mode():
            f = bb.model.encode_image(batch).float()
        f = f / f.norm(dim=-1, keepdim=True)
        if self.tta:
            n = len(pil_images)
            f = f[:n] + f[n:]
            f = f / f.norm(dim=-1, keepdim=True)
        return f.numpy()

    def embed(self, pil_images: list) -> list[np.ndarray]:
        """Per-backbone L2-normalised image embeddings, each of shape (n, d_k)."""
        with self._lock:
            return [self._embed_one(bb, pil_images) for bb in self.backbones]

    def _text(self, bb: _Backbone, cls: str) -> np.ndarray:
        with self._lock:
            cached = bb.text_cache.get(cls)
            if cached is not None:
                return cached
            torch = self._torch
            with torch.inference_mode():
                t = bb.model.encode_text(bb.tokenizer(_prompts(cls))).float()
            t = t / t.norm(dim=-1, keepdim=True)
            t = t.mean(0)
            t = (t / t.norm()).numpy()
            if len(bb.text_cache) > 256:  # bounded: targets come from untrusted input
                bb.text_cache.clear()
            bb.text_cache[cls] = t
            return t

    # -------------------------------------------------------------------- scoring
    def knows(self, target: str) -> bool:
        if target == "other":
            return False
        return target in KNOWN_TARGETS or (self.head is not None and target in self.head["classes"])

    def zero_shot_probs(self, feats: list[np.ndarray], target: str) -> np.ndarray:
        """P(target | tile) under a softmax over target + contrast classes (mean over backbones)."""
        own = set(SYNONYMS.get(target, (target,)))
        classes = [target] + [c for c in (*KNOWN_TARGETS, *DISTRACTORS) if c != target and c not in own]
        probs = []
        for bb, f in zip(self.backbones, feats):
            T = np.stack([self._text(bb, c) for c in classes])
            probs.append(_softmax(100.0 * f @ T.T)[:, 0])
        return np.mean(probs, axis=0)

    def head_probs(self, feats: list[np.ndarray], target: str) -> np.ndarray | None:
        if self.head is None or target == "other" or target not in self.head["classes"]:
            return None
        x = np.concatenate(feats, axis=1)
        p = _softmax(x @ self.head["W"].T + self.head["b"])
        return p[:, self.head["classes"].index(target)]

    def score_features(self, feats: list[np.ndarray], target: str) -> tuple[np.ndarray, str]:
        p = self.head_probs(feats, target)
        if p is not None:
            return p, "head"
        return self.zero_shot_probs(feats, target), "zeroshot"

    def scores(self, images: Sequence[TileInput], target: object,
               expected_tiles: int | None = GRID_TILES) -> tuple[np.ndarray, str]:
        """Return (P(target) per tile, method) for raw tiles."""
        t = normalize_target(target)
        if not self.knows(t):
            if _env_flag("AWS_CAPSOLVER_STRICT_TARGETS"):
                raise UnknownTarget(f"unknown grid target {t!r}")
            log.warning("unknown grid target %r: falling back to zero-shot", t)
        pil = decode_tiles(images, expected_tiles)
        return self.score_features(self.embed(pil), t)

    def default_threshold(self, target: str) -> float:
        if self.head is None:
            return DEFAULT_THRESHOLD
        return self.head["thresholds"].get(target, self.head["threshold"])


# ------------------------------------------------------------------- singleton / API
_engine: GridEngine | None = None
_engine_lock = threading.Lock()


def get_engine() -> GridEngine:
    """Lazily created, process-wide engine (thread-safe; concurrent callers wait for one load)."""
    global _engine
    if _engine is None:
        with _engine_lock:
            if _engine is None:
                _engine = GridEngine()
    return _engine


def warmup(background: bool = True) -> threading.Thread | None:
    """Load the engine and run one dummy inference.

    With ``background=True`` this returns immediately with the started daemon thread;
    requests arriving meanwhile simply wait for the load inside :func:`get_engine`.
    """
    def _run() -> None:
        from PIL import Image
        try:
            eng = get_engine()
            buf = io.BytesIO()
            Image.new("RGB", (100, 100), (128, 128, 128)).save(buf, "JPEG")
            eng.scores([buf.getvalue()] * GRID_TILES, KNOWN_TARGETS[0])
            log.info("grid engine warm-up done")
        except Exception:  # noqa: BLE001 - a failed warm-up must not kill the thread silently
            log.exception("grid engine warm-up failed")

    if not background:
        _run()
        return None
    th = threading.Thread(target=_run, name="grid-warmup", daemon=True)
    th.start()
    return th


def select_tiles(scores: Sequence[float], threshold: float | None = None, count: int | None = None) -> list[int]:
    """Pick tiles: the ``count`` best-scoring ones, or every tile scoring >= ``threshold``.

    Deterministic: ties are broken by the lower tile index. Never returns an empty list.
    """
    s = np.asarray(scores, dtype=np.float64)
    if s.ndim != 1 or s.size == 0:
        raise InvalidInput("scores must be a non-empty 1-D sequence")
    if count:
        count = max(1, min(int(count), s.size))
        return sorted(int(i) for i in np.argsort(-s, kind="stable")[:count])
    thr = DEFAULT_THRESHOLD if threshold is None else float(threshold)
    sel = [i for i, v in enumerate(s) if v >= thr]
    return sel or [int(np.argmax(s))]


def classify_grid(images: list[bytes], target: str, threshold: float | None = None, *,
                  count: int | None = None) -> list[int]:
    """Return 0-based, row-major indices of the tiles that contain ``target``.

    ``images``: the 9 tiles as raw JPEG/PNG bytes (base64 strings are accepted too).
    ``target``: 'clock', '["clock"]' or 'the clocks'.
    Selection: explicit ``threshold`` -> every tile with P >= threshold; otherwise the
    ``count`` most likely tiles (default AWS_CAPSOLVER_GRID_K = 5, see module docstring).

    Raises InvalidInput, UnknownTarget (strict mode only) or ModelUnavailable.
    """
    if threshold is not None:
        if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not 0.0 <= threshold <= 1.0:
            raise InvalidInput(f"threshold must be within [0, 1], got {threshold!r}")
    if count is not None:
        if isinstance(count, bool) or not isinstance(count, int) or not 1 <= count <= GRID_TILES:
            raise InvalidInput(f"count must be an int within [1, {GRID_TILES}], got {count!r}")
    normalize_target(target)  # validate before the (possibly slow) model load
    eng = get_engine()
    scores, _ = eng.scores(images, target)
    if threshold is not None:
        return select_tiles(scores, threshold=float(threshold))
    if count is None:
        env_k = os.environ.get("AWS_CAPSOLVER_GRID_K", "5").strip()
        count = int(env_k) if env_k.isdigit() else 5
        count = min(count, GRID_TILES) or None
    if count:
        return select_tiles(scores, count=count)
    return select_tiles(scores, threshold=eng.default_threshold(normalize_target(target)))
