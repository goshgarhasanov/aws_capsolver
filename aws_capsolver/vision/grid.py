"""AWS WAF ``gridcaptcha`` solver: pick every tile of a 3x3 grid that shows the target.

Pipeline
--------
1. Every tile is embedded with one or more frozen CLIP/SigLIP backbones (open_clip),
   optionally with horizontal-flip test-time augmentation.
2. If a trained head (``weights/aws_capsolver_head.pt``) matching the backbones is
   present and knows the target class, a multinomial logistic-regression head over the
   concatenated embeddings gives P(class | tile).
3. Otherwise (or for unseen targets) zero-shot CLIP is used: prompt-ensembled text
   embeddings of the target plus a contrast set of every other known class.
4. A tile is selected when the target is its most probable class, or its target
   probability exceeds ``threshold``. If nothing qualifies the best tile is returned.

Environment variables
---------------------
AWS_CAPSOLVER_CLIP_MODELS   comma separated ``model:pretrained`` list (backbone ensemble)
AWS_CAPSOLVER_TORCH_THREADS torch intra-op threads (default: min(8, cpu_count))
AWS_CAPSOLVER_HEAD          path to the trained head (default weights/aws_capsolver_head.pt;
                            "none" disables it)
AWS_CAPSOLVER_TTA           "1" (default) to average original + horizontally flipped tile
"""
from __future__ import annotations

import io
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np

from .errors import InvalidInput

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_HEAD_PATH = ROOT / "weights" / "aws_capsolver_head.pt"
DEFAULT_MODELS = "ViT-L-14:datacomp_xl_s13b_b90k"
DEFAULT_THRESHOLD = 0.5

# Targets observed in live AWS WAF grid problems plus frequent distractor objects.
# Every entry is part of the zero-shot contrast set; unknown targets are added on the fly.
KNOWN_TARGETS: tuple[str, ...] = (
    "bag", "bed", "bucket", "chair", "clock", "curtain", "hat",
)
DISTRACTORS: tuple[str, ...] = (
    "backpack", "table", "sofa", "lamp", "shoe", "mug", "box", "basket", "pillow",
    "window", "door", "bottle", "car", "dog", "tree", "flower", "book", "empty room",
)
# Extra phrasing per class that improves text embeddings.
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


def normalize_target(target: str) -> str:
    """Map a WAF target string ('clock', 'the clocks', '["clock"]') to a class name."""
    t = target.strip().strip('[]"\' ').lower()
    if t.startswith("the "):
        t = t[4:]
    if t in KNOWN_TARGETS:
        return t
    for cand in (t[:-2] if t.endswith("es") else None, t[:-1] if t.endswith("s") else None):
        if cand and cand in KNOWN_TARGETS:
            return cand
    return t


def _prompts(cls: str) -> list[str]:
    names = SYNONYMS.get(cls, (cls,))
    return [tpl.format(n) for n in names for tpl in TEMPLATES]


def _decode(images: Sequence[bytes]) -> list:
    from PIL import Image, UnidentifiedImageError

    out = []
    for i, b in enumerate(images):
        if not isinstance(b, (bytes, bytearray, memoryview)):
            raise InvalidInput(f"tile {i}: expected bytes, got {type(b).__name__}")
        try:
            img = Image.open(io.BytesIO(bytes(b)))
            img.load()
        except (UnidentifiedImageError, OSError) as exc:
            raise InvalidInput(f"tile {i}: cannot decode image: {exc}") from exc
        out.append(img.convert("RGB"))
    return out


@dataclass
class _Backbone:
    spec: str
    model: object
    preprocess: object
    tokenizer: object
    text_cache: dict[str, np.ndarray] = field(default_factory=dict)


class GridEngine:
    """Loaded backbones + optional head. Use :func:`get_engine` for the shared instance."""

    def __init__(self, specs: Sequence[str] | None = None, head_path: str | Path | None = None,
                 tta: bool | None = None, threads: int | None = None) -> None:
        import torch

        self._torch = torch
        threads = threads or int(os.environ.get("AWS_CAPSOLVER_TORCH_THREADS", "0") or 0) \
            or min(8, os.cpu_count() or 1)
        torch.set_num_threads(threads)
        if specs is None:
            specs = [s.strip() for s in os.environ.get("AWS_CAPSOLVER_CLIP_MODELS", DEFAULT_MODELS).split(",")
                     if s.strip()]
        self.tta = (os.environ.get("AWS_CAPSOLVER_TTA", "1") != "0") if tta is None else tta
        self.backbones: list[_Backbone] = [self._load_backbone(s) for s in specs]
        self.specs = [b.spec for b in self.backbones]
        self._lock = threading.Lock()
        self.head: dict | None = None
        if head_path is None:
            head_path = os.environ.get("AWS_CAPSOLVER_HEAD", str(DEFAULT_HEAD_PATH))
        if str(head_path).lower() != "none":
            self.head = self._load_head(Path(head_path))
        log.info("grid engine ready: backbones=%s head=%s tta=%s threads=%d",
                 self.specs, bool(self.head), self.tta, threads)

    # ------------------------------------------------------------------ loading
    def _load_backbone(self, spec: str) -> _Backbone:
        import open_clip

        name, _, pretrained = spec.partition(":")
        t0 = time.time()
        model, _, preprocess = open_clip.create_model_and_transforms(name, pretrained=pretrained or None)
        model.eval()
        tokenizer = open_clip.get_tokenizer(name)
        log.info("loaded %s in %.1fs", spec, time.time() - t0)
        return _Backbone(spec, model, preprocess, tokenizer)

    def _load_head(self, path: Path) -> dict | None:
        if not path.is_file():
            return None
        try:
            head = self._torch.load(path, map_location="cpu", weights_only=False)
        except (OSError, RuntimeError, ValueError) as exc:
            log.warning("cannot load head %s: %s", path, exc)
            return None
        if list(head.get("backbones", [])) != self.specs or bool(head.get("tta")) != self.tta:
            log.warning("head %s was trained for %s (tta=%s), engine uses %s (tta=%s); ignoring head",
                        path, head.get("backbones"), head.get("tta"), self.specs, self.tta)
            return None
        head["W"] = np.asarray(head["W"], dtype=np.float32)
        head["b"] = np.asarray(head["b"], dtype=np.float32)
        return head

    # ---------------------------------------------------------------- embedding
    def _embed_one(self, bb: _Backbone, pil_images: list) -> np.ndarray:
        torch = self._torch
        batch = torch.stack([bb.preprocess(im) for im in pil_images])
        with torch.inference_mode():
            f = bb.model.encode_image(batch)
            f = f / f.norm(dim=-1, keepdim=True)
            if self.tta:
                g = bb.model.encode_image(torch.flip(batch, dims=[3]))
                f = f + g / g.norm(dim=-1, keepdim=True)
                f = f / f.norm(dim=-1, keepdim=True)
        return f.float().numpy()

    def embed(self, pil_images: list) -> list[np.ndarray]:
        """Per-backbone L2-normalised image embeddings, each (n, d_k)."""
        with self._lock:
            return [self._embed_one(bb, pil_images) for bb in self.backbones]

    def embed_bytes(self, images: Sequence[bytes]) -> list[np.ndarray]:
        return self.embed(_decode(images))

    def _text(self, bb: _Backbone, cls: str) -> np.ndarray:
        cached = bb.text_cache.get(cls)
        if cached is not None:
            return cached
        torch = self._torch
        with torch.inference_mode():
            t = bb.model.encode_text(bb.tokenizer(_prompts(cls)))
            t = t / t.norm(dim=-1, keepdim=True)
            t = t.mean(0)
            t = (t / t.norm()).float().numpy()
        bb.text_cache[cls] = t
        return t

    # ------------------------------------------------------------------ scoring
    def zero_shot_probs(self, feats: list[np.ndarray], target: str) -> np.ndarray:
        """P(target | tile) under a softmax over target + contrast classes."""
        own = set(SYNONYMS.get(target, (target,)))
        classes = [target] + [c for c in (*KNOWN_TARGETS, *DISTRACTORS) if c != target and c not in own]
        probs = []
        with self._lock:
            for bb, f in zip(self.backbones, feats):
                T = np.stack([self._text(bb, c) for c in classes])
                logits = 100.0 * f @ T.T
                logits -= logits.max(1, keepdims=True)
                p = np.exp(logits)
                p /= p.sum(1, keepdims=True)
                probs.append(p[:, 0])
        return np.mean(probs, axis=0)

    def head_probs(self, feats: list[np.ndarray], target: str) -> np.ndarray | None:
        if self.head is None or target not in self.head["classes"]:
            return None
        x = np.concatenate(feats, axis=1)
        logits = x @ self.head["W"].T + self.head["b"]
        logits -= logits.max(1, keepdims=True)
        p = np.exp(logits)
        p /= p.sum(1, keepdims=True)
        return p[:, self.head["classes"].index(target)]

    def scores(self, images: Sequence[bytes], target: str) -> tuple[np.ndarray, str]:
        """Return (P(target) per tile, method name)."""
        target = normalize_target(target)
        feats = self.embed_bytes(images)
        p = self.head_probs(feats, target)
        if p is not None:
            return p, "head"
        return self.zero_shot_probs(feats, target), "zeroshot"

    def default_threshold(self, target: str) -> float:
        if self.head is not None:
            per = self.head.get("thresholds", {})
            return float(per.get(normalize_target(target), self.head.get("threshold", DEFAULT_THRESHOLD)))
        return DEFAULT_THRESHOLD


_engine: GridEngine | None = None
_engine_lock = threading.Lock()


def get_engine() -> GridEngine:
    """Lazily created, thread-safe process-wide engine."""
    global _engine
    if _engine is None:
        with _engine_lock:
            if _engine is None:
                _engine = GridEngine()
    return _engine


def select_tiles(scores: np.ndarray, threshold: float) -> list[int]:
    sel = [i for i, s in enumerate(scores) if s >= threshold]
    if not sel:  # every AWS grid has at least one target tile
        sel = [int(np.argmax(scores))]
    return sel


def classify_grid(images: list[bytes], target: str, threshold: float | None = None) -> list[int]:
    """Return 0-based, row-major indices of the tiles that contain ``target``.

    ``images`` are the raw JPEG bytes of the tiles (normally 9), ``target`` is the WAF
    target ('clock', '["clock"]' or 'the clocks').
    """
    if not images:
        raise InvalidInput("no tiles supplied")
    if not isinstance(target, str) or not target.strip():
        raise InvalidInput("empty target")
    eng = get_engine()
    scores, _ = eng.scores(images, target)
    thr = eng.default_threshold(target) if threshold is None else float(threshold)
    return select_tiles(scores, thr)
