"""reCAPTCHA v2 image classification engine (3x3 tiles, 4x4 cells, single tiles).

Pipeline
--------
1. The challenge picture is cut into crops (:func:`features.model_crops`): 9 tiles for a
   3x3 grid, 16 cells + the whole photo for a 4x4 grid, the tile itself for size 1.
2. Every crop is embedded by the frozen CLIP backbone(s) named in the head file
   (shared with the AWS grid solver when it runs the same model).
3. Trained heads (``weights/recaptcha_head.pt``) turn embeddings into per-class
   probabilities:
     * ``tile``: multi-label MLP for 3x3 tiles and single tiles,
     * ``cell``: MLP for 4x4 cells over [cell, whole photo, mean of the 4 neighbours]
       followed by a small neighbour-context stage (a cell is positive when ANY part of
       the object overlaps it, which depends on the neighbours).
   Each class has its own threshold tuned on validation data.
4. Classes the head does not know (or every class, when the weight file is missing) are
   scored zero-shot: CLIP text prompts of the target contrasted with the other reCAPTCHA
   classes and common street-scene distractors.

Environment variables
---------------------
RECAPTCHA_HEAD       head path (default weights/recaptcha_head.pt; "none" = zero-shot only)
RECAPTCHA_BACKBONES  backbones used without a head (default ViT-B-32:laion2b_s34b_b79k@224)
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import threading
import time
from pathlib import Path

import numpy as np

from . import features
from .backbone import get_encoder
from .errors import RecaptchaError
from .questions import CLASSES, prompt_name

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_HEAD = ROOT / "weights" / "recaptcha_head.pt"
DEFAULT_BACKBONES = "ViT-B-32:laion2b_s34b_b79k@224"
ZS_THRESHOLD = float(os.environ.get("RECAPTCHA_ZS_THRESHOLD", "0.5"))
VOCAB_TEMPLATES = ("a photo of a {}.", "a street view photo of a {}.", "a low quality photo of a {}.")
RELATED_TEXT_COS = 0.9
# Zero-shot calibration (weights, bias) for ViT-B-32 laion2b features: logistic regression
# fitted on bxding 3x3 validation tiles of bicycle / bus / car / crosswalk / fire hydrant.
ZS_CALIBRATION = ((0.056, 0.197, 0.378, 0.144, -0.137), 0.0)

DISTRACTORS = ("road", "street", "tree", "building", "house", "sky", "grass", "sidewalk",
               "parking lot", "street sign", "pole", "fence", "person", "wall", "window",
               "bush", "field", "river", "highway", "power lines", "roof")
SYNONYMS: dict[str, tuple[str, ...]] = {
    "car": ("car", "cars on a street", "suv"),
    "bus": ("bus", "city bus", "school bus"),
    "school_bus": ("yellow school bus",),
    "motorcycle": ("motorcycle", "motorbike", "scooter"),
    "bicycle": ("bicycle", "bike"),
    "traffic_light": ("traffic light", "traffic signal"),
    "fire_hydrant": ("fire hydrant",),
    "crosswalk": ("crosswalk", "zebra crossing", "pedestrian crossing"),
    "parking_meter": ("parking meter",),
    "stairs": ("stairs", "staircase", "steps"),
    "chimney": ("chimney", "chimney on a roof"),
    "palm_tree": ("palm tree",),
    "mountain": ("mountain", "hill", "mountains or hills"),
    "boat": ("boat", "ship"),
}
TEMPLATES = ("a photo of a {}.", "a street view photo of a {}.", "a low quality photo of a {}.",
             "a close-up photo of a {}.", "a cropped photo of a {}.")
NEIGHBOURS4 = [[r * 4 + c for r, c in ((i // 4 - 1, i % 4), (i // 4 + 1, i % 4), (i // 4, i % 4 - 1),
                                        (i // 4, i % 4 + 1)) if 0 <= r < 4 and 0 <= c < 4] for i in range(16)]


def _imagenet_names() -> list[str]:
    try:
        from open_clip import IMAGENET_CLASSNAMES
    except ImportError:  # pragma: no cover
        return []
    return list(IMAGENET_CLASSNAMES)


def _cache_dir() -> Path:
    base = os.environ.get("AWS_CAPSOLVER_CACHE_DIR") or os.path.join(
        os.environ.get("XDG_CACHE_HOME") or os.path.join(os.path.expanduser("~"), ".cache"), "aws_capsolver")
    return Path(base)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40, 40)))


def mlp_forward(layers: list[tuple[np.ndarray, np.ndarray]], x: np.ndarray) -> np.ndarray:
    """Plain numpy MLP: ReLU between layers, raw logits out."""
    for i, (W, b) in enumerate(layers):
        x = x @ W.T + b
        if i < len(layers) - 1:
            x = np.maximum(x, 0.0)
    return x


def cell_inputs(E: np.ndarray) -> np.ndarray:
    """4x4 cell features from crop embeddings E (n, 17, d): (n, 16, 3d)."""
    cells, whole = E[:, :16], E[:, 16:17]
    neigh = np.stack([cells[:, nb].mean(axis=1) for nb in NEIGHBOURS4], axis=1)
    return np.concatenate([cells, np.repeat(whole, 16, axis=1), neigh], axis=2)


def context_inputs(L: np.ndarray) -> np.ndarray:
    """Second-stage 4x4 features from first-stage logits L (n, 16) of ONE class: (n, 16, 7)."""
    n = L.shape[0]
    nb_max = np.stack([L[:, nb].max(axis=1) for nb in NEIGHBOURS4], axis=1)
    nb_mean = np.stack([L[:, nb].mean(axis=1) for nb in NEIGHBOURS4], axis=1)
    g_max = np.repeat(L.max(axis=1, keepdims=True), 16, axis=1)
    g_mean = np.repeat(L.mean(axis=1, keepdims=True), 16, axis=1)
    rows = np.tile(np.repeat(np.arange(4), 4), (n, 1)) / 3.0
    cols = np.tile(np.tile(np.arange(4), 4), (n, 1)) / 3.0
    return np.stack([L, nb_max, nb_mean, g_max, g_mean, rows, cols], axis=2)


class Engine:
    """Backbones + heads. Use :func:`get_engine` for the shared, lazily loaded instance.

    Three feature spaces may be in use: ``tile_specs`` (input of the tile head, usually the
    fine-tuned ``+tail`` backbone), ``cell_specs`` (input of the 4x4 cell head, frozen
    CLIP) and ``zs_specs`` (frozen CLIP, text-aligned, for zero-shot targets).
    """

    def __init__(self, head_path: str | Path | None = None, load_backbones: bool = True) -> None:
        if head_path is None:
            head_path = os.environ.get("RECAPTCHA_HEAD", str(DEFAULT_HEAD))
        self.head: dict | None = None
        if str(head_path).strip().lower() not in ("", "none", "off", "0"):
            self.head = self._load_head(Path(head_path))
        default = [s.strip() for s in os.environ.get("RECAPTCHA_BACKBONES", DEFAULT_BACKBONES).split(",") if s.strip()]
        self.tile_specs = self.head["backbones"] if self.head else default
        self.cell_specs = (self.head["cell_backbones"] or self.tile_specs) if self.head else default
        self.specs = self.tile_specs
        self._text_cache: dict[tuple[str, str], np.ndarray] = {}
        self._vocab_cache: dict[str, tuple[list[str], np.ndarray]] = {}
        self._lock = threading.Lock()
        self.encoders: dict[str, object] = {}
        if load_backbones:
            try:
                for s in dict.fromkeys(self.tile_specs + self.cell_specs):
                    self.encoders[s] = get_encoder(s)
            except RecaptchaError as exc:
                if self.head is None:
                    raise
                log.warning("recaptcha: %s; using zero-shot CLIP only", exc)
                self.head = None
                self.tile_specs = self.cell_specs = self.specs = default
            self.zs_specs = list(dict.fromkeys(s.split("+", 1)[0] for s in self.tile_specs + self.cell_specs))
            for s in self.zs_specs:
                if s not in self.encoders:
                    self.encoders[s] = get_encoder(s)
        log.info("recaptcha engine ready: tile=%s cell=%s head=%s", self.tile_specs, self.cell_specs,
                 self.head["trained_on"] if self.head else "none (zero-shot only)")

    # ------------------------------------------------------------------ loading
    @staticmethod
    def _load_head(path: Path) -> dict | None:
        if not path.is_file():
            log.warning("recaptcha head %s is missing; using zero-shot CLIP only", path)
            return None
        try:
            import torch

            raw = torch.load(path, map_location="cpu", weights_only=True)

            def layers(spec):
                return [(np.asarray(W, np.float32), np.asarray(b, np.float32)) for W, b in spec]

            head = {
                "backbones": [str(s) for s in raw["backbones"]],
                "cell_backbones": [str(s) for s in raw.get("cell_backbones") or []],
                "classes": [str(c) for c in raw["classes"]],
                "tile": layers(raw["tile"]),
                "tile_thr": {str(k): float(v) for k, v in dict(raw["tile_thr"]).items()},
                "cell": layers(raw["cell"]) if raw.get("cell") else None,
                "cell_ctx": layers(raw["cell_ctx"]) if raw.get("cell_ctx") else None,
                "cell_classes": [str(c) for c in raw.get("cell_classes", [])],
                "cell_thr": {str(k): float(v) for k, v in dict(raw.get("cell_thr", {})).items()},
                "trained_on": str(raw.get("trained_on", "unknown")),
            }
            k = len(head["classes"])
            if head["tile"][-1][0].shape[0] != k:
                raise ValueError("tile head output does not match the class list")
            if head["cell"] is not None and head["cell"][-1][0].shape[0] != len(head["cell_classes"]):
                raise ValueError("cell head output does not match the cell class list")
            return head
        except Exception as exc:  # noqa: BLE001 - a broken file must not take the service down
            log.warning("cannot load recaptcha head %s (%s); using zero-shot CLIP only", path, exc)
            return None

    # ---------------------------------------------------------------- embedding
    def embed(self, crops: list, specs: list[str]) -> np.ndarray:
        """(n, sum d) concatenated L2-normalised embeddings of ``crops`` for ``specs``."""
        return np.concatenate([self.encoders[s](features.to_tensor(crops, self.encoders[s].res)) for s in specs],
                              axis=1)

    # ---------------------------------------------------------------- zero-shot
    def _text(self, enc, name: str) -> np.ndarray:
        key = (enc.spec, name)
        with self._lock:
            v = self._text_cache.get(key)
        if v is None:
            words = SYNONYMS.get(name, (prompt_name(name),))
            t = enc.encode_text([tp.format(w) for w in words for tp in TEMPLATES]).mean(0)
            v = t / np.linalg.norm(t)
            with self._lock:
                if len(self._text_cache) > 512:  # bounded: targets may come from untrusted input
                    self._text_cache.clear()
                self._text_cache[key] = v
        return v

    def _vocab(self, enc) -> tuple[list[str], np.ndarray]:
        """Open vocabulary (reCAPTCHA classes, street-scene distractors, ImageNet-1k names)
        and its text embeddings for ``enc``; encoded once and cached on disk."""
        with self._lock:
            hit = self._vocab_cache.get(enc.spec)
        if hit is not None:
            return hit
        names: list[str] = []
        for n in (*(prompt_name(c) for c in CLASSES), *DISTRACTORS, *_imagenet_names()):
            n = n.split(" (")[0].strip().lower()
            if n and n not in names:
                names.append(n)
        key = hashlib.sha1(json.dumps([enc.spec.split("+")[0], names, VOCAB_TEMPLATES]).encode()).hexdigest()[:16]
        path = _cache_dir() / f"recaptcha_vocab_{key}.npy"
        V = None
        if path.is_file():
            try:
                V = np.load(path, allow_pickle=False)
                if V.shape[0] != len(names):
                    V = None
            except (OSError, ValueError):
                V = None
        if V is None:
            t0 = time.time()
            out = []
            for k in range(0, len(names), 64):
                chunk = names[k:k + 64]
                t = enc.encode_text([tp.format(n) for n in chunk for tp in VOCAB_TEMPLATES])
                t = t.reshape(len(chunk), len(VOCAB_TEMPLATES), -1).mean(1)
                out.append(t / np.linalg.norm(t, axis=1, keepdims=True))
            V = np.concatenate(out).astype(np.float32)
            log.info("recaptcha: encoded %d vocabulary names in %.1fs", len(names), time.time() - t0)
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
                with open(tmp, "wb") as fh:
                    np.save(fh, V)
                os.replace(tmp, path)
            except OSError as exc:
                log.warning("cannot write vocabulary cache %s: %s", path, exc)
        with self._lock:
            self._vocab_cache[enc.spec] = (names, V)
        return names, V

    def zero_shot_features(self, F: np.ndarray, enc, target: str) -> np.ndarray:
        """P(target | crop) for frozen-CLIP features F: softmax over the target prompt
        ensemble and an open vocabulary of ~1000 names minus those naming the target."""
        tvec = self._text(enc, target)
        names, V = self._vocab(enc)
        words = {prompt_name(target), *SYNONYMS.get(target, ())}
        if target == "school_bus":
            words |= {"bus", "school bus"}
        pats = [re.compile(rf"\b{re.escape(w)}(e?s)?\b") for w in words]
        related = np.array([any(p.search(n) for p in pats) for n in names]) | (V @ tvec > RELATED_TEXT_COS)
        C = np.vstack([tvec[None], V[~related]])
        z = 100.0 * F.astype(np.float32) @ C.T
        z -= z.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e[:, 0] / e.sum(axis=1)

    def zero_shot_scores(self, F: np.ndarray, enc, target: str) -> np.ndarray:
        """Class-agnostic calibrated zero-shot P(target | crop) for frozen-CLIP features F.

        Logistic regression over five similarity features (raw target similarity, margins
        against street-scene distractors / the other reCAPTCHA classes / both, open-vocab
        softmax). Fitted on the five big reCAPTCHA classes; leave-one-class-out it keeps
        84-96 % per-tile accuracy on the held-out class (see README).
        """
        def lse(z):
            m = z.max(axis=1, keepdims=True)
            return (m + np.log(np.exp(z - m).sum(axis=1, keepdims=True)))[:, 0]

        F = F.astype(np.float32)
        t = self._text(enc, target)
        sim = 100.0 * F @ t
        dist = 100.0 * F @ np.stack([self._text(enc, d) for d in DISTRACTORS]).T
        others = [c for c in CLASSES if c != target and prompt_name(c) != prompt_name(target)
                  and {c, target} != {"bus", "school_bus"}]
        oth = 100.0 * F @ np.stack([self._text(enc, o) for o in others]).T
        po = np.log(self.zero_shot_features(F, enc, target) + 1e-6)
        X = np.stack([sim, sim - lse(dist), sim - lse(oth), po, sim - lse(np.concatenate([dist, oth], 1))], 1)
        w, b = ZS_CALIBRATION
        return _sigmoid(X @ np.asarray(w, np.float32) + b)

    def zero_shot(self, crops: list, target: str) -> np.ndarray:
        """Zero-shot P(target | crop), averaged over the frozen backbones."""
        probs = []
        for s in self.zs_specs:
            enc = self.encoders[s]
            probs.append(self.zero_shot_scores(enc(features.to_tensor(crops, enc.res)), enc, target))
        return np.mean(probs, axis=0)

    def warm_text(self) -> None:
        for s in self.zs_specs:
            self._vocab(self.encoders[s])
            for c in CLASSES:
                self._text(self.encoders[s], c)

    # ------------------------------------------------------------------ scoring
    def knows_tile(self, target: str) -> bool:
        return self.head is not None and target in self.head["classes"]

    def knows_cell(self, target: str) -> bool:
        return self.head is not None and self.head["cell"] is not None and target in self.head["cell_classes"]

    def tile_probs(self, X: np.ndarray, target: str) -> tuple[np.ndarray, float, str]:
        """X: tile-head features (n, D) of a class the tile head knows."""
        h = self.head
        logits = mlp_forward(h["tile"], X.astype(np.float32))
        return _sigmoid(logits[:, h["classes"].index(target)]), h["tile_thr"].get(target, 0.5), "head"

    def cell_probs(self, X: np.ndarray, target: str) -> tuple[np.ndarray, float, str]:
        """X: (17, D) cell-head features of one 4x4 challenge -> P(target overlaps cell) (16,)."""
        h = self.head
        k = h["cell_classes"].index(target)
        L = mlp_forward(h["cell"], cell_inputs(X[None].astype(np.float32)).reshape(16, -1))[:, k][None]
        if h["cell_ctx"] is not None:
            C = context_inputs(L).reshape(16, -1)
            onehot = np.zeros((16, len(h["cell_classes"])), np.float32)
            onehot[:, k] = 1.0
            L = mlp_forward(h["cell_ctx"], np.concatenate([C, onehot], axis=1))[:, 0][None]
        return _sigmoid(L[0]), h["cell_thr"].get(target, 0.5), "head"

    def classify(self, img, target: str, size: int) -> dict:
        crops = features.model_crops(img, size)
        if size == 4 and self.knows_cell(target):
            p, thr, method = self.cell_probs(self.embed(crops, self.cell_specs), target)
        else:
            if size == 4:  # no 4x4 head for this class: score the 16 cells as tiles
                crops = crops[:16]
            if self.knows_tile(target):
                p, thr, method = self.tile_probs(self.embed(crops, self.tile_specs), target)
            else:
                p, thr, method = self.zero_shot(crops, target), ZS_THRESHOLD, "zeroshot"
        sel = [int(i) for i in np.flatnonzero(p >= thr)]
        return {"probs": p, "threshold": thr, "method": method, "selected": sel}


_engine: Engine | None = None
_engine_lock = threading.Lock()


def get_engine() -> Engine:
    global _engine
    if _engine is None:
        with _engine_lock:
            if _engine is None:
                t0 = time.time()
                try:
                    _engine = Engine()
                except RecaptchaError:
                    raise
                except Exception as exc:  # noqa: BLE001
                    raise RecaptchaError(f"cannot load the reCAPTCHA model: {exc}") from exc
                log.info("recaptcha engine loaded in %.1fs", time.time() - t0)
    return _engine
