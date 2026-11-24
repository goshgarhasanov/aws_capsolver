"""Frozen CLIP image encoders for the reCAPTCHA heads.

The same open_clip models as the AWS grid solver (:mod:`aws_capsolver.vision.grid`) are
used, and when that engine is already loaded in this process its model objects are
shared instead of loaded a second time. reCAPTCHA tiles are only 100-112 px wide, so an
encoder may run at a lower input resolution than it was trained at (e.g. ViT-B-16 at
112 px = 7x7 patches, ~4x faster than 224 px): the positional embedding is resampled
exactly like open_clip's ``force_image_size`` does, and every other weight is shared.

A backbone spec is ``"<model>:<pretrained>@<resolution>"``, e.g.
``"ViT-B-16:datacomp_xl_s13b_b90k@112"``. A ``"+tail"`` suffix selects the reCAPTCHA
fine-tuned variant: the shared frozen model runs blocks [0, K) and a fine-tuned copy of
the last blocks (``weights/recaptcha_tail.pt``, see tools/recaptcha_finetune.py) the rest.
"""
from __future__ import annotations

import copy
import logging
import os
import threading
import time
from pathlib import Path

import numpy as np

from .errors import RecaptchaError

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
TAIL_PATH = Path(os.environ.get("RECAPTCHA_TAIL", str(ROOT / "weights" / "recaptcha_tail.pt")))

_models: dict[str, object] = {}  # "model:pretrained" -> open_clip model (full CLIP)
_variants: dict[str, "Encoder"] = {}
_lock = threading.RLock()


def parse_spec(spec: str) -> tuple[str, str, int]:
    base, _, res = spec.split("+", 1)[0].partition("@")
    name, _, pretrained = base.partition(":")
    if not name or not pretrained:
        raise RecaptchaError(f"bad backbone spec {spec!r}")
    return name, pretrained, int(res or 224)


def _threads() -> int:
    env = os.environ.get("AWS_CAPSOLVER_TORCH_THREADS", "").strip()
    return int(env) if env.isdigit() and int(env) > 0 else min(8, os.cpu_count() or 1)


def _shared_from_grid_engine(base: str):
    """The model object of an already-loaded AWS grid engine, if it runs ``base``."""
    try:
        from aws_capsolver.vision import grid
    except ImportError:  # pragma: no cover
        return None
    eng = getattr(grid, "_engine", None)
    if eng is None:
        configured = [s.strip() for s in os.environ.get("AWS_CAPSOLVER_CLIP_MODELS", grid.DEFAULT_MODELS).split(",")]
        if base not in configured or os.environ.get("RECAPTCHA_SHARE_GRID", "1") == "0":
            return None
        try:  # load the grid engine once and share it (one copy of each backbone per process)
            eng = grid.get_engine()
        except Exception as exc:  # noqa: BLE001 - fall back to a private copy
            log.warning("recaptcha: cannot share the grid engine (%s); loading %s privately", exc, base)
            return None
    for bb in eng.backbones:
        if bb.spec == base:
            return bb.model
    return None


def _load_clip(name: str, pretrained: str):
    base = f"{name}:{pretrained}"
    with _lock:
        m = _models.get(base)
        if m is not None:
            return m
        m = _shared_from_grid_engine(base)
        if m is None:
            try:
                import open_clip
                import torch
            except ImportError as exc:
                raise RecaptchaError(f"torch / open_clip not installed: {exc}") from exc
            if torch.get_num_threads() > _threads():
                torch.set_num_threads(_threads())
            t0 = time.time()
            try:
                m, _, _ = open_clip.create_model_and_transforms(name, pretrained=pretrained)
            except (RuntimeError, OSError, ValueError, ImportError) as exc:
                raise RecaptchaError(f"cannot load backbone {base!r}: {exc}") from exc
            m.eval()
            log.info("recaptcha: loaded backbone %s in %.1fs", base, time.time() - t0)
        else:
            log.info("recaptcha: sharing backbone %s with the grid engine", base)
        _models[base] = m
        return m


def _resized_visual(visual, res: int):
    """Shallow copy of an open_clip VisionTransformer that runs at ``res`` px input."""
    import torch
    import torch.nn.functional as F

    patch = visual.conv1.kernel_size[0]
    grid = res // patch
    if tuple(visual.grid_size) == (grid, grid):
        return visual
    pos = visual.positional_embedding.detach()
    tok, img = pos[:1], pos[1:]
    old = int(round(len(img) ** 0.5))
    img = img.reshape(1, old, old, -1).permute(0, 3, 1, 2)
    img = F.interpolate(img, size=(grid, grid), mode="bicubic", antialias=True, align_corners=False)
    img = img.permute(0, 2, 3, 1).reshape(grid * grid, -1)
    v = copy.copy(visual)  # shares every submodule / weight tensor
    v._parameters = dict(visual._parameters)
    v.positional_embedding = torch.nn.Parameter(torch.cat([tok, img]), requires_grad=False)
    v.grid_size = (grid, grid)
    v.image_size = (res, res)
    return v


class Encoder:
    """Callable image encoder: tensor (n, 3, res, res) -> L2-normalised float32 (n, d)."""

    def __init__(self, spec: str) -> None:
        name, pretrained, res = parse_spec(spec)
        self.spec = spec
        self.res = res
        self.clip = _load_clip(name, pretrained)
        self.visual = _resized_visual(self.clip.visual, res)
        self.lock = threading.Lock()
        self.tail = _load_tail(self.visual, spec) if spec.endswith("+tail") else None

    def __call__(self, x) -> np.ndarray:
        import torch

        with self.lock, torch.inference_mode():
            if self.tail is None:
                f = self.visual(x).float()
            else:
                K, blocks, ln, proj = self.tail
                h = self.visual._embeds(x)
                for blk in self.visual.transformer.resblocks[:K]:
                    h = blk(h)
                for blk in blocks:
                    h = blk(h)
                f = (ln(h[:, 0]) @ proj).float()
        f = f / f.norm(dim=-1, keepdim=True)
        return f.numpy()

    def encode_text(self, prompts: list[str]) -> np.ndarray:
        import open_clip
        import torch

        name, _, _ = parse_spec(self.spec)
        tok = open_clip.get_tokenizer(name)
        with self.lock, torch.inference_mode():
            t = self.clip.encode_text(tok(prompts)).float()
        t = t / t.norm(dim=-1, keepdim=True)
        return t.numpy()


def _load_tail(visual, spec: str):
    """(K, blocks, ln_post, proj) of the fine-tuned tail; RecaptchaError when unusable."""
    import torch

    if not TAIL_PATH.is_file():
        raise RecaptchaError(f"fine-tuned tail {TAIL_PATH} is missing")
    try:
        ck = torch.load(TAIL_PATH, map_location="cpu", weights_only=True)
        base = spec[:-len("+tail")]
        if ck["base"] != base:
            raise ValueError(f"tail was trained for {ck['base']}, not {base}")
        K = int(ck["K"])
        st = ck["state"]
        blocks = copy.deepcopy(visual.transformer.resblocks[K:])
        blocks.load_state_dict({k[7:]: v.float() for k, v in st.items() if k.startswith("blocks.")})
        ln = copy.deepcopy(visual.ln_post)
        ln.load_state_dict({k[3:]: v.float() for k, v in st.items() if k.startswith("ln.")})
        proj = st["proj"].float()
    except Exception as exc:  # noqa: BLE001
        raise RecaptchaError(f"cannot load fine-tuned tail {TAIL_PATH}: {exc}") from exc
    blocks.eval()
    ln.eval()
    return K, blocks, ln, proj


def get_encoder(spec: str) -> Encoder:
    with _lock:
        enc = _variants.get(spec)
        if enc is None:
            enc = _variants[spec] = Encoder(spec)
        return enc
