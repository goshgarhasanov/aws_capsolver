"""Image decoding, grid splitting and CLIP preprocessing shared by inference and training.

Training (``tools/recaptcha_embed.py``) and :mod:`.model` must cut and preprocess the
challenge exactly the same way, so everything that touches pixels lives here.
"""
from __future__ import annotations

import base64
import binascii
import io

from .errors import InvalidRecaptchaInput

MAX_IMAGE_BYTES = 4 * 1024 * 1024
MAX_SIDE = 1200
MIN_SIDE = 40

# open_clip OPENAI_DATASET_MEAN / STD (all supported backbones use them).
MEAN = (0.48145466, 0.4578275, 0.40821073)
STD = (0.26862954, 0.26130258, 0.27577711)


def image_bytes(image: object) -> bytes:
    """Accept raw bytes or a base64 string (optionally a ``data:`` URL)."""
    if isinstance(image, (bytes, bytearray, memoryview)):
        data = bytes(image)
    elif isinstance(image, str):
        s = image.strip()
        if s.startswith("data:"):
            s = s.partition(",")[2]
        s = "".join(s.split())
        try:
            data = base64.b64decode(s + "=" * (-len(s) % 4), validate=True)
        except (binascii.Error, ValueError) as exc:
            raise InvalidRecaptchaInput("image is not valid base64") from exc
    else:
        raise InvalidRecaptchaInput(f"image must be bytes or a base64 string, got {type(image).__name__}")
    if not data:
        raise InvalidRecaptchaInput("image is empty")
    if len(data) > MAX_IMAGE_BYTES:
        raise InvalidRecaptchaInput(f"image is {len(data)} bytes, limit is {MAX_IMAGE_BYTES}")
    return data


def decode(image: object):
    """Decode to an RGB PIL image, validating its size."""
    from PIL import Image, UnidentifiedImageError

    data = image_bytes(image)
    try:
        img = Image.open(io.BytesIO(data))
        w, h = img.size
        if max(w, h) > MAX_SIDE or min(w, h) < MIN_SIDE:
            raise InvalidRecaptchaInput(f"implausible image size {w}x{h}")
        if max(w, h) > 1.6 * min(w, h):
            raise InvalidRecaptchaInput(f"image is not roughly square: {w}x{h}")
        img.load()
        return img.convert("RGB")
    except InvalidRecaptchaInput:
        raise
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError, ValueError) as exc:
        raise InvalidRecaptchaInput(f"cannot decode image: {exc}") from exc


def seam_score(img, n: int) -> float:
    """How strongly the picture looks like n x n separate photos.

    Ratio of the mean horizontal/vertical pixel jump across the n-grid lines to the
    median jump elsewhere. A 3x3 challenge (9 unrelated photos) scores >> 1 for n=3;
    a 4x4 challenge (one photo) scores ~1.
    """
    import numpy as np

    a = np.asarray(img.convert("L"), dtype=np.float32)
    h, w = a.shape
    dx = np.abs(np.diff(a, axis=1)).mean(axis=0)  # (w-1,)
    dy = np.abs(np.diff(a, axis=0)).mean(axis=1)  # (h-1,)
    xs = [round(w * k / n) - 1 for k in range(1, n)]
    ys = [round(h * k / n) - 1 for k in range(1, n)]

    def peak(d, idx):
        vals = [max(d[max(0, i - 1):i + 2]) for i in idx if 0 <= i < len(d)]
        return float(np.mean(vals)) if vals else 0.0

    base = float(np.median(np.concatenate([dx, dy]))) + 1e-3
    return (peak(dx, xs) + peak(dy, ys)) / 2.0 / base


def infer_size(img) -> int:
    """1 for a single tile, 3 or 4 for a grid (see module docs of :mod:`aws_capsolver.recaptcha`)."""
    w, h = img.size
    side = min(w, h)
    if side < 170:
        return 1
    s3 = seam_score(img, 3)
    if side >= 400:
        # 4x4 is one photo; a big 3x3 still shows strong seams at the thirds.
        return 3 if s3 > 2.5 and s3 > 1.5 * seam_score(img, 4) else 4
    return 3


def split_grid(img, n: int) -> list:
    """Cut into n*n cells, row-major. Cell borders are rounded so every pixel is used once."""
    w, h = img.size
    xs = [round(w * k / n) for k in range(n + 1)]
    ys = [round(h * k / n) for k in range(n + 1)]
    return [img.crop((xs[c], ys[r], xs[c + 1], ys[r + 1])) for r in range(n) for c in range(n)]


def model_crops(img, size: int) -> list:
    """The crops the model embeds: 1 -> [tile]; 3 -> 9 tiles; 4 -> 16 cells + whole photo."""
    if size == 1:
        return [img]
    if size == 3:
        return split_grid(img, 3)
    if size == 4:
        return split_grid(img, 4) + [img]
    raise InvalidRecaptchaInput(f"size must be 1, 3 or 4, got {size!r}")


def to_tensor(pils: list, res: int):
    """Bicubic resize to res x res and CLIP-normalise: (n, 3, res, res) float tensor."""
    import numpy as np
    import torch
    from PIL import Image

    arr = np.stack([np.asarray(p.resize((res, res), Image.BICUBIC), dtype=np.float32) for p in pils])
    x = torch.from_numpy(arr).permute(0, 3, 1, 2).div_(255.0)
    mean = torch.tensor(MEAN).view(1, 3, 1, 1)
    std = torch.tensor(STD).view(1, 3, 1, 1)
    return (x - mean) / std
