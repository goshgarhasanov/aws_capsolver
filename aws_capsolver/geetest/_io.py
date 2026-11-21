"""Input decoding / validation shared by the GeeTest solvers."""
from __future__ import annotations

import base64
import binascii
import io
import re
from typing import Union

from .errors import InvalidGeeTestInput

ImageInput = Union[bytes, bytearray, memoryview, str]

MAX_IMAGE_BYTES = 4 * 1024 * 1024   # encoded size limit per image
MAX_IMAGE_SIDE = 2048               # decoded width / height limit
MIN_IMAGE_SIDE = 8
_DATA_URL = re.compile(r"^data:[\w/+.-]+;base64,", re.I)


def to_bytes(value: object, what: str = "image") -> bytes:
    """Accept raw bytes or a base64 string (optionally a ``data:...;base64,`` URL)."""
    if isinstance(value, (bytes, bytearray, memoryview)):
        data = bytes(value)
    elif isinstance(value, str):
        s = _DATA_URL.sub("", value.strip())
        s = "".join(s.split())
        if not s:
            raise InvalidGeeTestInput(f"{what}: empty string")
        if len(s) > MAX_IMAGE_BYTES * 4 // 3 + 8:
            raise InvalidGeeTestInput(f"{what}: larger than {MAX_IMAGE_BYTES} bytes")
        s = s.replace("-", "+").replace("_", "/")
        s += "=" * (-len(s) % 4)
        try:
            data = base64.b64decode(s, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise InvalidGeeTestInput(f"{what}: not valid base64") from exc
    else:
        raise InvalidGeeTestInput(f"{what}: expected bytes or a base64 string, got {type(value).__name__}")
    if not data:
        raise InvalidGeeTestInput(f"{what}: empty")
    if len(data) > MAX_IMAGE_BYTES:
        raise InvalidGeeTestInput(f"{what}: larger than {MAX_IMAGE_BYTES} bytes")
    return data


def load_image(value: object, what: str = "image", mode: str = "RGB"):
    """Decode and validate an image; returns a PIL image converted to ``mode``.

    ``mode="RGBA"`` keeps transparency (puzzle pieces / hint icons). Images with a palette
    or LA mode are converted so transparency survives.
    """
    from PIL import Image, UnidentifiedImageError

    data = to_bytes(value, what)
    try:
        im = Image.open(io.BytesIO(data))
        w, h = im.size
        if w > MAX_IMAGE_SIDE or h > MAX_IMAGE_SIDE:
            raise InvalidGeeTestInput(f"{what}: {w}x{h} exceeds {MAX_IMAGE_SIDE}px")
        if w < MIN_IMAGE_SIDE or h < MIN_IMAGE_SIDE:
            raise InvalidGeeTestInput(f"{what}: {w}x{h} is too small")
        im.load()
    except InvalidGeeTestInput:
        raise
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError, SyntaxError) as exc:
        raise InvalidGeeTestInput(f"{what}: not a decodable image ({exc.__class__.__name__})") from exc
    if getattr(im, "n_frames", 1) > 1:
        im.seek(0)
    if mode == "RGBA":
        if im.mode == "P" and "transparency" in im.info:
            im = im.convert("RGBA")
        return im.convert("RGBA")
    if im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info):
        # flatten on white, like a browser would on a white page
        rgba = im.convert("RGBA")
        bg = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        return Image.alpha_composite(bg, rgba).convert(mode)
    return im.convert(mode)
