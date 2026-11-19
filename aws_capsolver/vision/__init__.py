"""Vision solvers for AWS WAF CAPTCHA problems.

Public interface (imported by the HTTP service):
    classify_grid(images, target, threshold=None) -> list[int]
    solve_toycar(image) -> tuple[int, int]
    transcribe(audio_b64) -> str
    UnsupportedProblem

Heavy dependencies (torch / open_clip) are imported only when a solver is first called.
"""
from __future__ import annotations

from .errors import InvalidInput, UnsupportedProblem, VisionError


def classify_grid(images: list[bytes], target: str, threshold: float | None = None) -> list[int]:
    """Return the 0-based, row-major indices of the tiles that contain ``target``."""
    from .grid import classify_grid as _impl

    return _impl(images, target, threshold)


def solve_toycar(image: bytes) -> tuple[int, int]:
    """Return the [x, y] end point of the car path ("toycarcity" problems)."""
    from .toycar import solve_toycar as _impl

    return _impl(image)


def transcribe(audio_b64: str) -> str:
    """Transcribe a "noisyneighbour" audio CAPTCHA (base64 AAC) to text."""
    try:
        from .audio import transcribe as _impl
    except ImportError as exc:
        raise UnsupportedProblem(f"audio CAPTCHA solver is not available: {exc}") from exc
    return _impl(audio_b64)


__all__ = ["classify_grid", "solve_toycar", "transcribe", "UnsupportedProblem",
           "VisionError", "InvalidInput"]
