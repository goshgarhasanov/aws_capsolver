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


def classify_grid(images: list[bytes], target: str, threshold: float | None = None, *,
                  count: int | None = None) -> list[int]:
    """Return the 0-based, row-major indices of the tiles that contain ``target``.

    By default the 5 most likely tiles are returned (``gridcaptcha-5-*`` grids always have
    5 target tiles); pass ``count`` (e.g. from ``count_from_problem_type``) or an explicit
    ``threshold`` to change that.
    """
    from .grid import classify_grid as _impl

    return _impl(images, target, threshold, count=count)


def count_from_problem_type(problem_type: str | None) -> int | None:
    """'gridcaptcha-5-0.1-0' -> 5 (number of tiles to select); None if unknown."""
    if not problem_type or not problem_type.startswith("gridcaptcha-"):
        return None
    part = problem_type.split("-")[1]
    return int(part) if part.isdigit() else None


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


__all__ = ["classify_grid", "count_from_problem_type", "solve_toycar", "transcribe", "UnsupportedProblem",
           "VisionError", "InvalidInput"]
