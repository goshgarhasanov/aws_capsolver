"""GeeTest v3 / v4 CAPTCHA solvers (models only; the HTTP task types live in the service).

Public interface:
    slide_offset(background, piece=None, full_background=None) -> int
    slide_details(background, piece=None, full_background=None, y=None) -> SlideResult
    click_points(image, kind, hints=None) -> list[(x, y)]
    nine_grid(image, target) -> list[int]
    gobang(board) -> ((row, col), (row, col))
    warmup(background=True)
    GeeTestError, UnsupportedGeeTest, InvalidGeeTestInput

Every image argument accepts raw bytes or a base64 string (a ``data:`` URL prefix is
fine). All functions are deterministic and thread-safe. Heavy dependencies (numpy,
OpenCV, torch, open_clip) are imported only when a solver is first called, so importing
this package is cheap and works without them.
"""
from __future__ import annotations

import threading
from typing import Sequence

from .errors import GeeTestError, InvalidGeeTestInput, UnsupportedGeeTest

CLICK_KINDS = ("icon", "word", "space")


def _deps(what: str):
    """Import numpy + OpenCV or raise UnsupportedGeeTest naming the missing dependency."""
    try:
        import cv2  # noqa: F401
        import numpy  # noqa: F401
    except ImportError as exc:  # pragma: no cover - depends on the install
        raise UnsupportedGeeTest(f"{what} needs numpy and opencv-python-headless: {exc}") from exc


def slide_offset(background: bytes | str, piece: bytes | str | None = None,
                 full_background: bytes | str | None = None, y: int | None = None) -> int:
    """Return the x (in pixels of ``background``) where the piece's left edge must end up.

    * v4 slide: ``background`` = ``bg`` PNG (with the real gap and decoy gaps),
      ``piece`` = ``slice`` PNG with alpha.
    * v3 slide: ``background`` = ``bg`` (raw shuffled 312x160 or the reassembled 260x160
      canvas), ``piece`` = ``slice`` PNG, ``full_background`` = ``fullbg`` (optional; makes
      the answer exact).

    With a piece, the value is the position of the piece IMAGE's left edge (the widget
    draws the piece image from x=0, so this is the drag distance in background pixels);
    without a piece it is the left edge of the visible gap. Scale it by
    ``displayed_width / background_width`` if the widget shows the picture at another size.

    ``y`` (optional) is the top of the piece image in background pixels (``ypos`` of the
    v3 ``get.php`` / v4 ``load`` answer, or the piece element's CSS top scaled to the
    picture). v4 draws decoy gaps; with ``y`` they are ruled out by row (held-out v4
    accuracy 110/110 with y vs 107/110 without).
    """
    return slide_details(background, piece, full_background, y).x


def slide_details(background: bytes | str, piece: bytes | str | None = None,
                  full_background: bytes | str | None = None, y: int | None = None):
    """Like :func:`slide_offset` but returns the full ``SlideResult``
    (x, gap_x, y, score, method). ``y`` = known top of the piece image (v3/v4 ``ypos``)
    restricts the search to that row band."""
    if background is None:
        raise InvalidGeeTestInput("background is required")
    if y is not None and (not isinstance(y, int) or isinstance(y, bool) or y < 0):
        raise InvalidGeeTestInput("y must be a non-negative int")
    _deps("slide")
    from .slide import solve

    return solve(background, piece, full_background, y=y)


def click_points(image: bytes | str, kind: str, hints: Sequence | None = None) -> list[tuple[int, int]]:
    """Return the points to click, in ``image`` pixel coordinates, in the required order.

    * ``kind="icon"`` (v3 / v4 icon click): ``hints`` = the small hint icon images (bytes or
      base64) in click order. For a v3 picture (344x384 with the hint strip at the bottom)
      ``hints`` may be omitted: they are cut from the strip.
    * ``kind="word"`` (v3 word click): ``hints`` = the characters (str) in click order.
    * ``kind="space"``: spatial-reasoning click (not supported).
    """
    if not isinstance(kind, str) or kind.strip().lower() not in CLICK_KINDS:
        raise InvalidGeeTestInput(f"kind must be one of {CLICK_KINDS}")
    kind = kind.strip().lower()
    if hints is not None and not isinstance(hints, (list, tuple)):
        raise InvalidGeeTestInput("hints must be a list")
    if hints is not None and len(hints) > 8:
        raise InvalidGeeTestInput("too many hints (max 8)")
    if image is None:
        raise InvalidGeeTestInput("image is required")
    _deps("click")
    from .click import click_points as _impl

    return _impl(image, kind, list(hints) if hints is not None else None)


def nine_grid(image: bytes | str, target: str | bytes, count: int | None = None) -> list[int]:
    """v4 "nine": 0-based row-major indices of the 3x3 cells that match ``target``.

    ``target`` is the prompt: a text label (e.g. "fish") or the prompt icon image (bytes or
    base64). ``count`` = number of cells to select (v4 ``nine_nums``; default 3)."""
    if image is None or target is None:
        raise InvalidGeeTestInput("image and target are required")
    if count is not None and (not isinstance(count, int) or isinstance(count, bool) or not 1 <= count <= 9):
        raise InvalidGeeTestInput("count must be an int in 1..9")
    _deps("nine")
    from .nine import nine_grid as _impl

    return _impl(image, target, count)


def gobang(board) -> tuple[tuple[int, int], tuple[int, int]]:
    """v4 "winlinze" (gobang): move one piece so that a full line of one kind appears.

    ``board`` is the board screenshot (bytes / base64) or the ``ques`` matrix of the v4
    ``load`` response (list of lists of ints, 0 = empty). Returns (from_cell, to_cell) as
    0-based (row, col)."""
    if board is None:
        raise InvalidGeeTestInput("board is required")
    from .board import solve_board

    return solve_board(board)


_warm_lock = threading.Lock()


def warmup(background: bool = True) -> threading.Thread | None:
    """Load the models (CLIP for nine/icon) and run one dummy inference of each solver.

    With ``background=True`` this returns the started daemon thread immediately; calls
    arriving meanwhile wait for the shared lazy singletons."""
    def _run() -> None:
        import logging

        log = logging.getLogger(__name__)
        with _warm_lock:
            try:
                from . import warm
                warm.run()
                log.info("geetest warm-up done")
            except Exception:  # noqa: BLE001 - a failed warm-up must not kill the thread silently
                log.exception("geetest warm-up failed")

    if not background:
        _run()
        return None
    th = threading.Thread(target=_run, name="geetest-warmup", daemon=True)
    th.start()
    return th


__all__ = ["slide_offset", "slide_details", "click_points", "nine_grid", "gobang", "warmup",
           "GeeTestError", "UnsupportedGeeTest", "InvalidGeeTestInput", "CLICK_KINDS"]
