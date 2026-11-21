"""GeeTest v4 "winlinze" (gobang): drag one piece so five identical pieces line up.

The board is an N x N grid (v4: 5 x 5). A move takes one piece from outside a line
(row, column or one of the two diagonals) that already holds N-1 pieces of that kind and
drops it into the line's single empty cell.

Input is either the ``ques`` matrix of the v4 ``load`` response (0 = empty) or a picture
of the board (the widget screenshot, or the cells drawn on a canvas). For a picture the
cells are compared with each other and grouped into kinds; which group is "empty" is
decided by the puzzle itself: GeeTest boards have exactly one valid move, so the
grouping/empty choice that yields exactly one move wins (ties: the plainest group).
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

from ._io import load_image
from .errors import InvalidGeeTestInput, UnsupportedGeeTest

Cell = tuple[int, int]
DEFAULT_SIDE = 5
MAX_SIDE = 10


def _lines(n: int) -> list[list[Cell]]:
    rows = [[(r, c) for c in range(n)] for r in range(n)]
    cols = [[(r, c) for r in range(n)] for c in range(n)]
    diags = [[(i, i) for i in range(n)], [(i, n - 1 - i) for i in range(n)]]
    return rows + cols + diags


def moves(board: Sequence[Sequence[int]], empty: int = 0) -> list[tuple[Cell, Cell]]:
    """All single moves that complete a line, sorted (deterministic)."""
    n = len(board)
    found: set[tuple[Cell, Cell]] = set()
    for line in _lines(n):
        vals = [board[r][c] for r, c in line]
        holes = [p for p, v in zip(line, vals) if v == empty]
        if len(holes) != 1:
            continue
        kinds = {v for v in vals if v != empty}
        if len(kinds) != 1:
            continue
        kind = kinds.pop()
        on_line = set(line)
        for r in range(n):
            for c in range(n):
                if board[r][c] == kind and (r, c) not in on_line:
                    found.add(((r, c), holes[0]))
    return sorted(found)


def _validate_matrix(board) -> list[list[int]]:
    if not isinstance(board, (list, tuple)) or not board:
        raise InvalidGeeTestInput("board matrix must be a non-empty list of rows")
    n = len(board)
    if n < 3 or n > MAX_SIDE:
        raise InvalidGeeTestInput(f"board side must be 3..{MAX_SIDE}")
    out = []
    for row in board:
        if not isinstance(row, (list, tuple)) or len(row) != n:
            raise InvalidGeeTestInput("board matrix must be square")
        vals = []
        for v in row:
            if isinstance(v, bool) or not isinstance(v, int) or v < 0 or v > 99:
                raise InvalidGeeTestInput("board cells must be ints 0..99 (0 = empty)")
            vals.append(v)
        out.append(vals)
    return out


def solve_matrix(board) -> tuple[Cell, Cell]:
    m = _validate_matrix(board)
    found = moves(m, 0)
    if not found:
        raise UnsupportedGeeTest("no single move completes a line on this board")
    return found[0]


# ----------------------------------------------------------------------- pictures
def _cell_features(im, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Shift-tolerant per-cell colour statistics and a 'plainness' measure.

    Pieces are not always centred in an exact 1/n slice of the picture (widget padding,
    screenshots), so position-free statistics are used: mean / spread / percentiles of
    each channel plus the mean gradient (shading)."""
    w, h = im.size
    a_all = np.asarray(im, dtype=np.float32) / 255.0
    feats, energy = [], []
    for r in range(n):
        for c in range(n):
            x0, y0, cw, ch = c * w / n, r * h / n, w / n, h / n
            a = a_all[int(y0 + 0.08 * ch):int(y0 + 0.92 * ch), int(x0 + 0.08 * cw):int(x0 + 0.92 * cw)]
            if a.size == 0:
                raise InvalidGeeTestInput("board picture is too small")
            px = a.reshape(-1, 3)
            g = a.mean(axis=2)
            grad = float(np.abs(np.diff(g, axis=0)).mean() + np.abs(np.diff(g, axis=1)).mean()) if min(g.shape) > 1 else 0.0
            f = np.concatenate([px.mean(0), px.std(0), np.percentile(px, 10, axis=0),
                                np.percentile(px, 50, axis=0), np.percentile(px, 90, axis=0), [grad * 4.0]])
            feats.append(f)
            sat = px.max(axis=1) - px.min(axis=1)
            energy.append(float(px.std()) + float(sat.mean()) + float(1.0 - px.mean()) * 0.5)
    return np.array(feats), np.array(energy)


def _cluster(feats: np.ndarray, threshold: float) -> list[int]:
    """Deterministic single-pass clustering by mean absolute difference to the cluster seed."""
    labels: list[int] = []
    seeds: list[np.ndarray] = []
    for f in feats:
        d = [float(np.abs(f - s).mean()) for s in seeds]
        if d and min(d) <= threshold:
            labels.append(int(np.argmin(d)))
        else:
            seeds.append(f)
            labels.append(len(seeds) - 1)
    # one refinement pass with cluster means as seeds
    for _ in range(2):
        means = [feats[[i for i, l in enumerate(labels) if l == k]].mean(axis=0) for k in range(len(seeds))]
        labels = [int(np.argmin([np.abs(f - m).mean() for m in means])) for f in feats]
        used = sorted(set(labels))
        remap = {k: i for i, k in enumerate(used)}
        labels = [remap[l] for l in labels]
        seeds = [means[k] for k in used]
    return labels


def solve_image(value, n: int = DEFAULT_SIDE) -> tuple[Cell, Cell]:
    im = load_image(value, "board", "RGB")
    feats, energy = _cell_features(im, n)
    best = None
    for threshold in (0.01, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12):
        labels = _cluster(feats, threshold)
        k = max(labels) + 1
        if k < 2:
            continue
        grid = [[labels[r * n + c] + 1 for c in range(n)] for r in range(n)]
        # try every group as "empty"; GeeTest boards have exactly one valid move
        for g in range(1, k + 1):
            found = moves(grid, g)
            if len(found) == 1:
                plain = float(np.mean([e for e, l in zip(energy, labels) if l + 1 == g]))
                cand = (plain, threshold, found[0])
                if best is None or cand[0] < best[0]:
                    best = cand
        if best is not None:
            break
    if best is None:
        raise UnsupportedGeeTest("could not read a solvable gobang board from the picture")
    return best[2]


def solve_board(board) -> tuple[Cell, Cell]:
    """Matrix (list of lists) or picture (bytes / base64)."""
    if isinstance(board, (list, tuple)):
        return solve_matrix(board)
    return solve_image(board)
