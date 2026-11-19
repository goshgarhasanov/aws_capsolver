"""Solver stub for the AWS WAF "toycarcity" problem (place a dot at the end of the car path).

No "toycarcity" problem was served by any of the sampled WAF endpoints (all returned
``gridcaptcha``), so there is no data to build or validate a solver against.
"""
from __future__ import annotations

from .errors import UnsupportedProblem


def solve_toycar(image: bytes) -> tuple[int, int]:
    raise UnsupportedProblem("toycarcity problems are not supported yet")
