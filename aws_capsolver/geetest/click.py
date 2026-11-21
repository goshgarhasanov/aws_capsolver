"""GeeTest click challenges (v3 / v4 icon click, v3 word click) - not available yet."""
from __future__ import annotations

from .errors import UnsupportedGeeTest


def click_points(image, kind: str, hints):
    raise UnsupportedGeeTest(f"GeeTest {kind} click is not supported by this build")
