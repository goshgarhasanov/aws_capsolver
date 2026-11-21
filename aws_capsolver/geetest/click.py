"""GeeTest click challenges: dispatch by kind.

* ``icon`` - v4 icon click / v3 icon click with hint icons: :mod:`.icon`.
* ``word`` - word (Chinese character) click: not supported by this build.
* ``space`` - spatial reasoning click: not supported by this build.
"""
from __future__ import annotations

from .errors import UnsupportedGeeTest


def click_points(image, kind: str, hints):
    if kind == "icon":
        from .icon import click_icons

        return click_icons(image, hints)
    if kind == "word":
        raise UnsupportedGeeTest("GeeTest word click is not supported by this build")
    raise UnsupportedGeeTest(f"GeeTest {kind} click is not supported by this build")
