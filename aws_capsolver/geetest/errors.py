"""Exceptions raised by the GeeTest solvers.

Kept free of heavy imports (torch / numpy / PIL) so any module can import them cheaply.
"""


class GeeTestError(RuntimeError):
    """Base class for GeeTest solver errors."""


class UnsupportedGeeTest(GeeTestError):
    """The GeeTest challenge kind (or its variant) cannot be solved by this build."""


class InvalidGeeTestInput(GeeTestError, ValueError):
    """The supplied images / parameters are malformed (bad base64, not an image, too big, ...)."""
