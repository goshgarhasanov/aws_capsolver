"""Exceptions raised by the vision solvers.

Kept free of heavy imports (torch / open_clip) so any module can import them cheaply.
"""


class VisionError(RuntimeError):
    """Base class for vision solver errors."""


class UnsupportedProblem(VisionError):
    """The problem type (or its configuration) cannot be solved by this build."""


class UnknownTarget(UnsupportedProblem):
    """The grid target is not a class the model knows (raised only in strict mode)."""


class InvalidInput(VisionError, ValueError):
    """The supplied images / audio / parameters are malformed (bad base64, tile count, ...)."""


class ModelUnavailable(VisionError):
    """The vision model could not be loaded (missing weights, no network for first download)."""
