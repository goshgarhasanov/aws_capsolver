"""Exceptions raised by the vision solvers."""


class VisionError(RuntimeError):
    """Base class for vision solver errors."""


class UnsupportedProblem(VisionError):
    """The problem type (or its configuration) cannot be solved by this build."""


class InvalidInput(VisionError, ValueError):
    """The supplied images / audio could not be decoded or have the wrong shape."""
