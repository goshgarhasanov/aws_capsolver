"""Errors raised by the text (ImageToTextTask) recognisers."""


class InvalidImage(ValueError):
    """The input bytes are not a usable image (undecodable, empty or absurdly large)."""


class ModelUnavailable(RuntimeError):
    """The recogniser cannot run: its package, runtime or weights are missing."""
