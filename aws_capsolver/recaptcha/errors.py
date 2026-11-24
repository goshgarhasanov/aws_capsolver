"""Exceptions raised by the reCAPTCHA v2 image classifier.

Kept free of heavy imports (torch / numpy / PIL) so any module can import them cheaply.
"""


class RecaptchaError(RuntimeError):
    """Base class for reCAPTCHA classifier errors."""


class UnsupportedQuestion(RecaptchaError):
    """The question (MID or text target) cannot be mapped to anything the model can score."""


class InvalidRecaptchaInput(RecaptchaError, ValueError):
    """The supplied image / size / question is malformed (bad base64, not an image, odd size, ...)."""
