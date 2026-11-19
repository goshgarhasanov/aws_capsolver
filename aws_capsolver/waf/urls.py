"""Derivation of the awswaf API base URLs from any of their script URLs."""

from typing import Optional
from urllib.parse import urlparse

TOKEN_HOST_SUFFIX = ".token.awswaf.com"
CAPTCHA_HOST_SUFFIX = ".captcha.awswaf.com"


def _split(script_url: str):
    if not script_url:
        return None, None
    parsed = urlparse(script_url.strip())
    host = (parsed.hostname or "").lower()
    if not host.endswith("awswaf.com"):
        return None, None
    path = parsed.path
    # strip a trailing file name (challenge.js / captcha.js / problem / verify ...)
    last = path.rsplit("/", 1)[-1]
    if last.endswith(".js") or last in ("problem", "verify", "inputs", "voucher", "mp_verify"):
        path = path.rsplit("/", 1)[0]
    return host, path.rstrip("/")


def token_base_url(script_url: str) -> Optional[str]:
    """
    https://<id>.<x>.<region>.captcha.awswaf.com/<a>/<b>/<c>/captcha.js
      -> https://<id>.<x>.<region>.token.awswaf.com/<a>/<b>/<c>
    """
    host, path = _split(script_url)
    if not host:
        return None
    if host.endswith(CAPTCHA_HOST_SUFFIX):
        host = host[: -len(CAPTCHA_HOST_SUFFIX)] + TOKEN_HOST_SUFFIX
    if not host.endswith(TOKEN_HOST_SUFFIX):
        return None
    return f"https://{host}{path}"


def captcha_base_url(script_url: str) -> Optional[str]:
    """Inverse of token_base_url: the base of /problem and /verify."""
    host, path = _split(script_url)
    if not host:
        return None
    if host.endswith(TOKEN_HOST_SUFFIX):
        host = host[: -len(TOKEN_HOST_SUFFIX)] + CAPTCHA_HOST_SUFFIX
    if not host.endswith(CAPTCHA_HOST_SUFFIX):
        return None
    return f"https://{host}{path}"
