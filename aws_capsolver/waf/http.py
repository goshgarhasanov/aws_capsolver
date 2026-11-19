"""HTTP session factory, proxy parsing and mapping of network failures onto error codes."""

import logging
from typing import Any, Dict, Optional
from urllib.parse import quote, urlparse

import requests

from .errors import (ERROR_INVALID_TASK_DATA, ERROR_PROXY_BANNED, ERROR_PROXY_CONNECT_REFUSED,
                     ERROR_SITE_UNREACHABLE, ERROR_TASK_TIMEOUT, SolverError)

logger = logging.getLogger(__name__)

DEFAULT_USER_AGENT = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                      "(KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36")
_SCHEMES = ("http", "https", "socks5", "socks5h", "socks4")


def parse_proxy(proxy: Any) -> Optional[str]:
    """
    Normalise a proxy into a URL usable by requests.

    Accepted forms:
      "http://user:pass@host:port", "socks5://host:port",
      "scheme:host:port:user:pass", "scheme:host:port", "host:port:user:pass", "host:port",
      {"proxyType","proxyAddress","proxyPort","proxyLogin","proxyPassword"} (CapSolver fields)
    """
    if not proxy:
        return None
    if isinstance(proxy, dict):
        address, port = proxy.get("proxyAddress"), proxy.get("proxyPort")
        if not address or not port:
            raise SolverError(ERROR_INVALID_TASK_DATA, "proxyAddress/proxyPort are required")
        scheme = str(proxy.get("proxyType") or "http").lower()
        login, password = proxy.get("proxyLogin"), proxy.get("proxyPassword")
        auth = f"{quote(str(login), safe='')}:{quote(str(password or ''), safe='')}@" if login else ""
        proxy = f"{scheme}://{auth}{address}:{port}"
    if not isinstance(proxy, str):
        raise SolverError(ERROR_INVALID_TASK_DATA, "proxy must be a string")
    text = proxy.strip()
    if "://" in text:
        parsed = urlparse(text)
        if parsed.scheme.lower() not in _SCHEMES or not parsed.hostname or not parsed.port:
            raise SolverError(ERROR_INVALID_TASK_DATA, "unsupported proxy URL")
        return text
    parts = text.split(":")
    scheme = "http"
    if parts and parts[0].lower() in _SCHEMES:
        scheme = parts.pop(0).lower()
    if len(parts) not in (2, 4) or not parts[1].isdigit():
        raise SolverError(ERROR_INVALID_TASK_DATA,
                          "proxy must be scheme:host:port[:user:pass] or a proxy URL")
    host, port = parts[0], parts[1]
    if len(parts) == 4:
        user, password = quote(parts[2], safe=""), quote(parts[3], safe="")
        return f"{scheme}://{user}:{password}@{host}:{port}"
    return f"{scheme}://{host}:{port}"


def make_session(proxy_url: Optional[str], user_agent: Optional[str]) -> requests.Session:
    session = requests.Session()
    session.trust_env = False
    session.headers.update({
        "user-agent": user_agent or DEFAULT_USER_AGENT,
        "accept-language": "en-US,en;q=0.9",
    })
    if proxy_url:
        session.proxies = {"http": proxy_url, "https": proxy_url}
    return session


def map_network_error(exc: requests.RequestException, what: str, via_proxy: bool) -> SolverError:
    """Translate a requests exception into the closest CapSolver error code."""
    if isinstance(exc, requests.exceptions.ProxyError):
        return SolverError(ERROR_PROXY_CONNECT_REFUSED, f"{what}: proxy failed ({exc.__class__.__name__})")
    if isinstance(exc, requests.exceptions.Timeout):
        code = ERROR_PROXY_CONNECT_REFUSED if via_proxy and isinstance(
            exc, requests.exceptions.ConnectTimeout) else ERROR_TASK_TIMEOUT
        return SolverError(code, f"{what}: timed out")
    if isinstance(exc, requests.exceptions.SSLError):
        return SolverError(ERROR_SITE_UNREACHABLE, f"{what}: TLS error")
    if isinstance(exc, requests.exceptions.ConnectionError):
        code = ERROR_PROXY_CONNECT_REFUSED if via_proxy else ERROR_SITE_UNREACHABLE
        return SolverError(code, f"{what}: connection failed")
    return SolverError(ERROR_SITE_UNREACHABLE, f"{what}: {exc.__class__.__name__}")


def request_json(session: requests.Session, method: str, url: str, what: str,
                 timeout: float, **kwargs) -> Dict[str, Any]:
    """Send one request and return its JSON body; every failure becomes a SolverError."""
    via_proxy = bool(session.proxies)
    try:
        resp = session.request(method, url, timeout=timeout, **kwargs)
    except requests.RequestException as exc:
        raise map_network_error(exc, what, via_proxy) from exc
    logger.debug("%s %s -> %s (%d bytes)", method, what, resp.status_code, len(resp.content))
    if resp.status_code == 407:
        raise SolverError(ERROR_PROXY_CONNECT_REFUSED, f"{what}: proxy authentication required")
    if resp.status_code in (403, 429) and via_proxy:
        raise SolverError(ERROR_PROXY_BANNED, f"{what}: HTTP {resp.status_code} through proxy")
    if resp.status_code >= 500:
        raise SolverError(ERROR_SITE_UNREACHABLE, f"{what}: HTTP {resp.status_code}")
    if resp.status_code != 200:
        raise SolverError("HTTP_%d" % resp.status_code, f"{what}: HTTP {resp.status_code} "
                          f"{resp.text[:200]!r}")
    try:
        data = resp.json()
    except ValueError as exc:
        raise SolverError(ERROR_SITE_UNREACHABLE, f"{what}: response is not JSON") from exc
    if not isinstance(data, dict):
        raise SolverError(ERROR_SITE_UNREACHABLE, f"{what}: unexpected JSON shape")
    return data
