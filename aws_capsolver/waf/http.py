"""
HTTP plumbing shared by the protocol layer: session factory, proxy parsing, the SSRF
guard, bounded reads, deadline-aware timeouts, mapping of network failures onto
CapSolver error codes and redaction of credentials in log/error text.
"""

import ipaddress
import logging
import re
import socket
import time
from typing import Any, Dict, Iterable, List, Optional
from urllib.parse import quote, urlparse

import requests

from .errors import (ERROR_CAPTCHA_UNSOLVABLE, ERROR_INVALID_TASK_DATA, ERROR_PROXY_BANNED,
                     ERROR_PROXY_CONNECT_REFUSED, ERROR_SITE_UNREACHABLE, ERROR_TASK_TIMEOUT,
                     SolverError)

logger = logging.getLogger(__name__)

DEFAULT_USER_AGENT = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                      "(KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36")
PROXY_SCHEMES = ("http", "https", "socks5", "socks5h", "socks4", "socks4a")
MAX_JSON_BYTES = 4 * 1024 * 1024    # awswaf API answers (captcha problems carry images)
MAX_PAGE_BYTES = 8 * 1024 * 1024    # the protected page itself
MIN_REQUEST_TIMEOUT = 0.5

_CREDENTIALS_RE = re.compile(r"(?i)\b([a-z][a-z0-9+.\-]*://)([^/\s@:]*):([^/\s@]*)@")
_HOST_RE = re.compile(r"^[A-Za-z0-9.\-_]+$")


# --- secrets ---------------------------------------------------------------------------
def redact(text: Any) -> str:
    """Mask the password of every `scheme://user:password@` in `text`."""
    return _CREDENTIALS_RE.sub(r"\1\2:***@", str(text))


class RedactingFilter(logging.Filter):
    """Logging filter that removes URL credentials from every formatted record."""

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            message = record.getMessage()
        except Exception:  # a broken format string must not kill logging
            return True
        clean = redact(message)
        if clean != message:
            record.msg, record.args = clean, None
        return True


def install_log_redaction(handlers: Optional[Iterable[logging.Handler]] = None) -> None:
    """Attach a RedactingFilter to the given (default: root) handlers, once each."""
    for handler in handlers if handlers is not None else logging.getLogger().handlers:
        if not any(isinstance(f, RedactingFilter) for f in handler.filters):
            handler.addFilter(RedactingFilter())


# --- proxies ---------------------------------------------------------------------------
def _valid_port(value: Any) -> int:
    try:
        port = int(str(value).strip())
    except (TypeError, ValueError):
        port = 0
    if not 0 < port < 65536:
        raise SolverError(ERROR_INVALID_TASK_DATA, "proxy port must be 1-65535")
    return port


def _valid_host(value: Any) -> str:
    host = str(value or "").strip()
    if not host or not _HOST_RE.match(host):
        raise SolverError(ERROR_INVALID_TASK_DATA, "proxy host is invalid")
    return host


def parse_proxy(proxy: Any) -> Optional[str]:
    """
    Normalise a proxy into a URL usable by requests.

    Accepted forms:
      "http://user:pass@host:port", "socks5://host:port", "http://user:pass@[::1]:port",
      "scheme:host:port:user:pass", "scheme:host:port", "host:port:user:pass", "host:port",
      {"proxyType","proxyAddress","proxyPort","proxyLogin","proxyPassword"} (CapSolver fields)
    Passwords may contain ':' in the colon form (everything after the 4th field).
    """
    if proxy is None or proxy == "" or proxy == {}:
        return None
    if isinstance(proxy, dict):
        address, port = proxy.get("proxyAddress"), proxy.get("proxyPort")
        if not address or port in (None, ""):
            raise SolverError(ERROR_INVALID_TASK_DATA, "proxyAddress/proxyPort are required")
        scheme = str(proxy.get("proxyType") or "http").strip().lower()
        if scheme not in PROXY_SCHEMES:
            raise SolverError(ERROR_INVALID_TASK_DATA, "unsupported proxyType")
        host = str(address).strip()
        if not (host.startswith("[") and host.endswith("]")):
            host = _valid_host(host)
        login, password = proxy.get("proxyLogin"), proxy.get("proxyPassword")
        auth = (f"{quote(str(login), safe='')}:{quote(str(password or ''), safe='')}@"
                if login not in (None, "") else "")
        return f"{scheme}://{auth}{host}:{_valid_port(port)}"
    if not isinstance(proxy, str):
        raise SolverError(ERROR_INVALID_TASK_DATA, "proxy must be a string")
    text = proxy.strip()
    if not text:
        return None
    if any(c.isspace() for c in text):
        raise SolverError(ERROR_INVALID_TASK_DATA, "proxy must not contain whitespace")
    if "://" in text:
        parsed = urlparse(text)
        try:
            port = parsed.port
        except ValueError:
            port = None
        if parsed.scheme.lower() not in PROXY_SCHEMES or not parsed.hostname or not port:
            raise SolverError(ERROR_INVALID_TASK_DATA, "unsupported proxy URL")
        return text
    parts = text.split(":")
    scheme = "http"
    if len(parts) > 2 and parts[0].lower() in PROXY_SCHEMES:
        scheme = parts.pop(0).lower()
    if len(parts) > 4:  # a password containing ':'
        parts = parts[:3] + [":".join(parts[3:])]
    if len(parts) not in (2, 4):
        raise SolverError(ERROR_INVALID_TASK_DATA,
                          "proxy must be scheme:host:port[:user:pass] or a proxy URL")
    host, port = _valid_host(parts[0]), _valid_port(parts[1])
    if len(parts) == 4:
        user, password = quote(parts[2], safe=""), quote(parts[3], safe="")
        return f"{scheme}://{user}:{password}@{host}:{port}"
    return f"{scheme}://{host}:{port}"


# --- SSRF guard ------------------------------------------------------------------------
def _resolve(host: str) -> List[str]:
    """All addresses `host` resolves to (module-level so tests can replace it)."""
    infos = socket.getaddrinfo(host, None, proto=socket.IPPROTO_TCP)
    return sorted({info[4][0] for info in infos})


def is_public_address(address: str) -> bool:
    try:
        ip = ipaddress.ip_address(address.split("%", 1)[0])
    except ValueError:
        return False
    mapped = getattr(ip, "ipv4_mapped", None)
    if mapped is not None:
        ip = mapped
    return ip.is_global and not ip.is_multicast


def check_public_host(host: Optional[str], *, what: str = "websiteURL",
                      resolve_failure_ok: bool = False) -> None:
    """
    Refuse hosts that are, or resolve to, private / loopback / link-local / reserved
    addresses. `resolve_failure_ok` lets names through that cannot be resolved locally
    (e.g. behind a proxy that does the DNS lookup itself).
    """
    host = (host or "").strip().strip("[]")
    if not host:
        raise SolverError(ERROR_INVALID_TASK_DATA, f"{what} has no host")
    try:
        ipaddress.ip_address(host.split("%", 1)[0])
        addresses = [host]
    except ValueError:
        try:
            addresses = _resolve(host)
        except (OSError, UnicodeError) as exc:
            if resolve_failure_ok:
                return
            raise SolverError(ERROR_SITE_UNREACHABLE, f"{what}: cannot resolve host") from exc
    if not addresses or not all(is_public_address(a) for a in addresses):
        raise SolverError(ERROR_INVALID_TASK_DATA,
                          f"{what} points to a private or reserved address "
                          "(set AWSCS_ALLOW_PRIVATE=1 to allow)")


# --- requests --------------------------------------------------------------------------
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


def remaining_timeout(timeout: float, deadline: Optional[float], what: str = "request") -> float:
    """The per-request timeout, shortened so a request never outlives the task deadline."""
    if deadline is None:
        return timeout
    left = deadline - time.monotonic()
    if left <= 0:
        raise SolverError(ERROR_TASK_TIMEOUT, f"{what}: deadline exceeded")
    return max(MIN_REQUEST_TIMEOUT, min(timeout, left))


def map_network_error(exc: requests.RequestException, what: str, via_proxy: bool) -> SolverError:
    """Translate a requests exception into the closest CapSolver error code."""
    name = exc.__class__.__name__
    if isinstance(exc, requests.exceptions.InvalidProxyURL):
        return SolverError(ERROR_INVALID_TASK_DATA, f"{what}: invalid proxy URL")
    if isinstance(exc, requests.exceptions.ProxyError):
        return SolverError(ERROR_PROXY_CONNECT_REFUSED, f"{what}: proxy failed ({name})")
    if isinstance(exc, requests.exceptions.ConnectTimeout):
        code = ERROR_PROXY_CONNECT_REFUSED if via_proxy else ERROR_TASK_TIMEOUT
        return SolverError(code, f"{what}: connect timed out")
    if isinstance(exc, requests.exceptions.Timeout):
        return SolverError(ERROR_TASK_TIMEOUT, f"{what}: timed out")
    if isinstance(exc, requests.exceptions.SSLError):
        return SolverError(ERROR_SITE_UNREACHABLE, f"{what}: TLS error")
    if isinstance(exc, requests.exceptions.ConnectionError):
        code = ERROR_PROXY_CONNECT_REFUSED if via_proxy else ERROR_SITE_UNREACHABLE
        return SolverError(code, f"{what}: connection failed")
    if isinstance(exc, (requests.exceptions.InvalidURL, requests.exceptions.MissingSchema,
                        requests.exceptions.InvalidSchema, requests.exceptions.InvalidHeader)):
        return SolverError(ERROR_INVALID_TASK_DATA, f"{what}: {name}")
    return SolverError(ERROR_SITE_UNREACHABLE, f"{what}: {name}")


def read_limited(resp: requests.Response, limit: int, what: str) -> bytes:
    """Read a (streamed) response body, refusing anything larger than `limit` bytes."""
    try:
        declared = int(resp.headers.get("content-length") or 0)
    except ValueError:
        declared = 0
    if declared > limit:
        resp.close()
        raise SolverError(ERROR_SITE_UNREACHABLE, f"{what}: response too large")
    chunks, size = [], 0
    try:
        for chunk in resp.iter_content(64 * 1024):
            size += len(chunk)
            if size > limit:
                raise SolverError(ERROR_SITE_UNREACHABLE, f"{what}: response too large")
            chunks.append(chunk)
    except requests.RequestException as exc:
        raise map_network_error(exc, what, False) from exc
    finally:
        resp.close()
    body = b"".join(chunks)
    resp._content = body  # make .text / .json() work after streaming
    resp._content_consumed = True
    return body


def send(session: requests.Session, method: str, url: str, what: str, timeout: float,
         deadline: Optional[float] = None, max_bytes: int = MAX_JSON_BYTES,
         **kwargs) -> requests.Response:
    """One request with a deadline-bounded timeout and a size-bounded body."""
    via_proxy = bool(session.proxies)
    t = remaining_timeout(timeout, deadline, what)
    try:
        resp = session.request(method, url, timeout=t, stream=True, **kwargs)
    except requests.RequestException as exc:
        raise map_network_error(exc, what, via_proxy) from exc
    read_limited(resp, max_bytes, what)
    return resp


def request_json(session: requests.Session, method: str, url: str, what: str,
                 timeout: float, deadline: Optional[float] = None,
                 **kwargs) -> Dict[str, Any]:
    """Send one request and return its JSON object; every failure becomes a SolverError."""
    via_proxy = bool(session.proxies)
    resp = send(session, method, url, what, timeout, deadline, **kwargs)
    status = resp.status_code
    logger.debug("%s %s -> %s (%d bytes)", method, what, status, len(resp.content))
    if status == 407:
        raise SolverError(ERROR_PROXY_CONNECT_REFUSED, f"{what}: proxy authentication required",
                          http_status=status)
    if status in (403, 429) and via_proxy:
        raise SolverError(ERROR_PROXY_BANNED, f"{what}: HTTP {status} through proxy",
                          http_status=status)
    if status >= 500:
        raise SolverError(ERROR_SITE_UNREACHABLE, f"{what}: HTTP {status}", http_status=status)
    if status != 200:
        snippet = redact(resp.text[:200])
        raise SolverError(ERROR_CAPTCHA_UNSOLVABLE, f"{what}: HTTP {status} {snippet!r}",
                          http_status=status)
    try:
        data = resp.json()
    except ValueError as exc:
        raise SolverError(ERROR_SITE_UNREACHABLE, f"{what}: response is not JSON") from exc
    if not isinstance(data, dict):
        raise SolverError(ERROR_SITE_UNREACHABLE, f"{what}: unexpected JSON shape")
    return data

