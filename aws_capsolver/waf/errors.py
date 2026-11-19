"""Error type carrying a CapSolver-style error code."""

from typing import Optional

# CapSolver-compatible error codes used across the service
ERROR_KEY_DENIED_ACCESS = "ERROR_KEY_DENIED_ACCESS"
ERROR_INVALID_TASK_DATA = "ERROR_INVALID_TASK_DATA"
ERROR_TASK_NOT_SUPPORTED = "ERROR_TASK_NOT_SUPPORTED"
ERROR_TASKID_INVALID = "ERROR_TASKID_INVALID"
ERROR_CAPTCHA_UNSOLVABLE = "ERROR_CAPTCHA_UNSOLVABLE"
ERROR_PROXY_CONNECT_REFUSED = "ERROR_PROXY_CONNECT_REFUSED"
ERROR_PROXY_BANNED = "ERROR_PROXY_BANNED"
ERROR_TASK_TIMEOUT = "ERROR_TASK_TIMEOUT"
ERROR_SITE_UNREACHABLE = "ERROR_SITE_UNREACHABLE"
ERROR_NO_SLOT_AVAILABLE = "ERROR_NO_SLOT_AVAILABLE"
ERROR_SERVICE_UNAVAILABLE = "ERROR_SERVICE_UNAVAILABLE"

ALL_CODES = frozenset({
    ERROR_KEY_DENIED_ACCESS, ERROR_INVALID_TASK_DATA, ERROR_TASK_NOT_SUPPORTED,
    ERROR_TASKID_INVALID, ERROR_CAPTCHA_UNSOLVABLE, ERROR_PROXY_CONNECT_REFUSED,
    ERROR_PROXY_BANNED, ERROR_TASK_TIMEOUT, ERROR_SITE_UNREACHABLE, ERROR_NO_SLOT_AVAILABLE,
    ERROR_SERVICE_UNAVAILABLE,
})


class SolverError(Exception):
    """
    A failure that maps to one CapSolver error code.

    `http_status` is set when the failure is an unexpected HTTP status from an upstream
    endpoint, so callers can react to specific statuses (e.g. 400 from a PoW verify)
    without inventing non-CapSolver error codes.
    """

    def __init__(self, code: str, description: str, http_status: Optional[int] = None):
        super().__init__(f"{code}: {description}")
        self.code = code
        self.description = description
        self.http_status = http_status
