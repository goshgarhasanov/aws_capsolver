"""Error type carrying a CapSolver-style error code."""

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


class SolverError(Exception):
    """A failure that maps to one CapSolver error code."""

    def __init__(self, code: str, description: str):
        super().__init__(f"{code}: {description}")
        self.code = code
        self.description = description
