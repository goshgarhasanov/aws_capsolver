"""Service settings from environment variables, client keys and the usage store."""

import hmac
import json
import logging
import os
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError:
        logger.warning("ignoring non-integer %s=%r", name, raw)
        return default


def _env_float(name: str, default: float, minimum: float = 0.1) -> float:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        value = float(raw)
    except ValueError:
        logger.warning("ignoring non-numeric %s=%r", name, raw)
        return default
    if not value >= minimum:  # also rejects NaN
        logger.warning("ignoring out-of-range %s=%r (minimum %s)", name, raw, minimum)
        return default
    return value


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


@dataclass
class Settings:
    keys_file: Path = ROOT / "keys.json"
    usage_db: Path = ROOT / "data" / "usage.db"
    workers: int = 8                # concurrent solving tasks
    max_pending: int = 200          # queued + running tasks before ERROR_NO_SLOT_AVAILABLE
    task_timeout: float = 120.0     # hard limit per task
    task_ttl: float = 600.0         # finished tasks are forgotten after this
    http_timeout: float = 20.0      # per network call
    classify_timeout: float = 60.0  # synchronous AwsWafClassification limit
    pow_processes: Optional[int] = None
    captcha_rounds: int = 5
    verify_token: bool = True
    warmup: bool = False            # preload the vision model at startup
    open_access: bool = False       # accept any clientKey (local development only)
    allow_private: bool = False     # let websiteURL / proxy target private networks
    max_body_bytes: int = 4 * 1024 * 1024  # request body limit (16 images of 2 MB won't fit)
    default_balance: float = 1000.0
    extra_keys: Dict[str, dict] = field(default_factory=dict)

    @classmethod
    def from_env(cls) -> "Settings":
        s = cls()
        s.keys_file = Path(os.environ.get("AWSCS_KEYS_FILE", str(s.keys_file)))
        s.usage_db = Path(os.environ.get("AWSCS_USAGE_DB", str(s.usage_db)))
        s.workers = max(1, _env_int("AWSCS_WORKERS", s.workers))
        s.max_pending = max(1, _env_int("AWSCS_MAX_PENDING", s.max_pending))
        s.task_timeout = _env_float("AWSCS_TASK_TIMEOUT", s.task_timeout)
        s.task_ttl = _env_float("AWSCS_TASK_TTL", s.task_ttl, minimum=1.0)
        s.http_timeout = _env_float("AWSCS_HTTP_TIMEOUT", s.http_timeout)
        s.classify_timeout = _env_float("AWSCS_CLASSIFY_TIMEOUT", s.classify_timeout)
        procs = _env_int("AWSCS_POW_PROCESSES", 0)
        s.pow_processes = procs if procs > 0 else None
        s.captcha_rounds = max(1, _env_int("AWSCS_CAPTCHA_ROUNDS", s.captcha_rounds))
        s.verify_token = _env_bool("AWSCS_VERIFY_TOKEN", True)
        s.warmup = _env_bool("AWSCS_WARMUP", True)
        s.open_access = _env_bool("AWSCS_OPEN_ACCESS", False)
        s.allow_private = _env_bool("AWSCS_ALLOW_PRIVATE", False)
        s.max_body_bytes = max(1024, _env_int("AWSCS_MAX_BODY_BYTES", s.max_body_bytes))
        for key in filter(None, (k.strip() for k in os.environ.get("AWSCS_KEYS", "").split(","))):
            s.extra_keys[key] = {"name": "env"}
        return s


class KeyStore:
    """
    Client keys. keys.json accepts either ["key1", "key2"] or
    {"key1": {"name": "bot-a", "balance": 100}, ...}.

    The file is re-read when its mtime/size change. A file that cannot be parsed (e.g.
    caught half-written) keeps the previous keys and is retried on the next lookup.
    Keys are compared in constant time against every configured key.
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self._keys: Dict[str, dict] = dict(settings.extra_keys)
        self._stamp: Optional[tuple] = None
        self._loaded = False
        self._lock = threading.Lock()
        with self._lock:
            self._maybe_reload()
        if not self._keys and not settings.open_access:
            logger.warning("no client keys configured (%s / AWSCS_KEYS); every request "
                           "will be denied", settings.keys_file)

    def _file_stamp(self) -> Optional[tuple]:
        try:
            st = self.settings.keys_file.stat()
        except OSError:
            return None
        return (st.st_mtime_ns, st.st_size)

    def _maybe_reload(self) -> None:
        stamp = self._file_stamp()
        if self._loaded and stamp == self._stamp:
            return
        keys: Dict[str, dict] = dict(self.settings.extra_keys)
        if stamp is not None:
            path = self.settings.keys_file
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                logger.error("cannot read keys file %s: %s (keeping the previous keys)",
                             path, exc.__class__.__name__)
                return  # _stamp unchanged: retried on the next lookup
            if isinstance(data, list):
                keys.update({str(k): {} for k in data if isinstance(k, (str, int))})
            elif isinstance(data, dict):
                keys.update({str(k): (v if isinstance(v, dict) else {}) for k, v in data.items()})
            else:
                logger.error("keys file %s must hold a JSON list or object", path)
        self._keys = {k: v for k, v in keys.items() if k}
        self._stamp, self._loaded = stamp, True

    def lookup(self, client_key: Any) -> Optional[dict]:
        if not isinstance(client_key, str) or not client_key:
            return None
        if len(client_key) > 512:  # longer than any real key; skip the comparison work
            return {} if self.settings.open_access else None
        with self._lock:
            self._maybe_reload()
            items = list(self._keys.items())
        candidate = client_key.encode("utf-8", "surrogatepass")
        found: Optional[dict] = None
        for key, info in items:  # no early exit: timing does not depend on the match
            if hmac.compare_digest(key.encode("utf-8", "surrogatepass"), candidate):
                found = info
        if found is not None:
            return found
        return {} if self.settings.open_access else None

    def balance(self, client_key: str) -> float:
        info = self.lookup(client_key) or {}
        try:
            value = float(info.get("balance", self.settings.default_balance))
        except (TypeError, ValueError):
            return self.settings.default_balance
        return value if value == value else self.settings.default_balance  # NaN guard


class UsageStore:
    """
    Per-key counters in SQLite: tasks created, solved, failed.

    One connection shared by all threads, serialised by a lock (sqlite3 objects are not
    safe for concurrent use). WAL mode plus a busy timeout let several service processes
    share the same file. Failures are logged and never break a request.
    """

    def __init__(self, path: Path):
        self.path = path
        self._lock = threading.Lock()
        if str(path) != ":memory:":
            path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: Optional[sqlite3.Connection] = sqlite3.connect(
            str(path), check_same_thread=False, timeout=10)
        try:
            self._conn.execute("PRAGMA journal_mode=WAL")
        except sqlite3.Error as exc:  # e.g. a read-only directory; the default mode still works
            logger.warning("usage store: WAL unavailable (%s)", exc)
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS usage (client_key TEXT, task_type TEXT, "
            "created INTEGER DEFAULT 0, solved INTEGER DEFAULT 0, failed INTEGER DEFAULT 0, "
            "last_used REAL, PRIMARY KEY (client_key, task_type))")
        self._conn.commit()

    def bump(self, client_key: str, task_type: str, column: str) -> None:
        if column not in ("created", "solved", "failed"):
            raise ValueError(column)
        now = time.time()
        with self._lock:
            if self._conn is None:
                return
            try:
                self._conn.execute(
                    f"INSERT INTO usage (client_key, task_type, {column}, last_used) "
                    f"VALUES (?, ?, 1, ?) ON CONFLICT(client_key, task_type) DO UPDATE SET "
                    f"{column} = {column} + 1, last_used = excluded.last_used",
                    (client_key, task_type, now))
                self._conn.commit()
            except sqlite3.Error as exc:
                logger.error("usage store write failed: %s", exc)
                try:
                    self._conn.rollback()
                except sqlite3.Error:
                    pass

    def stats(self, client_key: str) -> Dict[str, Dict[str, int]]:
        with self._lock:
            if self._conn is None:
                return {}
            try:
                rows = self._conn.execute(
                    "SELECT task_type, created, solved, failed FROM usage WHERE client_key = ?",
                    (client_key,)).fetchall()
            except sqlite3.Error as exc:
                logger.error("usage store read failed: %s", exc)
                return {}
        return {r[0]: {"created": r[1], "solved": r[2], "failed": r[3]} for r in rows}

    def close(self) -> None:
        with self._lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None
