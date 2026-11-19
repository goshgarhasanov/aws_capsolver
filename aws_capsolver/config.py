"""Service settings from environment variables, client keys and the usage store."""

import json
import logging
import os
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

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


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError:
        logger.warning("ignoring non-numeric %s=%r", name, raw)
        return default


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
        s.task_ttl = _env_float("AWSCS_TASK_TTL", s.task_ttl)
        s.http_timeout = _env_float("AWSCS_HTTP_TIMEOUT", s.http_timeout)
        s.classify_timeout = _env_float("AWSCS_CLASSIFY_TIMEOUT", s.classify_timeout)
        procs = _env_int("AWSCS_POW_PROCESSES", 0)
        s.pow_processes = procs or None
        s.captcha_rounds = max(1, _env_int("AWSCS_CAPTCHA_ROUNDS", s.captcha_rounds))
        s.verify_token = os.environ.get("AWSCS_VERIFY_TOKEN", "1") not in ("0", "false", "no")
        s.warmup = os.environ.get("AWSCS_WARMUP", "1") not in ("0", "false", "no")
        s.open_access = os.environ.get("AWSCS_OPEN_ACCESS", "0") in ("1", "true", "yes")
        for key in filter(None, (k.strip() for k in os.environ.get("AWSCS_KEYS", "").split(","))):
            s.extra_keys[key] = {"name": "env"}
        return s


class KeyStore:
    """
    Client keys. keys.json accepts either ["key1", "key2"] or
    {"key1": {"name": "bot-a", "balance": 100}, ...}.
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self._keys: Dict[str, dict] = {}
        self._mtime: Optional[float] = None
        self._lock = threading.Lock()
        self._reload()

    def _reload(self) -> None:
        keys: Dict[str, dict] = dict(self.settings.extra_keys)
        path = self.settings.keys_file
        try:
            mtime = path.stat().st_mtime
        except FileNotFoundError:
            mtime = None
        if mtime is not None:
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                logger.error("cannot read keys file %s: %s", path, exc)
                data = {}
            if isinstance(data, list):
                keys.update({str(k): {} for k in data})
            elif isinstance(data, dict):
                keys.update({str(k): (v if isinstance(v, dict) else {}) for k, v in data.items()})
        self._keys, self._mtime = keys, mtime
        if not keys and not self.settings.open_access:
            logger.warning("no client keys configured (%s / AWSCS_KEYS); every request "
                           "will be denied", path)

    def _maybe_reload(self) -> None:
        try:
            mtime = self.settings.keys_file.stat().st_mtime
        except FileNotFoundError:
            mtime = None
        if mtime != self._mtime:
            self._reload()

    def lookup(self, client_key: Optional[str]) -> Optional[dict]:
        if not isinstance(client_key, str) or not client_key:
            return None
        with self._lock:
            self._maybe_reload()
            if client_key in self._keys:
                return self._keys[client_key]
        return {} if self.settings.open_access else None

    def balance(self, client_key: str) -> float:
        info = self.lookup(client_key) or {}
        try:
            return float(info.get("balance", self.settings.default_balance))
        except (TypeError, ValueError):
            return self.settings.default_balance


class UsageStore:
    """Per-key counters in SQLite: tasks created, solved, failed."""

    def __init__(self, path: Path):
        self.path = path
        self._lock = threading.Lock()
        if str(path) != ":memory:":
            path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path), check_same_thread=False, timeout=10)
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS usage (client_key TEXT, task_type TEXT, "
            "created INTEGER DEFAULT 0, solved INTEGER DEFAULT 0, failed INTEGER DEFAULT 0, "
            "last_used REAL, PRIMARY KEY (client_key, task_type))")
        self._conn.commit()

    def bump(self, client_key: str, task_type: str, column: str) -> None:
        if column not in ("created", "solved", "failed"):
            raise ValueError(column)
        with self._lock:
            try:
                self._conn.execute(
                    "INSERT INTO usage (client_key, task_type, last_used) VALUES (?, ?, ?) "
                    "ON CONFLICT(client_key, task_type) DO NOTHING",
                    (client_key, task_type, time.time()))
                self._conn.execute(
                    f"UPDATE usage SET {column} = {column} + 1, last_used = ? "
                    "WHERE client_key = ? AND task_type = ?", (time.time(), client_key, task_type))
                self._conn.commit()
            except sqlite3.Error as exc:
                logger.error("usage store write failed: %s", exc)

    def stats(self, client_key: str) -> Dict[str, Dict[str, int]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT task_type, created, solved, failed FROM usage WHERE client_key = ?",
                (client_key,)).fetchall()
        return {r[0]: {"created": r[1], "solved": r[2], "failed": r[3]} for r in rows}

    def close(self) -> None:
        with self._lock:
            self._conn.close()
