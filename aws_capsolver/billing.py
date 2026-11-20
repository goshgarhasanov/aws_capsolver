"""
Token billing: every client key holds a token balance; each solve costs tokens.

* Tokens are bought in packages: ``tokens_per_usd`` tokens per US dollar (default 1000,
  so $10 = 10,000 tokens). The operator credits them with
  ``python -m aws_capsolver billing topup --key K --usd 10`` after a payment.
* Every task type has its own price (``DEFAULT_PRICES``, overridable with a JSON file).
  ``ImageToTextTask`` may be priced per module (``"ImageToTextTask:bls_poland"``).
* The price is reserved when a task is accepted (so concurrent tasks can never overdraw a
  key) and refunded when the task fails - only solved tasks are paid for, like CapSolver.
* A key seen for the first time starts with its ``keys.json`` ``"balance"`` (or the
  configured default) as an opening grant.
* Every movement is a row in an append-only ledger; refunds are idempotent per task.

SQLite (WAL) in the usage database file; one connection guarded by a lock, every
movement in a ``BEGIN IMMEDIATE`` transaction so several processes stay consistent.
"""

import json
import logging
import sqlite3
import threading
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

# tokens per solved task
DEFAULT_PRICES: Dict[str, int] = {
    "AntiAwsWafTask": 30,
    "AntiAwsWafTaskProxyLess": 30,
    "AwsWafClassification": 10,
    "BlsClassification": 5,
    "GeeTestClassification": 8,
    "ImageToTextTask": 2,
}
DEFAULT_TOKENS_PER_USD = 1000
MAX_TOKENS = 10 ** 12


class BillingError(Exception):
    """Base class of billing failures."""


class InsufficientTokens(BillingError):
    def __init__(self, balance: int, price: int):
        super().__init__(f"not enough tokens: balance {balance}, this task costs {price}")
        self.balance, self.price = balance, price


def load_pricing(path: Optional[Path]) -> tuple:
    """(prices, tokens_per_usd) from an optional JSON file
    ``{"prices": {"AwsWafClassification": 10, ...}, "tokens_per_usd": 1000}``."""
    prices, per_usd = dict(DEFAULT_PRICES), DEFAULT_TOKENS_PER_USD
    if path is None or not Path(path).is_file():
        return prices, per_usd
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        logger.error("cannot read pricing file %s (%s); using the defaults", path, exc)
        return prices, per_usd
    for k, v in (data.get("prices") or {}).items():
        if isinstance(k, str) and isinstance(v, int) and not isinstance(v, bool) and 0 <= v <= 10 ** 6:
            prices[k] = v
        else:
            logger.warning("pricing: ignoring %r=%r (need a whole number of tokens)", k, v)
    v = data.get("tokens_per_usd")
    if isinstance(v, int) and not isinstance(v, bool) and v > 0:
        per_usd = v
    return prices, per_usd


class Billing:
    def __init__(self, path: Path, prices: Optional[Dict[str, int]] = None,
                 tokens_per_usd: int = DEFAULT_TOKENS_PER_USD,
                 opening_grant: Callable[[str], int] = lambda key: 0):
        self.path = Path(path)
        self.prices = dict(DEFAULT_PRICES if prices is None else prices)
        self.tokens_per_usd = int(tokens_per_usd)
        self._opening_grant = opening_grant
        self._lock = threading.Lock()
        if str(path) != ":memory:":
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: Optional[sqlite3.Connection] = sqlite3.connect(
            str(path), check_same_thread=False, timeout=10, isolation_level=None)
        try:
            self._conn.execute("PRAGMA journal_mode=WAL")
        except sqlite3.Error as exc:
            logger.warning("billing: WAL unavailable (%s)", exc)
        self._conn.executescript(
            "CREATE TABLE IF NOT EXISTS accounts (client_key TEXT PRIMARY KEY, "
            "  tokens INTEGER NOT NULL CHECK (tokens >= 0), updated REAL);"
            "CREATE TABLE IF NOT EXISTS ledger (id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, "
            "  client_key TEXT NOT NULL, kind TEXT NOT NULL, delta INTEGER NOT NULL, "
            "  balance INTEGER NOT NULL, task_type TEXT, task_id TEXT, note TEXT);"
            "CREATE UNIQUE INDEX IF NOT EXISTS ledger_task_kind ON ledger (task_id, kind) "
            "  WHERE task_id IS NOT NULL;"
            "CREATE INDEX IF NOT EXISTS ledger_key ON ledger (client_key, id);")

    # ------------------------------------------------------------------ helpers
    def price(self, task_type: str, module: Optional[str] = None) -> int:
        if module and f"{task_type}:{module}" in self.prices:
            return self.prices[f"{task_type}:{module}"]
        return self.prices.get(task_type, 0)

    def _ensure(self, cur: sqlite3.Cursor, client_key: str) -> int:
        row = cur.execute("SELECT tokens FROM accounts WHERE client_key = ?", (client_key,)).fetchone()
        if row is not None:
            return int(row[0])
        grant = max(0, min(MAX_TOKENS, int(self._opening_grant(client_key) or 0)))
        now = time.time()
        cur.execute("INSERT INTO accounts (client_key, tokens, updated) VALUES (?, ?, ?)",
                    (client_key, grant, now))
        cur.execute("INSERT INTO ledger (ts, client_key, kind, delta, balance, note) "
                    "VALUES (?, ?, 'grant', ?, ?, 'opening balance')", (now, client_key, grant, grant))
        return grant

    def _tx(self, fn):
        """Run fn(cursor) in one BEGIN IMMEDIATE transaction."""
        with self._lock:
            if self._conn is None:
                raise BillingError("billing store is closed")
            cur = self._conn.cursor()
            cur.execute("BEGIN IMMEDIATE")
            try:
                out = fn(cur)
            except BaseException:
                cur.execute("ROLLBACK")
                raise
            cur.execute("COMMIT")
            return out

    def _move(self, cur, client_key: str, delta: int, kind: str, task_type=None, task_id=None,
              note=None) -> int:
        bal = self._ensure(cur, client_key)
        new = bal + delta
        if new < 0:
            raise InsufficientTokens(bal, -delta)
        if new > MAX_TOKENS:
            raise BillingError("balance would exceed the maximum")
        now = time.time()
        cur.execute("UPDATE accounts SET tokens = ?, updated = ? WHERE client_key = ?",
                    (new, now, client_key))
        cur.execute("INSERT INTO ledger (ts, client_key, kind, delta, balance, task_type, task_id, note) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (now, client_key, kind, delta, new, task_type, task_id, note))
        return new

    # ------------------------------------------------------------------ API
    def balance(self, client_key: str) -> int:
        return self._tx(lambda cur: self._ensure(cur, client_key))

    def charge(self, client_key: str, task_type: str, task_id: str, module: Optional[str] = None) -> int:
        """Reserve the task's price; raises InsufficientTokens. Returns the price charged."""
        price = self.price(task_type, module)
        if price <= 0:
            return 0
        self._tx(lambda cur: self._move(cur, client_key, -price, "charge", task_type, task_id))
        return price

    def refund(self, task_id: str) -> int:
        """Give back the charge of a failed task (once). Returns the tokens refunded."""
        def fn(cur):
            row = cur.execute("SELECT client_key, delta, task_type FROM ledger "
                              "WHERE task_id = ? AND kind = 'charge'", (task_id,)).fetchone()
            if row is None:
                return 0
            if cur.execute("SELECT 1 FROM ledger WHERE task_id = ? AND kind = 'refund'",
                           (task_id,)).fetchone():
                return 0
            self._move(cur, row[0], -int(row[1]), "refund", row[2], task_id)
            return -int(row[1])
        return self._tx(fn)

    def topup(self, client_key: str, tokens: Optional[int] = None, usd: Optional[float] = None,
              note: str = "") -> int:
        """Credit a purchase (``usd`` converted at tokens_per_usd) or a manual amount."""
        if (tokens is None) == (usd is None):
            raise ValueError("give exactly one of tokens / usd")
        if usd is not None:
            if not (usd > 0) or usd > 10 ** 7:
                raise ValueError("usd must be positive")
            tokens = int(round(usd * self.tokens_per_usd))
            note = note or f"${usd:g} package"
        if not isinstance(tokens, int) or tokens == 0 or abs(tokens) > MAX_TOKENS:
            raise ValueError("tokens must be a non-zero whole number")
        kind = "topup" if tokens > 0 else "adjust"
        return self._tx(lambda cur: self._move(cur, client_key, tokens, kind, note=note or None))

    def history(self, client_key: str, limit: int = 50) -> List[dict]:
        with self._lock:
            if self._conn is None:
                return []
            rows = self._conn.execute(
                "SELECT ts, kind, delta, balance, task_type, task_id, note FROM ledger "
                "WHERE client_key = ? ORDER BY id DESC LIMIT ?", (client_key, int(limit))).fetchall()
        return [dict(zip(("ts", "kind", "delta", "balance", "task_type", "task_id", "note"), r)) for r in rows]

    def accounts(self) -> List[dict]:
        with self._lock:
            if self._conn is None:
                return []
            rows = self._conn.execute("SELECT client_key, tokens, updated FROM accounts ORDER BY client_key").fetchall()
        return [{"client_key": r[0], "tokens": r[1], "updated": r[2]} for r in rows]

    def close(self) -> None:
        with self._lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None
