"""SQLite cache for LLM responses. Reruns are free and the cache ships with the artifact."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class CachedResponse:
    text: str
    prompt_tokens: int
    completion_tokens: int
    latency_s: float
    # Why generation stopped: "stop", or "length" when cut off at max_tokens.
    finish_reason: str = "stop"

    @property
    def truncated(self) -> bool:
        return self.finish_reason == "length"


def _retry(fn, tries: int = 120):
    """Run a database call, waiting out "database is locked" from other writers."""
    for attempt in range(tries):
        try:
            return fn()
        except sqlite3.OperationalError as err:
            if "locked" not in str(err) or attempt == tries - 1:
                raise
            time.sleep(1)


def cache_key(payload: dict[str, Any]) -> str:
    """Stable hash of everything that determines a response."""
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()


class ResponseCache:
    """`path=None` keeps the cache in memory (used by tests)."""

    def __init__(self, path: Path | None) -> None:
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        # Several workers (threads or processes) may share one cache file: wait for the lock
        # instead of failing, and let readers proceed during a write.
        self._db = sqlite3.connect(str(path) if path else ":memory:", check_same_thread=False,
                                   timeout=120)
        def setup() -> None:
            if path is not None:
                self._db.execute("PRAGMA journal_mode=WAL")
            self._db.execute(
                "CREATE TABLE IF NOT EXISTS responses ("
                " key TEXT PRIMARY KEY, model TEXT, text TEXT,"
                " prompt_tokens INTEGER, completion_tokens INTEGER, latency_s REAL,"
                " created TEXT DEFAULT CURRENT_TIMESTAMP, finish_reason TEXT DEFAULT 'stop')"
            )
            # Caches written before finish_reason existed: add the column.
            columns = {row[1] for row in self._db.execute("PRAGMA table_info(responses)")}
            if "finish_reason" not in columns:
                self._db.execute(
                    "ALTER TABLE responses ADD COLUMN finish_reason TEXT DEFAULT 'stop'")
            self._db.commit()

        _retry(setup)

    def get(self, key: str) -> CachedResponse | None:
        with self._lock:
            row = _retry(lambda: self._db.execute(
                "SELECT text, prompt_tokens, completion_tokens, latency_s, finish_reason"
                " FROM responses WHERE key = ?", (key,)
            ).fetchone())
        return CachedResponse(*row) if row else None

    def put(self, key: str, model: str, resp: CachedResponse) -> None:
        with self._lock:
            def write() -> None:
                self._db.execute(
                    "INSERT OR REPLACE INTO responses"
                    " (key, model, text, prompt_tokens, completion_tokens, latency_s,"
                    " finish_reason) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (key, model, resp.text, resp.prompt_tokens, resp.completion_tokens,
                     resp.latency_s, resp.finish_reason),
                )
                self._db.commit()

            _retry(write)
