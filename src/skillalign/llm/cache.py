"""SQLite cache for LLM responses. Reruns are free and the cache ships with the artifact."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class CachedResponse:
    text: str
    prompt_tokens: int
    completion_tokens: int
    latency_s: float


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
        self._db = sqlite3.connect(str(path) if path else ":memory:", check_same_thread=False)
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS responses ("
            " key TEXT PRIMARY KEY, model TEXT, text TEXT,"
            " prompt_tokens INTEGER, completion_tokens INTEGER, latency_s REAL,"
            " created TEXT DEFAULT CURRENT_TIMESTAMP)"
        )
        self._db.commit()

    def get(self, key: str) -> CachedResponse | None:
        with self._lock:
            row = self._db.execute(
                "SELECT text, prompt_tokens, completion_tokens, latency_s"
                " FROM responses WHERE key = ?", (key,)
            ).fetchone()
        return CachedResponse(*row) if row else None

    def put(self, key: str, model: str, resp: CachedResponse) -> None:
        with self._lock:
            self._db.execute(
                "INSERT OR REPLACE INTO responses"
                " (key, model, text, prompt_tokens, completion_tokens, latency_s)"
                " VALUES (?, ?, ?, ?, ?, ?)",
                (key, model, resp.text, resp.prompt_tokens, resp.completion_tokens,
                 resp.latency_s),
            )
            self._db.commit()
