"""Settings, read from environment variables (prefix SKILLALIGN_).

Variables can also live in a `.env` file at the repo root (gitignored; see `.env.example`).
Real environment variables take precedence over the file.

The LLM endpoint is any OpenAI-compatible server: vLLM on the GPU box
(e.g. via an SSH tunnel to localhost:8000) or ollama on a laptop.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def load_dotenv(path: Path = REPO_ROOT / ".env") -> None:
    """Load KEY=value lines from `path` into os.environ without overriding existing values."""
    if not path.is_file():
        return
    for line in path.read_text().splitlines():
        line = line.strip().removeprefix("export ")
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        os.environ.setdefault(key.strip(), value)


load_dotenv()


def _env(name: str, default: str) -> str:
    return os.environ.get(f"SKILLALIGN_{name}", default)


@dataclass(frozen=True)
class Settings:
    llm_base_url: str = field(default_factory=lambda: _env("LLM_BASE_URL",
                                                           "http://localhost:8000/v1"))
    llm_model: str = field(default_factory=lambda: _env("LLM_MODEL", "Qwen/Qwen3-8B"))
    llm_api_key: str = field(default_factory=lambda: _env("LLM_API_KEY", "EMPTY"))
    # Qwen3 thinking mode: off by default; the judge has a thinking variant for RQ4.
    llm_thinking: bool = field(default_factory=lambda: _env("LLM_THINKING", "0") == "1")
    llm_timeout_s: float = field(default_factory=lambda: float(_env("LLM_TIMEOUT_S", "300")))
    data_dir: Path = field(default_factory=lambda: Path(_env("DATA_DIR",
                                                             str(REPO_ROOT / "data"))))
    seed: int = field(default_factory=lambda: int(_env("SEED", "13")))

    @property
    def cache_path(self) -> Path:
        return self.data_dir / "cache" / "llm.sqlite"


def get_settings() -> Settings:
    return Settings()
