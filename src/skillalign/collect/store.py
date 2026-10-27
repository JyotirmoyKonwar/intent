"""Write skill snapshots to the corpus safely, with one manifest per skill.

Layout (under the data dir):
    corpus/skills/<key>/...        the skill's files, byte-for-byte, never executable
    corpus/manifests/<key>.json    where it came from, file hashes, what was skipped

Crawled content is untrusted: paths are checked, symlinks and binaries are not
stored, size limits apply, and nothing is ever imported or run.
"""

from __future__ import annotations

import hashlib
import re
import shutil
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath

from pydantic import BaseModel, Field

MAX_FILE_BYTES = 1_000_000
MAX_SKILL_BYTES = 2_000_000
MAX_FILES = 400
IGNORED_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv", ".mypy_cache",
                ".pytest_cache", "dist", "build", ".next", ".DS_Store"}


class FileEntry(BaseModel):
    path: str
    size: int
    sha256: str


class Skipped(BaseModel):
    path: str
    reason: str  # "binary", "too-large", "symlink", "unsafe-path", "skill-size-limit", ...


class Manifest(BaseModel):
    key: str
    skill_id: str
    platform: str  # "official", "github", ...
    repo: str | None = None  # "owner/name"
    commit: str | None = None
    path: str = ""  # skill directory inside the repo ("" = repo root)
    source_url: str | None = None
    license: str | None = None  # SPDX-like guess, or "unknown"
    license_file: str | None = None  # repo-relative path of the license used
    repo_stars: int | None = None
    is_fork: bool | None = None
    crawled_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat(timespec="seconds"))
    files: list[FileEntry] = Field(default_factory=list)
    skipped: list[Skipped] = Field(default_factory=list)
    has_frontmatter: bool = False
    name: str | None = None
    description: str | None = None

    @property
    def total_bytes(self) -> int:
        return sum(f.size for f in self.files)


def safe_relpath(path: str) -> str | None:
    """Normalized relative POSIX path, or None if it could escape the skill directory."""
    if not path or "\x00" in path or "\\" in path:
        return None
    p = PurePosixPath(path)
    if p.is_absolute() or any(part in ("..", "") for part in p.parts):
        return None
    return p.as_posix()


def is_binary(data: bytes) -> bool:
    return b"\x00" in data[:8192]


def skill_key(platform: str, repo: str | None, path: str) -> str:
    """Filesystem-safe, stable, unique key for one skill directory."""
    raw = f"{platform}/{repo or ''}/{path}"
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", raw.replace("/", "--")).strip("_-")[:120]
    return f"{slug}-{hashlib.sha1(raw.encode()).hexdigest()[:8]}"


class CorpusStore:
    def __init__(self, data_dir: Path) -> None:
        self.skills_dir = data_dir / "corpus" / "skills"
        self.manifest_dir = data_dir / "corpus" / "manifests"

    def skill_dir(self, key: str) -> Path:
        return self.skills_dir / key

    def has(self, key: str) -> bool:
        return (self.manifest_dir / f"{key}.json").exists()

    def manifests(self) -> list[Manifest]:
        return [Manifest.model_validate_json(p.read_text())
                for p in sorted(self.manifest_dir.glob("*.json"))]

    def write(self, manifest: Manifest, files: Iterable[tuple[str, bytes]]) -> Manifest:
        """Store files (relative path, content) for one skill and its manifest.

        Files already in `manifest.skipped` are kept there; new skips are appended.
        """
        target = self.skill_dir(manifest.key)
        if target.exists():
            shutil.rmtree(target)
        target.mkdir(parents=True)
        total = 0
        for raw_path, data in sorted(files, key=lambda x: x[0]):
            rel = safe_relpath(raw_path)
            if rel is None:
                manifest.skipped.append(Skipped(path=raw_path[:200], reason="unsafe-path"))
                continue
            reason = None
            if any(part in IGNORED_DIRS for part in PurePosixPath(rel).parts):
                reason = "ignored-dir"
            elif len(data) > MAX_FILE_BYTES:
                reason = "too-large"
            elif is_binary(data):
                reason = "binary"
            elif total + len(data) > MAX_SKILL_BYTES:
                reason = "skill-size-limit"
            elif len(manifest.files) >= MAX_FILES:
                reason = "file-count-limit"
            if reason:
                manifest.skipped.append(Skipped(path=rel, reason=reason))
                continue
            out = target / rel
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(data)
            out.chmod(0o644)
            total += len(data)
            manifest.files.append(FileEntry(path=rel, size=len(data),
                                            sha256=hashlib.sha256(data).hexdigest()))
        self.manifest_dir.mkdir(parents=True, exist_ok=True)
        (self.manifest_dir / f"{manifest.key}.json").write_text(
            manifest.model_dump_json(indent=2) + "\n")
        return manifest


# ---------------------------------------------------------------- licenses

_LICENSE_PATTERNS = [
    ("Apache-2.0", r"apache license,?\s+version 2\.0"),
    ("MIT", r"\bmit license\b|permission is hereby granted, free of charge"),
    ("BSD-3-Clause", r"redistribution and use in source and binary forms.*neither the name"),
    ("BSD-2-Clause", r"redistribution and use in source and binary forms"),
    ("GPL-3.0", r"gnu general public license\s+version 3"),
    ("AGPL-3.0", r"gnu affero general public license"),
    ("MPL-2.0", r"mozilla public license,?\s+(version|v\.?)\s*2\.0"),
    ("CC-BY-4.0", r"creative commons attribution 4\.0"),
    ("Unlicense", r"this is free and unencumbered software"),
    ("proprietary", r"all rights reserved|proprietary|may not (use|copy|distribute)"),
]


def guess_license(text: str) -> str:
    low = " ".join(text.lower().split())[:20000]
    for spdx, pattern in _LICENSE_PATTERNS:
        if re.search(pattern, low, re.DOTALL):
            return spdx
    return "unknown"


LICENSE_NAMES = re.compile(r"^(license|licence|copying)(\.(md|txt|rst))?$", re.IGNORECASE)
