"""Collect skills from git repositories: shallow-clone, find every SKILL.md, snapshot its folder.

Used for official vendor repos and any repo given by hand. Clones keep `.git`, so
history can be mined later. Nothing in a clone is executed: hooks are disabled
and symlinks are checked out as small plain-text files holding the link target.
"""

from __future__ import annotations

import fnmatch
import os
import subprocess
from collections.abc import Iterator
from pathlib import Path, PurePosixPath

from skillalign.collect.store import (
    IGNORED_DIRS,
    LICENSE_NAMES,
    CorpusStore,
    Manifest,
    Skipped,
    guess_license,
    skill_key,
)
from skillalign.normalize.markdown import parse_markdown

#: Official / vendor skill repos and paths to leave out (templates, specs).
OFFICIAL_REPOS: dict[str, list[str]] = {
    "anthropics/skills": ["template", "template/*", "spec/*"],
    "openai/skills": [],
    "vercel-labs/agent-skills": [],
    "huggingface/skills": [],
}


def is_skill_md(name: str) -> bool:
    return name.lower() == "skill.md"


def describe(manifest: Manifest, skill_md: bytes) -> None:
    """Fill name/description/has_frontmatter from the SKILL.md text."""
    fm = parse_markdown(skill_md.decode("utf-8", errors="replace")).frontmatter
    manifest.has_frontmatter = fm.data is not None
    manifest.name = fm.get_str("name")
    desc = fm.get_str("description")
    manifest.description = desc[:2000] if desc else None


def clone(url: str, dest: Path) -> str:
    """Shallow clone (or reuse an existing clone). Returns the HEAD commit."""
    env = {**os.environ, "GIT_TERMINAL_PROMPT": "0", "GIT_LFS_SKIP_SMUDGE": "1"}
    safe = ["-c", "core.symlinks=false", "-c", "core.hooksPath=/dev/null"]
    if not (dest / ".git").exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", *safe, "clone", "-q", "--depth", "1", url, str(dest)],
                       check=True, env=env, timeout=600)
    out = subprocess.run(["git", "-C", str(dest), "rev-parse", "HEAD"], check=True,
                         capture_output=True, text=True, env=env)
    return out.stdout.strip()


def find_skill_dirs(repo_root: Path, exclude: list[str] = ()) -> list[str]:
    """Repo-relative POSIX paths of folders that contain a SKILL.md ("" = root)."""
    found = []
    for dirpath, dirnames, filenames in os.walk(repo_root):
        dirnames[:] = sorted(d for d in dirnames if d not in IGNORED_DIRS)
        if any(is_skill_md(f) for f in filenames):
            rel = Path(dirpath).relative_to(repo_root).as_posix()
            rel = "" if rel == "." else rel
            if not any(fnmatch.fnmatch(rel, pat) for pat in exclude):
                found.append(rel)
    return found


def _walk_files(root: Path) -> Iterator[tuple[str, Path]]:
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        dirnames[:] = sorted(d for d in dirnames if d not in IGNORED_DIRS)
        for f in sorted(filenames):
            p = Path(dirpath) / f
            yield p.relative_to(root).as_posix(), p


def find_license(repo_root: Path, skill_rel: str) -> tuple[str | None, str | None]:
    """Nearest license file from the skill folder up to the repo root."""
    parts = PurePosixPath(skill_rel).parts if skill_rel else ()
    for depth in range(len(parts), -1, -1):
        folder = repo_root.joinpath(*parts[:depth])
        for f in sorted(folder.iterdir()) if folder.is_dir() else []:
            if f.is_file() and not f.is_symlink() and LICENSE_NAMES.match(f.name):
                rel = f.relative_to(repo_root).as_posix()
                return guess_license(f.read_text(errors="replace")), rel
    return None, None


def collect_repo(store: CorpusStore, repo_root: Path, *, platform: str, repo: str,
                 commit: str, exclude: list[str] = (), repo_url: str | None = None,
                 stars: int | None = None, is_fork: bool | None = None) -> list[Manifest]:
    out = []
    base = repo_url or f"https://github.com/{repo}"
    for rel in find_skill_dirs(repo_root, exclude):
        skill_root = repo_root / rel if rel else repo_root
        lic, lic_file = find_license(repo_root, rel)
        m = Manifest(
            key=skill_key(platform, repo, rel), skill_id=f"{platform}/{repo}/{rel}@{commit[:12]}",
            platform=platform, repo=repo, commit=commit, path=rel,
            source_url=f"{base}/tree/{commit}/{rel}".rstrip("/"),
            license=lic or "unknown", license_file=lic_file, repo_stars=stars, is_fork=is_fork)
        files: list[tuple[str, bytes]] = []
        for relf, p in _walk_files(skill_root):
            if p.is_symlink():
                m.skipped.append(Skipped(path=relf, reason="symlink"))
                continue
            data = p.read_bytes()
            files.append((relf, data))
            if is_skill_md(relf):
                describe(m, data)
        out.append(store.write(m, files))
    return out


def collect_official(store: CorpusStore, clones_dir: Path,
                     repos: dict[str, list[str]] | None = None) -> list[Manifest]:
    out = []
    for repo, exclude in (repos or OFFICIAL_REPOS).items():
        dest = clones_dir / repo.replace("/", "__")
        commit = clone(f"https://github.com/{repo}", dest)
        out += collect_repo(store, dest, platform="official", repo=repo, commit=commit,
                            exclude=exclude)
    return out
