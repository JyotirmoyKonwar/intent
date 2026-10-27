"""Collect long-tail skills from GitHub via code search for SKILL.md files.

Needs a token in GITHUB_TOKEN (a fine-grained token with public read-only access is enough):
code search does not work anonymously.

GitHub returns at most 1,000 results per query, so the query is split by file size
until every shard fits. For each repo we take its default-branch HEAD commit, list
the tree once, and snapshot every SKILL.md folder in it. File contents come from
raw.githubusercontent.com, which does not count against the API rate limit.
Every search hit is logged to data/raw/github/search_hits.jsonl for reproducibility.
"""

from __future__ import annotations

import json
import logging
import os
import time
from collections.abc import Callable, Iterator
from pathlib import Path, PurePosixPath

import httpx

from skillalign.collect.repos import describe, is_skill_md
from skillalign.collect.store import (
    IGNORED_DIRS,
    LICENSE_NAMES,
    MAX_FILE_BYTES,
    CorpusStore,
    Manifest,
    Skipped,
    guess_license,
    skill_key,
)

log = logging.getLogger(__name__)

API = "https://api.github.com"
RAW = "https://raw.githubusercontent.com"
MAX_RESULTS_PER_QUERY = 1000
MAX_SIZE = 384_000  # GitHub only indexes files below this size
# One repo can hold thousands of near-identical skills (javimosch/supercli has 5,432),
# which would swamp the corpus. Keep the first N per repo, in path order.
MAX_SKILLS_PER_REPO = 50


class GitHubClient:
    def __init__(self, token: str | None = None, *, transport: httpx.BaseTransport | None = None,
                 sleep: Callable[[float], None] = time.sleep, max_wait_s: float = 900) -> None:
        token = token or os.environ.get("GITHUB_TOKEN")
        if not token:
            raise RuntimeError("GITHUB_TOKEN is not set; GitHub code search needs a token")
        headers = {"Authorization": f"Bearer {token}",
                   "Accept": "application/vnd.github+json",
                   "X-GitHub-Api-Version": "2022-11-28",
                   "User-Agent": "skillalign-research-crawler"}
        self.http = httpx.Client(headers=headers, transport=transport, timeout=60,
                                 follow_redirects=True)
        self.sleep = sleep
        self.max_wait_s = max_wait_s
        self.requests = 0

    def get(self, url: str, **kw) -> httpx.Response:
        for attempt in range(6):
            resp = self.http.get(url, **kw)
            self.requests += 1
            if resp.status_code in (403, 429) and self._rate_limited(resp):
                self.sleep(self._wait_time(resp, attempt))
                continue
            if resp.status_code >= 500:
                self.sleep(2 ** attempt)
                continue
            return resp
        resp.raise_for_status()
        return resp

    @staticmethod
    def _rate_limited(resp: httpx.Response) -> bool:
        return (resp.headers.get("x-ratelimit-remaining") == "0"
                or "retry-after" in resp.headers
                or "rate limit" in resp.text.lower())

    def _wait_time(self, resp: httpx.Response, attempt: int) -> float:
        if "retry-after" in resp.headers:
            wait = float(resp.headers["retry-after"])
        elif "x-ratelimit-reset" in resp.headers:
            wait = float(resp.headers["x-ratelimit-reset"]) - time.time() + 2
        else:
            wait = 60 * (attempt + 1)
        return min(max(wait, 1.0), self.max_wait_s)

    def json(self, path: str, **params) -> dict:
        resp = self.get(f"{API}{path}", params=params or None)
        resp.raise_for_status()
        return resp.json()

    # -- search ------------------------------------------------------------

    def _count(self, query: str) -> int:
        return self.json("/search/code", q=query, per_page=1)["total_count"]

    def shards(self, query: str, lo: int = 0, hi: int = MAX_SIZE) -> Iterator[str]:
        """Split `query` by file size until each part has at most 1,000 results."""
        q = f"{query} size:{lo}..{hi}"
        n = self._count(q)
        if n == 0:
            return
        if n <= MAX_RESULTS_PER_QUERY or hi - lo <= 1:
            yield q
            return
        mid = (lo + hi) // 2
        yield from self.shards(query, lo, mid)
        yield from self.shards(query, mid + 1, hi)

    def search_code(self, query: str) -> Iterator[dict]:
        for shard in self.shards(query):
            for page in range(1, MAX_RESULTS_PER_QUERY // 100 + 1):
                items = self.json("/search/code", q=shard, per_page=100, page=page)["items"]
                yield from items
                if len(items) < 100:
                    break

    # -- repo contents -----------------------------------------------------

    def head_commit(self, repo: str, branch: str) -> str:
        return self.json(f"/repos/{repo}/commits/{branch}")["sha"]

    def tree(self, repo: str, commit: str) -> list[dict]:
        data = self.json(f"/repos/{repo}/git/trees/{commit}", recursive="1")
        if data.get("truncated"):
            log.warning("tree of %s is truncated; some skills may be missed", repo)
        return data["tree"]

    def raw(self, repo: str, commit: str, path: str) -> bytes:
        resp = self.get(f"{RAW}/{repo}/{commit}/{path}")
        resp.raise_for_status()
        return resp.content


def _skill_dirs(tree: list[dict]) -> list[str]:
    dirs = []
    for e in tree:
        p = PurePosixPath(e["path"])
        if e["type"] == "blob" and is_skill_md(p.name) and not IGNORED_DIRS & set(p.parts):
            dirs.append("" if str(p.parent) == "." else p.parent.as_posix())
    return sorted(set(dirs))


def _license_for(client: GitHubClient, repo: str, commit: str, tree: list[dict],
                 skill_dir: str, repo_spdx: str | None) -> tuple[str, str | None]:
    blobs = {e["path"] for e in tree if e["type"] == "blob"}
    parts = PurePosixPath(skill_dir).parts if skill_dir else ()
    for depth in range(len(parts), -1, -1):
        prefix = "/".join(parts[:depth])
        for path in sorted(blobs):
            folder, _, name = path.rpartition("/")
            if folder == prefix and LICENSE_NAMES.match(name):
                if depth == 0 and repo_spdx and repo_spdx != "NOASSERTION":
                    return repo_spdx, path
                text = client.raw(repo, commit, path).decode("utf-8", errors="replace")
                return guess_license(text), path
    return (repo_spdx if repo_spdx and repo_spdx != "NOASSERTION" else "unknown"), None


def collect_repo_via_api(client: GitHubClient, store: CorpusStore, repo: str,
                         max_skills: int = MAX_SKILLS_PER_REPO) -> list[Manifest]:
    info = client.json(f"/repos/{repo}")
    if info.get("fork") or info.get("size", 1) == 0:  # forks duplicate their parent
        return []
    commit = client.head_commit(repo, info["default_branch"])
    tree = client.tree(repo, commit)
    stars = info.get("stargazers_count")
    spdx = (info.get("license") or {}).get("spdx_id")
    out = []
    for skill_dir in _skill_dirs(tree)[:max_skills]:
        key = skill_key("github", repo, skill_dir)
        if store.has(key):
            continue
        lic, lic_file = _license_for(client, repo, commit, tree, skill_dir, spdx)
        m = Manifest(key=key, skill_id=f"github/{repo}/{skill_dir}@{commit[:12]}",
                     platform="github", repo=repo, commit=commit, path=skill_dir,
                     source_url=f"https://github.com/{repo}/tree/{commit}/{skill_dir}".rstrip("/"),
                     license=lic, license_file=lic_file, repo_stars=stars, is_fork=False)
        prefix = f"{skill_dir}/" if skill_dir else ""
        files: list[tuple[str, bytes]] = []
        for e in tree:
            if not e["path"].startswith(prefix) or e["type"] != "blob":
                continue
            rel = e["path"][len(prefix):]
            if IGNORED_DIRS & set(PurePosixPath(rel).parts):
                continue
            if e.get("mode") == "120000":
                m.skipped.append(Skipped(path=rel, reason="symlink"))
            elif e.get("size", 0) > MAX_FILE_BYTES:
                m.skipped.append(Skipped(path=rel, reason="too-large"))
            else:
                data = client.raw(repo, commit, e["path"])
                files.append((rel, data))
                if is_skill_md(rel):
                    describe(m, data)
        out.append(store.write(m, files))
    return out


def collect_github(client: GitHubClient, store: CorpusStore, log_dir: Path, *,
                   query: str = "filename:SKILL.md", max_repos: int = 1000,
                   max_skills_per_repo: int = MAX_SKILLS_PER_REPO) -> list[Manifest]:
    """Search, then snapshot up to `max_skills_per_repo` skills from each of `max_repos` repos."""
    log_dir.mkdir(parents=True, exist_ok=True)
    repos: list[str] = []
    with (log_dir / "search_hits.jsonl").open("a") as hits:
        for item in client.search_code(query):
            repo = item["repository"]["full_name"]
            hits.write(json.dumps({"repo": repo, "path": item["path"], "sha": item["sha"],
                                   "query": query}) + "\n")
            if repo not in repos:
                repos.append(repo)
                if len(repos) >= max_repos:
                    break
    out = []
    for i, repo in enumerate(repos, 1):
        try:
            out += collect_repo_via_api(client, store, repo, max_skills_per_repo)
        except httpx.HTTPError as err:
            log.warning("skipping %s: %s", repo, err)
        if i % 25 == 0:
            log.info("%d/%d repos, %d skills, %d API requests", i, len(repos), len(out),
                     client.requests)
    return out
