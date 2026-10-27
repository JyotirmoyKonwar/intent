import json
import subprocess
from pathlib import Path

import httpx
import pytest

from skillalign.collect import github
from skillalign.collect.dedup import dedup
from skillalign.collect.profile import profile
from skillalign.collect.repos import clone, collect_repo, find_skill_dirs
from skillalign.collect.store import (
    CorpusStore,
    Manifest,
    guess_license,
    safe_relpath,
    skill_key,
)

SKILL = "---\nname: {name}\ndescription: {desc}\n---\n\n# {name}\n\n{body}\n"
MIT = "MIT License\n\nPermission is hereby granted, free of charge, to any person..."


def write(root: Path, files: dict[str, str | bytes]) -> None:
    for rel, content in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(content if isinstance(content, bytes) else content.encode())


# ---------------------------------------------------------------- store


@pytest.mark.parametrize("path, ok", [
    ("a/b.py", True), ("./a.py", True), ("../x", False), ("/etc/passwd", False),
    ("a/../../x", False), ("a\\b", False), ("", False), ("a\x00b", False),
])
def test_safe_relpath(path, ok):
    assert (safe_relpath(path) is not None) == ok


def test_store_writes_safely(tmp_path):
    store = CorpusStore(tmp_path)
    m = Manifest(key="k1", skill_id="x", platform="test")
    big = b"x" * 1_000_001
    store.write(m, [("SKILL.md", b"---\nname: a\n---\n"), ("../escape", b"x"),
                    ("img.png", b"\x89PNG\x00\x00"), ("big.txt", big),
                    ("node_modules/x.js", b"1"), ("run.sh", b"#!/bin/sh\necho hi\n")])
    root = store.skill_dir("k1")
    assert sorted(p.name for p in root.rglob("*") if p.is_file()) == ["SKILL.md", "run.sh"]
    assert not (root / "run.sh").stat().st_mode & 0o111  # never executable
    assert {s.reason for s in m.skipped} == {"unsafe-path", "binary", "too-large", "ignored-dir"}
    assert store.has("k1") and store.manifests()[0].files[0].path == "SKILL.md"


def test_skill_key_is_stable_and_safe():
    a = skill_key("github", "o/r", "skills/x y")
    assert a == skill_key("github", "o/r", "skills/x y") != skill_key("github", "o/r", "skills/x")
    assert "/" not in a and " " not in a


def test_guess_license():
    assert guess_license(MIT) == "MIT"
    assert guess_license("Apache License\n  Version 2.0, January 2004") == "Apache-2.0"
    assert guess_license("Some custom terms.") == "unknown"
    assert guess_license("© Acme. All rights reserved.") == "proprietary"


# ---------------------------------------------------------------- repos


@pytest.fixture
def git_repo(tmp_path):
    repo = tmp_path / "src"
    write(repo, {
        "LICENSE": MIT,
        "skills/alpha/SKILL.md": SKILL.format(name="alpha", desc="Alpha skill.", body="Hi."),
        "skills/alpha/scripts/run.py": "print('alpha')\n",
        "skills/beta/SKILL.md": SKILL.format(name="beta", desc="Beta skill.", body="Yo."),
        "skills/beta/LICENSE.txt": "All rights reserved.",
        "template/SKILL.md": SKILL.format(name="t", desc="Template.", body="..."),
        "node_modules/pkg/SKILL.md": "ignored",
    })
    git = ["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t"]
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run([*git, "add", "."], check=True)
    subprocess.run([*git, "commit", "-qm", "init"], check=True)
    return repo


def test_find_skill_dirs(git_repo):
    assert find_skill_dirs(git_repo) == ["skills/alpha", "skills/beta", "template"]
    assert find_skill_dirs(git_repo, ["template"]) == ["skills/alpha", "skills/beta"]


def test_clone_and_collect_repo(git_repo, tmp_path):
    commit = clone(str(git_repo), tmp_path / "clone")
    store = CorpusStore(tmp_path / "data")
    ms = {m.path: m for m in collect_repo(store, tmp_path / "clone", platform="official",
                                          repo="o/r", commit=commit, exclude=["template"])}
    assert set(ms) == {"skills/alpha", "skills/beta"}
    alpha, beta = ms["skills/alpha"], ms["skills/beta"]
    assert alpha.name == "alpha" and alpha.description == "Alpha skill." and alpha.has_frontmatter
    assert [f.path for f in alpha.files] == ["SKILL.md", "scripts/run.py"]
    assert (alpha.license, alpha.license_file) == ("MIT", "LICENSE")
    assert (beta.license, beta.license_file) == ("proprietary", "skills/beta/LICENSE.txt")
    assert alpha.skill_id == f"official/o/r/skills/alpha@{commit[:12]}"


# ---------------------------------------------------------------- github


def fake_github(total_by_range: dict[str, int], rate_limit_once: bool = False):
    """A MockTransport that serves search, repo, commit, tree and raw endpoints."""
    state = {"limited": rate_limit_once}
    files = {"skills/a/SKILL.md": SKILL.format(name="a", desc="A.", body="x"),
             "skills/a/run.sh": "echo a\n", "LICENSE": MIT}

    def handler(req: httpx.Request) -> httpx.Response:
        path = req.url.path
        if req.url.host == "raw.githubusercontent.com":
            return httpx.Response(200, content=files[path.split("/", 4)[4]].encode())
        if path == "/search/code":
            if state["limited"]:
                state["limited"] = False
                return httpx.Response(403, headers={"x-ratelimit-remaining": "0",
                                                    "x-ratelimit-reset": "0"},
                                      text="API rate limit exceeded")
            q = req.url.params["q"]
            size = q.split("size:")[1]
            total = total_by_range.get(size, 0)
            if req.url.params.get("per_page") == "1":
                return httpx.Response(200, json={"total_count": total, "items": []})
            page = int(req.url.params.get("page", 1))
            n = max(0, min(100, total - (page - 1) * 100))
            items = [{"repository": {"full_name": f"o/r{size}"}, "path": "skills/a/SKILL.md",
                      "sha": "s"} for _ in range(n)]
            return httpx.Response(200, json={"total_count": total, "items": items})
        if path.startswith("/repos/") and path.count("/") == 3:
            return httpx.Response(200, json={"default_branch": "main", "fork": False,
                                             "size": 10, "stargazers_count": 5,
                                             "license": {"spdx_id": "MIT"}})
        if "/commits/" in path:
            return httpx.Response(200, json={"sha": "c0ffee" * 6 + "abcd"})
        if "/git/trees/" in path:
            tree = [{"path": p, "type": "blob", "mode": "100644", "size": len(c)}
                    for p, c in files.items()]
            tree.append({"path": "skills/a/link", "type": "blob", "mode": "120000", "size": 5})
            return httpx.Response(200, json={"tree": tree, "truncated": False})
        return httpx.Response(404)

    return httpx.MockTransport(handler)


def test_github_requires_token(monkeypatch):
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    with pytest.raises(RuntimeError, match="GITHUB_TOKEN"):
        github.GitHubClient()


def test_search_shards_large_queries():
    half = github.MAX_SIZE // 2
    totals = {f"0..{github.MAX_SIZE}": 1500, f"0..{half}": 700,
              f"{half + 1}..{github.MAX_SIZE}": 800}
    sleeps = []
    client = github.GitHubClient("t", transport=fake_github(totals, rate_limit_once=True),
                                 sleep=sleeps.append)
    shards = list(client.shards("filename:SKILL.md"))
    assert shards == [f"filename:SKILL.md size:0..{half}",
                      f"filename:SKILL.md size:{half + 1}..{github.MAX_SIZE}"]
    assert len(sleeps) == 1  # waited once for the rate limit, then carried on
    assert sum(1 for _ in client.search_code("filename:SKILL.md")) == 1500


def test_collect_github_end_to_end(tmp_path):
    totals = {f"0..{github.MAX_SIZE}": 3}
    client = github.GitHubClient("t", transport=fake_github(totals), sleep=lambda s: None)
    store = CorpusStore(tmp_path)
    [m] = github.collect_github(client, store, tmp_path / "log")
    assert m.repo == f"o/r0..{github.MAX_SIZE}" and m.path == "skills/a"
    assert [f.path for f in m.files] == ["SKILL.md", "run.sh"]
    assert [(s.path, s.reason) for s in m.skipped] == [("link", "symlink")]
    assert m.license == "MIT" and m.license_file == "LICENSE" and m.name == "a"
    hits = (tmp_path / "log" / "search_hits.jsonl").read_text().splitlines()
    assert len(hits) == 3 and json.loads(hits[0])["path"] == "skills/a/SKILL.md"
    # A second run skips skills already in the store.
    assert github.collect_github(client, store, tmp_path / "log") == []


# ---------------------------------------------------------------- dedup + profile

LONG = " ".join(f"word{i}" for i in range(400))


def add(store, key, text, platform="github", stars=0, extra=None):
    m = Manifest(key=key, skill_id=key, platform=platform, repo_stars=stars,
                 has_frontmatter="---" in text, description="d" if "---" in text else None)
    store.write(m, [("SKILL.md", text.encode()), *(extra or {}).items()])


def test_dedup_exact_near_and_filter(tmp_path):
    store = CorpusStore(tmp_path)
    base = SKILL.format(name="x", desc="d", body=LONG)
    add(store, "a-copy", base, stars=1)
    add(store, "b-official", base, platform="official")
    add(store, "c-near", base.replace("word200", "changed"), stars=50)
    add(store, "d-other", SKILL.format(name="y", desc="d", body=LONG[::-1]))
    add(store, "e-nofm", "# just markdown\n")
    res = dedup(store)
    assert res.filtered == {"e-nofm": "no-frontmatter"}
    assert res.canonical == ["b-official", "d-other"]
    [cluster] = res.clusters
    assert cluster["canonical"] == "b-official" and cluster["kind"] == "near"
    assert cluster["aliases"] == ["c-near", "a-copy"]  # ranked by stars

    summary = profile(store, res.canonical, tmp_path / "p.jsonl")
    assert summary["skills"] == 2 and summary["instruction_only"] == 2
