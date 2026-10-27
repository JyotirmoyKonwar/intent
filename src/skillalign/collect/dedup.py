"""Exact and near-duplicate detection over the corpus (MinHash + LSH), and the filter.

A skill copied into many repos (often with tiny edits) would otherwise dominate
the corpus and leak between dev and test. Each cluster keeps one canonical skill:
official first, then the most-starred repo, then the smallest key.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from skillalign.collect.store import CorpusStore, Manifest

NUM_PERM = 128
BANDS, ROWS = 16, 8  # candidate threshold ≈ (1/16)^(1/8) ≈ 0.71
NEAR_DUP_JACCARD = 0.8
SHINGLE = 5
_PRIME = (1 << 31) - 1
_rng = np.random.default_rng(13)
_A = _rng.integers(1, _PRIME, NUM_PERM, dtype=np.uint64)
_B = _rng.integers(0, _PRIME, NUM_PERM, dtype=np.uint64)
_WORD = re.compile(r"[a-z0-9_]+")

TEXT_SUFFIXES = {".md", ".py", ".sh", ".js", ".ts", ".mjs", ".cjs", ".json", ".yaml", ".yml",
                 ".toml", ".txt", ".bash", ".tsx", ".jsx", ".ps1", ".rb", ".go", ".rst"}


def content_hash(m: Manifest) -> str:
    blob = "\n".join(sorted(f"{f.path}:{f.sha256}" for f in m.files))
    return hashlib.sha256(blob.encode()).hexdigest()


def shingles(text: str) -> np.ndarray:
    words = _WORD.findall(text.lower())
    if len(words) < SHINGLE:
        words = words + [""] * (SHINGLE - len(words))
    grams = {" ".join(words[i:i + SHINGLE]) for i in range(len(words) - SHINGLE + 1)}
    return np.array([int.from_bytes(hashlib.blake2b(g.encode(), digest_size=8).digest())
                     % _PRIME for g in grams], dtype=np.uint64)


def minhash(text: str) -> np.ndarray:
    x = shingles(text)
    return ((np.outer(_A, x) + _B[:, None]) % _PRIME).min(axis=1)


def skill_text(store: CorpusStore, m: Manifest) -> str:
    root = store.skill_dir(m.key)
    parts = []
    for f in sorted(m.files, key=lambda f: f.path):
        if Path(f.path).suffix.lower() in TEXT_SUFFIXES:
            parts.append((root / f.path).read_text(errors="replace"))
    return "\n".join(parts)


def filter_reason(m: Manifest) -> str | None:
    """Why a skill is excluded from the corpus, or None to keep it (PLAN.md §5.2)."""
    if not m.has_frontmatter:
        return "no-frontmatter"
    if not m.description:
        return "no-description"
    if not any(f.path.lower() == "skill.md" for f in m.files):
        return "no-root-skill-md"
    return None


@dataclass
class DedupResult:
    canonical: list[str]
    clusters: list[dict]  # {"canonical", "aliases", "kind"}
    filtered: dict[str, str]  # key → reason

    def save(self, path: Path) -> None:
        path.write_text(json.dumps({"canonical": self.canonical, "clusters": self.clusters,
                                    "filtered": self.filtered}, indent=1) + "\n")


def _rank(m: Manifest) -> tuple:
    return (m.platform != "official", -(m.repo_stars or 0), m.key)


def dedup(store: CorpusStore) -> DedupResult:
    manifests = {m.key: m for m in store.manifests()}
    filtered = {k: r for k, m in manifests.items() if (r := filter_reason(m))}
    keys = sorted(k for k in manifests if k not in filtered)

    parent = {k: k for k in keys}
    kind: dict[str, str] = {}

    def find(k: str) -> str:
        while parent[k] != k:
            parent[k] = parent[parent[k]]
            k = parent[k]
        return k

    def union(a: str, b: str, how: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra
            kind[ra] = "near" if "near" in (how, kind.get(ra), kind.get(rb)) else "exact"

    by_hash: dict[str, str] = {}
    for k in keys:
        h = content_hash(manifests[k])
        if h in by_hash:
            union(by_hash[h], k, "exact")
        else:
            by_hash[h] = k

    sigs = {k: minhash(skill_text(store, manifests[k])) for k in keys}
    buckets: dict[tuple, list[str]] = defaultdict(list)
    for k, sig in sigs.items():
        for b in range(BANDS):
            buckets[(b, sig[b * ROWS:(b + 1) * ROWS].tobytes())].append(k)
    checked: set[tuple[str, str]] = set()
    for members in buckets.values():
        for i, a in enumerate(members):
            for b in members[i + 1:]:
                pair = (a, b) if a < b else (b, a)
                if pair in checked:
                    continue
                checked.add(pair)
                if float(np.mean(sigs[a] == sigs[b])) >= NEAR_DUP_JACCARD:
                    union(a, b, "near")

    groups: dict[str, list[str]] = defaultdict(list)
    for k in keys:
        groups[find(k)].append(k)
    canonical, clusters = [], []
    for root, members in groups.items():
        members.sort(key=lambda k: _rank(manifests[k]))
        canonical.append(members[0])
        if len(members) > 1:
            clusters.append({"canonical": members[0], "aliases": members[1:],
                             "kind": kind.get(root, "exact")})
    return DedupResult(sorted(canonical), clusters, filtered)
