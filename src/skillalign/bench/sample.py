"""Benchmark pilot sample: which corpus skills get annotated first (IMPLEMENTATION.md §4 M2).

The pool is a seeded random draw from the canonical corpus (the natural-random stratum),
plus a seeded draw of skills that have code (the enriched stratum). The profile has no
detector-hit field, so "enriched" uses the proxy `has_code_files or has_inline_code`.
The two strata never overlap. Same seed and corpus, same output.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

N_RANDOM = 60
N_ENRICHED = 20


def is_enriched(row: dict) -> bool:
    return bool(row.get("has_code_files") or row.get("has_inline_code"))


def draw_sample(canonical: list[str], profile: list[dict], seed: int,
                n_random: int = N_RANDOM, n_enriched: int = N_ENRICHED) -> list[dict]:
    """Pure draw. Returns entries `{key, skill_id, path, stratum}` in draw order."""
    rows = {r["key"]: r for r in profile}
    keys = sorted(k for k in canonical if k in rows)
    rng = random.Random(seed)
    random_keys = rng.sample(keys, n_random)
    taken = set(random_keys)
    enriched_pool = [k for k in keys if k not in taken and is_enriched(rows[k])]
    enriched_keys = rng.sample(enriched_pool, n_enriched)
    return ([_entry(k, rows[k], "random") for k in random_keys]
            + [_entry(k, rows[k], "enriched") for k in enriched_keys])


def extend_sample(existing: list[dict], canonical: list[str], profile: list[dict], seed: int,
                  n_random: int, n_enriched: int) -> list[dict]:
    """Append more skills to a sample without changing the entries already in it.

    The new draw uses its own seeded stream and skips every skill already sampled, so the
    first `len(existing)` entries (and the work done on them) stay exactly as they were.
    """
    rows = {r["key"]: r for r in profile}
    taken = {e["key"] for e in existing}
    keys = sorted(k for k in canonical if k in rows and k not in taken)
    rng = random.Random(f"{seed}-extend-{len(existing)}")
    random_keys = rng.sample(keys, n_random)
    taken |= set(random_keys)
    pool = [k for k in keys if k not in taken and is_enriched(rows[k])]
    enriched_keys = rng.sample(pool, n_enriched)
    return (existing + [_entry(k, rows[k], "random") for k in random_keys]
            + [_entry(k, rows[k], "enriched") for k in enriched_keys])


def _entry(key: str, row: dict, stratum: str) -> dict:
    return {"key": key, "skill_id": row["skill_id"], "path": f"skills/{key}",
            "stratum": stratum}


def build_sample(corpus_dir: Path, seed: int) -> list[dict]:
    """Read `dedup.json` and `profile.jsonl` from the corpus folder and draw the sample."""
    dedup = json.loads((corpus_dir / "dedup.json").read_text())
    profile = [json.loads(line)
               for line in (corpus_dir / "profile.jsonl").read_text().splitlines() if line]
    return draw_sample(dedup["canonical"], profile, seed)


def write_sample(corpus_dir: Path, out_path: Path, seed: int) -> list[dict]:
    entries = build_sample(corpus_dir, seed)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(entries, indent=2) + "\n")
    return entries
