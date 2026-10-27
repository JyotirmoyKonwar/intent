"""Profile the deduplicated corpus: languages, code vs. instruction-only, size (PLAN.md §5.2).

Writes corpus/profile.jsonl (one row per canonical skill) and returns summary tables
for the paper's corpus statistics. The category classifier (LLM) is added in M2.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np

from skillalign.collect.store import CorpusStore, Manifest
from skillalign.models import FileRole
from skillalign.normalize.languages import DEEP_LANGS
from skillalign.normalize.loader import load_skill


def profile_row(store: CorpusStore, m: Manifest) -> dict:
    s = load_skill(store.skill_dir(m.key), skill_id=m.skill_id, platform=m.platform)
    langs = s.languages
    real_code = [f for f in s.files(FileRole.CODE)]
    return {
        "key": m.key, "skill_id": m.skill_id, "platform": m.platform, "repo": m.repo,
        "license": m.license, "stars": m.repo_stars,
        "n_files": len(m.files), "n_code_files": len(real_code),
        "has_code_files": bool(real_code),
        "has_inline_code": any(f.role is FileRole.VIRTUAL for f in s.skill.files),
        "instruction_only": not real_code,
        "loc": langs, "loc_total": sum(langs.values()),
        "deep_lang_only": all(lang in DEEP_LANGS for lang in langs),
        "skill_md_lines": len(s.texts.get(s.skill_md or "", "").splitlines()),
        "n_doc_files": len(s.files(FileRole.DOC)),
        "n_config_files": len(s.files(FileRole.CONFIG)),
        "n_html_comments": len(s.comments),
        "allowed_tools": s.allowed_tools,
        "frontmatter_keys": sorted((s.frontmatter or {}).keys()),
    }


def profile(store: CorpusStore, keys: list[str], out: Path) -> dict:
    by_key = {m.key: m for m in store.manifests()}
    rows = [profile_row(store, by_key[k]) for k in keys]
    with out.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return summarize(rows)


def summarize(rows: list[dict]) -> dict:
    n = len(rows) or 1
    lang_skills = Counter(lang for r in rows for lang in r["loc"])
    loc = np.array([r["loc_total"] for r in rows if r["loc_total"]] or [0])
    return {
        "skills": len(rows),
        "by_platform": dict(Counter(r["platform"] for r in rows)),
        "with_code_files": sum(r["has_code_files"] for r in rows),
        "with_inline_code": sum(r["has_inline_code"] for r in rows),
        "instruction_only": sum(r["instruction_only"] for r in rows),
        "share_deep_lang_only": round(sum(r["deep_lang_only"] for r in rows) / n, 3),
        "skills_per_language": dict(lang_skills.most_common()),
        "loc_percentiles_p25_p50_p75_p95": [int(np.percentile(loc, q)) for q in (25, 50, 75, 95)],
        "with_html_comments": sum(r["n_html_comments"] > 0 for r in rows),
        "with_allowed_tools": sum(bool(r["allowed_tools"]) for r in rows),
        "licenses": dict(Counter(r["license"] for r in rows).most_common()),
    }
