import json

from skillalign.bench.render import render_markdown
from skillalign.bench.sample import draw_sample, is_enriched, write_sample
from skillalign.models import Annotation, Evidence

from .conftest import GOLD, SKILLS


def fake_profile(n: int) -> list[dict]:
    return [{"key": f"k{i:03d}", "skill_id": f"github/o/r{i}", "has_code_files": i % 3 == 0,
             "has_inline_code": False} for i in range(n)]


def test_draw_sample_shape_and_strata():
    profile = fake_profile(300)
    keys = [r["key"] for r in profile]
    entries = draw_sample(keys, profile, seed=13)
    random_part = [e for e in entries if e["stratum"] == "random"]
    enriched_part = [e for e in entries if e["stratum"] == "enriched"]
    assert len(random_part) == 60 and len(enriched_part) == 20
    assert not {e["key"] for e in random_part} & {e["key"] for e in enriched_part}
    rows = {r["key"]: r for r in profile}
    assert all(is_enriched(rows[e["key"]]) for e in enriched_part)
    assert entries[0]["path"] == f"skills/{entries[0]['key']}"


def test_draw_sample_is_deterministic_and_seed_sensitive():
    profile = fake_profile(300)
    keys = [r["key"] for r in profile]
    assert draw_sample(keys, profile, 13) == draw_sample(keys, profile, 13)
    assert draw_sample(keys, profile, 13) != draw_sample(keys, profile, 14)


def test_write_sample_reads_corpus_files(tmp_path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    profile = fake_profile(300)
    (corpus / "dedup.json").write_text(json.dumps({"canonical": [r["key"] for r in profile]}))
    (corpus / "profile.jsonl").write_text("\n".join(json.dumps(r) for r in profile) + "\n")
    out = tmp_path / "bench" / "sample.json"
    entries = write_sample(corpus, out, 13)
    assert json.loads(out.read_text()) == entries
    assert len(entries) == 80


def test_render_quotes_every_evidence_quote():
    ann = Annotation.model_validate_json((GOLD / "11-sd-destination.json").read_text())
    md = render_markdown(ann, SKILLS / "11-sd-destination")
    assert f"# {ann.skill_id}" in md
    quotes = [ev.quote for i in ann.intents for ev in i.sources]
    quotes += [ev.quote for b in ann.behaviors for ev in b.evidence]
    quotes += [ev.quote for f in ann.findings for ev in f.evidence if isinstance(ev, Evidence)]
    assert quotes
    for q in quotes:
        assert q in md
    assert "Lines:" in md
