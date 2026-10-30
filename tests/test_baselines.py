from pathlib import Path

from skillalign.align.similarity import LexicalEmbedder
from skillalign.baselines.m1 import M1Config, build_m1, run_m1, tune
from skillalign.baselines.r2 import doc_intents, run_r2
from skillalign.behaviors import detect_behaviors
from skillalign.eval.metrics import evaluate
from skillalign.models import Annotation, FindingType, SkillLabel
from skillalign.normalize.loader import load_skill
from skillalign.spans import check_annotation

FIX = Path(__file__).parent / "fixtures"
NAMES = sorted(p.name for p in (FIX / "skills").iterdir())


def golds() -> dict[str, Annotation]:
    out = {}
    for p in sorted((FIX / "gold").glob("*.json")):
        a = Annotation.model_validate_json(p.read_text())
        out[a.skill_id] = a
    return out


def test_similarity_orders_related_text_first():
    sim = LexicalEmbedder().similarity(["downloads a file from a URL"],
                                       ["download file from url", "format markdown"])
    assert sim[0, 0] > sim[0, 1] and sim[0, 1] == 0


def test_r2_output_is_valid_on_every_fixture():
    for name in NAMES:
        ann = run_r2(FIX / "skills" / name, skill_id=f"fixture/{name}")
        assert ann.provenance.system == "baseline:R2"
        assert check_annotation(ann, FIX / "skills" / name) == [], name
        assert {f.type for f in ann.findings} <= {FindingType.UI, FindingType.UB}


def test_r2_flags_exfiltration_and_leaves_aligned_csv_alone():
    ub = run_r2(FIX / "skills" / "07-ub-exfil-crossfile")
    assert ub.skill_label is SkillLabel.MISALIGNED
    assert any(f.type is FindingType.UB for f in ub.findings)
    ok = run_r2(FIX / "skills" / "01-aligned-csv-to-json")
    assert ok.skill_label is SkillLabel.ALIGNED


def test_r2_reports_a_doc_verb_with_no_behavior_as_ui(tmp_path):
    (tmp_path / "SKILL.md").write_text(
        "---\nname: grab\ndescription: Grab pages.\n---\n\n# Grab\n\n"
        "Downloads the page at the given URL.\nWrites it to out.html.\n")
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "grab.py").write_text(
        "from pathlib import Path\nPath('out.html').write_text('<html></html>')\n")
    ann = run_r2(tmp_path)
    ui = [f for f in ann.findings if f.type is FindingType.UI]
    assert [ann.intent(f.intent).action for f in ui] == ["download"]
    assert check_annotation(ann, tmp_path) == []


def test_doc_intents_have_valid_sources():
    loaded = load_skill(FIX / "skills" / "01-aligned-csv-to-json")
    intents = doc_intents(loaded)
    assert intents and all(i.expected_capabilities for i in intents)


def test_m1_runs_on_every_fixture_and_thresholds_matter():
    for name in NAMES:
        ann = run_m1(FIX / "skills" / name, skill_id=f"fixture/{name}")
        assert ann.provenance.system == "baseline:M1"
        assert check_annotation(ann, FIX / "skills" / name) == [], name
    strict = run_m1(FIX / "skills" / "01-aligned-csv-to-json", M1Config(0.05, 0.99))
    lenient = run_m1(FIX / "skills" / "01-aligned-csv-to-json", M1Config(0.01, 0.02))
    assert len(strict.findings) >= len(lenient.findings)


def test_m1_tune_picks_the_best_config_and_scores_with_the_harness():
    g = golds()
    items = {}
    for name in NAMES:
        loaded = load_skill(FIX / "skills" / name)
        items[f"fixture/{name}"] = (g[f"fixture/{name}"].intents,
                                    detect_behaviors(loaded).behaviors)
    cfg, score = tune(items, g)
    assert 0 <= score <= 1 and cfg.match > cfg.low
    preds = {sid: build_m1(sid, i, b, cfg, LexicalEmbedder()) for sid, (i, b) in items.items()}
    assert evaluate(g, preds).macro_f1() == score
