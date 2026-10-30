import ast
import json
import re
from pathlib import Path

import pytest

from skillalign.behaviors import detect_behaviors
from skillalign.bench.inject import OPERATORS, NotApplicable, build_set, inject
from skillalign.eval.metrics import evaluate
from skillalign.models import Annotation, Capability, FindingType, SkillLabel
from skillalign.normalize.loader import load_skill
from skillalign.spans import check_annotation

FIX = Path(__file__).parent / "fixtures"


def base(name: str) -> tuple[Path, Annotation]:
    gold = Annotation.model_validate_json((FIX / "gold" / f"{name}.json").read_text())
    return FIX / "skills" / name, gold


def run(op, name, tmp_path, templates="train", seed=13):
    d, g = base(name)
    return inject(op, d, g, tmp_path / "out", seed=seed, templates=templates)


def caps(path: Path):
    return [(b.capability, b.destination, b.reachability.value)
            for b in detect_behaviors(load_skill(path)).behaviors]


APPLICABLE = [
    ("DROP_IMPL", "01-aligned-csv-to-json", FindingType.UI),
    ("DROP_VARIANT", "01-aligned-csv-to-json", FindingType.UI_PARTIAL),
    ("ADD_SINK_SPLIT", "01-aligned-csv-to-json", FindingType.UB),
    ("ADD_INSTRUCTION", "01-aligned-csv-to-json", FindingType.UB),
    ("RETARGET", "02-aligned-weather-api", FindingType.SD),
    ("NEGATE_CLAIM", "02-aligned-weather-api", FindingType.SD),
]


@pytest.mark.parametrize("templates", ["train", "test"])
@pytest.mark.parametrize(("op", "name", "ftype"), APPLICABLE)
def test_operator_gold_is_valid_and_has_the_expected_finding(op, name, ftype, templates,
                                                             tmp_path):
    inj = run(op, name, tmp_path, templates)
    ann = inj.annotation
    assert [f.type for f in ann.findings] == [ftype]
    assert ann.skill_label is SkillLabel.MISALIGNED
    assert ann.skill_id.startswith(f"injected/{op.lower()}/") and ann.split.value == "injected"
    assert check_annotation(ann, tmp_path / "out") == []
    for rel in inj.changed:
        if rel.endswith(".py"):
            ast.parse((tmp_path / "out" / rel).read_text())
    ev = evaluate({ann.skill_id: ann}, {ann.skill_id: ann})
    assert ev.skill_level.f1 == 1.0


@pytest.mark.parametrize("templates", ["train", "test"])
def test_every_written_host_is_a_canary(templates, tmp_path):
    for i, (op, name, _) in enumerate(APPLICABLE):
        d, g = base(name)
        inj = inject(op, d, g, tmp_path / f"o{i}", templates=templates)
        for rel in inj.changed:
            text = (tmp_path / f"o{i}" / rel).read_text()
            hosts = re.findall(r"https?://([\w.-]+)|HTTPSConnection\(\"([\w.-]+)\"", text)
            for host in (h for pair in hosts for h in pair if h):
                assert host.endswith(".canary.local") or host == "api.openweathermap.org", host


@pytest.mark.parametrize("templates", ["train", "test"])
def test_split_sink_is_found_as_a_composite_by_the_detectors(templates, tmp_path):
    inj = run("ADD_SINK_SPLIT", "01-aligned-csv-to-json", tmp_path, templates)
    found = caps(tmp_path / "out")
    assert any(c is Capability.COMPOSITE and dest and dest.endswith(".canary.local")
               for c, dest, _ in found), found
    assert len([c for c, _, _ in found if c is Capability.COMPOSITE]) == 1
    assert len(inj.changed) >= 3


def test_hidden_instruction_is_found_by_the_detectors(tmp_path):
    run("ADD_INSTRUCTION", "01-aligned-csv-to-json", tmp_path, "test")
    assert any(c is Capability.NET_OUTBOUND and dest == "beacon.canary.local"
               for c, dest, _ in caps(tmp_path / "out"))


def test_drop_impl_removes_the_behaviors_but_keeps_the_file_valid(tmp_path):
    inj = run("DROP_IMPL", "01-aligned-csv-to-json", tmp_path)
    assert not [c for c, _, _ in caps(tmp_path / "out")
                if c in (Capability.FS_READ, Capability.FS_WRITE)]
    assert inj.annotation.alignment[0].status.value == "unfulfilled"
    assert inj.annotation.behaviors == []


def test_drop_variant_leaves_one_behavior_and_marks_partial(tmp_path):
    inj = run("DROP_VARIANT", "01-aligned-csv-to-json", tmp_path)
    assert len(inj.annotation.behaviors) == 1
    assert inj.annotation.alignment[0].status.value == "partial"
    assert len(caps(tmp_path / "out")) == 1


def test_retarget_changes_the_destination(tmp_path):
    inj = run("RETARGET", "02-aligned-weather-api", tmp_path)
    assert any(dest and dest.endswith("mirror.canary.local")
               for _, dest, _ in caps(tmp_path / "out"))
    [f] = inj.annotation.findings
    assert f.divergence_axis.value == "destination" and f.intent == "I1"


def test_negate_claim_adds_a_negative_intent_with_the_network_behavior(tmp_path):
    inj = run("NEGATE_CLAIM", "02-aligned-weather-api", tmp_path)
    new = inj.annotation.intents[-1]
    assert new.polarity.value == "negative" and new.id == "I3"
    [f] = inj.annotation.findings
    assert f.divergence_axis.value == "negative-claim" and f.behaviors == ["B2"]


@pytest.mark.parametrize("templates", ["train", "test"])
def test_vague_doc_is_a_control_with_no_findings(templates, tmp_path):
    inj = run("VAGUE_DOC", "01-aligned-csv-to-json", tmp_path, templates)
    ann = inj.annotation
    assert ann.findings == [] and ann.skill_label is SkillLabel.ALIGNED
    assert "csv" not in load_skill(tmp_path / "out").description.lower()
    assert [s.line_start for i in ann.intents for s in i.sources] == [8]  # description gone
    assert check_annotation(ann, tmp_path / "out") == []


def test_not_applicable_leaves_nothing_behind(tmp_path):
    d, g = base("01-aligned-csv-to-json")
    for op in ("RETARGET", "NEGATE_CLAIM"):  # 01 makes no network call
        with pytest.raises(NotApplicable):
            inject(op, d, g, tmp_path / op)
        assert not (tmp_path / op).exists()
    bad = Annotation.model_validate_json(
        (FIX / "gold" / "11-sd-destination.json").read_text())
    with pytest.raises(NotApplicable):
        inject("VAGUE_DOC", FIX / "skills" / "11-sd-destination", bad, tmp_path / "x")
    with pytest.raises(KeyError):
        inject("NOPE", d, g, tmp_path / "y")
    with pytest.raises(FileExistsError):
        (tmp_path / "z").mkdir()
        inject("VAGUE_DOC", d, g, tmp_path / "z")


def test_same_seed_same_output_and_templates_differ(tmp_path):
    a = run("ADD_SINK_SPLIT", "01-aligned-csv-to-json", tmp_path / "a")
    b = run("ADD_SINK_SPLIT", "01-aligned-csv-to-json", tmp_path / "b")
    assert a.annotation == b.annotation
    train = run("ADD_INSTRUCTION", "01-aligned-csv-to-json", tmp_path / "c", "train")
    test = run("ADD_INSTRUCTION", "01-aligned-csv-to-json", tmp_path / "d", "test")
    assert (tmp_path / "c/out/SKILL.md").read_text() != (tmp_path / "d/out/SKILL.md").read_text()
    assert train.annotation.behaviors[-2].destination != test.annotation.behaviors[-2].destination


def test_build_set_writes_skills_gold_and_manifest(tmp_path):
    bases = [base("01-aligned-csv-to-json"), base("02-aligned-weather-api")]
    rows = build_set(bases, tmp_path, templates="test")
    assert {r["operator"] for r in rows} == set(OPERATORS)
    ok = [r for r in rows if r["status"] == "ok"]
    assert ok and all((tmp_path / "skills" / r["stem"]).is_dir() for r in ok)
    for r in ok:
        gold = Annotation.model_validate_json((tmp_path / "gold" / f"{r['stem']}.json").read_text())
        assert check_annotation(gold, tmp_path / "skills" / r["stem"]) == []
    assert any(r["status"] == "skipped" and r["operator"] == "RETARGET" for r in rows)
    assert json.loads((tmp_path / "manifest.json").read_text()) == rows
