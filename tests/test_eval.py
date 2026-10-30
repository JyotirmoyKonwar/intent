import json
from pathlib import Path

import pytest

from skillalign.eval.match import match_skill
from skillalign.eval.metrics import confusion_matrix, control_fpr, evaluate
from skillalign.eval.stats import bootstrap_ci, mcnemar
from skillalign.models import Annotation, FindingType, SkillLabel

GOLD_DIR = Path(__file__).parent / "fixtures" / "gold"


def load_golds() -> dict[str, Annotation]:
    out = {}
    for p in sorted(GOLD_DIR.glob("*.json")):
        a = Annotation.model_validate(json.loads(p.read_text()))
        out[a.skill_id] = a
    return out


def retype(ann: Annotation, finding_id: str, new_type: FindingType) -> Annotation:
    """A copy of `ann` whose finding `finding_id` has another type (and valid fields)."""
    data = ann.model_dump(mode="json")
    for f in data["findings"]:
        if f["id"] == finding_id:
            f["type"] = new_type.value
            if new_type is not FindingType.SD:
                f["divergence_axis"] = None
            if new_type is FindingType.UB:
                f["intent"] = None
    return Annotation.model_validate(data)


def test_gold_against_itself_is_perfect():
    golds = load_golds()
    ev = evaluate(golds, golds)
    assert ev.skill_level.f1 == 1.0
    assert all(c.fp == 0 and c.fn == 0 for c in ev.per_type().values())
    assert ev.mean_iou() == 1.0
    assert ev.axis_accuracy() == 1.0
    assert control_fpr(golds, golds) == 0.0


def test_missing_predictions_are_all_false_negatives():
    golds = load_golds()
    ev = evaluate(golds, {})
    n_find = sum(len(g.findings) for g in golds.values())
    assert sum(c.fn for c in ev.per_type().values()) == n_find
    assert ev.skill_level.recall == 0.0 and ev.skill_level.fp == 0


def test_type_confusion_strict_vs_lenient():
    golds = load_golds()
    gold = golds["fixture/11-sd-destination"]
    pred = retype(gold, "F1", FindingType.UB)
    strict = match_skill(gold, pred, lenient=False)
    assert not strict.pairs and len(strict.false_neg) == 1 and len(strict.false_pos) == 1
    lenient = match_skill(gold, pred, lenient=True)
    assert len(lenient.pairs) == 1 and lenient.pairs[0].wrong_type
    ev = evaluate({gold.skill_id: gold}, {gold.skill_id: pred}, lenient=True)
    assert ev.per_type()["SD"].tp == 1
    cm = confusion_matrix({gold.skill_id: gold}, {gold.skill_id: pred})
    assert cm["SD"]["UB"] == 1


def test_wrong_axis_is_scored_separately():
    gold = load_golds()["fixture/11-sd-destination"]
    data = gold.model_dump(mode="json")
    data["findings"][0]["divergence_axis"] = "scope"
    pred = Annotation.model_validate(data)
    [pair] = match_skill(gold, pred).pairs
    assert pair.axis_ok is False and not pair.wrong_type


def test_spurious_finding_on_aligned_skill_is_a_false_positive():
    golds = load_golds()
    aligned = golds["fixture/01-aligned-csv-to-json"]
    bad = golds["fixture/11-sd-destination"]
    pred = aligned.model_copy(update={"findings": [bad.findings[0]],
                                      "skill_label": SkillLabel.MISALIGNED})
    ev = evaluate({aligned.skill_id: aligned}, {aligned.skill_id: pred})
    assert ev.skill_level.fp == 1
    assert control_fpr({aligned.skill_id: aligned}, {aligned.skill_id: pred}) == 1.0


def test_intent_fallback_matches_on_action_and_object():
    gold = load_golds()["fixture/05-ui-pdf-encrypt"]
    data = gold.model_dump(mode="json")
    ui = next(f for f in data["findings"] if f["type"] == "UI")
    intent = next(i for i in data["intents"] if i["id"] == ui["intent"])
    for s in intent["sources"]:
        s["line_start"] = s["line_end"] = 1  # no longer overlaps the gold spans
        s["file"] = "elsewhere.md"
    pred = Annotation.model_validate(data)
    pairs = match_skill(gold, pred).pairs
    assert any(p.gold.id == ui["id"] for p in pairs)


def test_bootstrap_and_mcnemar():
    point, lo, hi = bootstrap_ci([1.0] * 20, lambda xs: sum(xs) / len(xs))
    assert point == lo == hi == 1.0
    point, lo, hi = bootstrap_ci([0, 1] * 20, lambda xs: sum(xs) / len(xs))
    assert lo < point < hi
    b, c, p = mcnemar([True] * 10 + [False] * 2, [False] * 10 + [True] * 2)
    assert (b, c) == (10, 2) and p == pytest.approx(0.0386, abs=1e-3)
    assert mcnemar([True, False], [True, False]) == (0, 0, 1.0)
    with pytest.raises(ValueError):
        mcnemar([True], [True, False])


def test_confidence_intervals_from_evaluation():
    golds = load_golds()
    ev = evaluate(golds, golds)
    assert ev.finding_f1_ci(n_boot=50) == (1.0, 1.0, 1.0)
    assert ev.finding_f1_ci("UB", n_boot=50)[0] == 1.0
    assert ev.skill_f1_ci(n_boot=50)[0] == 1.0
    assert json.dumps(ev.summary())
