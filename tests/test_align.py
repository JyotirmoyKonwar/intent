"""D1-D4, the judge and the pipeline, on the fixtures, with a scripted judge (no GPU)."""

import json
import re
from pathlib import Path

import pytest

from skillalign.align.candidates import candidates
from skillalign.align.cards import build_cards, render_card
from skillalign.align.implied import table_says
from skillalign.align.judge import JudgeResult, combine, judge_intent
from skillalign.align.pipeline import DetectOptions, detect_skill
from skillalign.align.report import render_report, report_json
from skillalign.align.rules import behavior_risk, decide
from skillalign.behaviors import detect_behaviors
from skillalign.eval.metrics import evaluate
from skillalign.models import (
    Annotation,
    Capability,
    ChecklistAnswer,
    ConstraintAnswer,
    FindingType,
    Polarity,
    Reachability,
    Risk,
    SkillLabel,
)
from skillalign.normalize.loader import load_skill
from skillalign.spans import check_annotation

from .fake_llm import Router, client_for

FIX = Path(__file__).parent / "fixtures"


def gold(name: str) -> Annotation:
    return Annotation.model_validate_json((FIX / "gold" / f"{name}.json").read_text())


def cards_in(prompt: str) -> dict[str, str]:
    """`{card id: capability}` for the cards printed in a judge prompt."""
    return dict(re.findall(r"^\[(B\d+)\] (\w+)", prompt, re.M))


def scripted_judge(prompt: str):
    """Answers like a good judge: cites what carries the claim out, flags a destination
    mismatch for the 'save locally' claim and a contradiction for the offline claim."""
    cards = cards_in(prompt)
    if "implied_d3" in prompt or "would expect" in prompt or "expected" in prompt[:200]:
        return json.dumps({"expected": False}), "stop"
    action = re.search(r"action: (.*)", prompt).group(1)
    negative = "polarity: negative" in prompt
    net = [b for b, c in cards.items() if c == "NET_OUTBOUND"]
    comp = [b for b, c in cards.items() if c == "COMPOSITE"]
    ans = {"achieves_action": "yes", "constraints": [], "axis_mismatch": None,
           "contradicts_negative": False, "cited_cards": list(cards)[:1]}
    if negative:
        ans.update(contradicts_negative=bool(net), cited_cards=net[:1] or [])
        ans["achieves_action"] = "yes" if net else "no"
    elif "save" in action and net:
        ans.update(axis_mismatch="destination", cited_cards=list(cards))
    elif not cards:
        ans.update(achieves_action="no", cited_cards=[])
    if comp and not negative:
        ans["cited_cards"] = list(cards)[:1]
    return json.dumps(ans), "stop"


def run(name: str, opts: DetectOptions | None = None, route=scripted_judge):
    router = Router(route)
    res = detect_skill(client_for(router), FIX / "skills" / name,
                       opts or DetectOptions(votes=1, prose=False),
                       skill_id=gold(name).skill_id, intents=gold(name).intents)
    return res, router


# ---------------------------------------------------------------- cards and candidates


def test_cards_show_summary_chain_and_snippet():
    loaded = load_skill(FIX / "skills" / "07-ub-exfil-crossfile")
    behaviors = detect_behaviors(loaded).behaviors
    cards = build_cards(loaded, behaviors)
    comp = next(c for c in cards.values() if "COMPOSITE" in c.text)
    text = render_card(comp)
    assert "composite of: B4 (ENV_READ), B5 (ENCODE_OBFUSCATE), B1 (NET_OUTBOUND)" in text
    assert "origin=code" in text and "18|" in text.replace(" ", "")[:400] or "|" in text
    assert all(len(c.snippet.splitlines()) <= 14 for c in cards.values())
    raw = build_cards(loaded, behaviors, mode="raw")
    assert "origin=" not in render_card(next(iter(raw.values())))


def test_candidates_use_expected_capabilities_and_skip_dormant():
    g = gold("11-sd-destination")
    behaviors = detect_behaviors(load_skill(FIX / "skills" / "11-sd-destination")).behaviors
    save = g.intent("I2")
    assert set(candidates(save, behaviors, top_k=0)) == {"B3", "B4", "B5"}  # B5: the upload
    dormant = [b.model_copy(update={"reachability": Reachability.DORMANT}) if b.id == "B4" else b
               for b in behaviors]
    assert "B4" not in candidates(save, dormant, top_k=0)


def test_negative_claim_candidates_include_composites():
    g = gold("07-ub-exfil-crossfile")
    behaviors = detect_behaviors(load_skill(FIX / "skills" / "07-ub-exfil-crossfile")).behaviors
    neg = g.intents[0].model_copy(update={
        "polarity": Polarity.NEGATIVE, "expected_capabilities": [Capability.NET_OUTBOUND]})
    assert {"B1", "B6"} <= set(candidates(neg, behaviors, top_k=0))


# ---------------------------------------------------------------- judge voting


def answer(achieves="yes", cited=("B1",), axis=None, contra=False, cons=()):
    return ChecklistAnswer(achieves_action=achieves, axis_mismatch=axis,
                           contradicts_negative=contra, cited_cards=list(cited),
                           constraints=[ConstraintAnswer(constraint="c", respected=r)
                                        for r in cons])


def test_combine_takes_the_majority_and_reports_agreement():
    intent = gold("11-sd-destination").intent("I2").model_copy(update={"constraints": ["c"]})
    got, agreement = combine([answer(cons=["yes"]), answer("partly", cons=["no"]),
                              answer(cons=["yes"])], intent)
    assert got.achieves_action == "yes" and got.constraints[0].respected == "yes"
    assert agreement == pytest.approx(2 / 3)


def test_judge_drops_samples_that_cite_missing_cards():
    loaded = load_skill(FIX / "skills" / "11-sd-destination")
    behaviors = detect_behaviors(loaded).behaviors
    cards = build_cards(loaded, behaviors)
    intent = gold("11-sd-destination").intent("I2")
    def judged(reply: ChecklistAnswer):
        router = Router(lambda p: (reply.model_dump_json(), "stop"))
        return judge_intent(client_for(router), intent, ["B3", "B4"], cards, votes=1)

    r = judged(answer(cited=["B9"]))  # B9 is not a candidate
    assert r.answer is None and "unknown cards" in r.note
    r = judged(answer(cited=["B3"]))
    assert r.answer and r.answer.cited_cards == ["B3"]
    assert judged(answer(cited=[])).answer is None  # "yes" with no card is not accepted


def test_no_candidates_means_no_without_asking_the_model():
    router = Router(lambda p: ("{}", "stop"))
    r = judge_intent(client_for(router), gold("11-sd-destination").intent("I2"), [], {})
    assert r.answer.achieves_action == "no" and not router.requests


def test_holistic_verdict_maps_to_the_same_answer():
    loaded = load_skill(FIX / "skills" / "11-sd-destination")
    cards = build_cards(loaded, detect_behaviors(loaded).behaviors)
    router = Router(lambda p: (json.dumps({"verdict": "divergent", "axis_mismatch": "scope",
                                           "cited_cards": ["B3"]}), "stop"))
    r = judge_intent(client_for(router), gold("11-sd-destination").intent("I2"), ["B3"],
                     cards, votes=1, holistic=True)
    assert r.answer.achieves_action == "yes" and r.answer.axis_mismatch.value == "scope"


# ---------------------------------------------------------------- D4 rules


def results(**by_intent: ChecklistAnswer) -> dict[str, JudgeResult]:
    return {i: JudgeResult(i, a, 1.0, list(a.cited_cards), 1) for i, a in by_intent.items()}


def test_rules_ui_partial_sd_and_ub():
    g = gold("11-sd-destination")
    b = detect_behaviors(load_skill(FIX / "skills" / "11-sd-destination")).behaviors
    _, f = decide(g.intents, b, results(I1=answer("no", []), I2=answer("partly", ["B3"])), {})
    types = {x.type for x in f}
    assert FindingType.UI in types and FindingType.UI_PARTIAL in types
    assert FindingType.UB in types  # B1/B2/B5 are left over
    _, f = decide(g.intents, b, results(I1=answer(cited=["B1", "B2"]), I2=answer(
        cited=["B3", "B4", "B5"], axis="destination")), {})
    [sd] = [x for x in f if x.type is FindingType.SD]
    assert sd.divergence_axis.value == "destination" and not [
        x for x in f if x.type is FindingType.UB]


def test_constraint_no_is_ui_partial_and_unknown_is_not():
    g = gold("11-sd-destination")
    intent = g.intent("I2").model_copy(update={"constraints": ["c"]})
    b = detect_behaviors(load_skill(FIX / "skills" / "11-sd-destination")).behaviors
    for respected, expect in (("no", 1), ("unknown", 0)):
        _, f = decide([intent], b, results(I2=answer(cited=["B3"], cons=[respected])), {})
        assert len([x for x in f if x.type is FindingType.UI_PARTIAL]) == expect


def test_dormant_behavior_is_not_ub_and_implied_is_not_ub():
    g = gold("11-sd-destination")
    b = detect_behaviors(load_skill(FIX / "skills" / "11-sd-destination")).behaviors
    b = [x.model_copy(update={"reachability": Reachability.DORMANT}) if x.id == "B5" else x
         for x in b]
    res = results(I1=answer(cited=["B1"]), I2=answer(cited=["B3"]))
    _, f = decide(g.intents, b, res, {"B2": True, "B4": True})
    assert f == []


def test_composite_parts_are_not_reported_twice():
    g = gold("07-ub-exfil-crossfile")
    b = detect_behaviors(load_skill(FIX / "skills" / "07-ub-exfil-crossfile")).behaviors
    res = results(**{i.id: answer("yes", ["B2"]) for i in g.intents})
    _, f = decide(g.intents, b, res, {"B3": True, "B2": True})
    ub = [x for x in f if x.type is FindingType.UB]
    assert [x.behaviors for x in ub] == [["B6"]]
    assert ub[0].risk in (Risk.CRITICAL, Risk.HIGH) and ub[0].intent_suspected.value == "deliberate"


def test_risk_levels():
    b = detect_behaviors(load_skill(FIX / "skills" / "07-ub-exfil-crossfile")).behaviors
    by_id = {x.id: x for x in b}
    assert behavior_risk(by_id["B3"], by_id) is Risk.LOW
    assert behavior_risk(by_id["B6"], by_id) is Risk.CRITICAL
    dormant = by_id["B1"].model_copy(update={"reachability": Reachability.DORMANT})
    assert behavior_risk(dormant, by_id) is Risk.LOW
    assert behavior_risk(by_id["B1"], by_id) is Risk.HIGH


def test_implied_table_rows():
    b = detect_behaviors(load_skill(FIX / "skills" / "01-aligned-csv-to-json")).behaviors
    assert all(table_says("file-conversion", x) for x in b)
    net = detect_behaviors(load_skill(FIX / "skills" / "11-sd-destination")).behaviors[-1]
    assert not table_says("data-analysis", net)
    assert table_says("web-fetch", net.model_copy(update={"destination": None}))
    assert not table_says("web-fetch", net)


# ---------------------------------------------------------------- pipeline end to end


@pytest.mark.parametrize("name", ["11-sd-destination", "12-sd-negative-claim",
                                  "01-aligned-csv-to-json", "07-ub-exfil-crossfile"])
def test_pipeline_matches_the_gold_label(name):
    res, _ = run(name)
    g = gold(name)
    assert res.annotation.skill_label is g.skill_label
    assert check_annotation(res.annotation, FIX / "skills" / name) == []
    ev = evaluate({g.skill_id: g}, {g.skill_id: res.annotation})
    assert ev.skill_level.fp == 0 and ev.skill_level.fn == 0


def test_pipeline_finds_sd_and_negative_claim():
    res, _ = run("11-sd-destination")
    [sd] = [f for f in res.annotation.findings if f.type is FindingType.SD]
    assert sd.divergence_axis.value == "destination"
    res, _ = run("12-sd-negative-claim")
    [sd] = [f for f in res.annotation.findings if f.type is FindingType.SD]
    assert sd.divergence_axis.value == "negative-claim"
    assert res.annotation.skill_label is SkillLabel.MISALIGNED


def test_ablations_change_the_output():
    full, _ = run("07-ub-exfil-crossfile")
    no_comp, _ = run("07-ub-exfil-crossfile", DetectOptions(votes=1, prose=False,
                                                            composite=False))
    assert not any(b.capability is Capability.COMPOSITE for b in no_comp.annotation.behaviors)
    assert any(b.capability is Capability.COMPOSITE for b in full.annotation.behaviors)
    no_table, _ = run("01-aligned-csv-to-json", DetectOptions(votes=1, prose=False,
                                                              implied_table=False))
    assert no_table.annotation.skill_label in SkillLabel
    raw, router = run("11-sd-destination", DetectOptions(votes=1, prose=False, cards="raw"))
    assert all("origin=" not in r["messages"][0]["content"] for r in router.requests)


def test_report_json_and_markdown():
    res, _ = run("11-sd-destination")
    data = report_json(res)
    assert Annotation.model_validate(data["annotation"]).skill_id == res.annotation.skill_id
    md = render_report(res)
    assert "SD (destination)" in md and "paste.canary.local" in md and "Alignment" in md
    assert "provenance" not in md
    assert res.annotation.provenance.system == "skillalign"
    assert res.annotation.provenance.prompt_hash


def test_unverifiable_claims_are_not_reported_as_ui(tmp_path):
    (tmp_path / "SKILL.md").write_text(
        "---\nname: g\ndescription: Guides the agent.\n---\n\n# G\n\nSummarize the notes.\n")
    intents = gold("11-sd-destination").intents[:1]  # a claim, but the skill has no code
    router = Router(lambda p: ("{}", "stop"))
    res = detect_skill(client_for(router), tmp_path, DetectOptions(votes=1, prose=False),
                       skill_id="x", intents=intents)
    assert res.annotation.findings == [] and not router.requests
    assert res.judged["I1"].note == "not verifiable"


def test_running_the_skills_own_script_is_not_a_verifiable_claim():
    from skillalign.align.pipeline import verifiable
    run_own = gold("01-aligned-csv-to-json").intents[0].model_copy(update={
        "action": "Run", "object": "`python scripts/convert.py data.csv`",
        "expected_capabilities": [Capability.PROC_EXEC]})
    assert not verifiable(run_own, True)
    other = run_own.model_copy(update={"object": "the linter"})
    assert verifiable(other, True)
