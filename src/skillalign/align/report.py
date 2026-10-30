"""E: the report. JSON is the `Annotation` plus run details; Markdown is for a person."""

from __future__ import annotations

from skillalign.align.cards import render_card
from skillalign.align.pipeline import DetectResult
from skillalign.models import FindingType

_TITLES = {
    FindingType.UI: "Unimplemented intent",
    FindingType.UI_PARTIAL: "Partly implemented intent",
    FindingType.UB: "Undeclared behavior",
    FindingType.SD: "Semantically divergent behavior",
}


def report_json(res: DetectResult) -> dict:
    return {
        "annotation": res.annotation.model_dump(mode="json"),
        "judge": {i: {"agreement": r.agreement, "samples": r.samples,
                      "candidates": r.candidates, "note": r.note}
                  for i, r in res.judged.items()},
        "implied": res.implied,
        "dormant": res.dormant,
        "cards": {k: v.model_dump(mode="json") for k, v in res.cards.items()},
        "notes": res.notes,
    }


def _excerpt(res: DetectResult, file: str, start: int, end: int, limit: int = 8) -> str:
    text = res.loaded.texts.get(file)
    if text is None:
        return ""
    lines = text.splitlines()[start - 1:min(end, start + limit - 1)]
    return "\n".join(f"{start + n}| {ln}" for n, ln in enumerate(lines))


def render_report(res: DetectResult) -> str:
    ann = res.annotation
    out = [f"# SkillAlign report: {ann.skill_id}", "",
           f"**Verdict: {ann.skill_label.value}** · category `{ann.category}` · "
           f"{len(ann.intents)} intents · {len(ann.behaviors)} behaviors · "
           f"{len(ann.findings)} findings", ""]
    for f in ann.findings:
        axis = f" ({f.divergence_axis.value})" if f.divergence_axis else ""
        out += [f"## {f.id} {f.type.value}{axis}: {_TITLES[f.type]}",
                f"risk **{f.risk.value}** · suspected {f.intent_suspected.value}", ""]
        if f.intent:
            it = ann.intent(f.intent)
            out.append(f"- Claim ({f.intent}): {it.action} {it.object}"
                       + "".join(f"; {c}" for c in it.constraints))
        if f.expected_result:
            out.append(f"- Expected: {f.expected_result}")
        if f.actual_result:
            out.append(f"- Actual: {f.actual_result}")
        if f.call_chain:
            out.append(f"- Call chain: {' → '.join(f.call_chain)}")
        out += [f"- Why: {f.rationale}", "", "Evidence:", ""]
        for s in f.evidence:
            out += [f"`{s.file}:{s.line_start}-{s.line_end}`", "```",
                    _excerpt(res, s.file, s.line_start, s.line_end), "```"]
        for b in f.behaviors:
            out += ["", "```", render_card(res.cards[b]), "```"]
        out.append("")
    if not ann.findings:
        out += ["No findings.", ""]
    if res.dormant:
        out += ["## Dormant behaviors (not reported as findings)", ""]
        out += [f"- {b}: {ann.behavior(b).action} ({ann.behavior(b).evidence[0].file}:"
                f"{ann.behavior(b).evidence[0].line_start})" for b in res.dormant]
        out.append("")
    out += ["## Alignment", ""]
    out += [f"- {a.intent} → {a.status.value}" + (f" via {', '.join(a.behaviors)}"
                                                    if a.behaviors else "")
            for a in ann.alignment]
    if res.notes:
        out += ["", "## Run notes", ""] + [f"- {n}" for n in res.notes]
    return "\n".join(out) + "\n"
