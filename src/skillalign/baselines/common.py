"""Shared pieces for the baselines: findings built from behaviors and intents."""

from __future__ import annotations

from collections.abc import Sequence

from skillalign.models import (
    Annotation,
    Axis,
    Behavior,
    Finding,
    FindingType,
    Intent,
    Provenance,
    Risk,
    SkillLabel,
    Span,
    Suspected,
)


def spans(items) -> list[Span]:
    return [Span(file=e.file, line_start=e.line_start, line_end=e.line_end) for e in items]


class FindingBuilder:
    def __init__(self) -> None:
        self.findings: list[Finding] = []

    def _add(self, **kw) -> None:
        self.findings.append(Finding(id=f"F{len(self.findings) + 1}", **kw))

    def ui(self, intent: Intent, why: str) -> None:
        self._add(type=FindingType.UI, intent=intent.id, evidence=spans(intent.sources),
                  expected_result=f"{intent.action} {intent.object}", rationale=why)

    def ub(self, b: Behavior, why: str) -> None:
        self._add(type=FindingType.UB, behaviors=[b.id], evidence=spans(b.evidence),
                  actual_result=b.action, risk=Risk.LOW, rationale=why,
                  call_chain=list(b.call_chain))

    def sd(self, intent: Intent, b: Behavior, axis: Axis, why: str) -> None:
        self._add(type=FindingType.SD, intent=intent.id, behaviors=[b.id], divergence_axis=axis,
                  evidence=spans(b.evidence), expected_result=f"{intent.action} {intent.object}",
                  actual_result=b.action, intent_suspected=Suspected.UNCLEAR, rationale=why)


def annotation(skill_id: str, system: str, intents: Sequence[Intent],
               behaviors: Sequence[Behavior], findings: list[Finding],
               notes: str | None = None) -> Annotation:
    return Annotation(
        skill_id=skill_id, intents=list(intents), behaviors=list(behaviors), findings=findings,
        skill_label=SkillLabel.MISALIGNED if findings else SkillLabel.ALIGNED,
        provenance=Provenance(system=system, notes=notes))
