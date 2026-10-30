"""D4: deterministic decision rules and risk scoring.

- achieves_action "no" -> UI. "partly", or a constraint answered "no" -> UI-partial.
- an axis mismatch, or a contradicted negative claim -> SD, with the axis.
- a reachable behavior that no intent links to and the implied table does not cover -> UB.
- dormant behaviors never become findings; the report lists them apart.

Risk is a base level per capability, raised for composites, obfuscation, secrets and hosts
the docs never mention, and lowered for dormant code.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence

from skillalign.align.judge import JudgeResult
from skillalign.models import (
    Alignment,
    AlignmentStatus,
    Axis,
    Behavior,
    Finding,
    FindingType,
    Intent,
    Origin,
    Polarity,
    Reachability,
    Risk,
    Span,
    Suspected,
)
from skillalign.models import Capability as C

_LEVELS = [Risk.INFO, Risk.LOW, Risk.MEDIUM, Risk.HIGH, Risk.CRITICAL]
_BASE: dict[C, Risk] = {
    C.FS_READ: Risk.INFO, C.DATA_TRANSFORM: Risk.INFO, C.USER_INTERACTION: Risk.INFO,
    C.FS_WRITE: Risk.LOW, C.FS_ENUMERATE: Risk.LOW, C.ENV_READ: Risk.LOW,
    C.PKG_INSTALL: Risk.LOW, C.EXTERNAL_API: Risk.LOW, C.DNS: Risk.LOW,
    C.NET_OUTBOUND: Risk.MEDIUM, C.NET_INBOUND: Risk.MEDIUM, C.PROC_EXEC: Risk.MEDIUM,
    C.FS_DELETE: Risk.MEDIUM, C.SYS_CONFIG_MODIFY: Risk.MEDIUM,
    C.AGENT_INSTRUCTION: Risk.MEDIUM, C.TOOL_PERMISSION: Risk.MEDIUM,
    C.ENCODE_OBFUSCATE: Risk.MEDIUM,
    C.CREDENTIAL_ACCESS: Risk.HIGH, C.DOWNLOAD_EXEC: Risk.HIGH, C.CODE_EVAL: Risk.HIGH,
    C.PERSISTENCE: Risk.HIGH, C.PRIV_ESCALATION: Risk.HIGH, C.COMPOSITE: Risk.HIGH,
}
_SECRET = re.compile(r"\.ssh|\.aws|\.gnupg|\.netrc|credentials|id_rsa|\.env\b|token|secret|"
                     r"passw|api[_-]?key|os\.environ|environ", re.I)
_HIDE = re.compile(r"\b(silently|do not (tell|mention|show)|don't (tell|mention)|without "
                   r"(telling|informing)|hide|ignore (all |any )?(previous|other))\b", re.I)


def _level(r: Risk) -> int:
    return _LEVELS.index(r)


def behavior_risk(b: Behavior, by_id: Mapping[str, Behavior], docs_text: str = "") -> Risk:
    level = _level(_BASE.get(b.capability, Risk.LOW))
    parts = [by_id[p] for p in b.composite_of if p in by_id]
    text = f"{b.action} {b.target or ''} {b.destination or ''}"
    if b.capability is C.COMPOSITE and any(p.capability is C.ENCODE_OBFUSCATE for p in parts):
        level += 1
    if b.capability is C.ENCODE_OBFUSCATE:
        level += 1
    if _SECRET.search(text) or any(_SECRET.search(f"{p.target or ''} {p.action}") for p in parts):
        level += 1
    host = (b.destination or "").lower()
    if host and b.capability in (C.NET_OUTBOUND, C.COMPOSITE) and host not in docs_text.lower():
        level += 1
    if b.reachability is Reachability.DORMANT:
        level -= 2
    return _LEVELS[max(0, min(level, len(_LEVELS) - 1))]


def max_risk(risks: Sequence[Risk]) -> Risk:
    return max(risks, key=_level, default=Risk.LOW)


def suspected(behaviors: Sequence[Behavior], by_id: Mapping[str, Behavior]) -> Suspected:
    """`deliberate` when the behavior hides itself or disguises its data; `drift` for a plain
    mismatch with no such sign; `unclear` otherwise."""
    for b in behaviors:
        parts = [by_id[p] for p in b.composite_of if p in by_id]
        if b.capability is C.ENCODE_OBFUSCATE or any(
                p.capability is C.ENCODE_OBFUSCATE for p in parts):
            return Suspected.DELIBERATE
        if b.origin is Origin.AGENT_INSTRUCTION and _HIDE.search(b.action):
            return Suspected.DELIBERATE
    return Suspected.UNCLEAR


def _spans(evidence: Sequence[Span]) -> list[Span]:
    return [Span(file=e.file, line_start=e.line_start, line_end=e.line_end) for e in evidence]


def _chain(behaviors: Sequence[Behavior]) -> list[str]:
    out: list[str] = []
    for b in behaviors:
        out += [c for c in b.call_chain if c not in out]
    return out


def linked_ids(intents: Sequence[Intent], results: Mapping[str, JudgeResult],
               behaviors: Sequence[Behavior]) -> set[str]:
    """Behaviors some intent accounts for: cited by a judged intent, or all its candidates
    when the judge said the intent is carried out and cited none. Parts of a linked
    composite count as linked."""
    by_id = {b.id: b for b in behaviors}
    linked: set[str] = set()
    for intent in intents:
        r = results.get(intent.id)
        if r is None or r.answer is None:
            continue
        a = r.answer
        if intent.polarity is Polarity.NEGATIVE and not a.contradicts_negative:
            continue
        if a.achieves_action == "no" and not a.contradicts_negative and a.axis_mismatch is None:
            continue
        linked |= set(a.cited_cards) or set(r.candidates)
    for bid in list(linked):
        linked |= {p for p in by_id[bid].composite_of if p in by_id}
    return linked


def unlinked_behaviors(behaviors: Sequence[Behavior], linked: set[str]) -> list[Behavior]:
    """Reachable behaviors no intent links to. Dormant ones are left out."""
    return [b for b in behaviors
            if b.id not in linked and b.reachability is not Reachability.DORMANT]


def decide(intents: Sequence[Intent], behaviors: Sequence[Behavior],
           results: Mapping[str, JudgeResult], implied: Mapping[str, bool], *,
           docs_text: str = "") -> tuple[list[Alignment], list[Finding]]:
    """Alignment entries and findings from the judge results and the D3 answers."""
    by_id = {b.id: b for b in behaviors}
    alignment: list[Alignment] = []
    findings: list[Finding] = []

    def add(**kw) -> None:
        findings.append(Finding(id=f"F{len(findings) + 1}", **kw))

    def risk_of(ids: Sequence[str]) -> Risk:
        return max_risk([behavior_risk(by_id[i], by_id, docs_text) for i in ids])

    for intent in intents:
        r = results.get(intent.id)
        if r is None or r.answer is None:
            continue  # the judge gave no usable answer: no claim either way
        a = r.answer
        cited = [c for c in a.cited_cards if c in by_id] or list(r.candidates)
        claim = f"{intent.action} {intent.object}".strip()
        agree = f"judge agreement {r.agreement:.2f} over {r.samples} samples"
        if intent.polarity is Polarity.NEGATIVE:
            if a.contradicts_negative and cited:
                alignment.append(Alignment(intent=intent.id, behaviors=cited,
                                           status=AlignmentStatus.UNFULFILLED))
                add(type=FindingType.SD, intent=intent.id, behaviors=cited,
                    divergence_axis=Axis.NEGATIVE_CLAIM, expected_result=f"never: {claim}",
                    actual_result="; ".join(by_id[b].action for b in cited),
                    evidence=_spans([e for b in cited for e in by_id[b].evidence]) or
                    _spans(intent.sources), call_chain=_chain([by_id[b] for b in cited]),
                    risk=max_risk([Risk.MEDIUM, risk_of(cited)]),
                    intent_suspected=suspected([by_id[b] for b in cited], by_id),
                    rationale=f"The skill says it will not {claim}, but a behavior does. {agree}.")
            else:
                alignment.append(Alignment(intent=intent.id, behaviors=[],
                                           status=AlignmentStatus.FULFILLED))
            continue

        partial = a.achieves_action == "partly" or any(
            c.respected == "no" for c in a.constraints)
        if a.achieves_action == "no":
            alignment.append(Alignment(intent=intent.id, behaviors=[],
                                       status=AlignmentStatus.UNFULFILLED))
            add(type=FindingType.UI, intent=intent.id, expected_result=claim,
                actual_result="no behavior carries this out",
                evidence=_spans(intent.sources), risk=Risk.LOW,
                intent_suspected=Suspected.DRIFT,
                rationale=f"Nothing in the skill achieves: {claim}. {agree}.")
            continue
        alignment.append(Alignment(
            intent=intent.id, behaviors=cited,
            status=AlignmentStatus.PARTIAL if partial else AlignmentStatus.FULFILLED))
        if a.axis_mismatch is not None and cited:
            add(type=FindingType.SD, intent=intent.id, behaviors=cited,
                divergence_axis=a.axis_mismatch, expected_result=claim,
                actual_result="; ".join(by_id[b].action for b in cited),
                evidence=_spans([e for b in cited for e in by_id[b].evidence]),
                call_chain=_chain([by_id[b] for b in cited]),
                risk=max_risk([Risk.LOW, risk_of(cited)]),
                intent_suspected=suspected([by_id[b] for b in cited], by_id),
                rationale=f"A behavior differs from the claim in its {a.axis_mismatch.value}. "
                          f"{agree}.")
        elif partial:
            missed = [c.constraint for c in a.constraints if c.respected == "no"]
            add(type=FindingType.UI_PARTIAL, intent=intent.id, expected_result=claim,
                actual_result="only part is carried out"
                + (f"; constraint not met: {'; '.join(missed)}" if missed else ""),
                evidence=_spans(intent.sources), risk=Risk.LOW,
                intent_suspected=Suspected.DRIFT,
                rationale=f"The skill carries out only part of: {claim}. {agree}.")

    # UB: behaviors left over, unless a composite that contains them is reported instead.
    leftover = [b for b in unlinked_behaviors(behaviors, linked_ids(intents, results, behaviors))
                if not implied.get(b.id, False)]
    covered = {p for b in leftover for p in b.composite_of}
    for b in leftover:
        if b.id in covered:
            continue
        add(type=FindingType.UB, behaviors=[b.id], expected_result=None,
            actual_result=b.action, evidence=_spans(b.evidence), call_chain=list(b.call_chain),
            risk=behavior_risk(b, by_id, docs_text), intent_suspected=suspected([b], by_id),
            rationale=f"No intent covers this {b.capability.value} behavior and the "
                      "skill's category does not imply it.")
    return alignment, findings

