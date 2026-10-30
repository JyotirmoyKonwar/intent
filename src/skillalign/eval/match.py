"""Match predicted findings to gold findings (PLAN.md §8.1).

- UI / UI-partial: the type matches and the intents' source spans overlap (same file, any
  shared line). With no span overlap, the intents' action and object must match after
  normalization.
- UB / SD: the type matches and the evidence of a predicted behavior overlaps the evidence of
  a gold behavior. For SD the linked intents must match too.
- Strict scoring needs the exact type. Lenient scoring also pairs UI with UI-partial and
  SD with UB, and flags the pair as `wrong_type`. The SD axis is scored on its own.

Matching is one-to-one and greedy: exact-type pairs are taken before lenient ones.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from skillalign.models import Annotation, Finding, FindingType, Intent, Span

_INTENT_TYPES = {FindingType.UI, FindingType.UI_PARTIAL}
_STOP = {"a", "an", "the", "to", "of", "and", "or", "in", "on", "for", "from", "into", "with",
         "by", "is", "are", "be", "it", "its", "that", "this", "any", "all", "each", "user"}


def overlap(a: Span, b: Span) -> bool:
    return a.file == b.file and a.line_start <= b.line_end and b.line_start <= a.line_end


def any_overlap(xs: list[Span], ys: list[Span]) -> bool:
    return any(overlap(x, y) for x in xs for y in ys)


def _words(text: str) -> set[str]:
    out = set()
    for w in re.findall(r"[a-z0-9]+", text.lower()):
        if w not in _STOP:
            out.add(w[:-1] if len(w) > 3 and w.endswith("s") else w)
    return out


def _same_claim(a: Intent, b: Intent) -> bool:
    """Action/object match after normalization: same action word set, object overlap >= 0.5."""
    wa, wb = _words(a.action), _words(b.action)
    if not (wa & wb):
        return False
    oa, ob = _words(a.object), _words(b.object)
    if not oa or not ob:
        return True
    return len(oa & ob) / min(len(oa), len(ob)) >= 0.5


def intents_match(a: Intent, b: Intent) -> bool:
    return any_overlap(list(a.sources), list(b.sources)) or _same_claim(a, b)


def _behavior_spans(ann: Annotation, f: Finding) -> list[Span]:
    spans: list[Span] = []
    for bid in f.behaviors:
        spans += list(ann.behavior(bid).evidence)
    return spans or list(f.evidence)


def _locates(g_ann: Annotation, g: Finding, p_ann: Annotation, p: Finding) -> bool:
    """Do `g` and `p` point at the same place? The type is checked elsewhere."""
    if g.type in _INTENT_TYPES and p.type in _INTENT_TYPES:
        if g.intent is None or p.intent is None:
            return False
        return intents_match(g_ann.intent(g.intent), p_ann.intent(p.intent))
    if g.type in _INTENT_TYPES or p.type in _INTENT_TYPES:
        return False
    if not any_overlap(_behavior_spans(g_ann, g), _behavior_spans(p_ann, p)):
        return False
    if g.type is FindingType.SD and p.type is FindingType.SD:
        return intents_match(g_ann.intent(g.intent), p_ann.intent(p.intent))  # type: ignore[arg-type]
    return True


def _lenient_ok(a: FindingType, b: FindingType) -> bool:
    return (a in _INTENT_TYPES and b in _INTENT_TYPES) or (
        a not in _INTENT_TYPES and b not in _INTENT_TYPES)


@dataclass
class Pair:
    gold: Finding
    pred: Finding
    wrong_type: bool
    axis_ok: bool | None  # SD only
    iou: float


@dataclass
class SkillMatch:
    skill_id: str
    pairs: list[Pair] = field(default_factory=list)
    false_pos: list[Finding] = field(default_factory=list)
    false_neg: list[Finding] = field(default_factory=list)


def line_iou(xs: list[Span], ys: list[Span]) -> float:
    def lines(spans: list[Span]) -> set[tuple[str, int]]:
        return {(s.file, n) for s in spans for n in range(s.line_start, s.line_end + 1)}

    a, b = lines(xs), lines(ys)
    return len(a & b) / len(a | b) if a | b else 0.0


def match_skill(gold: Annotation, pred: Annotation, *, lenient: bool = False) -> SkillMatch:
    cands: list[tuple[int, Finding, Finding]] = []
    for g in gold.findings:
        for p in pred.findings:
            exact = g.type is p.type
            if (exact or (lenient and _lenient_ok(g.type, p.type))) and _locates(
                    gold, g, pred, p):
                cands.append((0 if exact else 1, g, p))
    cands.sort(key=lambda c: c[0])  # stable: exact pairs first
    used_g: set[str] = set()
    used_p: set[str] = set()
    out = SkillMatch(gold.skill_id)
    for _, g, p in cands:
        if g.id in used_g or p.id in used_p:
            continue
        used_g.add(g.id)
        used_p.add(p.id)
        both_sd = g.type is FindingType.SD and p.type is FindingType.SD
        out.pairs.append(Pair(g, p, g.type is not p.type,
                              (g.divergence_axis == p.divergence_axis) if both_sd else None,
                              line_iou(list(g.evidence), list(p.evidence))))
    out.false_neg = [g for g in gold.findings if g.id not in used_g]
    out.false_pos = [p for p in pred.findings if p.id not in used_p]
    return out
