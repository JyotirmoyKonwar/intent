"""D2: the checklist judge, one call per (intent, candidate cards), voted over n samples.

A sample is kept only if every card it cites is one of the candidates and it cites at least
one card unless it answers "no" with nothing contradicted. The code checks this, not the
model. Each field is decided by majority over the kept samples. With `holistic=True`
(ablation `--judge holistic`) the model gives one overall verdict, mapped to the same
`ChecklistAnswer` so the rules in `rules.py` stay the same.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel

from skillalign.align.cards import render_card
from skillalign.llm import LLMClient, StructuredOutputError
from skillalign.llm.templates import PromptTemplate, load_prompt
from skillalign.models import Axis, Card, ChecklistAnswer, ConstraintAnswer, Intent, Polarity

JUDGE_MAX_TOKENS = 1024


class HolisticAnswer(BaseModel):
    verdict: Literal["fulfilled", "partial", "unfulfilled", "divergent"]
    axis_mismatch: Axis | None = None
    cited_cards: list[str] = []


@dataclass
class JudgeResult:
    intent_id: str
    answer: ChecklistAnswer | None
    agreement: float = 0.0
    candidates: list[str] = field(default_factory=list)
    samples: int = 0  # samples kept after the card-id check
    note: str = ""


def prompt_for(holistic: bool) -> PromptTemplate:
    return load_prompt("judge_holistic" if holistic else "judge_d2")


def _message(template: PromptTemplate, intent: Intent, cards: list[Card]) -> str:
    constraints = ("\n".join(f"    {n}. {c}" for n, c in enumerate(intent.constraints, 1))
                   or "    (none)")
    return template.render(
        polarity=intent.polarity.value, action=intent.action, object=intent.object,
        constraints=constraints, cards="\n\n".join(render_card(c) for c in cards))


def from_holistic(h: HolisticAnswer) -> ChecklistAnswer:
    axis = h.axis_mismatch
    return ChecklistAnswer(
        achieves_action={"fulfilled": "yes", "divergent": "yes", "partial": "partly",
                         "unfulfilled": "no"}[h.verdict],
        axis_mismatch=axis if h.verdict == "divergent" else None,
        contradicts_negative=h.verdict == "divergent" and axis is Axis.NEGATIVE_CLAIM,
        cited_cards=h.cited_cards)


def _usable(a: ChecklistAnswer, allowed: set[str]) -> bool:
    if any(c not in allowed for c in a.cited_cards):
        return False
    quiet = a.achieves_action == "no" and not a.contradicts_negative and a.axis_mismatch is None
    return bool(a.cited_cards) or quiet


def _top(values: list):
    """The most common value; ties go to the one seen first."""
    counts = Counter(values)
    best = max(counts.values())
    return next(v for v in values if counts[v] == best)


def combine(samples: list[ChecklistAnswer], intent: Intent) -> tuple[ChecklistAnswer, float]:
    """Majority per field over `samples`, plus the share of samples that match the winner."""
    achieves = _top([s.achieves_action for s in samples])
    axis = _top([s.axis_mismatch for s in samples])
    contradicts = _top([s.contradicts_negative for s in samples])
    constraints = []
    for n, text in enumerate(intent.constraints):
        votes = [s.constraints[n].respected for s in samples if n < len(s.constraints)]
        constraints.append(ConstraintAnswer(constraint=text, respected=_top(votes)
                                            if votes else "unknown"))
    agreeing = [s for s in samples if (s.achieves_action, s.axis_mismatch,
                                       s.contradicts_negative) == (achieves, axis, contradicts)]
    cited: list[str] = []
    for s in agreeing:
        cited += [c for c in s.cited_cards if c not in cited]
    answer = ChecklistAnswer(achieves_action=achieves, constraints=constraints,
                             axis_mismatch=axis, contradicts_negative=contradicts,
                             cited_cards=cited)
    return answer, len(agreeing) / len(samples)


def judge_intent(client: LLMClient, intent: Intent, candidate_ids: list[str],
                 cards: Mapping[str, Card], *, votes: int = 5,
                 holistic: bool = False) -> JudgeResult:
    """Judge one intent against its candidate behaviors."""
    if not candidate_ids:
        # Nothing could fulfill it, so the model has nothing to read.
        return JudgeResult(intent.id, ChecklistAnswer(achieves_action="no"), 1.0, [], 0,
                           "no candidate behaviors")
    shown = [cards[b] for b in candidate_ids]
    template = prompt_for(holistic)
    schema = HolisticAnswer if holistic else ChecklistAnswer
    messages = [{"role": "user", "content": _message(template, intent, shown)}]
    try:
        raw = client.structured(messages, schema, n=votes, max_tokens=JUDGE_MAX_TOKENS)
    except StructuredOutputError as err:
        return JudgeResult(intent.id, None, 0.0, candidate_ids, 0, f"no valid reply: {err}")
    answers = [from_holistic(r) if isinstance(r, HolisticAnswer) else r for r in raw]
    kept = [a for a in answers if _usable(a, set(candidate_ids))]
    if not kept:
        return JudgeResult(intent.id, None, 0.0, candidate_ids, 0,
                           f"all {len(answers)} samples cited missing or unknown cards")
    answer, agreement = combine(kept, intent)
    if intent.polarity is Polarity.POSITIVE:
        answer = answer.model_copy(update={"contradicts_negative": False})
    return JudgeResult(intent.id, answer, agreement, candidate_ids, len(kept))
