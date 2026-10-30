"""M4 end to end: one skill folder in, one `Annotation` (the same schema as gold) out.

Order: normalize, B1 intents (+ B2 capabilities, category), C1-C5 behaviors (+ C4 prose),
cards, D1 candidates, D2 judge, D3 implied check for what is left, D4 rules. The flags in
`DetectOptions` are the ablations from PLAN.md §8.3.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from skillalign.align.candidates import candidates
from skillalign.align.cards import build_cards, render_card
from skillalign.align.implied import table_says, would_user_expect
from skillalign.align.judge import JudgeResult, judge_intent, prompt_for
from skillalign.align.rules import decide, linked_ids, unlinked_behaviors
from skillalign.align.similarity import Embedder
from skillalign.behaviors import detect_behaviors
from skillalign.behaviors.prose import extract_prose_behaviors
from skillalign.intents.extract import extract_claims
from skillalign.intents.vocab import classify_category, enrich_intents
from skillalign.llm import LLMClient
from skillalign.llm.templates import load_prompt
from skillalign.models import (
    Annotation,
    Behavior,
    Capability,
    Card,
    Intent,
    Origin,
    Polarity,
    Provenance,
    Reachability,
    SkillLabel,
)
from skillalign.normalize.loader import LoadedSkill, load_skill


@dataclass
class DetectOptions:
    """Ablation switches. The defaults are the full system."""

    instructions: bool = True  # --no-instructions: no agent-instruction behaviors
    composite: bool = True  # --no-composite: drop composite behaviors
    cards: Literal["cards", "raw"] = "cards"  # --cards raw
    judge: Literal["checklist", "holistic"] = "checklist"  # --judge holistic
    implied_table: bool = True  # --no-implied-table: every leftover behavior is UB
    votes: int = 5  # --votes
    prose: bool = True  # run the C4 model pass (needs `instructions`)

    def describe(self) -> str:
        return (f"instructions={self.instructions} composite={self.composite} "
                f"cards={self.cards} judge={self.judge} implied_table={self.implied_table} "
                f"votes={self.votes}")


@dataclass
class DetectResult:
    annotation: Annotation
    loaded: LoadedSkill
    cards: dict[str, Card]
    judged: dict[str, JudgeResult]
    implied: dict[str, bool]
    dormant: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def _apply_ablations(behaviors: list[Behavior], opts: DetectOptions) -> list[Behavior]:
    out = behaviors
    if not opts.instructions:
        out = [b for b in out if b.origin is not Origin.AGENT_INSTRUCTION
               and b.capability is not Capability.AGENT_INSTRUCTION]
    if not opts.composite:
        out = [b for b in out if b.capability is not Capability.COMPOSITE]
    return out


#: Capabilities the detectors never report, so "no behavior has it" proves nothing.
UNDETECTED = {Capability.DATA_TRANSFORM, Capability.USER_INTERACTION,
              Capability.AGENT_INSTRUCTION}


_SCRIPT = re.compile(r"\b[\w./-]+\.(py|sh|js|ts|rb|ps1)\b")


def verifiable(intent: Intent, has_code: bool) -> bool:
    """Can a missing behavior show that this claim is unimplemented (UI)?"""
    if not has_code or not set(intent.expected_capabilities) - UNDETECTED:
        return False
    # "Run `python scripts/x.py`" starts the skill itself; it is not a claim about an effect.
    return not (set(intent.expected_capabilities) == {Capability.PROC_EXEC}
                and _SCRIPT.search(f"{intent.action} {intent.object}"))


def prompt_hash(opts: DetectOptions) -> str:
    """One hash over the prompts this configuration uses, for `Provenance.prompt_hash`."""
    names = ["claims_b1", "c4_prose", "implied_d3"]
    parts = [load_prompt(n).sha256 for n in names] + [prompt_for(opts.judge == "holistic").sha256]
    return hashlib.sha256("".join(parts).encode()).hexdigest()


def detect_skill(client: LLMClient, skill_dir: Path, opts: DetectOptions | None = None, *,
                 skill_id: str | None = None, intents: list[Intent] | None = None,
                 embedder: Embedder | None = None) -> DetectResult:
    """Run the pipeline on one skill. Pass `intents` to skip B1 (for example, cached)."""
    opts = opts or DetectOptions()
    loaded = load_skill(skill_dir, skill_id=skill_id)
    notes: list[str] = []

    if intents is None:
        extracted = extract_claims(client, skill_dir, skill_id=skill_id)
        intents = extracted.intents
        s = extracted.stats
        notes.append(f"B1: {s.claims_kept} claims kept, {s.dropped_bad_span} dropped, "
                     f"{s.failed_chunks} chunks failed")
    intents = enrich_intents(intents)

    detection = detect_behaviors(loaded)
    behaviors = list(detection.behaviors)
    notes += detection.notes
    if opts.instructions and opts.prose:
        prose, ps = extract_prose_behaviors(client, loaded, behaviors)
        behaviors += prose
        notes.append(f"C4: {ps.kept} prose behaviors, {ps.dropped_bad_span} dropped, "
                     f"{ps.failed_chunks} chunks failed")
    behaviors = _apply_ablations(behaviors, opts)

    category = classify_category(loaded.name, loaded.description,
                                 has_code=bool(loaded.code_files()),
                                 body=loaded.texts.get(loaded.skill_md or "", ""))
    cards = build_cards(loaded, behaviors, mode=opts.cards)

    judged: dict[str, JudgeResult] = {}
    has_code = bool(loaded.code_files())
    for intent in intents:
        cand = candidates(intent, behaviors, embedder=embedder)
        if not cand and intent.polarity is Polarity.POSITIVE and not verifiable(intent, has_code):
            # Nothing to check it against: a guidance sentence, a verb no detector reports
            # (convert, summarize), or a skill with no code. Silence is not "unimplemented".
            judged[intent.id] = JudgeResult(intent.id, None, 0.0, [], 0, "not verifiable")
            continue
        judged[intent.id] = judge_intent(client, intent, cand, cards, votes=opts.votes,
                                         holistic=opts.judge == "holistic")
        if judged[intent.id].answer is None:
            notes.append(f"{intent.id}: {judged[intent.id].note}")

    left = unlinked_behaviors(behaviors, linked_ids(intents, judged, behaviors))
    implied: dict[str, bool] = {}
    if opts.implied_table:
        for b in left:
            implied[b.id] = table_says(category, b)
            # Only a skill the table has no row for gets the model question, and never a
            # composite or an instruction: the table's "still counts" column is final.
            if (not implied[b.id] and category in (None, "other") and not b.composite_of
                    and b.origin is Origin.CODE):
                implied[b.id] = would_user_expect(client, loaded.description or "",
                                                  render_card(cards[b.id]))
    docs = "\n".join(t for p, t in loaded.texts.items() if "#" not in p)
    alignment, findings = decide(intents, behaviors, judged, implied, docs_text=docs)

    annotation = Annotation(
        skill_id=skill_id or loaded.skill.skill_id, category=category, intents=intents,
        behaviors=behaviors, alignment=alignment, findings=findings,
        skill_label=SkillLabel.MISALIGNED if findings else SkillLabel.ALIGNED,
        provenance=Provenance(system="skillalign", model=client.model,
                              prompt_hash=prompt_hash(opts), seed=client.settings.seed,
                              notes=opts.describe()))
    dormant = [b.id for b in behaviors if b.reachability is Reachability.DORMANT]
    return DetectResult(annotation, loaded, cards, judged, implied, dormant, notes)
