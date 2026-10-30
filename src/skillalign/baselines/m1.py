"""M1: semantic similarity between intents and behaviors (DocChunk-style, PLAN.md §7).

Embed each intent and each behavior summary and compare by cosine:
- an intent with no behavior above `low` is UI;
- a behavior with no intent above `low` is UB;
- a pair that matches best but scores between `low` and `match` is SD (axis: object).
The thresholds are tuned on dev only (`tune`). The default embedder is lexical (no model);
any object with a `similarity(a, b)` method can replace it, for example Qwen3-Embedding.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from itertools import product
from pathlib import Path

from skillalign.align.similarity import Embedder, LexicalEmbedder, behavior_text, intent_text
from skillalign.baselines.common import FindingBuilder, annotation
from skillalign.baselines.r2 import doc_intents
from skillalign.behaviors import detect_behaviors
from skillalign.eval.metrics import evaluate
from skillalign.models import Annotation, Axis, Behavior, Intent, Reachability
from skillalign.normalize.loader import load_skill


@dataclass(frozen=True)
class M1Config:
    low: float = 0.15
    match: float = 0.4


def description_intents(loaded) -> list[Intent]:
    """Intents without a model: the doc sentences that start with an action verb."""
    return doc_intents(loaded)


def run_m1(skill_dir: Path, config: M1Config | None = None, *, embedder: Embedder | None = None,
           intents: Sequence[Intent] | None = None, skill_id: str | None = None) -> Annotation:
    config = config or M1Config()
    loaded = load_skill(skill_dir, skill_id=skill_id)
    behaviors = [b for b in detect_behaviors(loaded).behaviors
                 if b.reachability is not Reachability.DORMANT]
    its = list(intents) if intents is not None else description_intents(loaded)
    return build_m1(loaded.skill.skill_id if skill_id is None else skill_id, its, behaviors,
                    config, embedder or LexicalEmbedder())


def build_m1(skill_id: str, intents: Sequence[Intent], behaviors: Sequence[Behavior],
             config: M1Config, embedder: Embedder) -> Annotation:
    out = FindingBuilder()
    if intents and behaviors:
        sim = embedder.similarity([intent_text(i) for i in intents],
                                  [behavior_text(b) for b in behaviors])
    else:
        sim = None
    parts = {p for b in behaviors for p in b.composite_of}
    for r, intent in enumerate(intents):
        best = float(sim[r].max()) if sim is not None else 0.0
        if best < config.low:
            out.ui(intent, f"No behavior is similar to this claim (best {best:.2f}).")
        elif best < config.match and sim is not None:
            b = behaviors[int(sim[r].argmax())]
            out.sd(intent, b, Axis.OBJECT,
                   f"The closest behavior is only weakly similar (best {best:.2f}).")
    for c, b in enumerate(behaviors):
        if b.id in parts:
            continue
        best = float(sim[:, c].max()) if sim is not None else 0.0
        if best < config.low:
            out.ub(b, f"No claim is similar to this behavior (best {best:.2f}).")
    return annotation(skill_id, "baseline:M1", intents, behaviors, out.findings,
                      notes=f"low={config.low} match={config.match}")


def tune(items: Mapping[str, tuple[Sequence[Intent], Sequence[Behavior]]],
         golds: Mapping[str, Annotation], *, embedder: Embedder | None = None,
         lows: Sequence[float] = (0.05, 0.1, 0.15, 0.2, 0.25, 0.3),
         matches: Sequence[float] = (0.3, 0.4, 0.5, 0.6, 0.7),
         score: Callable[[Mapping, Mapping], float] | None = None) -> tuple[M1Config, float]:
    """Grid-search the thresholds for the best macro F1 on `golds` (a dev set only).

    `items[skill_id]` is `(intents, behaviors)` for that skill, so the detectors and the
    embedder run once per setting rather than reading the files again.
    """
    emb = embedder or LexicalEmbedder()
    scorer = score or (lambda g, p: evaluate(g, p).macro_f1())
    best, best_score = M1Config(), -1.0
    for low, match in product(lows, matches):
        if match <= low:
            continue
        cfg = M1Config(low, match)
        preds = {sid: build_m1(sid, i, b, cfg, emb) for sid, (i, b) in items.items()}
        s = scorer(golds, preds)
        if s > best_score:
            best, best_score = cfg, s
    return best, best_score


