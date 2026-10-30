"""D1: which behaviors could fulfill an intent.

Candidates are the reachable behaviors whose capability is in the intent's
`expected_capabilities`, plus the top-k by text similarity (recall only). For a negative claim
the candidates are the behaviors that could break it: the forbidden capabilities and the
composites that contain them.
"""

from __future__ import annotations

from collections.abc import Sequence

from skillalign.align.similarity import Embedder, LexicalEmbedder, behavior_text, intent_text
from skillalign.models import Behavior, Capability, Intent, Polarity, Reachability

TOP_K = 2
#: An intent that writes or sends data also sees the other behaviors that move data out. A
#: skill that "saves locally" and also uploads is a destination mismatch (SD), which the judge
#: can only find if the upload is among the cards it reads.
SINKS = {Capability.FS_WRITE, Capability.NET_OUTBOUND}
MIN_SIMILARITY = 0.15


def candidates(intent: Intent, behaviors: Sequence[Behavior], *, embedder: Embedder | None = None,
               top_k: int = TOP_K) -> list[str]:
    """Behavior ids, in behavior order."""
    live = [b for b in behaviors if b.reachability is not Reachability.DORMANT]
    caps = set(intent.expected_capabilities)
    chosen = {b.id for b in live if b.capability in caps}
    if intent.polarity is Polarity.NEGATIVE:
        by_id = {b.id: b for b in behaviors}
        chosen |= {b.id for b in live if b.capability is Capability.COMPOSITE and any(
            by_id[p].capability in caps for p in b.composite_of if p in by_id)}
    else:
        if caps & SINKS:
            chosen |= {b.id for b in live if b.capability in SINKS}
    if intent.polarity is Polarity.POSITIVE and live and top_k > 0:
        sims = (embedder or LexicalEmbedder()).similarity(
            [intent_text(intent)], [behavior_text(b) for b in live])[0]
        for idx in sims.argsort()[::-1][:top_k]:
            if sims[idx] >= MIN_SIMILARITY:
                chosen.add(live[idx].id)
    return [b.id for b in live if b.id in chosen]
