"""Text similarity for recall (D1) and the M1 baseline.

`LexicalEmbedder` is a bag-of-words cosine over stemmed words, with action verbs mapped to
their vocabulary form. It needs no model. Anything with the same `similarity` method (for
example a Qwen3-Embedding client) can replace it.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Sequence
from typing import Protocol

import numpy as np

from skillalign.intents.vocab import canonical_verb
from skillalign.models import Behavior, Intent

_STOP = {"a", "an", "the", "to", "of", "and", "or", "in", "on", "for", "from", "into", "with",
         "by", "is", "are", "be", "it", "its", "that", "this", "any", "all", "each", "user",
         "file", "files", "data", "target", "when", "then", "also", "only", "not", "no"}


class Embedder(Protocol):
    def similarity(self, a: Sequence[str], b: Sequence[str]) -> np.ndarray:
        """A `len(a) x len(b)` matrix of cosine similarities in [0, 1]."""


def tokens(text: str) -> list[str]:
    out = []
    for w in re.findall(r"[A-Za-z][A-Za-z0-9]+", text):
        w = w.lower()
        if w in _STOP:
            continue
        verb = canonical_verb(w)
        if verb:
            w = verb
        elif len(w) > 3 and w.endswith("s"):
            w = w[:-1]
        out.append(w)
    return out


class LexicalEmbedder:
    def similarity(self, a: Sequence[str], b: Sequence[str]) -> np.ndarray:
        bags_a = [Counter(tokens(t)) for t in a]
        bags_b = [Counter(tokens(t)) for t in b]
        vocab = {w: i for i, w in enumerate({w for bag in (*bags_a, *bags_b) for w in bag})}

        def matrix(bags: list[Counter]) -> np.ndarray:
            m = np.zeros((len(bags), max(len(vocab), 1)))
            for r, bag in enumerate(bags):
                for w, n in bag.items():
                    m[r, vocab[w]] = np.sqrt(n)
            norm = np.linalg.norm(m, axis=1, keepdims=True)
            return m / np.where(norm == 0, 1, norm)

        return matrix(bags_a) @ matrix(bags_b).T


def intent_text(i: Intent) -> str:
    return f"{i.action} {i.object} {' '.join(i.constraints)}"


def behavior_text(b: Behavior) -> str:
    return f"{b.action} {b.target or ''} {b.destination or ''}"
