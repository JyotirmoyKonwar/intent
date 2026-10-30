"""Bootstrap confidence intervals and McNemar's test."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from math import comb

import numpy as np


def bootstrap_ci(items: Sequence, stat: Callable[[Sequence], float], *, n_boot: int = 1000,
                 alpha: float = 0.05, seed: int = 13) -> tuple[float, float, float]:
    """`(point, low, high)`: resample `items` (one per skill) with replacement and apply `stat`."""
    if not items:
        return 0.0, 0.0, 0.0
    rng = np.random.default_rng(seed)
    n = len(items)
    stats = [stat([items[i] for i in rng.integers(0, n, n)]) for _ in range(n_boot)]
    lo, hi = np.quantile(stats, [alpha / 2, 1 - alpha / 2])
    return float(stat(items)), float(lo), float(hi)


def mcnemar(a_correct: Sequence[bool], b_correct: Sequence[bool]) -> tuple[int, int, float]:
    """Exact two-sided McNemar test on paired per-skill outcomes.

    Returns `(b, c, p)`: `b` skills only A got right, `c` only B got right.
    """
    if len(a_correct) != len(b_correct):
        raise ValueError("the two systems must be scored on the same skills")
    b = sum(x and not y for x, y in zip(a_correct, b_correct, strict=True))
    c = sum(y and not x for x, y in zip(a_correct, b_correct, strict=True))
    n = b + c
    if n == 0:
        return b, c, 1.0
    tail = sum(comb(n, k) for k in range(min(b, c) + 1)) / 2**n
    return b, c, min(1.0, 2 * tail)
