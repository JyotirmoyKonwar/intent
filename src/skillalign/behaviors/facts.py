"""One detected operation, before reachability and composites are decided."""

from __future__ import annotations

from dataclasses import dataclass, field

from skillalign.models import Capability, Origin

#: Tags a fact can carry.
SENSITIVE = "sensitive"  # a source worth guarding: credentials, secrets, the whole environment
NET_IN = "net-in"  # data that came back from the network
ENCODE = "encode"  # a transform that hides data (base64 and the like)
DYNAMIC = "dynamic"  # the argument is not a constant


@dataclass(eq=False)
class Fact:
    cap: Capability
    action: str
    file: str  # real path, or a virtual path for a fenced block
    l0: int
    l1: int
    target: str | None = None
    destination: str | None = None
    origin: Origin = Origin.CODE
    scope: str = "<module>"  # "<module>", "<main>", "func" or "Class.method"
    tags: set[str] = field(default_factory=set)
    also: list[tuple[str, int, int]] = field(default_factory=list)  # more evidence lines
    conditions: list[str] = field(default_factory=list)
    chain: list[str] = field(default_factory=list)
    idx: int = -1

    @property
    def key(self) -> tuple:
        return (self.cap, self.file, self.l0, self.target, self.destination)


class FactSet:
    """Facts in the order they were found. Adding the same operation twice returns the first."""

    def __init__(self) -> None:
        self.items: list[Fact] = []
        self._seen: dict[tuple, Fact] = {}

    def add(self, fact: Fact) -> Fact:
        old = self._seen.get(fact.key)
        if old is not None:
            old.tags |= fact.tags
            for span in fact.also:
                if span not in old.also:
                    old.also.append(span)
            return old
        fact.idx = len(self.items)
        self.items.append(fact)
        self._seen[fact.key] = fact
        return fact

    def __iter__(self):
        return iter(self.items)

    def __len__(self) -> int:
        return len(self.items)
