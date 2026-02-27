"""Finite dependency-set lattice utilities used by TS/SCI-style experiments.

This module models `(Φ, ⊆, ∪, ∅)` where `Φ` is a powerset of `DepVar`.
"""

from __future__ import annotations

from dataclasses import dataclass
import itertools
import json
import random
from typing import Iterable, Sequence


@dataclass(frozen=True, order=True)
class DepVar:
    """String-tagged dependency source identifier."""

    name: str

    def __str__(self) -> str:
        return self.name


DepCtx = "DepSet"


@dataclass(frozen=True)
class DepSet:
    """Finite set of dependency variables with semilattice operations."""

    vars: frozenset[DepVar]

    @staticmethod
    def empty() -> "DepSet":
        return DepSet(frozenset())

    @staticmethod
    def from_names(names: Iterable[str]) -> "DepSet":
        return DepSet(frozenset(DepVar(n) for n in names))

    def join(self, other: "DepSet") -> "DepSet":
        return DepSet(self.vars | other.vars)

    def le(self, other: "DepSet") -> bool:
        return self.vars <= other.vars

    def glb(self, other: "DepSet") -> "DepSet":
        return DepSet(self.vars & other.vars)

    def difference(self, other: "DepSet") -> "DepSet":
        return DepSet(self.vars - other.vars)

    def project(self, ctx: "DepCtx") -> "DepSet":
        return self.glb(ctx)

    def insert(self, var: DepVar) -> "DepSet":
        return DepSet(self.vars | {var})

    def remove(self, var: DepVar) -> "DepSet":
        return DepSet(self.vars - {var})

    def member(self, var: DepVar) -> bool:
        return var in self.vars

    def depends_on(self, var: DepVar) -> bool:
        return self.member(var)

    def dep_eq(self, other: "DepSet") -> bool:
        return self.le(other) and other.le(self)

    def is_bottom(self) -> bool:
        return self == DepSet.empty()

    def leakage_size_against(self, observer: "DepSet") -> int:
        return len(self.difference(observer).vars)

    def to_json(self) -> str:
        return json.dumps(sorted(v.name for v in self.vars))

    @staticmethod
    def from_json(raw: str) -> "DepSet":
        data = json.loads(raw)
        if not isinstance(data, list) or not all(isinstance(x, str) for x in data):
            raise ValueError("DepSet JSON must be a list[str]")
        return DepSet.from_names(data)

    def __len__(self) -> int:
        return len(self.vars)

    def __str__(self) -> str:
        # Mathematical style: {α, β, γ}
        if not self.vars:
            return "∅"
        return "{" + ", ".join(sorted(v.name for v in self.vars)) + "}"


# Function-style API aliases

def depEmpty() -> DepSet:
    return DepSet.empty()


def depJoin(left: DepSet, right: DepSet) -> DepSet:
    return left.join(right)


def depLe(left: DepSet, right: DepSet) -> bool:
    return left.le(right)


def isSubset(left: DepSet, right: DepSet) -> bool:
    return depLe(left, right)


def ctxInsert(delta: DepCtx, var: DepVar) -> DepCtx:
    return delta.insert(var)


def ctxMember(delta: DepCtx, var: DepVar) -> bool:
    return delta.member(var)


def wellScopedDepSet(delta: DepCtx, phi: DepSet) -> bool:
    return phi.le(delta)


def depJoinList(items: Iterable[DepSet]) -> DepSet:
    acc = depEmpty()
    for item in items:
        acc = depJoin(acc, item)
    return acc


def projectDep(phi: DepSet, delta: DepCtx) -> DepSet:
    return phi.project(delta)


def lub(phi1: DepSet, phi2: DepSet) -> DepSet:
    return depJoin(phi1, phi2)


def glb(phi1: DepSet, phi2: DepSet) -> DepSet:
    return phi1.glb(phi2)


def dependsOn(phi: DepSet, alpha: DepVar) -> bool:
    return phi.depends_on(alpha)


def removeDep(phi: DepSet, alpha: DepVar) -> DepSet:
    return phi.remove(alpha)


def depDifference(phi1: DepSet, phi2: DepSet) -> DepSet:
    return phi1.difference(phi2)


def isIndependent(phi1: DepSet, phi2: DepSet) -> bool:
    return len(glb(phi1, phi2)) == 0


def allDepSets(universe: Sequence[DepVar]) -> list[DepSet]:
    out: list[DepSet] = []
    for r in range(len(universe) + 1):
        for combo in itertools.combinations(universe, r):
            out.append(DepSet(frozenset(combo)))
    return out


def closureUnderJoin(seed: Iterable[DepSet]) -> list[DepSet]:
    closed = set(seed)
    changed = True
    while changed:
        changed = False
        curr = list(closed)
        for a in curr:
            for b in curr:
                j = depJoin(a, b)
                if j not in closed:
                    closed.add(j)
                    changed = True
    return sorted(closed, key=lambda s: (len(s.vars), str(s)))


def randomDepSet(n: int, universe: Sequence[DepVar] | None = None, *, rng: random.Random | None = None) -> DepSet:
    if n < 0:
        raise ValueError("n must be >= 0")
    if universe is None:
        universe = [DepVar(chr(ord("α") + i)) for i in range(max(8, n))]
    if n > len(universe):
        raise ValueError("n cannot exceed universe size")
    rr = rng or random
    return DepSet(frozenset(rr.sample(list(universe), n)))
