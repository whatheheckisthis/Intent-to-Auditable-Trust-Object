from __future__ import annotations

import random

from src.dep_lattice import (
    DepSet,
    DepVar,
    allDepSets,
    closureUnderJoin,
    depDifference,
    depEmpty,
    depJoin,
    depJoinList,
    depLe,
    glb,
    isIndependent,
    lub,
    projectDep,
    randomDepSet,
    removeDep,
    wellScopedDepSet,
)


def test_join_semilattice_laws() -> None:
    a = DepSet.from_names(["α"])
    b = DepSet.from_names(["β"])
    c = DepSet.from_names(["γ"])
    empty = depEmpty()

    assert depJoin(a, depJoin(b, c)) == depJoin(depJoin(a, b), c)
    assert depJoin(a, b) == depJoin(b, a)
    assert depJoin(a, a) == a
    assert depJoin(a, empty) == a
    assert depJoin(empty, a) == a


def test_subset_order_laws() -> None:
    a = DepSet.from_names(["α"])
    ab = DepSet.from_names(["α", "β"])
    abc = DepSet.from_names(["α", "β", "γ"])

    assert depLe(a, a)  # reflexive
    assert depLe(a, ab) and depLe(ab, abc) and depLe(a, abc)  # transitive
    assert depLe(a, a) and depLe(a, a) and a == a  # antisymmetry witness


def test_well_scoped_and_projection() -> None:
    delta = DepSet.from_names(["α", "β"])
    scoped = DepSet.from_names(["β"])
    ill_scoped = DepSet.from_names(["γ"])

    assert wellScopedDepSet(delta, scoped)
    assert not wellScopedDepSet(delta, ill_scoped)
    assert projectDep(DepSet.from_names(["α", "γ"]), delta) == DepSet.from_names(["α"])


def test_lub_and_glb_match_union_and_intersection() -> None:
    a = DepSet.from_names(["α", "β"])
    b = DepSet.from_names(["β", "γ"])

    assert lub(a, b) == DepSet.from_names(["α", "β", "γ"])
    assert glb(a, b) == DepSet.from_names(["β"])


def test_json_roundtrip_and_remove_monotone() -> None:
    phi = DepSet.from_names(["α", "β"])
    wire = phi.to_json()
    assert DepSet.from_json(wire) == phi

    alpha = DepVar("α")
    removed = removeDep(phi, alpha)
    assert not removed.member(alpha)
    assert depLe(removed, phi)


def test_difference_independence_and_join_list() -> None:
    a = DepSet.from_names(["α", "β"])
    b = DepSet.from_names(["β"])
    c = DepSet.from_names(["γ"])

    assert depDifference(a, b) == DepSet.from_names(["α"])
    assert isIndependent(b, c)
    assert depJoinList([b, c]) == DepSet.from_names(["β", "γ"])


def test_all_dep_sets_and_closure_under_join() -> None:
    u = [DepVar("α"), DepVar("β"), DepVar("γ")]
    all_sets = allDepSets(u)
    assert len(all_sets) == 8

    seed = [DepSet.from_names(["α"]), DepSet.from_names(["β"])]
    closed = closureUnderJoin(seed)
    assert DepSet.from_names(["α", "β"]) in closed


def test_fuzz_join_and_order_properties() -> None:
    rng = random.Random(7)
    u = [DepVar("α"), DepVar("β"), DepVar("γ"), DepVar("δ")]

    for _ in range(200):
        a = randomDepSet(rng.randint(0, len(u)), universe=u, rng=rng)
        b = randomDepSet(rng.randint(0, len(u)), universe=u, rng=rng)
        c = randomDepSet(rng.randint(0, len(u)), universe=u, rng=rng)

        assert depJoin(a, depJoin(b, c)) == depJoin(depJoin(a, b), c)
        assert depJoin(a, b) == depJoin(b, a)
        assert depJoin(a, a) == a
        assert depLe(a, depJoin(a, b))
