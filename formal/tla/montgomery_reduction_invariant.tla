---------------------------- MODULE montgomery_reduction_invariant ----------------------------
EXTENDS Naturals, TLC

CONSTANT Q
ASSUME Q > 1

VARIABLES t, reduced

MontgomeryReduce(x) == x % Q

Init ==
    /\ t \in Nat
    /\ reduced = MontgomeryReduce(t)

Next ==
    \E x \in Nat:
        /\ t' = x
        /\ reduced' = MontgomeryReduce(x)

RangeBound == reduced < Q

Spec == Init /\ [][Next]_<<t, reduced>>

THEOREM MontgomeryReductionAlwaysBounded == Spec => []RangeBound

===============================================================================================
