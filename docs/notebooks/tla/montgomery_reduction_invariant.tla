---------------------------- MODULE montgomery_reduction_invariant ----------------------------
EXTENDS Naturals, Sequences, TLC

CONSTANTS Q, Bound, PiSet, WitnessSet
ASSUME Q > 1
ASSUME Bound \/ Bound = 0

VARIABLES transitions, pi, witness

InBounds(t) == t \in Nat /\ t <= Bound

MontgomeryReduce(x) == x % Q

AllTransitionsInBounds == \A i \in DOMAIN transitions: InBounds(transitions[i])

verify(p, w) ==
    /\ p \in PiSet
    /\ w \in WitnessSet
    /\ AllTransitionsInBounds

Init ==
    /\ transitions \in Seq(Nat)
    /\ Len(transitions) > 0
    /\ pi \in PiSet
    /\ witness \in WitnessSet

Next ==
    \E newTransitions \in Seq(Nat), p2 \in PiSet, w2 \in WitnessSet:
        /\ transitions' = newTransitions
        /\ pi' = p2
        /\ witness' = w2

Spec == Init /\ [][Next]_<<transitions, pi, witness>>

InvariantIff == verify(pi, witness) <=> AllTransitionsInBounds

THEOREM VerifyIffTransitionsBounded == Spec => []InvariantIff

===============================================================================================
