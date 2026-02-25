---------------------------- MODULE post_boundary_stability_safety ----------------------------
EXTENDS Naturals, TLC

VARIABLES t, M, inputFromNonRealm

Init ==
  /\ t \in Nat
  /\ M \in Nat
  /\ inputFromNonRealm \in BOOLEAN

Next ==
  /\ t' \in Nat
  /\ M' = M
  /\ inputFromNonRealm' \in BOOLEAN

Spec == Init /\ [][Next]_<<t, M, inputFromNonRealm>>

PostBoundaryStability == (t >= M) => (inputFromNonRealm = FALSE)

THEOREM PostBoundaryInvariant == Spec => []PostBoundaryStability
===============================================================================================
