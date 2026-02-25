---------------------------- MODULE post_boundary_stability_safety ----------------------------
EXTENDS Naturals, TLC

VARIABLES t, M, inputFromNonRealm

PostBoundaryStability == (t >= M) => (inputFromNonRealm = FALSE)

Init ==
  /\ t \in Nat
  /\ M \in Nat
  /\ inputFromNonRealm \in BOOLEAN
  /\ PostBoundaryStability

Next ==
  /\ t' \in Nat
  /\ M' = M
  /\ inputFromNonRealm' \in BOOLEAN
  /\ ((t' >= M') => (inputFromNonRealm' = FALSE))

Spec == Init /\ [][Next]_<<t, M, inputFromNonRealm>>

THEOREM PostBoundaryInvariant == Spec => []PostBoundaryStability
===============================================================================================
