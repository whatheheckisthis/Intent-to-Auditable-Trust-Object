---------------------------- MODULE speculative_blindfold_safety ----------------------------
EXTENDS Naturals, Sequences, TLC

VARIABLES ssbs, executionWorld, speculativeEncodesSecret

Init ==
  /\ ssbs \in BOOLEAN
  /\ executionWorld \in {"Realm", "NonRealm", "Secure"}
  /\ speculativeEncodesSecret \in BOOLEAN

Next ==
  /\ ssbs' \in BOOLEAN
  /\ executionWorld' \in {"Realm", "NonRealm", "Secure"}
  /\ speculativeEncodesSecret' \in BOOLEAN

Spec == Init /\ [][Next]_<<ssbs, executionWorld, speculativeEncodesSecret>>

BlindfoldCondition == (ssbs = TRUE /\ executionWorld = "Realm") => (speculativeEncodesSecret = FALSE)

THEOREM BlindfoldInvariant == Spec => []BlindfoldCondition
===============================================================================================
