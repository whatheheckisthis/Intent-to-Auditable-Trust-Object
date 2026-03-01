---------------------------- MODULE speculative_blindfold_safety ----------------------------
EXTENDS Naturals, Sequences, TLC

VARIABLES ssbs, executionWorld, speculativeEncodesSecret

BlindfoldCondition == (ssbs = TRUE /\ executionWorld = "Realm") => (speculativeEncodesSecret = FALSE)

Init ==
  /\ ssbs \in BOOLEAN
  /\ executionWorld \in {"Realm", "NonRealm", "Secure"}
  /\ speculativeEncodesSecret \in BOOLEAN
  /\ BlindfoldCondition

Next ==
  /\ ssbs' \in BOOLEAN
  /\ executionWorld' \in {"Realm", "NonRealm", "Secure"}
  /\ speculativeEncodesSecret' \in BOOLEAN
  /\ ((ssbs' = TRUE /\ executionWorld' = "Realm") => (speculativeEncodesSecret' = FALSE))

Spec == Init /\ [][Next]_<<ssbs, executionWorld, speculativeEncodesSecret>>

THEOREM BlindfoldInvariant == Spec => []BlindfoldCondition
===============================================================================================
