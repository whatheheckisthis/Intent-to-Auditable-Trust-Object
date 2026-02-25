---------------------------- MODULE ssbs_store_load_ordering_safety ----------------------------
EXTENDS Naturals, TLC

CONSTANT AddrSet
VARIABLES ssbs, unresolvedStore, storeAddr, loadAddr, specExec

StoreLoadOrderingSafety ==
  (ssbs = TRUE /\ unresolvedStore = TRUE /\ storeAddr = loadAddr) => (specExec = FALSE)

Init ==
  /\ ssbs \in BOOLEAN
  /\ unresolvedStore \in BOOLEAN
  /\ storeAddr \in AddrSet
  /\ loadAddr \in AddrSet
  /\ specExec \in BOOLEAN
  /\ StoreLoadOrderingSafety

Next ==
  /\ ssbs' \in BOOLEAN
  /\ unresolvedStore' \in BOOLEAN
  /\ storeAddr' \in AddrSet
  /\ loadAddr' \in AddrSet
  /\ specExec' \in BOOLEAN
  /\ ((ssbs' = TRUE /\ unresolvedStore' = TRUE /\ storeAddr' = loadAddr') => (specExec' = FALSE))

Spec == Init /\ [][Next]_<<ssbs, unresolvedStore, storeAddr, loadAddr, specExec>>

THEOREM SSBSOrderingInvariant == Spec => []StoreLoadOrderingSafety
===============================================================================================
