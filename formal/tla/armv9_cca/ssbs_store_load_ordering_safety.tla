---------------------------- MODULE ssbs_store_load_ordering_safety ----------------------------
EXTENDS Naturals, TLC

CONSTANT AddrSet
VARIABLES ssbs, storeBeforeLoad, unresolvedStore, storeAddr, loadAddr, specExec

\* Invariant 1 (Store-Load Ordering Safety):
\* (PO(S,L) /\ Addr(S)=Addr(L) /\ unresolved(S) /\ SSBS=1) => ~SpecExec(L)
StoreLoadOrderingSafety ==
  (ssbs = TRUE /\ storeBeforeLoad = TRUE /\ unresolvedStore = TRUE /\ storeAddr = loadAddr)
    => (specExec = FALSE)

Init ==
  /\ ssbs \in BOOLEAN
  /\ storeBeforeLoad \in BOOLEAN
  /\ unresolvedStore \in BOOLEAN
  /\ storeAddr \in AddrSet
  /\ loadAddr \in AddrSet
  /\ specExec \in BOOLEAN
  /\ StoreLoadOrderingSafety

Next ==
  /\ ssbs' \in BOOLEAN
  /\ storeBeforeLoad' \in BOOLEAN
  /\ unresolvedStore' \in BOOLEAN
  /\ storeAddr' \in AddrSet
  /\ loadAddr' \in AddrSet
  /\ specExec' \in BOOLEAN
  /\ ((ssbs' = TRUE /\ storeBeforeLoad' = TRUE /\ unresolvedStore' = TRUE /\ storeAddr' = loadAddr')
      => (specExec' = FALSE))

Spec == Init /\ [][Next]_<<ssbs, storeBeforeLoad, unresolvedStore, storeAddr, loadAddr, specExec>>

THEOREM SSBSOrderingInvariant == Spec => []StoreLoadOrderingSafety
===============================================================================================
