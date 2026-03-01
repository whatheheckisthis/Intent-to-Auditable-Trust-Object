---------------------------- MODULE ssbs_store_load_ordering_safety ----------------------------
EXTENDS Naturals, TLC

CONSTANT AddrSet
VARIABLES PSTATE_SSBS, storeBeforeLoad, unresolvedStore, storeAddr, loadAddr, speculativeLoadBypassesStore

\* Speculation Control: PSTATE.SSBS specifically constrains store-to-load forwarding.
\* If PSTATE.SSBS == 1 and a younger Load targets the same address as an older unresolved Store,
\* the load must not bypass the store in the microarchitectural forwarding logic.
StoreLoadOrderingSafety ==
  (PSTATE_SSBS = 1 /\ storeBeforeLoad = TRUE /\ unresolvedStore = TRUE /\ storeAddr = loadAddr)
    => (speculativeLoadBypassesStore = FALSE)

Init ==
  /\ PSTATE_SSBS \in {0, 1}
  /\ storeBeforeLoad \in BOOLEAN
  /\ unresolvedStore \in BOOLEAN
  /\ storeAddr \in AddrSet
  /\ loadAddr \in AddrSet
  /\ speculativeLoadBypassesStore \in BOOLEAN
  /\ StoreLoadOrderingSafety

Next ==
  /\ PSTATE_SSBS' \in {0, 1}
  /\ storeBeforeLoad' \in BOOLEAN
  /\ unresolvedStore' \in BOOLEAN
  /\ storeAddr' \in AddrSet
  /\ loadAddr' \in AddrSet
  /\ speculativeLoadBypassesStore' \in BOOLEAN
  /\ ((PSTATE_SSBS' = 1 /\ storeBeforeLoad' = TRUE /\ unresolvedStore' = TRUE /\ storeAddr' = loadAddr')
      => (speculativeLoadBypassesStore' = FALSE))

Spec == Init /\ [][Next]_<<PSTATE_SSBS, storeBeforeLoad, unresolvedStore, storeAddr, loadAddr, speculativeLoadBypassesStore>>

THEOREM SSBSOrderingInvariant == Spec => []StoreLoadOrderingSafety
===============================================================================================
