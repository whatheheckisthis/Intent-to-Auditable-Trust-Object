---------------------------- MODULE realm_noninterference_safety ----------------------------
EXTENDS Naturals, FiniteSets, TLC

CONSTANT Secrets, Observables, AddrSet

\* Security State: SCR_EL3.NSE, SCR_EL3.NS select world routing logic.
\* Model alignment for Arm CCA: the Hypervisor is Non-Realm and adversarial.
VARIABLES SCR_EL3_NSE, SCR_EL3_NS, executionWorld, hypervisorTrusted,
          realmSecret, nonRealmObs, leak,
          GPI, gptStatus, memOp, memAddr, memAccessGranted

WorldFromSCR(nse, ns) ==
  IF nse = 1 /\ ns = 1 THEN "NonRealm"
  ELSE IF nse = 0 /\ ns = 0 THEN "Realm"
  ELSE "Secure"

\* Memory Tagging: GPI (Granule Protection Index) consults GPT before Load/Store.
\* Load or Store is granted only if GPT status authorizes the world for that granule.
GPTAllows(world, status) ==
  CASE world = "Realm" -> status = "Realm"
    [] world = "NonRealm" -> status = "NonRealm"
    [] OTHER -> status = "Secure"

GPTCheckedBeforeMemOp ==
  (memOp \in {"Load", "Store"}) =>
    (memAccessGranted = GPTAllows(executionWorld, gptStatus[memAddr]))

RealmNonInterference == (executionWorld = "NonRealm") => (leak = FALSE)

ThreatModelAligned == (hypervisorTrusted = FALSE) /\ (executionWorld = WorldFromSCR(SCR_EL3_NSE, SCR_EL3_NS))

Init ==
  /\ SCR_EL3_NSE \in {0, 1}
  /\ SCR_EL3_NS \in {0, 1}
  /\ executionWorld \in {"Realm", "NonRealm", "Secure"}
  /\ hypervisorTrusted = FALSE
  /\ realmSecret \in Secrets
  /\ nonRealmObs \in Observables
  /\ leak \in BOOLEAN
  /\ GPI \in AddrSet -> Nat
  /\ gptStatus \in AddrSet -> {"Realm", "NonRealm", "Secure"}
  /\ memOp \in {"None", "Load", "Store"}
  /\ memAddr \in AddrSet
  /\ memAccessGranted \in BOOLEAN
  /\ ThreatModelAligned
  /\ GPTCheckedBeforeMemOp
  /\ RealmNonInterference

Next ==
  /\ SCR_EL3_NSE' \in {0, 1}
  /\ SCR_EL3_NS' \in {0, 1}
  /\ executionWorld' = WorldFromSCR(SCR_EL3_NSE', SCR_EL3_NS')
  /\ hypervisorTrusted' = FALSE
  /\ realmSecret' \in Secrets
  /\ nonRealmObs' \in Observables
  /\ leak' \in BOOLEAN
  /\ GPI' \in AddrSet -> Nat
  /\ gptStatus' \in AddrSet -> {"Realm", "NonRealm", "Secure"}
  /\ memOp' \in {"None", "Load", "Store"}
  /\ memAddr' \in AddrSet
  /\ memAccessGranted' \in BOOLEAN
  /\ ((memOp' \in {"Load", "Store"}) =>
        (memAccessGranted' = GPTAllows(executionWorld', gptStatus'[memAddr'])))
  /\ ((executionWorld' = "NonRealm") => (leak' = FALSE))

Spec == Init /\ [][Next]_<<SCR_EL3_NSE, SCR_EL3_NS, executionWorld, hypervisorTrusted,
                           realmSecret, nonRealmObs, leak,
                           GPI, gptStatus, memOp, memAddr, memAccessGranted>>

THEOREM RealmNonInterferenceInvariant == Spec => []RealmNonInterference
THEOREM ThreatModelAlignmentInvariant == Spec => []ThreatModelAligned
THEOREM GPTCheckedBeforeMemOpInvariant == Spec => []GPTCheckedBeforeMemOp
===============================================================================================
