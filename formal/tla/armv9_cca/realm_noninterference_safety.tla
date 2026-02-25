---------------------------- MODULE realm_noninterference_safety ----------------------------
EXTENDS Naturals, FiniteSets, TLC

CONSTANT Secrets, Observables
VARIABLES world, realmSecret, nonRealmObs, leak

Init ==
  /\ world \in {"Realm", "NonRealm", "Secure"}
  /\ realmSecret \in Secrets
  /\ nonRealmObs \in Observables
  /\ leak \in BOOLEAN

Next ==
  /\ world' \in {"Realm", "NonRealm", "Secure"}
  /\ realmSecret' \in Secrets
  /\ nonRealmObs' \in Observables
  /\ leak' \in BOOLEAN

Spec == Init /\ [][Next]_<<world, realmSecret, nonRealmObs, leak>>

RealmNonInterference == (world = "NonRealm") => (leak = FALSE)

THEOREM RealmNonInterferenceInvariant == Spec => []RealmNonInterference
===============================================================================================
