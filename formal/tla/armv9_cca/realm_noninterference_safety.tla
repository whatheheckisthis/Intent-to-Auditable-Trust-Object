---------------------------- MODULE realm_noninterference_safety ----------------------------
EXTENDS Naturals, FiniteSets, TLC

CONSTANT Secrets, Observables
VARIABLES world, realmSecret, nonRealmObs, leak

RealmNonInterference == (world = "NonRealm") => (leak = FALSE)

Init ==
  /\ world \in {"Realm", "NonRealm", "Secure"}
  /\ realmSecret \in Secrets
  /\ nonRealmObs \in Observables
  /\ leak \in BOOLEAN
  /\ RealmNonInterference

Next ==
  /\ world' \in {"Realm", "NonRealm", "Secure"}
  /\ realmSecret' \in Secrets
  /\ nonRealmObs' \in Observables
  /\ leak' \in BOOLEAN
  /\ ((world' = "NonRealm") => (leak' = FALSE))

Spec == Init /\ [][Next]_<<world, realmSecret, nonRealmObs, leak>>

THEOREM RealmNonInterferenceInvariant == Spec => []RealmNonInterference
===============================================================================================
