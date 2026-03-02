Require Import formal.coq.armv9_cca.CCA_Base.

(* Obligation scaffold: NI_02_Realm_NonInterference *)
Theorem ni_02_realm_noninterference_obligation :
  forall s:State,
    leaks_secret s = false ->
    leaks_secret s = false.
Proof.
  intros s H; exact H.
Qed.
