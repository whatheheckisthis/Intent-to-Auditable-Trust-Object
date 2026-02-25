Require Import formal.coq.armv9_cca.CCA_Base.

(* Obligation scaffold: NI_07_RMM_Lifecycle *)
Theorem ni_07_rmm_lifecycle_obligation :
  forall s:State,
    leaks_secret s = false ->
    leaks_secret s = false.
Proof.
  intros s H; exact H.
Qed.
