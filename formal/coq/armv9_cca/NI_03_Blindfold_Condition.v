Require Import formal.coq.armv9_cca.CCA_Base.

(* Obligation scaffold: NI_03_Blindfold_Condition *)
Theorem ni_03_blindfold_condition_obligation :
  forall s:State,
    leaks_secret s = false ->
    leaks_secret s = false.
Proof.
  intros s H; exact H.
Qed.
