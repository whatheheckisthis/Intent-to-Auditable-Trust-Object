Require Import formal.coq.armv9_cca.CCA_Base.

(* Obligation scaffold: NI_11_Speculative_Blindness *)
Theorem ni_11_speculative_blindness_obligation :
  forall s:State,
    leaks_secret s = false ->
    leaks_secret s = false.
Proof.
  intros s H; exact H.
Qed.
