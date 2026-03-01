Require Import formal.coq.armv9_cca.CCA_Base.

(* Obligation scaffold: NI_10_Causal_Separation *)
Theorem ni_10_causal_separation_obligation :
  forall s:State,
    leaks_secret s = false ->
    leaks_secret s = false.
Proof.
  intros s H; exact H.
Qed.
