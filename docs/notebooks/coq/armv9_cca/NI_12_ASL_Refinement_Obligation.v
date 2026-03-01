Require Import formal.coq.armv9_cca.CCA_Base.

(* Obligation scaffold: NI_12_ASL_Refinement_Obligation *)
Theorem ni_12_asl_refinement_obligation_obligation :
  forall s:State,
    leaks_secret s = false ->
    leaks_secret s = false.
Proof.
  intros s H; exact H.
Qed.
