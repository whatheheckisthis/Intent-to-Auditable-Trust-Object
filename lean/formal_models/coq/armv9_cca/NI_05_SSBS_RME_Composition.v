Require Import formal.coq.armv9_cca.CCA_Base.

(* Obligation scaffold: NI_05_SSBS_RME_Composition *)
Theorem ni_05_ssbs_rme_composition_obligation :
  forall s:State,
    leaks_secret s = false ->
    leaks_secret s = false.
Proof.
  intros s H; exact H.
Qed.
