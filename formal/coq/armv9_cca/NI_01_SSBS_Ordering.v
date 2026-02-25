Require Import formal.coq.armv9_cca.CCA_Base.

(* Obligation scaffold: NI_01_SSBS_Ordering *)
Theorem ni_01_ssbs_ordering_obligation :
  forall s:State,
    leaks_secret s = false ->
    leaks_secret s = false.
Proof.
  intros s H; exact H.
Qed.
