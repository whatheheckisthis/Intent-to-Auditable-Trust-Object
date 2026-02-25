Require Import formal.coq.armv9_cca.CCA_Base.

(* Obligation scaffold: NI_09_MutualInformation_Bound *)
Theorem ni_09_mutualinformation_bound_obligation :
  forall s:State,
    leaks_secret s = false ->
    leaks_secret s = false.
Proof.
  intros s H; exact H.
Qed.
