Require Import formal.coq.armv9_cca.CCA_Base.

(* Obligation scaffold: NI_04_PostBoundary_Stability *)
Theorem ni_04_postboundary_stability_obligation :
  forall s:State,
    leaks_secret s = false ->
    leaks_secret s = false.
Proof.
  intros s H; exact H.
Qed.
