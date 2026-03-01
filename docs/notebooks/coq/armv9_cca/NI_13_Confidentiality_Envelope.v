Require Import formal.coq.armv9_cca.CCA_Base.

(* Obligation scaffold: NI_13_Confidentiality_Envelope *)
Theorem ni_13_confidentiality_envelope_obligation :
  forall s:State,
    leaks_secret s = false ->
    leaks_secret s = false.
Proof.
  intros s H; exact H.
Qed.
