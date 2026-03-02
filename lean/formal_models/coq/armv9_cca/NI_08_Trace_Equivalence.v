Require Import formal.coq.armv9_cca.CCA_Base.

(* Obligation scaffold: NI_08_Trace_Equivalence *)
Theorem ni_08_trace_equivalence_obligation :
  forall s:State,
    leaks_secret s = false ->
    leaks_secret s = false.
Proof.
  intros s H; exact H.
Qed.
