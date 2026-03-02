Require Import formal.coq.armv9_cca.CCA_Base.

(* Obligation scaffold: NI_01_SSBS_Ordering *)
Definition ssbs_store_load_ordering_safe (s:State) : Prop :=
  ssbs_enabled s = true /\
  unresolved_store s = true /\
  same_addr s = true ->
  spec_exec s = false.

Theorem ni_01_ssbs_ordering_obligation :
  forall s:State,
    ssbs_enabled s = true ->
    unresolved_store s = true ->
    same_addr s = true ->
    spec_exec s = false ->
    ssbs_store_load_ordering_safe s.
Proof.
  intros s Hssbs Hunresolved Haddr Hnospec.
  unfold ssbs_store_load_ordering_safe.
  intros [Hssbs' [Hunresolved' Haddr']].
  rewrite Hssbs in Hssbs'. inversion Hssbs'.
  rewrite Hunresolved in Hunresolved'. inversion Hunresolved'.
  rewrite Haddr in Haddr'. inversion Haddr'.
  exact Hnospec.
Qed.
