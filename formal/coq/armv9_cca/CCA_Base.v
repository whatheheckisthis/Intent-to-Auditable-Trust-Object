(* Base definitions for Armv9-A CCA non-interference scaffolding. *)
Inductive World := Realm | NonRealm | Secure.

Record State := {
  ssbs_enabled : bool;
  exec_world : World;
  unresolved_store : bool;
  same_addr : bool;
  spec_exec : bool;
  leaks_secret : bool;
  t_now : nat;
  t_boundary : nat;
  nonrealm_input : bool
}.

Definition blindfold (s:State) : Prop :=
  ssbs_enabled s = true /\ exec_world s = Realm.

Definition non_interference (s:State) : Prop :=
  exec_world s = NonRealm -> leaks_secret s = false.
