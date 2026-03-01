import Mathlib.Data.Finset.Basic

namespace IATO.V7

inductive GranuleState where
  | Undelegated
  | Delegated
  | RealmDesc (rid : Nat)
  | Data (rid : Nat)
  deriving DecidableEq, Repr

structure RmeState where
  realms : Finset Nat
  granules : Nat → GranuleState

instance : Inhabited RmeState where
  default := ⟨∅, fun _ => GranuleState.Undelegated⟩

def RmeState.wf (σ : RmeState) : Prop :=
  ∀ gid rid, σ.granules gid = GranuleState.Data rid → rid ∈ σ.realms

def writeGranule (σ : RmeState) (gid : Nat) (gs : GranuleState) : RmeState :=
  { σ with granules := Function.update σ.granules gid gs }

lemma writeGranule_eq (σ : RmeState) (gid : Nat) (gs : GranuleState) :
    (writeGranule σ gid gs).granules gid = gs := by
  simp [writeGranule]

lemma writeGranule_ne (σ : RmeState) {gid other : Nat} (h : other ≠ gid) (gs : GranuleState) :
    (writeGranule σ gid gs).granules other = σ.granules other := by
  simp [writeGranule, h]

inductive RmiCall where
  | RealmCreate (rid : Nat)
  | GranuleDelegate (gid : Nat)
  | DataCreate (rid gid : Nat)
  | RealmDestroy (rid : Nat)
  deriving DecidableEq, Repr

def realmDestroyGranule (rid : Nat) (gs : GranuleState) : GranuleState :=
  match gs with
  | GranuleState.RealmDesc owner => if owner = rid then GranuleState.Delegated else gs
  | GranuleState.Data owner => if owner = rid then GranuleState.Delegated else gs
  | _ => gs

def step (σ : RmeState) : RmiCall → RmeState
  | RmiCall.RealmCreate rid =>
      { σ with realms := σ.realms.insert rid }
  | RmiCall.GranuleDelegate gid =>
      if σ.granules gid = GranuleState.Undelegated then
        writeGranule σ gid GranuleState.Delegated
      else
        σ
  | RmiCall.DataCreate rid gid =>
      if rid ∈ σ.realms ∧ σ.granules gid = GranuleState.Delegated then
        writeGranule σ gid (GranuleState.Data rid)
      else
        σ
  | RmiCall.RealmDestroy rid =>
      { realms := σ.realms.erase rid
      , granules := fun gid => realmDestroyGranule rid (σ.granules gid)
      }

lemma wf_default : (default : RmeState).wf := by
  intro gid rid h
  simp at h

lemma wf_realmCreate (σ : RmeState) (hσ : σ.wf) (rid : Nat) :
    (step σ (RmiCall.RealmCreate rid)).wf := by
  intro gid owner hdata
  exact Finset.mem_insert_of_mem (hσ gid owner hdata)

lemma wf_granuleDelegate (σ : RmeState) (hσ : σ.wf) (gid : Nat) :
    (step σ (RmiCall.GranuleDelegate gid)).wf := by
  by_cases h : σ.granules gid = GranuleState.Undelegated
  · intro g owner hg
    by_cases hEq : g = gid
    · subst hEq
      simp [step, h, writeGranule_eq] at hg
    · have hs : (step σ (RmiCall.GranuleDelegate gid)).granules g = σ.granules g := by
        simp [step, h, writeGranule_ne, hEq]
      exact hσ g owner (by simpa [hs] using hg)
  · simpa [step, h] using hσ

lemma wf_dataCreate (σ : RmeState) (hσ : σ.wf) (rid gid : Nat) :
    (step σ (RmiCall.DataCreate rid gid)).wf := by
  by_cases h : rid ∈ σ.realms ∧ σ.granules gid = GranuleState.Delegated
  · intro g owner hg
    by_cases hEq : g = gid
    · subst hEq
      simp [step, h, writeGranule_eq] at hg
      rcases hg with rfl
      exact h.left
    · have hs : (step σ (RmiCall.DataCreate rid gid)).granules g = σ.granules g := by
        simp [step, h, writeGranule_ne, hEq]
      exact hσ g owner (by simpa [hs] using hg)
  · simpa [step, h] using hσ

lemma realmDestroy_no_data_owner (σ : RmeState) (rid gid : Nat) :
    (step σ (RmiCall.RealmDestroy rid)).granules gid ≠ GranuleState.Data rid := by
  simp [step, realmDestroyGranule]

lemma wf_realmDestroy (σ : RmeState) (hσ : σ.wf) (rid : Nat) :
    (step σ (RmiCall.RealmDestroy rid)).wf := by
  intro gid owner hg
  by_cases hOwner : owner = rid
  · subst hOwner
    exact False.elim ((realmDestroy_no_data_owner σ rid gid) hg)
  · have hpre : σ.granules gid = GranuleState.Data owner := by
      cases hgs : σ.granules gid <;> simp [step, realmDestroyGranule] at hg
      case Data x =>
        by_cases hx : x = rid <;> simp [hx] at hg
  have hmem : owner ∈ σ.realms := hσ gid owner hpre
  exact Finset.mem_erase_of_ne hOwner hmem

lemma wf_step (σ : RmeState) (hσ : σ.wf) (call : RmiCall) : (step σ call).wf := by
  cases call with
  | RealmCreate rid => exact wf_realmCreate σ hσ rid
  | GranuleDelegate gid => exact wf_granuleDelegate σ hσ gid
  | DataCreate rid gid => exact wf_dataCreate σ hσ rid gid
  | RealmDestroy rid => exact wf_realmDestroy σ hσ rid

end IATO.V7
