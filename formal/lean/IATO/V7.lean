import Mathlib.Data.Finset.Basic
import Mathlib.Data.Finset.Lattice.Basic
import Mathlib.Order.BoundedOrder

namespace IATO.V7

open Finset

structure DepVar where
  name : String
  deriving DecidableEq

abbrev DepSet : Type := Finset DepVar

namespace DepSet

def join (φ₁ φ₂ : DepSet) : DepSet := φ₁ ∪ φ₂

def le (φ₁ φ₂ : DepSet) : Prop := φ₁ ⊆ φ₂

def empty : DepSet := ∅

lemma le_refl (φ : DepSet) : le φ φ := by
  intro x hx
  exact hx

lemma le_trans {φ₁ φ₂ φ₃ : DepSet} : le φ₁ φ₂ → le φ₂ φ₃ → le φ₁ φ₃ := by
  intro h₁₂ h₂₃ x hx
  exact h₂₃ (h₁₂ hx)

lemma join_comm (φ₁ φ₂ : DepSet) : join φ₁ φ₂ = join φ₂ φ₁ := by
  exact Finset.union_comm φ₁ φ₂

lemma join_idem (φ : DepSet) : join φ φ = φ := by
  exact Finset.union_idem

instance : PartialOrder DepSet := inferInstance
instance : SemilatticeSup DepSet := inferInstance
instance : OrderBot DepSet := inferInstance

lemma join_eq_sup (φ₁ φ₂ : DepSet) : join φ₁ φ₂ = φ₁ ⊔ φ₂ := rfl
lemma le_iff_le (φ₁ φ₂ : DepSet) : le φ₁ φ₂ ↔ φ₁ ≤ φ₂ := Iff.rfl
lemma empty_eq_bot : empty = (⊥ : DepSet) := rfl

end DepSet

end IATO.V7
