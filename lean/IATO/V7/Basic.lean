import Mathlib.Data.Finset.Basic

namespace IATO.V7

structure DepVar where
  name : String
  deriving DecidableEq, Repr

abbrev DepSet := Finset DepVar

def DepSet.join (φ₁ φ₂ : DepSet) : DepSet := φ₁ ∪ φ₂
def DepSet.le (φ₁ φ₂ : DepSet) : Prop := φ₁ ⊆ φ₂
def DepSet.empty : DepSet := ∅

theorem DepSet.le_refl (φ : DepSet) : DepSet.le φ φ := by
  intro x hx
  exact hx

theorem DepSet.join_comm (φ₁ φ₂ : DepSet) :
    DepSet.join φ₁ φ₂ = DepSet.join φ₂ φ₁ := by
  simp [DepSet.join, Finset.union_comm]

end IATO.V7
