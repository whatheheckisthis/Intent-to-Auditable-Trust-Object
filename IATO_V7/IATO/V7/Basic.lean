import Mathlib.Data.Finset.Basic
import Mathlib.Data.Finset.Lattice.Basic
import Mathlib.Order.BoundedOrder

namespace IATO.V7

structure DepVar where
  name : String
  deriving DecidableEq

abbrev DepSet : Type := Finset DepVar

def DepSet.join (φ₁ φ₂ : DepSet) : DepSet := φ₁ ∪ φ₂
def DepSet.le (φ₁ φ₂ : DepSet) : Prop := φ₁ ⊆ φ₂
def DepSet.empty : DepSet := ∅

instance : LE DepSet := inferInstance
instance : LT DepSet := inferInstance
instance : PartialOrder DepSet := inferInstance
instance : Sup DepSet := inferInstance
instance : SemilatticeSup DepSet := inferInstance
instance : OrderBot DepSet := inferInstance

theorem DepSet.join_eq_sup (φ₁ φ₂ : DepSet) :
    DepSet.join φ₁ φ₂ = φ₁ ⊔ φ₂ := rfl

theorem DepSet.le_iff_le (φ₁ φ₂ : DepSet) :
    DepSet.le φ₁ φ₂ ↔ φ₁ ≤ φ₂ := Iff.rfl

theorem DepSet.empty_eq_bot : DepSet.empty = ⊥ := rfl

theorem DepSet.le_refl (φ : DepSet) : DepSet.le φ φ := by simp [DepSet.le]

theorem DepSet.le_trans (φ₁ φ₂ φ₃ : DepSet) :
    DepSet.le φ₁ φ₂ → DepSet.le φ₂ φ₃ → DepSet.le φ₁ φ₃ := by
  simp [DepSet.le]; exact subset_trans

end IATO.V7
