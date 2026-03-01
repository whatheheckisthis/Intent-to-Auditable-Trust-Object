/-!
`Basic.lean` — Lattice and security foundations.

Defines dependency variables, dependency sets, and the bounded semilattice
operations used throughout the IATO-V7 worker compatibility model.
-/

import Mathlib.Data.Finset.Basic
import Mathlib.Data.Finset.Lattice.Basic
import Mathlib.Order.BoundedOrder

namespace IATO.V7

structure DepVar where
  name : String
  deriving DecidableEq

abbrev DepSet := Finset DepVar

-- Basic order
instance : LE DepSet := inferInstance
instance : PartialOrder DepSet := inferInstance

-- SemilatticeSup (join)
instance : Sup DepSet := inferInstance
instance : SemilatticeSup DepSet := inferInstance

-- OrderBot
instance : OrderBot DepSet := inferInstance

def join (φ₁ φ₂ : DepSet) := φ₁ ⊔ φ₂

theorem join_eq_union (φ₁ φ₂ : DepSet) : join φ₁ φ₂ = φ₁ ∪ φ₂ := rfl

end IATO.V7

open IATO.V7

example : ⊥ = (∅ : DepSet) := rfl
example (φ₁ φ₂ : DepSet) : φ₁ ⊔ φ₂ = φ₁ ∪ φ₂ := rfl
example (φ : DepSet) : φ ≤ φ := le_rfl φ
example (φ : DepSet) : φ ≤ φ := by decide
