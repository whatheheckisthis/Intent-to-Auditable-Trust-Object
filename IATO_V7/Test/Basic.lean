import Mathlib.Data.Finset.Basic
import IATO.V7.Basic


def test_finset_import : Finset Nat := {1, 2, 3}
#eval test_finset_import

open IATO.V7

example : ⊥ = (∅ : DepSet) := rfl
example (φ₁ φ₂ : DepSet) : φ₁ ⊔ φ₂ = φ₁ ∪ φ₂ := rfl
example (φ : DepSet) : φ ≤ φ := le_rfl φ


def main : IO Unit := do
  IO.println s!"Finset.Basic import test: {test_finset_import}"
  IO.println "IATO.V7 Basic tests compiled"
