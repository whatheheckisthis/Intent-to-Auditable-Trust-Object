import IATO_V7

open IATO.V7

/-- Compile-time proof test: `DepSet.le` is reflexive. -/
theorem test_depSet_le_refl :
    let φ : DepSet := ∅
    DepSet.le φ φ := by
  intro φ
  exact DepSet.le_refl φ

/-- Compile-time proof test: `DepSet.join` is commutative. -/
theorem test_join_comm :
    let α : DepVar := ⟨"α"⟩
    let β : DepVar := ⟨"β"⟩
    let φ₁ : DepSet := {α}
    let φ₂ : DepSet := {β}
    DepSet.join φ₁ φ₂ = DepSet.join φ₂ φ₁ := by
  intro α β φ₁ φ₂
  simp [DepSet.join, Finset.union_comm]

/-- Runtime smoke test to mirror the compile-time test facts. -/
def smoke : Bool :=
  let α : DepVar := ⟨"α"⟩
  let β : DepVar := ⟨"β"⟩
  let φ₁ : DepSet := {α}
  let φ₂ : DepSet := {β}
  decide (DepSet.le (∅ : DepSet) (∅ : DepSet) ∧
    DepSet.join φ₁ φ₂ = DepSet.join φ₂ φ₁)

def main : IO Unit := do
  if smoke then
    IO.println "IATO_V7 tests passed"
  else
    throw <| IO.userError "IATO_V7 tests failed"
