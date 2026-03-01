import Legacy.IATO.V7.Basic

open IATO.V7

/-- Compile-time check: reflexive ordering. -/
theorem test_depSet_order :
    let α : DepVar := ⟨"α"⟩
    let φ : DepSet := {α}
    DepSet.le φ φ := by
  intro α φ
  exact DepSet.le_refl φ

/-- Compile-time check: join commutativity. -/
theorem test_join_comm :
    let α : DepVar := ⟨"α"⟩
    let β : DepVar := ⟨"β"⟩
    let φ₁ : DepSet := {α}
    let φ₂ : DepSet := {β}
    DepSet.join φ₁ φ₂ = DepSet.join φ₂ φ₁ := by
  intro α β φ₁ φ₂
  exact DepSet.join_comm φ₁ φ₂

def main : IO Unit :=
  IO.println "Lean test module compiled successfully"
