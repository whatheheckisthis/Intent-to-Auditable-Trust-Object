import IATO.V7.Basic


def main : IO Unit := do
  let α : IATO.V7.DepVar := ⟨"α"⟩
  let φ₁ : IATO.V7.DepSet := {α}
  let φ₂ : IATO.V7.DepSet := ∅

  IO.println s!"φ₁ = {φ₁}"
  IO.println s!"φ₁ ≤ φ₁: {φ₁ ≤ φ₁}"
  IO.println s!"φ₁ ⊔ φ₂ = {φ₁ ⊔ φ₂}"
