/-!
`Architecture.lean` — System invariants aligned to SOC2 and ISM.

Assembles reference workers and proves pairwise compatibility invariants that
support governance, integrity, and isolation assurance goals.
-/

import IATO.V7.Scanner

namespace IATO.V7

def referenceArchitecture : List Worker :=
  [
    ⟨1, Domain.SecureWorld, {⟨"secure_secret"⟩}⟩,
    ⟨2, Domain.NormalWorld, {⟨"public_data"⟩}⟩
  ]

def architectureSecure (workers : List Worker) : Prop :=
  ∀ i j,
    i < workers.length →
    j < workers.length →
    i ≠ j →
    Worker.compatible (workers.get ⟨i, by assumption⟩) (workers.get ⟨j, by assumption⟩)

def assessIncompatibility (w : Worker) : String :=
  if w.domain = Domain.SecureWorld then
    s!"Worker #{w.id}: split secure dependencies from shared path"
  else
    s!"Worker #{w.id}: isolate domain and remove overlapping deps"

def generateMigrationPlan (report : ScanReport) : List String :=
  let header :=
    if report.incompatible.isEmpty then
      ["No migration required; all scanned workers are compatible."]
    else
      ["Refactor incompatible workers with dedicated domain/dependency isolation:"]
  header ++ report.incompatible.map assessIncompatibility

theorem referenceArchitecture_secure : architectureSecure referenceArchitecture := by
  intro i j hi hj hij
  have hi' : i = 0 ∨ i = 1 := by omega
  have hj' : j = 0 ∨ j = 1 := by omega
  rcases hi' with rfl | rfl
  · rcases hj' with rfl | rfl
    · exact (hij rfl).elim
    · constructor
      · decide
      · decide
  · rcases hj' with rfl | rfl
    · constructor
      · decide
      · decide
    · exact (hij rfl).elim

end IATO.V7
