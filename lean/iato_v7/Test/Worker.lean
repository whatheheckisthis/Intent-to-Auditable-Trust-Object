import IATO.V7.Architecture

open IATO.V7

private def wSecure : Worker := ⟨10, Domain.SecureWorld, {⟨"secure_secret"⟩}⟩
private def wNormal : Worker := ⟨11, Domain.NormalWorld, {⟨"public_data"⟩}⟩
private def wConflict : Worker := ⟨12, Domain.NormalWorld, {⟨"secure_secret"⟩}⟩

def test_worker_compatible : Bool :=
  decide (Worker.compatible wSecure wNormal)

def test_legacy_scan : Bool :=
  let lw : LegacyWorker := ⟨21, "Normal", ["public_data"]⟩
  let (_, ok) := scanLegacy lw
  ok

def test_architecture_secure : Bool :=
  decide (architectureSecure referenceArchitecture)


def main : IO Unit := do
  IO.println s!"test_worker_compatible = {test_worker_compatible}"
  IO.println s!"test_legacy_scan = {test_legacy_scan}"
  IO.println s!"test_architecture_secure = {test_architecture_secure}"

  if !(test_worker_compatible && test_legacy_scan && test_architecture_secure) then
    throw <| IO.userError "Worker test suite failed"
