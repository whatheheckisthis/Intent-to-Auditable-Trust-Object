import IATO.V7.WorkerPackage

open IATO.V7

private def wSecure : Worker := ⟨100, Domain.SecureWorld, {⟨"secure_secret"⟩}⟩
private def wNormal : Worker := ⟨101, Domain.NormalWorld, {⟨"public_data"⟩}⟩
private def wConflict : Worker := ⟨102, Domain.NormalWorld, {⟨"secure_secret"⟩}⟩

def pkgIsolated : WorkerPackage :=
  WorkerPackage.reconfigure "isolated" [wSecure, wNormal]

def pkgConflict : WorkerPackage :=
  WorkerPackage.reconfigure "conflict" [wSecure, wConflict]

def test_pkg_isolated : Bool := decide pkgIsolated.isIsolated

def test_pkg_conflict : Bool := decide (¬ pkgConflict.isIsolated)

def test_pkg_summary_nonempty : Bool :=
  !(WorkerPackage.summary pkgIsolated).isEmpty

def main : IO Unit := do
  IO.println s!"test_pkg_isolated = {test_pkg_isolated}"
  IO.println s!"test_pkg_conflict = {test_pkg_conflict}"
  IO.println s!"test_pkg_summary_nonempty = {test_pkg_summary_nonempty}"

  if !(test_pkg_isolated && test_pkg_conflict && test_pkg_summary_nonempty) then
    throw <| IO.userError "WorkerPackage test suite failed"
