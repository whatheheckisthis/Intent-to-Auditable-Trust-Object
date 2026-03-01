/-!
`Scanner.lean` — Dependency/conflict detection.

Parses legacy worker records, normalizes fields, and computes compatibility
reports against a secure reference worker profile.
-/

import IATO.V7.Worker

namespace IATO.V7

structure LegacyWorker where
  id : Nat
  domain : String
  deps : List String
  deriving Repr

private def parseDomainNormalized : String → Option Domain
  | "root" => some Domain.RootWorld
  | "secure" => some Domain.SecureWorld
  | "normal" => some Domain.NormalWorld
  | "peripheral" => some Domain.PeripheralWorld
  | _ => none

def parseDomain (s : String) : Option Domain :=
  parseDomainNormalized s.trim.toLower

def parseDeps (xs : List String) : Option DepSet :=
  if xs.any (fun x => x.trim.isEmpty) then
    none
  else
    some <| (xs.map (fun x => DepVar.mk x.trim)).eraseDups.toFinset

def newReferenceWorker : Worker :=
  ⟨999, Domain.SecureWorld, {⟨"secure_secret"⟩}⟩

def scanLegacy (lw : LegacyWorker) : Option Worker × Bool :=
  match parseDomain lw.domain, parseDeps lw.deps with
  | some d, some φ =>
    let w : Worker := ⟨lw.id, d, φ⟩
    (some w, Worker.compatible w newReferenceWorker)
  | _, _ => (none, false)

structure ScanReport where
  total : Nat
  compatible : Nat
  incompatible : List Worker
  deriving Repr

def scanAll (lws : List LegacyWorker) : ScanReport := Id.run do
  let mut compatible := 0
  let mut incompatible : List Worker := []
  for lw in lws do
    let (ow, ok) := scanLegacy lw
    if let some w := ow then
      if ok then
        compatible := compatible + 1
      else
        incompatible := incompatible.concat w
  return ⟨lws.length, compatible, incompatible⟩

theorem scanAll_total_coverage (lws : List LegacyWorker) : (scanAll lws).total = lws.length := rfl

end IATO.V7
