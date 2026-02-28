import IATO.V7.Architecture

open IATO.V7

private def parseLegacyCsvLine? (line : String) : Option LegacyWorker :=
  match line.splitOn "," with
  | [idStr, domain, depsRaw] =>
    match idStr.trim.toNat? with
    | none => none
    | some wid =>
      let deps := depsRaw.splitOn ";" |> List.map String.trim
      some ⟨wid, domain.trim, deps⟩
  | _ => none

private def loadLegacyWorkers (path : Option String) : IO (List LegacyWorker) := do
  match path with
  | none =>
    pure
      [ ⟨1, "Secure", ["secure_secret"]⟩
      , ⟨2, "Normal", ["public_data"]⟩
      , ⟨3, "Normal", ["secure_secret", "shared_cache"]⟩
      ]
  | some p =>
    if p.endsWith ".json" then
      IO.eprintln "JSON parsing is not enabled in this scanner; provide CSV (id,domain,deps) instead."
      pure []
    else
      let txt ← IO.FS.readFile p
      pure <| (txt.splitOn "\n").filterMap parseLegacyCsvLine?

private def printReport (report : ScanReport) : IO Unit := do
  IO.println "Worker Scan Report:"
  IO.println s!"✅ {report.compatible}/{report.total} workers compatible"
  if report.incompatible.isEmpty then
    IO.println "❌ 0 workers need migration"
  else
    IO.println s!"❌ {report.incompatible.length} workers need migration:"
    for w in report.incompatible do
      IO.println s!"  - Worker #{w.id}: domain={repr w.domain}, deps={w.deps.toList.map DepVar.name}"


def main (args : List String) : IO Unit := do
  let workers ← loadLegacyWorkers args.head?
  let report := scanAll workers
  let plan := generateMigrationPlan report

  printReport report
  IO.println "Migration plan:"
  for (idx, item) in plan.enum do
    IO.println s!"  {idx + 1}. {item}"
