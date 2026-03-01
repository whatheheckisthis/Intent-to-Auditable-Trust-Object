import Lake
open Lake DSL

package «IATO_V7» where
  defaultFacet := `pkg

require mathlib from git
  "https://github.com/leanprover-community/mathlib4" @ "main"

@[default_target]
lean_lib «IATO_V7» where
  globs := #[.submodules `IATO]

lean_exe «IATO_V7» where
  root := `Main

lean_exe «worker-scan» where
  root := `Main

lean_exe «test-basic» where
  root := `Test.Basic

lean_exe «test-worker» where
  root := `Test.Worker

lean_exe «test-rme» where
  root := `Test.RME

lean_exe «cache» where
  root := `Cache

script test do
  let buildOut ← IO.Process.output {
    cmd := "lake"
    args := # ["build", "test-basic", "test-worker", "test-rme"]
  }
  IO.print buildOut.stdout
  if buildOut.exitCode != 0 then
    IO.eprintln buildOut.stderr
    return buildOut.exitCode

  let runBasic ← IO.Process.output {
    cmd := "lake"
    args := #["exe", "test-basic"]
  }
  IO.print runBasic.stdout
  if runBasic.exitCode != 0 then
    IO.eprintln runBasic.stderr
    return runBasic.exitCode

  let runWorker ← IO.Process.output {
    cmd := "lake"
    args := #["exe", "test-worker"]
  }
  IO.print runWorker.stdout
  if runWorker.exitCode != 0 then
    IO.eprintln runWorker.stderr
    return runWorker.exitCode

  let runRme ← IO.Process.output {
    cmd := "lake"
    args := #["exe", "test-rme"]
  }
  IO.print runRme.stdout
  if runRme.exitCode != 0 then
    IO.eprintln runRme.stderr
  return runRme.exitCode
