import Lake
open Lake DSL

package «IATO_V7» where
  defaultFacet := `pkg

require mathlib from git
  "https://github.com/leanprover-community/mathlib4" @ "main"

@[default_target]
lean_lib «IATO_V7» where
  globs := #[.submodules `IATO]

/-- Test executable target. -/
lean_exe «test» where
  root := `Test

/-- Cache helper executable target so `lake exe cache` works out of the box. -/
lean_exe «cache» where
  root := `Cache

script test do
  let buildOut ← IO.Process.output {
    cmd := "lake"
    args := #["build", "test"]
  }
  IO.print buildOut.stdout
  if buildOut.exitCode ≠ 0 then
    IO.eprintln buildOut.stderr
    return buildOut.exitCode

  let runOut ← IO.Process.output {
    cmd := "lake"
    args := #["exe", "test"]
  }
  IO.print runOut.stdout
  if runOut.exitCode ≠ 0 then
    IO.eprintln runOut.stderr
  return runOut.exitCode

/-- AOT helper: compile all configured targets to native executables. -/
script aot do
  let out ← IO.Process.output {
    cmd := "lake"
    args := #["build", "IATO_V7", "test", "cache"]
  }
  IO.print out.stdout
  if out.exitCode ≠ 0 then
    IO.eprintln out.stderr
  return out.exitCode
