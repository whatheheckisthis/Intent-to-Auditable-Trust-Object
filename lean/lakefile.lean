import Lake
open Lake DSL

package «iato-v7» where
  defaultFacet := `pkg

require mathlib from git
  "https://github.com/leanprover-community/mathlib4" @ "main"

@[default_target]
lean_lib IATO where
  globs := #[.submodules `IATO]

lean_exe tests where
  root := `Test.Basic
