/-!
`Worker.lean` — Worker-domain non-interference.

Defines worker/domain structures and composition/compatibility relations used to
state separation properties between execution domains.
-/

import IATO.V7.Basic

namespace IATO.V7

inductive Domain where
  | RootWorld
  | SecureWorld
  | NormalWorld
  | PeripheralWorld
  deriving DecidableEq, Repr

instance : Inhabited Domain where
  default := Domain.RootWorld

structure Worker where
  id : Nat
  domain : Domain
  deps : DepSet
  deriving Repr

instance : Inhabited Worker where
  default := ⟨0, Domain.RootWorld, ⊥⟩

def Worker.compatible (w1 w2 : Worker) : Prop :=
  w1.domain ≠ w2.domain ∧ w1.deps ∩ w2.deps = (∅ : DepSet)

def Worker.compose (ws : List Worker) : Worker :=
  ⟨0, Domain.RootWorld, ws.foldl (fun acc w => acc ⊔ w.deps) ⊥⟩

end IATO.V7
