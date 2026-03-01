import IATO.V7.Basic

namespace IATO.V7

structure Worker where
  id : String
  deps : DepSet
  world : String
  deriving Repr

def Worker.compatible (w : Worker) : Prop :=
  w.world = "rme" ∨ w.world = "normal"

end IATO.V7
