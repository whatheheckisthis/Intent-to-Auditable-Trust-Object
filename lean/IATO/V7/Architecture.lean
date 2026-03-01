import IATO.V7.Worker

namespace IATO.V7

structure Architecture where
  workers : List Worker

/-- Placeholder non-interference invariant marker. -/
def Architecture.noninterference (_a : Architecture) : Prop := True

end IATO.V7
