import IATO.V7.Worker

namespace IATO.V7

structure WorkerPackage where
  name : String
  workers : List Worker
  deriving Repr

def WorkerPackage.dependencyFootprint (pkg : WorkerPackage) : DepSet :=
  pkg.workers.foldl (fun acc w => acc ⊔ w.deps) ⊥

def WorkerPackage.isIsolated (pkg : WorkerPackage) : Prop :=
  pkg.workers.Pairwise Worker.compatible

def WorkerPackage.reconfigure (name : String) (workers : List Worker) : WorkerPackage :=
  ⟨name, workers⟩

def WorkerPackage.summary (pkg : WorkerPackage) : String :=
  s!"worker-package={pkg.name}, workers={pkg.workers.length}, deps={pkg.dependencyFootprint.toList.length}"

lemma workerPackage_isolated_nil : (WorkerPackage.reconfigure "empty" []).isIsolated := by
  simp [WorkerPackage.reconfigure, WorkerPackage.isIsolated]

lemma workerPackage_isolated_singleton (w : Worker) :
    (WorkerPackage.reconfigure "single" [w]).isIsolated := by
  simp [WorkerPackage.reconfigure, WorkerPackage.isIsolated]

end IATO.V7
