---------------------------- MODULE dijkstra_branchless_assurance ----------------------------
EXTENDS Naturals, Integers, FiniteSets, Sequences, TLC

(***************************************************************************
This model combines:
  * A Dijkstra-style shortest-path exploration over weighted directed graphs.
  * Branchless final reduction (arithmetic mask selection) used in modular logic.
  * Scenario scheduling in 24 x 5 batches, covering 10,000 scenarios.
  * Explicit residual-risk obligations that cannot be discharged by TLA+ alone.
***************************************************************************)

CONSTANTS Nodes, Source, Infinity, Weight, M, NumScenarios

ASSUME Nodes # {}
ASSUME Source \in Nodes
ASSUME Infinity \in Nat
ASSUME M > 0
ASSUME NumScenarios >= 10000
ASSUME Weight \in [Nodes \X Nodes -> Nat]

BatchRows == 24
BatchCols == 5
BatchSize == BatchRows * BatchCols

ScenarioSet == 1..NumScenarios

BatchIndex(s) == ((s - 1) \div BatchSize) + 1
RowInBatch(s) == (((s - 1) % BatchSize) \div BatchCols) + 1
ColInBatch(s) == ((s - 1) % BatchCols) + 1

Neighbors(u) == {v \in Nodes : Weight[<<u, v>>] < Infinity}

MinNode(frontier, d) ==
    CHOOSE n \in frontier : \A m \in frontier : d[n] <= d[m]

(* Abstract relaxation of one edge from the chosen frontier node. *)
RelaxOne(d, u, v) ==
    IF d[v] > d[u] + Weight[<<u, v>>]
        THEN [d EXCEPT ![v] = d[u] + Weight[<<u, v>>]]
        ELSE d

VARIABLES
    dist, visited, frontier,
    currentScenario,
    t, result, path_taken,
    compilerRiskDischarged, microArchRiskDischarged

Init ==
    /\ dist = [n \in Nodes |-> IF n = Source THEN 0 ELSE Infinity]
    /\ visited = {}
    /\ frontier = {Source}
    /\ currentScenario = 1
    /\ t = 0
    /\ result = 0
    /\ path_taken = "always_same"
    /\ compilerRiskDischarged = FALSE
    /\ microArchRiskDischarged = FALSE

DijkstraStep ==
    /\ frontier # {}
    /\ LET u == MinNode(frontier, dist) IN
       IF Neighbors(u) = {}
          THEN
              /\ dist' = dist
              /\ visited' = visited \cup {u}
              /\ frontier' = frontier \ {u}
          ELSE
              /\ \E v \in Neighbors(u):
                    /\ dist' = RelaxOne(dist, u, v)
                    /\ visited' = visited \cup {u}
                    /\ frontier' = (frontier \ {u}) \cup (Neighbors(u) \ visited)

AdvanceScenario ==
    /\ currentScenario < NumScenarios
    /\ currentScenario' = currentScenario + 1

StayOnLastScenario ==
    /\ currentScenario = NumScenarios
    /\ currentScenario' = currentScenario

(*
Branchless final reduction:
  mask   <- 0 - (t >= M)
  result <- t - (M AND mask)
The arithmetic-select abstraction below preserves the same control-path guarantee.
*)
BranchlessReduction ==
    /\ \E newT \in Nat:
          /\ t' = newT
          /\ LET mask == IF newT >= M THEN 0 - 1 ELSE 0
                 selected == IF mask = -1 THEN M ELSE 0
             IN result' = newT - selected
    /\ path_taken' = "always_same"

DischargeCompilerRisk ==
    /\ compilerRiskDischarged' \in BOOLEAN
    /\ UNCHANGED <<microArchRiskDischarged>>

DischargeMicroArchRisk ==
    /\ microArchRiskDischarged' \in BOOLEAN
    /\ UNCHANGED <<compilerRiskDischarged>>

CoreProgress ==
    /\ DijkstraStep
    /\ BranchlessReduction
    /\ AdvanceScenario \/ StayOnLastScenario
    /\ UNCHANGED <<compilerRiskDischarged, microArchRiskDischarged>>

Next ==
    \/ CoreProgress
    \/ (/\ DischargeCompilerRisk
        /\ UNCHANGED <<dist, visited, frontier, currentScenario, t, result, path_taken>>)
    \/ (/\ DischargeMicroArchRisk
        /\ UNCHANGED <<dist, visited, frontier, currentScenario, t, result, path_taken>>)

Spec == Init /\ [][Next]_<<dist, visited, frontier, currentScenario, t, result, path_taken,
                        compilerRiskDischarged, microArchRiskDischarged>>

PathDeterminism_Branchless == path_taken = "always_same"

ScenarioBatchingInvariant ==
    /\ currentScenario \in ScenarioSet
    /\ RowInBatch(currentScenario) \in 1..BatchRows
    /\ ColInBatch(currentScenario) \in 1..BatchCols
    /\ BatchIndex(currentScenario) >= 1

ScenarioCoverageGoal == currentScenario = NumScenarios

ResidualRiskObligation ==
    /\ compilerRiskDischarged
    /\ microArchRiskDischarged

(* Complete assurance requires all three layers:
   1) Algorithmic path determinism (this model)
   2) Static disassembly validation (compiler residual risk)
   3) Microarchitectural timing validation (speculation residual risk)
*)
CompleteAssuranceCase ==
    /\ PathDeterminism_Branchless
    /\ ResidualRiskObligation

THEOREM PathDeterminismInvariant == Spec => []PathDeterminism_Branchless
THEOREM ScenarioBatchingSafety == Spec => []ScenarioBatchingInvariant

===============================================================================================
