---- MODULE NTTTraversalOffline ----
EXTENDS Naturals, Sequences

CONSTANTS PrecomputedOrder, Nodes
VARIABLE pc

Init == pc = 1

Next ==
  /\ pc <= Len(PrecomputedOrder)
  /\ PrecomputedOrder[pc] \in Nodes
  /\ pc' = pc + 1

Spec == Init /\ [][Next]_pc

Deterministic == \A i \in 1..Len(PrecomputedOrder): PrecomputedOrder[i] \in Nodes

====
