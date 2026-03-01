---- MODULE NTTCacheGraph ----
EXTENDS Naturals, Sequences, TLC

CONSTANTS Nodes, Start, Goal, Weights

VARIABLES frontier, dist, visited, order

Init ==
  /\ frontier = <<Start>>
  /\ dist = [n \in Nodes |-> IF n = Start THEN 0 ELSE 999999]
  /\ visited = {}
  /\ order = << >>

Neighbors(n) == {m \in Nodes : <<n,m>> \in DOMAIN Weights}
W(n,m) == Weights[<<n,m>>]

PickMin(S) == CHOOSE x \in S : \A y \in S : dist[x] <= dist[y]

Step ==
  \E u \in (Nodes \ visited):
    /\ u = PickMin(Nodes \ visited)
    /\ visited' = visited \cup {u}
    /\ order' = Append(order, u)
    /\ dist' = [n \in Nodes |->
         IF n \in Neighbors(u) /\ dist[u] + W(u,n) < dist[n]
         THEN dist[u] + W(u,n)
         ELSE dist[n]]
    /\ frontier' = frontier

Next == Step

Spec == Init /\ [][Next]_<<frontier, dist, visited, order>>

DeterministicOrder == \A i,j \in 1..Len(order): i < j => order[i] # order[j]

====
