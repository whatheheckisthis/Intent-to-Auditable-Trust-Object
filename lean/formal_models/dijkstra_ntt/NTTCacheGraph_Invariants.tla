---- MODULE NTTCacheGraph_Invariants ----
EXTENDS NTTCacheGraph, TLC

NoSecretDependency == TRUE 

MonotoneVisited == \A n \in visited : n \in Nodes

DistNonNegative == \A n \in Nodes : dist[n] >= 0

Inv == NoSecretDependency /\ MonotoneVisited /\ DistNonNegative

THEOREM Spec => []Inv

====
