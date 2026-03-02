---- MODULE NTTTieBreaking ----
EXTENDS Naturals, Sequences

CONSTANTS Candidates, Dist, NodeId, EdgeId

Rank(c) == <<Dist[c], NodeId[c], EdgeId[c]>>

MinCandidate == CHOOSE c \in Candidates : \A d \in Candidates : Rank(c) <= Rank(d)

DeterministicTieBreak == \A c,d \in Candidates : Rank(c)=Rank(d) => c=d

====
