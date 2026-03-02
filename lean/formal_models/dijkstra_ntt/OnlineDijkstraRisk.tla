---- MODULE OnlineDijkstraRisk ----
EXTENDS Naturals

CONSTANT RuntimePriorityQueueBranches

Risky == RuntimePriorityQueueBranches > 0

NoOnlineRequirement == ~Risky

====
