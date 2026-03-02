---- MODULE RefinementMapping ----
EXTENDS Naturals, Sequences

CONSTANTS HighLevelOrder, LowLevelTrace

Refines == \A i \in 1..Len(HighLevelOrder): HighLevelOrder[i] = LowLevelTrace[i]

====
