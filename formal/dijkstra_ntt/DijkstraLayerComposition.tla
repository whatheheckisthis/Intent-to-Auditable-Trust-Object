---- MODULE DijkstraLayerComposition ----
EXTENDS Naturals, Sequences

CONSTANTS CacheOrder, InstrOrder

ComposedSchedule == <<CacheOrder, InstrOrder>>

BothOffline == Len(CacheOrder) > 0 /\ Len(InstrOrder) > 0

====
