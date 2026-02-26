---- MODULE Twinddle_batch1_01 ----
EXTENDS Naturals, Sequences

(*** Proof sketch set 1-4 ***)

CONSTANTS
  CachePenalty, PipelineHazard, TableStride, FactoringOverhead

AxiomStaticTopology == TableStride > 0
AxiomDeterministicTieBreak == TRUE
AxiomOfflineSchedule == TRUE
AxiomFactoringPenalty == FactoringOverhead >= 1

TwinddleBetterThanFactoring ==
  CachePenalty + PipelineHazard <= CachePenalty + PipelineHazard + FactoringOverhead

CorrelationPreserved ==
  /\ TableStride > 0
  /\ CachePenalty >= 0
  /\ PipelineHazard >= 0

Lemma1 ==
  /\ AxiomStaticTopology
  /\ AxiomDeterministicTieBreak
  /\ AxiomOfflineSchedule
  => Theorem1

Theorem1 ==
  TwinddleBetterThanFactoring /\ CorrelationPreserved

Lemma2 ==
  /\ AxiomStaticTopology
  /\ AxiomDeterministicTieBreak
  /\ AxiomOfflineSchedule
  => Theorem2

Theorem2 ==
  TwinddleBetterThanFactoring /\ CorrelationPreserved

Lemma3 ==
  /\ AxiomStaticTopology
  /\ AxiomDeterministicTieBreak
  /\ AxiomOfflineSchedule
  => Theorem3

Theorem3 ==
  TwinddleBetterThanFactoring /\ CorrelationPreserved

Lemma4 ==
  /\ AxiomStaticTopology
  /\ AxiomDeterministicTieBreak
  /\ AxiomOfflineSchedule
  => Theorem4

Theorem4 ==
  TwinddleBetterThanFactoring /\ CorrelationPreserved

====
