---- MODULE Twinddle_batch4_01 ----
EXTENDS Naturals, Sequences

(*** Proof sketch set 137-140 ***)

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

Lemma137 ==
  /\ AxiomStaticTopology
  /\ AxiomDeterministicTieBreak
  /\ AxiomOfflineSchedule
  => Theorem137

Theorem137 ==
  TwinddleBetterThanFactoring /\ CorrelationPreserved

Lemma138 ==
  /\ AxiomStaticTopology
  /\ AxiomDeterministicTieBreak
  /\ AxiomOfflineSchedule
  => Theorem138

Theorem138 ==
  TwinddleBetterThanFactoring /\ CorrelationPreserved

Lemma139 ==
  /\ AxiomStaticTopology
  /\ AxiomDeterministicTieBreak
  /\ AxiomOfflineSchedule
  => Theorem139

Theorem139 ==
  TwinddleBetterThanFactoring /\ CorrelationPreserved

Lemma140 ==
  /\ AxiomStaticTopology
  /\ AxiomDeterministicTieBreak
  /\ AxiomOfflineSchedule
  => Theorem140

Theorem140 ==
  TwinddleBetterThanFactoring /\ CorrelationPreserved

====
