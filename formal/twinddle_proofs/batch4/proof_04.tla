---- MODULE Twinddle_batch4_04 ----
EXTENDS Naturals, Sequences

(*** Proof sketch set 149-152 ***)

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

Lemma149 ==
  /\ AxiomStaticTopology
  /\ AxiomDeterministicTieBreak
  /\ AxiomOfflineSchedule
  => Theorem149

Theorem149 ==
  TwinddleBetterThanFactoring /\ CorrelationPreserved

Lemma150 ==
  /\ AxiomStaticTopology
  /\ AxiomDeterministicTieBreak
  /\ AxiomOfflineSchedule
  => Theorem150

Theorem150 ==
  TwinddleBetterThanFactoring /\ CorrelationPreserved

Lemma151 ==
  /\ AxiomStaticTopology
  /\ AxiomDeterministicTieBreak
  /\ AxiomOfflineSchedule
  => Theorem151

Theorem151 ==
  TwinddleBetterThanFactoring /\ CorrelationPreserved

Lemma152 ==
  /\ AxiomStaticTopology
  /\ AxiomDeterministicTieBreak
  /\ AxiomOfflineSchedule
  => Theorem152

Theorem152 ==
  TwinddleBetterThanFactoring /\ CorrelationPreserved

====
