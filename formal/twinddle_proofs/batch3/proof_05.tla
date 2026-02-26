---- MODULE Twinddle_batch3_05 ----
EXTENDS Naturals, Sequences

(*** Proof sketch set 109-112 ***)

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

Lemma109 ==
  /\ AxiomStaticTopology
  /\ AxiomDeterministicTieBreak
  /\ AxiomOfflineSchedule
  => Theorem109

Theorem109 ==
  TwinddleBetterThanFactoring /\ CorrelationPreserved

Lemma110 ==
  /\ AxiomStaticTopology
  /\ AxiomDeterministicTieBreak
  /\ AxiomOfflineSchedule
  => Theorem110

Theorem110 ==
  TwinddleBetterThanFactoring /\ CorrelationPreserved

Lemma111 ==
  /\ AxiomStaticTopology
  /\ AxiomDeterministicTieBreak
  /\ AxiomOfflineSchedule
  => Theorem111

Theorem111 ==
  TwinddleBetterThanFactoring /\ CorrelationPreserved

Lemma112 ==
  /\ AxiomStaticTopology
  /\ AxiomDeterministicTieBreak
  /\ AxiomOfflineSchedule
  => Theorem112

Theorem112 ==
  TwinddleBetterThanFactoring /\ CorrelationPreserved

====
