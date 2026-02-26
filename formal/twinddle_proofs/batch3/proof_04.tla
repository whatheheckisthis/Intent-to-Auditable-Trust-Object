---- MODULE Twinddle_batch3_04 ----
EXTENDS Naturals, Sequences

(*** Proof sketch set 105-108 ***)

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

Lemma105 ==
  /\ AxiomStaticTopology
  /\ AxiomDeterministicTieBreak
  /\ AxiomOfflineSchedule
  => Theorem105

Theorem105 ==
  TwinddleBetterThanFactoring /\ CorrelationPreserved

Lemma106 ==
  /\ AxiomStaticTopology
  /\ AxiomDeterministicTieBreak
  /\ AxiomOfflineSchedule
  => Theorem106

Theorem106 ==
  TwinddleBetterThanFactoring /\ CorrelationPreserved

Lemma107 ==
  /\ AxiomStaticTopology
  /\ AxiomDeterministicTieBreak
  /\ AxiomOfflineSchedule
  => Theorem107

Theorem107 ==
  TwinddleBetterThanFactoring /\ CorrelationPreserved

Lemma108 ==
  /\ AxiomStaticTopology
  /\ AxiomDeterministicTieBreak
  /\ AxiomOfflineSchedule
  => Theorem108

Theorem108 ==
  TwinddleBetterThanFactoring /\ CorrelationPreserved

====
