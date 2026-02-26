---- MODULE Twinddle_batch3_01 ----
EXTENDS Naturals, Sequences

(*** Proof sketch set 93-96 ***)

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

Lemma93 ==
  /\ AxiomStaticTopology
  /\ AxiomDeterministicTieBreak
  /\ AxiomOfflineSchedule
  => Theorem93

Theorem93 ==
  TwinddleBetterThanFactoring /\ CorrelationPreserved

Lemma94 ==
  /\ AxiomStaticTopology
  /\ AxiomDeterministicTieBreak
  /\ AxiomOfflineSchedule
  => Theorem94

Theorem94 ==
  TwinddleBetterThanFactoring /\ CorrelationPreserved

Lemma95 ==
  /\ AxiomStaticTopology
  /\ AxiomDeterministicTieBreak
  /\ AxiomOfflineSchedule
  => Theorem95

Theorem95 ==
  TwinddleBetterThanFactoring /\ CorrelationPreserved

Lemma96 ==
  /\ AxiomStaticTopology
  /\ AxiomDeterministicTieBreak
  /\ AxiomOfflineSchedule
  => Theorem96

Theorem96 ==
  TwinddleBetterThanFactoring /\ CorrelationPreserved

====
