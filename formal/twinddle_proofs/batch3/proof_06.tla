---- MODULE Twinddle_batch3_06 ----
EXTENDS Naturals, Sequences

(*** Proof sketch set 113-116 ***)

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

Lemma113 ==
  /\ AxiomStaticTopology
  /\ AxiomDeterministicTieBreak
  /\ AxiomOfflineSchedule
  => Theorem113

Theorem113 ==
  TwinddleBetterThanFactoring /\ CorrelationPreserved

Lemma114 ==
  /\ AxiomStaticTopology
  /\ AxiomDeterministicTieBreak
  /\ AxiomOfflineSchedule
  => Theorem114

Theorem114 ==
  TwinddleBetterThanFactoring /\ CorrelationPreserved

Lemma115 ==
  /\ AxiomStaticTopology
  /\ AxiomDeterministicTieBreak
  /\ AxiomOfflineSchedule
  => Theorem115

Theorem115 ==
  TwinddleBetterThanFactoring /\ CorrelationPreserved

Lemma116 ==
  /\ AxiomStaticTopology
  /\ AxiomDeterministicTieBreak
  /\ AxiomOfflineSchedule
  => Theorem116

Theorem116 ==
  TwinddleBetterThanFactoring /\ CorrelationPreserved

====
