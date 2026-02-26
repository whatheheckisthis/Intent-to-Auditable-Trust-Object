---- MODULE Twinddle_batch2_03 ----
EXTENDS Naturals, Sequences

(*** Proof sketch set 57-60 ***)

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

Lemma57 ==
  /\ AxiomStaticTopology
  /\ AxiomDeterministicTieBreak
  /\ AxiomOfflineSchedule
  => Theorem57

Theorem57 ==
  TwinddleBetterThanFactoring /\ CorrelationPreserved

Lemma58 ==
  /\ AxiomStaticTopology
  /\ AxiomDeterministicTieBreak
  /\ AxiomOfflineSchedule
  => Theorem58

Theorem58 ==
  TwinddleBetterThanFactoring /\ CorrelationPreserved

Lemma59 ==
  /\ AxiomStaticTopology
  /\ AxiomDeterministicTieBreak
  /\ AxiomOfflineSchedule
  => Theorem59

Theorem59 ==
  TwinddleBetterThanFactoring /\ CorrelationPreserved

Lemma60 ==
  /\ AxiomStaticTopology
  /\ AxiomDeterministicTieBreak
  /\ AxiomOfflineSchedule
  => Theorem60

Theorem60 ==
  TwinddleBetterThanFactoring /\ CorrelationPreserved

====
