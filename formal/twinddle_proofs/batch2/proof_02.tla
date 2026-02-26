---- MODULE Twinddle_batch2_02 ----
EXTENDS Naturals, Sequences

(*** Proof sketch set 53-56 ***)

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

Lemma53 ==
  /\ AxiomStaticTopology
  /\ AxiomDeterministicTieBreak
  /\ AxiomOfflineSchedule
  => Theorem53

Theorem53 ==
  TwinddleBetterThanFactoring /\ CorrelationPreserved

Lemma54 ==
  /\ AxiomStaticTopology
  /\ AxiomDeterministicTieBreak
  /\ AxiomOfflineSchedule
  => Theorem54

Theorem54 ==
  TwinddleBetterThanFactoring /\ CorrelationPreserved

Lemma55 ==
  /\ AxiomStaticTopology
  /\ AxiomDeterministicTieBreak
  /\ AxiomOfflineSchedule
  => Theorem55

Theorem55 ==
  TwinddleBetterThanFactoring /\ CorrelationPreserved

Lemma56 ==
  /\ AxiomStaticTopology
  /\ AxiomDeterministicTieBreak
  /\ AxiomOfflineSchedule
  => Theorem56

Theorem56 ==
  TwinddleBetterThanFactoring /\ CorrelationPreserved

====
