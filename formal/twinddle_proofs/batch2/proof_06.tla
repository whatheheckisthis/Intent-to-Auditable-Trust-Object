---- MODULE Twinddle_batch2_06 ----
EXTENDS Naturals, Sequences

(*** Proof sketch set 69-72 ***)

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

Lemma69 ==
  /\ AxiomStaticTopology
  /\ AxiomDeterministicTieBreak
  /\ AxiomOfflineSchedule
  => Theorem69

Theorem69 ==
  TwinddleBetterThanFactoring /\ CorrelationPreserved

Lemma70 ==
  /\ AxiomStaticTopology
  /\ AxiomDeterministicTieBreak
  /\ AxiomOfflineSchedule
  => Theorem70

Theorem70 ==
  TwinddleBetterThanFactoring /\ CorrelationPreserved

Lemma71 ==
  /\ AxiomStaticTopology
  /\ AxiomDeterministicTieBreak
  /\ AxiomOfflineSchedule
  => Theorem71

Theorem71 ==
  TwinddleBetterThanFactoring /\ CorrelationPreserved

Lemma72 ==
  /\ AxiomStaticTopology
  /\ AxiomDeterministicTieBreak
  /\ AxiomOfflineSchedule
  => Theorem72

Theorem72 ==
  TwinddleBetterThanFactoring /\ CorrelationPreserved

====
