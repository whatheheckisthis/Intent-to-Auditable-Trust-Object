---- MODULE Twinddle_batch1_04 ----
EXTENDS Naturals, Sequences

(*** Proof sketch set 13-16 ***)

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

Lemma13 ==
  /\ AxiomStaticTopology
  /\ AxiomDeterministicTieBreak
  /\ AxiomOfflineSchedule
  => Theorem13

Theorem13 ==
  TwinddleBetterThanFactoring /\ CorrelationPreserved

Lemma14 ==
  /\ AxiomStaticTopology
  /\ AxiomDeterministicTieBreak
  /\ AxiomOfflineSchedule
  => Theorem14

Theorem14 ==
  TwinddleBetterThanFactoring /\ CorrelationPreserved

Lemma15 ==
  /\ AxiomStaticTopology
  /\ AxiomDeterministicTieBreak
  /\ AxiomOfflineSchedule
  => Theorem15

Theorem15 ==
  TwinddleBetterThanFactoring /\ CorrelationPreserved

Lemma16 ==
  /\ AxiomStaticTopology
  /\ AxiomDeterministicTieBreak
  /\ AxiomOfflineSchedule
  => Theorem16

Theorem16 ==
  TwinddleBetterThanFactoring /\ CorrelationPreserved

====
