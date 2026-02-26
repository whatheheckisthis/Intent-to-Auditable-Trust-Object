---- MODULE Twinddle_batch1_03 ----
EXTENDS Naturals, Sequences

(*** Proof sketch set 9-12 ***)

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

Lemma9 ==
  /\ AxiomStaticTopology
  /\ AxiomDeterministicTieBreak
  /\ AxiomOfflineSchedule
  => Theorem9

Theorem9 ==
  TwinddleBetterThanFactoring /\ CorrelationPreserved

Lemma10 ==
  /\ AxiomStaticTopology
  /\ AxiomDeterministicTieBreak
  /\ AxiomOfflineSchedule
  => Theorem10

Theorem10 ==
  TwinddleBetterThanFactoring /\ CorrelationPreserved

Lemma11 ==
  /\ AxiomStaticTopology
  /\ AxiomDeterministicTieBreak
  /\ AxiomOfflineSchedule
  => Theorem11

Theorem11 ==
  TwinddleBetterThanFactoring /\ CorrelationPreserved

Lemma12 ==
  /\ AxiomStaticTopology
  /\ AxiomDeterministicTieBreak
  /\ AxiomOfflineSchedule
  => Theorem12

Theorem12 ==
  TwinddleBetterThanFactoring /\ CorrelationPreserved

====
