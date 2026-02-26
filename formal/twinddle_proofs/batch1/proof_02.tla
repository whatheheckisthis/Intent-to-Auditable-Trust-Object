---- MODULE Twinddle_batch1_02 ----
EXTENDS Naturals, Sequences

(*** Proof sketch set 5-8 ***)

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

Lemma5 ==
  /\ AxiomStaticTopology
  /\ AxiomDeterministicTieBreak
  /\ AxiomOfflineSchedule
  => Theorem5

Theorem5 ==
  TwinddleBetterThanFactoring /\ CorrelationPreserved

Lemma6 ==
  /\ AxiomStaticTopology
  /\ AxiomDeterministicTieBreak
  /\ AxiomOfflineSchedule
  => Theorem6

Theorem6 ==
  TwinddleBetterThanFactoring /\ CorrelationPreserved

Lemma7 ==
  /\ AxiomStaticTopology
  /\ AxiomDeterministicTieBreak
  /\ AxiomOfflineSchedule
  => Theorem7

Theorem7 ==
  TwinddleBetterThanFactoring /\ CorrelationPreserved

Lemma8 ==
  /\ AxiomStaticTopology
  /\ AxiomDeterministicTieBreak
  /\ AxiomOfflineSchedule
  => Theorem8

Theorem8 ==
  TwinddleBetterThanFactoring /\ CorrelationPreserved

====
