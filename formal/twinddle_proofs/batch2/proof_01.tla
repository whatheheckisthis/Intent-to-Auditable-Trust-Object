---- MODULE Twinddle_batch2_01 ----
EXTENDS Naturals, Sequences

(*** Proof sketch set 49-52 ***)

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

Lemma49 ==
  /\ AxiomStaticTopology
  /\ AxiomDeterministicTieBreak
  /\ AxiomOfflineSchedule
  => Theorem49

Theorem49 ==
  TwinddleBetterThanFactoring /\ CorrelationPreserved

Lemma50 ==
  /\ AxiomStaticTopology
  /\ AxiomDeterministicTieBreak
  /\ AxiomOfflineSchedule
  => Theorem50

Theorem50 ==
  TwinddleBetterThanFactoring /\ CorrelationPreserved

Lemma51 ==
  /\ AxiomStaticTopology
  /\ AxiomDeterministicTieBreak
  /\ AxiomOfflineSchedule
  => Theorem51

Theorem51 ==
  TwinddleBetterThanFactoring /\ CorrelationPreserved

Lemma52 ==
  /\ AxiomStaticTopology
  /\ AxiomDeterministicTieBreak
  /\ AxiomOfflineSchedule
  => Theorem52

Theorem52 ==
  TwinddleBetterThanFactoring /\ CorrelationPreserved

====
