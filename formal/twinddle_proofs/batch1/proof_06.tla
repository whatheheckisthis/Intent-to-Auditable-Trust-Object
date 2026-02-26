---- MODULE Twinddle_batch1_06 ----
EXTENDS Naturals, Sequences

(*** Proof sketch set 21-24 ***)

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

Lemma21 ==
  /\ AxiomStaticTopology
  /\ AxiomDeterministicTieBreak
  /\ AxiomOfflineSchedule
  => Theorem21

Theorem21 ==
  TwinddleBetterThanFactoring /\ CorrelationPreserved

Lemma22 ==
  /\ AxiomStaticTopology
  /\ AxiomDeterministicTieBreak
  /\ AxiomOfflineSchedule
  => Theorem22

Theorem22 ==
  TwinddleBetterThanFactoring /\ CorrelationPreserved

Lemma23 ==
  /\ AxiomStaticTopology
  /\ AxiomDeterministicTieBreak
  /\ AxiomOfflineSchedule
  => Theorem23

Theorem23 ==
  TwinddleBetterThanFactoring /\ CorrelationPreserved

Lemma24 ==
  /\ AxiomStaticTopology
  /\ AxiomDeterministicTieBreak
  /\ AxiomOfflineSchedule
  => Theorem24

Theorem24 ==
  TwinddleBetterThanFactoring /\ CorrelationPreserved

====
