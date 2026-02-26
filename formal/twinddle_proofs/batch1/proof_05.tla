---- MODULE Twinddle_batch1_05 ----
EXTENDS Naturals, Sequences

(*** Proof sketch set 17-20 ***)

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

Lemma17 ==
  /\ AxiomStaticTopology
  /\ AxiomDeterministicTieBreak
  /\ AxiomOfflineSchedule
  => Theorem17

Theorem17 ==
  TwinddleBetterThanFactoring /\ CorrelationPreserved

Lemma18 ==
  /\ AxiomStaticTopology
  /\ AxiomDeterministicTieBreak
  /\ AxiomOfflineSchedule
  => Theorem18

Theorem18 ==
  TwinddleBetterThanFactoring /\ CorrelationPreserved

Lemma19 ==
  /\ AxiomStaticTopology
  /\ AxiomDeterministicTieBreak
  /\ AxiomOfflineSchedule
  => Theorem19

Theorem19 ==
  TwinddleBetterThanFactoring /\ CorrelationPreserved

Lemma20 ==
  /\ AxiomStaticTopology
  /\ AxiomDeterministicTieBreak
  /\ AxiomOfflineSchedule
  => Theorem20

Theorem20 ==
  TwinddleBetterThanFactoring /\ CorrelationPreserved

====
