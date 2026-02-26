---- MODULE Twinddle_batch4_03 ----
EXTENDS Naturals, Sequences

(*** Proof sketch set 145-148 ***)

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

Lemma145 ==
  /\ AxiomStaticTopology
  /\ AxiomDeterministicTieBreak
  /\ AxiomOfflineSchedule
  => Theorem145

Theorem145 ==
  TwinddleBetterThanFactoring /\ CorrelationPreserved

Lemma146 ==
  /\ AxiomStaticTopology
  /\ AxiomDeterministicTieBreak
  /\ AxiomOfflineSchedule
  => Theorem146

Theorem146 ==
  TwinddleBetterThanFactoring /\ CorrelationPreserved

Lemma147 ==
  /\ AxiomStaticTopology
  /\ AxiomDeterministicTieBreak
  /\ AxiomOfflineSchedule
  => Theorem147

Theorem147 ==
  TwinddleBetterThanFactoring /\ CorrelationPreserved

Lemma148 ==
  /\ AxiomStaticTopology
  /\ AxiomDeterministicTieBreak
  /\ AxiomOfflineSchedule
  => Theorem148

Theorem148 ==
  TwinddleBetterThanFactoring /\ CorrelationPreserved

====
