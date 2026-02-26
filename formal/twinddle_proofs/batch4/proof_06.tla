---- MODULE Twinddle_batch4_06 ----
EXTENDS Naturals, Sequences

(*** Proof sketch set 157-160 ***)

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

Lemma157 ==
  /\ AxiomStaticTopology
  /\ AxiomDeterministicTieBreak
  /\ AxiomOfflineSchedule
  => Theorem157

Theorem157 ==
  TwinddleBetterThanFactoring /\ CorrelationPreserved

Lemma158 ==
  /\ AxiomStaticTopology
  /\ AxiomDeterministicTieBreak
  /\ AxiomOfflineSchedule
  => Theorem158

Theorem158 ==
  TwinddleBetterThanFactoring /\ CorrelationPreserved

Lemma159 ==
  /\ AxiomStaticTopology
  /\ AxiomDeterministicTieBreak
  /\ AxiomOfflineSchedule
  => Theorem159

Theorem159 ==
  TwinddleBetterThanFactoring /\ CorrelationPreserved

Lemma160 ==
  /\ AxiomStaticTopology
  /\ AxiomDeterministicTieBreak
  /\ AxiomOfflineSchedule
  => Theorem160

Theorem160 ==
  TwinddleBetterThanFactoring /\ CorrelationPreserved

====
