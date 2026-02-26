---- MODULE Twinddle_batch4_05 ----
EXTENDS Naturals, Sequences

(*** Proof sketch set 153-156 ***)

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

Lemma153 ==
  /\ AxiomStaticTopology
  /\ AxiomDeterministicTieBreak
  /\ AxiomOfflineSchedule
  => Theorem153

Theorem153 ==
  TwinddleBetterThanFactoring /\ CorrelationPreserved

Lemma154 ==
  /\ AxiomStaticTopology
  /\ AxiomDeterministicTieBreak
  /\ AxiomOfflineSchedule
  => Theorem154

Theorem154 ==
  TwinddleBetterThanFactoring /\ CorrelationPreserved

Lemma155 ==
  /\ AxiomStaticTopology
  /\ AxiomDeterministicTieBreak
  /\ AxiomOfflineSchedule
  => Theorem155

Theorem155 ==
  TwinddleBetterThanFactoring /\ CorrelationPreserved

Lemma156 ==
  /\ AxiomStaticTopology
  /\ AxiomDeterministicTieBreak
  /\ AxiomOfflineSchedule
  => Theorem156

Theorem156 ==
  TwinddleBetterThanFactoring /\ CorrelationPreserved

====
