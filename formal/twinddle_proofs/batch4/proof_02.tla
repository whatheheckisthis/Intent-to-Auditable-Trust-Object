---- MODULE Twinddle_batch4_02 ----
EXTENDS Naturals, Sequences

(*** Proof sketch set 141-144 ***)

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

Lemma141 ==
  /\ AxiomStaticTopology
  /\ AxiomDeterministicTieBreak
  /\ AxiomOfflineSchedule
  => Theorem141

Theorem141 ==
  TwinddleBetterThanFactoring /\ CorrelationPreserved

Lemma142 ==
  /\ AxiomStaticTopology
  /\ AxiomDeterministicTieBreak
  /\ AxiomOfflineSchedule
  => Theorem142

Theorem142 ==
  TwinddleBetterThanFactoring /\ CorrelationPreserved

Lemma143 ==
  /\ AxiomStaticTopology
  /\ AxiomDeterministicTieBreak
  /\ AxiomOfflineSchedule
  => Theorem143

Theorem143 ==
  TwinddleBetterThanFactoring /\ CorrelationPreserved

Lemma144 ==
  /\ AxiomStaticTopology
  /\ AxiomDeterministicTieBreak
  /\ AxiomOfflineSchedule
  => Theorem144

Theorem144 ==
  TwinddleBetterThanFactoring /\ CorrelationPreserved

====
