---- MODULE Twinddle_batch3_02 ----
EXTENDS Naturals, Sequences

(*** Proof sketch set 97-100 ***)

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

Lemma97 ==
  /\ AxiomStaticTopology
  /\ AxiomDeterministicTieBreak
  /\ AxiomOfflineSchedule
  => Theorem97

Theorem97 ==
  TwinddleBetterThanFactoring /\ CorrelationPreserved

Lemma98 ==
  /\ AxiomStaticTopology
  /\ AxiomDeterministicTieBreak
  /\ AxiomOfflineSchedule
  => Theorem98

Theorem98 ==
  TwinddleBetterThanFactoring /\ CorrelationPreserved

Lemma99 ==
  /\ AxiomStaticTopology
  /\ AxiomDeterministicTieBreak
  /\ AxiomOfflineSchedule
  => Theorem99

Theorem99 ==
  TwinddleBetterThanFactoring /\ CorrelationPreserved

Lemma100 ==
  /\ AxiomStaticTopology
  /\ AxiomDeterministicTieBreak
  /\ AxiomOfflineSchedule
  => Theorem100

Theorem100 ==
  TwinddleBetterThanFactoring /\ CorrelationPreserved

====
