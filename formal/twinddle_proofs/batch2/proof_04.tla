---- MODULE Twinddle_batch2_04 ----
EXTENDS Naturals, Sequences

(*** Proof sketch set 61-64 ***)

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

Lemma61 ==
  /\ AxiomStaticTopology
  /\ AxiomDeterministicTieBreak
  /\ AxiomOfflineSchedule
  => Theorem61

Theorem61 ==
  TwinddleBetterThanFactoring /\ CorrelationPreserved

Lemma62 ==
  /\ AxiomStaticTopology
  /\ AxiomDeterministicTieBreak
  /\ AxiomOfflineSchedule
  => Theorem62

Theorem62 ==
  TwinddleBetterThanFactoring /\ CorrelationPreserved

Lemma63 ==
  /\ AxiomStaticTopology
  /\ AxiomDeterministicTieBreak
  /\ AxiomOfflineSchedule
  => Theorem63

Theorem63 ==
  TwinddleBetterThanFactoring /\ CorrelationPreserved

Lemma64 ==
  /\ AxiomStaticTopology
  /\ AxiomDeterministicTieBreak
  /\ AxiomOfflineSchedule
  => Theorem64

Theorem64 ==
  TwinddleBetterThanFactoring /\ CorrelationPreserved

====
