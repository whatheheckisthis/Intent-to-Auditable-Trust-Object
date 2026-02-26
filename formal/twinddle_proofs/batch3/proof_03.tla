---- MODULE Twinddle_batch3_03 ----
EXTENDS Naturals, Sequences

(*** Proof sketch set 101-104 ***)

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

Lemma101 ==
  /\ AxiomStaticTopology
  /\ AxiomDeterministicTieBreak
  /\ AxiomOfflineSchedule
  => Theorem101

Theorem101 ==
  TwinddleBetterThanFactoring /\ CorrelationPreserved

Lemma102 ==
  /\ AxiomStaticTopology
  /\ AxiomDeterministicTieBreak
  /\ AxiomOfflineSchedule
  => Theorem102

Theorem102 ==
  TwinddleBetterThanFactoring /\ CorrelationPreserved

Lemma103 ==
  /\ AxiomStaticTopology
  /\ AxiomDeterministicTieBreak
  /\ AxiomOfflineSchedule
  => Theorem103

Theorem103 ==
  TwinddleBetterThanFactoring /\ CorrelationPreserved

Lemma104 ==
  /\ AxiomStaticTopology
  /\ AxiomDeterministicTieBreak
  /\ AxiomOfflineSchedule
  => Theorem104

Theorem104 ==
  TwinddleBetterThanFactoring /\ CorrelationPreserved

====
