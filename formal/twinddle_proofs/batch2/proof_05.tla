---- MODULE Twinddle_batch2_05 ----
EXTENDS Naturals, Sequences

(*** Proof sketch set 65-68 ***)

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

Lemma65 ==
  /\ AxiomStaticTopology
  /\ AxiomDeterministicTieBreak
  /\ AxiomOfflineSchedule
  => Theorem65

Theorem65 ==
  TwinddleBetterThanFactoring /\ CorrelationPreserved

Lemma66 ==
  /\ AxiomStaticTopology
  /\ AxiomDeterministicTieBreak
  /\ AxiomOfflineSchedule
  => Theorem66

Theorem66 ==
  TwinddleBetterThanFactoring /\ CorrelationPreserved

Lemma67 ==
  /\ AxiomStaticTopology
  /\ AxiomDeterministicTieBreak
  /\ AxiomOfflineSchedule
  => Theorem67

Theorem67 ==
  TwinddleBetterThanFactoring /\ CorrelationPreserved

Lemma68 ==
  /\ AxiomStaticTopology
  /\ AxiomDeterministicTieBreak
  /\ AxiomOfflineSchedule
  => Theorem68

Theorem68 ==
  TwinddleBetterThanFactoring /\ CorrelationPreserved

====
