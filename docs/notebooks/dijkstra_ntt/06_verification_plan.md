# Verification Plan

1. Model cache-layer traversal in TLA+ (`NTTCacheGraph.tla`, `NTTTraversalOffline.tla`).
2. Model pipeline-layer scheduling in TLA+ (`SVE2ScheduleGraph.tla`).
3. Assert non-interference: no transition guard references `SecretCoeff`.
4. Assert deterministic tie-breaking and schedule uniqueness.
5. Run TLC configs for small parameter sets.
6. Link resulting traces to implementation code generation checks.

Outcome target: proof obligations supporting the statement that Dijkstra contributes performance optimization while preserving constant-time only when used as an offline static synthesis step.
