# Constant-Time Conditions for Both Layers

Necessary conditions:

1. Graph topology is generated from algorithm parameters (`N`, radix, vector width), not runtime coefficient values.
2. Edge weights are static calibration constants from offline profiling/architecture manuals.
3. Tie-breaking is deterministic and canonical (e.g., smallest node id, then lexical edge id).
4. Dijkstra runs offline; runtime executes precomputed schedule only.
5. SVE2 predicates that affect issued instructions are not secret-derived.

If any condition fails, shortest-path output may become data-dependent and invalidate constant-time claims.
