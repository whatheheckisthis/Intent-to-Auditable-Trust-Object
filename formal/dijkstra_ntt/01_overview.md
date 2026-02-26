# Dijkstra Framing for NTT Twiddle Traversal and SVE2 Scheduling

This package separates two valid uses of Dijkstra:

1. **Interpretation A (cache layer):** choose a fixed traversal order for NTT butterfly/twiddle accesses that minimizes cache-cost over a static graph.
2. **Interpretation B (pipeline layer):** choose a fixed instruction order over a static SVE2 dependency graph to minimize hazard penalties.

The key constant-time conclusion is the same for both layers: the shortest-path output must be computed **offline** and compiled as a fixed schedule. Runtime data (coefficients, secrets, predicates derived from secrets) must not alter graph topology, edge weights, tie-breaking, or emitted order.

See `05_answers_to_reviewer_questions.md` for direct answers to the skeptic reviewer questions.
