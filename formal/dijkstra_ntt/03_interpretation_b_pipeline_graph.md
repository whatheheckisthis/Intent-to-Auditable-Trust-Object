# Interpretation B: Execution Path as Shortest Path (SVE2)

## Graph model

- Vertices: instruction instances in an unrolled/partially-unrolled NTT kernel.
- Edges: dependency and issue-order constraints.
- Edge weights: static hazard model (latency, structural conflicts, forwarding penalties).

## Correct framing

Dijkstra (or a shortest-path dynamic variant) can be used to select an instruction schedule with minimal modeled stalls while preserving data dependencies.

## Constant-time relevance

The selected schedule must be branchless and fixed once compiled. Predicate manipulation may occur, but predicate values must be derived from public lane layout/loop counters rather than secret coefficients.
