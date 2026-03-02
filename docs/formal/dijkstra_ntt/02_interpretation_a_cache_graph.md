# Interpretation A: Cache-Optimal Twiddle Traversal

## Graph model

- Vertices: coefficient indices or butterfly operations at each NTT stage.
- Directed edges: legal next butterfly operations preserving stage dependencies.
- Edge weight: static cache-cost estimate for fetching the required twiddle factor (e.g., line distance, reuse-distance class, page crossing penalty).

## Correct framing

Dijkstra is used as a **design-time optimizer** producing a total order (or stage-local order) that minimizes modeled twiddle-access cost.

## Constant-time relevance

If the resulting order is compiled into tables/loops, runtime twiddle accesses are deterministic and independent of coefficient values. This supports constant-time memory behavior under the usual microarchitectural caveats.

## Forbidden variants

- Weighting edges by runtime coefficient values.
- Re-running Dijkstra per input polynomial.
- Runtime tie-breaking with secret-dependent state.
