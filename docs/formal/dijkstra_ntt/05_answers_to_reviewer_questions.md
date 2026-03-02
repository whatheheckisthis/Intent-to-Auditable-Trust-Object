# Direct Answers to Reviewer Questions

## 1) Which graph gets Dijkstra?

**Answer:** Both can use Dijkstra, but at different layers:

- Layer A: twiddle-access graph for cache optimization.
- Layer B: instruction dependency graph for pipeline scheduling.

They are separable optimization passes; either can be used alone.

## 2) Static vs runtime graph construction?

**Answer:** For constant-time safety, graph construction must be entirely static (design/build time). Runtime coefficient values must not affect vertices, edges, or edge weights.

## 3) Tie-breaking rule?

**Answer:** Use a canonical, data-independent rule: `(distance, node_id, edge_id)` lexicographic order. No runtime entropy, pointer-order dependence, or value-based branching.

## 4) Offline or online Dijkstra?

**Answer:** Required framing is **offline** Dijkstra with emitted fixed schedules/tables. Online Dijkstra per invocation is discouraged because standard priority-queue behavior introduces data-dependent control flow unless a specialized constant-time implementation is proven.

## 5) SVE2 predicates and Dijkstra output?

**Answer:** Predicates may be arranged by the offline schedule, but runtime predicate values must depend only on public structure (lane index, stage shape, bounds masks) and not on coefficient values. Otherwise, lane-uniform execution claims need qualification.
