# Chapter 23 — Fusion II: legality

Chapter 22 argued that fusing a producer into its consumer removes a full tensor round trip, and measured 2.55×. So the question is why a compiler would ever decline.

There is a profitability answer — the fused kernel might need too many registers, or duplicate too much work — and Chapter 24 is about that. This chapter is about the other answer, the one that is not a judgement call: some merges produce a program that **cannot be scheduled at all**, and no cost model can rescue them.

## 23.1 The problem: two adjacent operations that must not be merged

Here is a program with three operations:

```js
const p = a.add(b);      // elementwise
const q = p.matmul(c);   // a matrix multiply — its own kernel
return p.mul(q);         // elementwise
```

`add` and `mul` are both elementwise, they have the same shape, and `mul` consumes `add`'s result directly. By every rule in Chapter 22 they should be fused: one intermediate internalized, one launch saved.

Try it. The fused kernel would compute `p` and then `p * q` in the same loop. But `q` is the matmul, and the matmul reads `p`. So the fused kernel needs `q` before it can finish, and `q` needs the fused kernel to have finished producing `p`. There is no order in which these two kernels can run.

Nothing about *memory* was violated here — the shapes match, the dependency is a clean SSA edge, no aliasing is involved. What was violated is the acyclicity of the *kernel* graph, which is a property of the grouping and not of any individual edge.

## 23.2 Intuition: fusing is contracting an edge

Think of the graph with one node per operation and one edge per dataflow dependency. Chapter 8 established this is a DAG.

Fusing a set of operations means treating them as **one node**. In graph terms that is *contraction*: replace the set with a single vertex whose in-edges are the union of the set's in-edges from outside, and whose out-edges are the union of its out-edges to outside.

Contraction can create a cycle. In the example: contract `{add, mul}` into a node `F`. `F → matmul` survives (the matmul reads `add`'s result), and `matmul → F` survives (the `mul` reads the matmul's result). Two nodes, two edges, opposite directions — a cycle in a graph that had none.

That is the whole legality question, and it has a pleasing property: it does not depend on what the operations *do*. Contraction is legal exactly when it leaves a DAG, and everything else the fusion engine checks — shapes, patterns, region nesting — is either a lowering restriction or a cost heuristic wearing legality's clothes.

## 23.3 Theory

> **Definition 23.1 (Contraction).** Let `G = (V, E)` be a DAG and `S ⊆ V`. The *contraction* `G/S` has vertex set `(V \ S) ∪ {s}` and an edge `u → s` for every `u → v ∈ E` with `u ∉ S, v ∈ S`, and `s → w` for every `v → w ∈ E` with `v ∈ S, w ∉ S`.

> **Theorem 23.2 (Legality of fusion).** Fusing `S` into one kernel admits a valid execution order if and only if `G/S` is acyclic.

*Proof sketch.* (⇐) If `G/S` is acyclic it has a topological order; running the kernels in that order satisfies every dependency, because every dependency of the fused kernel is an in-edge of `s` and every dependent is an out-edge. (⇒) A cycle in `G/S` through `s` means there is a path `s → w → … → u → s`: some kernel `w` needs a value the fused kernel produces, and the fused kernel needs a value `u` produces downstream of `w`. Since the fused kernel is atomic — it is one kernel launch — neither can be scheduled first. ∎

Theorem 23.2 also explains why this problem does not exist before fusion: contracting a *single* vertex is the identity, so an unfused DAG is always schedulable. Cycles are created by grouping, not by dependencies.

Now the algorithmic question. A fusion engine considers thousands of candidate merges and performs hundreds. Recomputing acyclicity from scratch is a full traversal per candidate — `O(V + E)` each, `O(V·E)` overall. The standard fix is to maintain a topological ordering *incrementally*.

> **Definition 23.3 (Incremental topological order).** Maintain a bijection `rank : V → {0..n−1}` such that `u → v` implies `rank(u) < rank(v)`. An operation that would violate it repairs the order by reordering only the vertices whose ranks lie between the two endpoints.

> **Theorem 23.4 (Pearce–Kelly).** *(Pearce and Kelly, 2006.)* Maintaining a topological order under edge insertion by reordering only the *affected region* — the vertices reachable from the lower endpoint with rank below the upper endpoint, and vice versa — costs `O(m^{3/2})` over `m` insertions in the worst case, against `O(m·n)` for recomputing.

The insight the bound rests on is that if `rank(u) < rank(v)` already, the edge `u → v` needs no repair at all; and if not, everything that must move lies strictly between the two ranks. The search never leaves that window, which is why the cost is proportional to how *wrong* the order was rather than to how big the graph is.

> **Corollary 23.5 (Cycle detection, stated here).** With a maintained rank, adding `u → v` creates a cycle if and only if there is a path from the lower-ranked endpoint to the higher-ranked one that passes through a vertex strictly inside the window. The check is a bounded search, not a global one.

## 23.4 In mlfw: 194 lines of union-find and ranks

[`passes/fusion/graph_cycles.ts`](../../../src/compiler/passes/fusion/graph_cycles.ts) implements Definition 23.3 for contraction rather than edge insertion, which is the operation fusion actually performs. The state is five typed arrays and two adjacency sets ([`graph_cycles.ts:3`](../../../src/compiler/passes/fusion/graph_cycles.ts)):

```ts
export class GraphCycles {
  private _n: number;
  private _parent: Int32Array;
  private _rank: Int32Array;
  private _nodeAtRank: Int32Array;
  private _out: Set<number>[];
  private _in: Set<number>[];
```

`_parent` is union-find: after a merge, every operation in a group resolves to the group's representative via `find` ([`graph_cycles.ts:34`](../../../src/compiler/passes/fusion/graph_cycles.ts)), with path compression. That is Definition 23.1's contraction, done lazily — the graph is never rebuilt, only the equivalence relation changes.

`_rank` and `_nodeAtRank` are the bijection of Definition 23.3, in both directions. The query is Corollary 23.5 ([`graph_cycles.ts:46`](../../../src/compiler/passes/fusion/graph_cycles.ts)):

```ts
  wouldCreateCycle(a: number, b: number): boolean {
    const ra = this.find(a);
    const rb = this.find(b);
    if (ra === rb) return false;
    const lo = this._rank[ra] < this._rank[rb] ? ra : rb;
    const hi = lo === ra ? rb : ra;
    return this._hasIntermediatePath(lo, hi);
  }
```

Already in the same group: no cycle, trivially. Otherwise search from the lower-ranked representative for a path to the higher one — and the search is windowed ([`graph_cycles.ts:55`](../../../src/compiler/passes/fusion/graph_cycles.ts)):

```ts
  _hasIntermediatePath(lo: number, hi: number): boolean {
    const limit = this._rank[hi];
    const visited = new Set([lo]);
    const stack = [lo];
    while (stack.length > 0) {
      const u = stack.pop() as number;
      for (const raw of this._out[u]) {
        const v = this.find(raw);
        if (v === u || v === lo) continue;
        if (v === hi) {
          if (u !== lo) return true;
          continue;
        }
        if (this._rank[v] >= limit) continue;
```

Two lines carry the whole idea. `if (this._rank[v] >= limit) continue` is the window: anything ranked at or beyond `hi` cannot be on a path that gets *back* to `hi` from inside, so the search never expands past it. And `if (v === hi) { if (u !== lo) return true; continue; }` is the distinction between a direct edge and an intermediate path: `lo → hi` is the producer-consumer edge you are trying to fuse and is not a cycle; `lo → … → x → hi` with `x` in between is.

The merge itself ([`graph_cycles.ts:77`](../../../src/compiler/passes/fusion/graph_cycles.ts)) does three things: rewires the smaller vertex's neighbours onto the larger one — chosen by degree, `const s = degRa >= degRb ? ra : rb`, so the rewiring cost is the smaller side — unions the two in the parent array, and repairs the rank. The repair is conditional ([`graph_cycles.ts:116`](../../../src/compiler/passes/fusion/graph_cycles.ts)):

```ts
    if (this._reachesWithinWindow(hi, loRank, hiRank)) {
      this._reorder(loRank, hiRank);
    } else {
      const sRank = this._rank[s];
      if (sRank !== loRank) this._nodeAtRank[sRank] = -1;
      this._rank[s] = loRank;
      this._nodeAtRank[loRank] = s;
    }
```

If nothing between the two ranks feeds the upper endpoint, the merged group simply takes the lower rank and no reordering is needed. Otherwise `_reorder` ([`graph_cycles.ts:137`](../../../src/compiler/passes/fusion/graph_cycles.ts)) collects the live representatives in the window, topologically sorts *only those* with Kahn's algorithm, and writes them back into the same slots. Theorem 23.4's affected region, exactly.

### The other checks, and which of them are legality

`GraphCycles` answers one question. [`fusion_analysis.ts`](../../../src/compiler/passes/fusion/fusion_analysis.ts) answers the rest, and it is worth separating them by kind:

| Check | Kind | Where |
|---|---|---|
| would create a cycle | **legality** | `graph_cycles.ts` |
| producer or consumer has a control-flow region | legality | [`fusion_analysis.ts:100`](../../../src/compiler/passes/fusion/fusion_analysis.ts) |
| operation has no lowering rule | implementation limit | [`fusion_analysis.ts:112`](../../../src/compiler/passes/fusion/fusion_analysis.ts) |
| operation is `OPAQUE` (`dot`, `conv`) | implementation limit | [`fusion_analysis.ts:119`](../../../src/compiler/passes/fusion/fusion_analysis.ts) |
| pattern kinds incompatible | implementation limit | [`fusion_analysis.ts:129`](../../../src/compiler/passes/fusion/fusion_analysis.ts) |
| shapes incompatible | legality | [`fusion_analysis.ts:174`](../../../src/compiler/passes/fusion/fusion_analysis.ts) |
| group exceeds `maxFusionSize` | cost heuristic | [`fusion_analysis.ts:149`](../../../src/compiler/passes/fusion/fusion_analysis.ts) |
| more than one reduction in the group | implementation limit | [`fusion_analysis.ts:164`](../../../src/compiler/passes/fusion/fusion_analysis.ts) |

The middle rows matter because they are all reported through the same `{legal, reason}` channel, and a reader debugging a missing fusion will read "cannot fuse pattern kReduction -> kReduction" as a law of nature when it is a statement about what the lowering can currently emit. The pattern lattice is the clearest case ([`fusion_analysis.ts:32`](../../../src/compiler/passes/fusion/fusion_analysis.ts)):

```ts
export function canFusePatterns(pKind: string, cKind: string): boolean {
  const pr = PATTERN_RANK[pKind];
  const cr = PATTERN_RANK[cKind];
  if (pr === undefined || cr === undefined) return false;
  if (pKind === FusionKind.REDUCTION) return cKind === FusionKind.ELEMENTWISE;
  if (cKind === FusionKind.REDUCTION) return pr <= PATTERN_RANK[FusionKind.INJECTIVE];
  return true;
}
```

A reduction may be followed by elementwise work and preceded by anything up to injective. Two reductions in a row may not be fused, which is why Chapter 21's softmax kept its max reduction outside the region. This is XLA's fusion-kind lattice in miniature, and like XLA's it encodes what the emitter can generate.

## 23.5 Lab 1 — The cycle that blocks a fusion

```bash
node docs/part4/ch23-fusion-legality/labs/01-the-cycle-that-blocks-fusion.mjs
```

Two programs. They differ in one word.

```js
class CycleCreating extends Module {
  forward(a, b, c) {
    const p = a.add(b);
    const q = p.matmul(c);   // <- reads p
    return p.mul(q);
  }
}

class NoCycle extends Module {
  forward(a, b, c) {
    const p = a.add(b);
    const q = c.matmul(c);   // <- does not read p
    return p.mul(q);
  }
}
```

Same operations, same shapes, same dataflow between `add` and `mul`. And opposite outcomes:

```
=== q depends on p: fusing p with its consumer would create a cycle ===
module @CycleCreating {
  func @CycleCreating(%0: tensor<4x4xf32>, %1: tensor<4x4xf32>, %2: tensor<4x4xf32>) -> (tensor<4x4xf32>) {
    %3 = add(%0, %1) : tensor<4x4xf32>
    %4 = dot(%3, %2) {lhs_batch = [], lhs_contracting = [1], rhs_batch = [], rhs_contracting = [0]} : tensor<4x4xf32>
    %5 = mul(%3, %4) : tensor<4x4xf32>
    return(%5)
  }
}
fusion regions: 0

=== q does not depend on p: the same two operations fuse ===
module @NoCycle {
  func @NoCycle(%0: tensor<4x4xf32>, %1: tensor<4x4xf32>, %2: tensor<4x4xf32>) -> (tensor<4x4xf32>) {
    %3 = dot(%2, %2) {lhs_batch = [], lhs_contracting = [1], rhs_batch = [], rhs_contracting = [0]} : tensor<4x4xf32>
    %4 = fusion(%0, %1, %3) {fusion_kind = "kElementwise"} : tensor<4x4xf32>
    {
      ^bb(%5: tensor<4x4xf32>, %6: tensor<4x4xf32>, %7: tensor<4x4xf32>):
      %8 = add(%5, %6) : tensor<4x4xf32>
      %9 = mul(%8, %7) : tensor<4x4xf32>
      yield(%9)
    }
    return(%4)
  }
}
fusion regions: 1
```

In the second program the `dot` is a source with respect to the group: it feeds the fusion and reads nothing from it, so contracting `{add, mul}` leaves a DAG and the merge goes through. In the first the `dot` sits *inside* the dependency chain, and contracting the two ends around it closes a loop.

**This is the shape to remember.** The obstacle is never the pair you are looking at; it is the path between them. `add` and `mul` are as fusible as any two operations in the IR, and whether they fuse is decided by a third operation neither of them mentions.

**Try this.** Add a fourth operation `q.add(1)` after the matmul in `CycleCreating`, so the matmul has two consumers. Predict whether anything fuses, then check.

## 23.6 Lab 2 — What the check costs

```bash
node docs/part4/ch23-fusion-legality/labs/02-how-the-check-scales.mjs
```

Theorem 23.4 promises the incremental order beats recomputation. The lab measures the fusion pass on chains from 65 to 513 operations — an eight-fold span, which is the smallest span over which a log-log exponent means anything.

```
graph ops   PriorityFusionPass (ms)   exponent over previous point
       65        0.828                              -
      129        2.514                           1.62
      257        9.334                           1.90
      513       22.009                           1.24

over the whole 8x span: exponent 1.59
1.0 would be linear, 2.0 quadratic
```

Sub-quadratic, and not by much. A genuinely `O(n²)` algorithm on this workload would read 2.0 and hold there; 1.59 with a middle segment at 1.90 says something in the pass is close to quadratic and something else is not.

The something is not the cycle check. This program is the worst case for a different part of the engine: a single chain in which *every* operation ends up in one group, so after each merge the engine re-evaluates candidate edges by walking the whole group again ([`priority_fusion.ts:142`](../../../src/compiler/passes/fusion/priority_fusion.ts)):

```ts
    const reEval = (group: FusionGroup, root: number): void => {
      const dedup = new Set<number>();
      for (const op of group.ops) {
        for (let r = 0; r < op.numResults; r++) {
          for (const use of op.getResult(r).uses()) {
```

`k` merges, each scanning a group that has grown to size `k`, is `O(k²)` in the size of the final group — independent of how good the cycle check is. The cycle check is doing its job; the candidate bookkeeping around it is what the exponent is measuring.

That is worth stating as a general point about performance work on compilers: **a data structure with a good amortized bound does not make the pass that uses it fast.** The bound applies to the operation it covers, and Theorem 23.4 covers rank maintenance, not candidate enumeration.

**Try this.** Widen the graph instead of deepening it — `width` independent three-operation chains rather than one chain of `2·depth` — so group sizes stay bounded, and measure the exponent again.

## 23.7 Traps and limits

- **`maxFusionSize` is 512 and it silently caps the measurement above.** [`fusion_analysis.ts:81`](../../../src/compiler/passes/fusion/fusion_analysis.ts) defaults it, so a chain longer than 512 operations stops merging partway and the timing flattens. §23.6 stops at 513 operations for that reason; extending the sweep further measures the cap, not the algorithm.
- **The cycle check is the *only* check `GraphCycles` performs.** It knows nothing about shapes, kinds, or cost. Everything in §23.4's table above the cycle row is enforced elsewhere, and the engine calls them in a specific order — `legality.canFuse`, then `costModel.shouldFuse`, then `wouldCreateCycle` ([`priority_fusion.ts:198`](../../../src/compiler/passes/fusion/priority_fusion.ts)) — so the cheapest-to-evaluate check runs last. Reversing that order would save work on graphs where cycles are common.
- **`_reorder` uses Kahn's algorithm on the window and writes back into the same slots.** If the window contains a vertex whose representative has changed (`this.find(nd) !== nd`), its slot is released ([`graph_cycles.ts:143`](../../../src/compiler/passes/fusion/graph_cycles.ts)). Slots therefore accumulate holes as merges proceed, and `_nodeAtRank` is a sparse array with `-1` entries — correct, and it means the rank space never compacts.
- **Two reductions never fuse, and that is an emitter limitation.** The `maxReductions` default is 1 ([`priority_fusion.ts:50`](../../../src/compiler/passes/fusion/priority_fusion.ts)). A softmax's two reductions could in principle share a pass over the data; they do not, because the lowering has no way to emit two reduction loops in one block.
- **A refused fusion is silent under the default strategy.** Chapter 18 covered this: `priority` explains only its successes. On the `CycleCreating` program above, the compiler makes a decision with real performance consequences and emits no event at any trace level to say it did.

## 23.8 Read the tests

- [`tests/compiler/passes/fusion/graph-cycles.test.js`](../../../tests/compiler/passes/fusion/graph-cycles.test.js) — `wouldCreateCycle` and `merge` on hand-built graphs, including the diamond that motivates the window search.
- [`tests/compiler/passes/fusion/graph-cycles-scaling.test.js`](../../../tests/compiler/passes/fusion/graph-cycles-scaling.test.js) — the log-log exponent asserted as a test, which is the executable form of §23.6.
- [`tests/compiler/passes/fusion/analysis.test.js`](../../../tests/compiler/passes/fusion/analysis.test.js) — the legality table: pattern kinds, shapes, regions, and the reasons each refusal reports.

---

**Next:** [Chapter 24 — Fusion III: the three strategies](../ch24-fusion-strategies/README.md), which takes the legality rules and the cost model and asks the remaining question: given that many merges are legal and profitable, in what order should you make them?
