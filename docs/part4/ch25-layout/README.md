# Chapter 25 — Layout

Every optimization so far changed *what* is computed or *in what groups*. This one changes neither. The same operations run on the same values in the same order, and the only difference is which memory address each element lives at.

It is also the first optimization in this book that is switched off by default, and the first where turning it on made the program slower. Both facts are worth a chapter, because layout is where the gap between "the compiler can do this" and "the compiler should do this" is widest.

> **Read this chapter in the conditional.** Sections 25.2 to 25.4 describe a layout pipeline: a preference per operation, a propagation, a cost model, and a conversion inserted where two decisions disagree. All of that machinery exists and is what the source says. **None of it currently has an effect on any compilation this repository can produce**, because the benefit term is gated on `target.layoutAwareOps` and no target populates it (§25.5 measures exactly this). So the mechanism is real, the pipeline is described accurately, and the pass is inert. Where a sentence in the next three sections reads as though the compiler routinely relays layouts through a model, take it as *how the pass behaves when the gate is opened by hand*, which is what §25.5's third row does.

## 25.1 The problem: a shape is not an address

A tensor's type says `tensor<1x3x16x16xf32>` — four dimensions, and nothing about memory. Chapter 10 introduced `Layout` as the missing half: a permutation saying which dimension varies fastest.

For a 4-D activation with dimensions (batch, channel, height, width), two conventions dominate:

- **NCHW** — row-major over `[N, C, H, W]`. Element `(n, c, h, w)` sits at `((n·C + c)·H + h)·W + w`. Consecutive `w` are adjacent; consecutive `c` are `H·W` apart.
- **NHWC** — the permutation `[0, 2, 3, 1]`. Element `(n, c, h, w)` sits at `((n·H + h)·W + w)·C + c`. Consecutive `c` are adjacent.

Both hold identical numbers. They differ only in which loop of a kernel walks memory sequentially, and that decides everything about cache behaviour. A convolution accumulating over input channels reads `C` values per output element: under NHWC those are contiguous, under NCHW they are `H·W` apart. On a 16×16 feature map that is a 1-KiB stride between consecutive reads — a cache miss per channel.

Which is better is not a property of the operation. It is a property of the *kernel*, and therefore of the target: a CPU with 64-byte cache lines wants the reduction axis contiguous; a GPU wants the axis that varies across threads contiguous; a tensor-core kernel wants a *blocked* layout where a 16×16 tile is contiguous.

So a compiler cannot pick one layout and be done. It has to decide per operation, and then reconcile decisions that disagree.

## 25.2 Intuition: preferences, propagation, and a bill

Three steps.

**Preference.** Each operation states which layout it would like for each operand and each result. A convolution says "I want NHWC input". An elementwise add says nothing — it is equally happy either way.

**Propagation.** Walk the graph in topological order assigning each value a layout: from the preference if the producer stated one, otherwise inherited from the operation's inputs. Elementwise operations pass layout through, which is what makes long chains agree without anybody negotiating.

**Reconciliation.** Wherever a value's assigned layout differs from what its consumer wants, insert a conversion — a physical transpose. That costs a pass over the tensor. So the question becomes arithmetic: does the consumer save more from the layout it wants than the conversion costs?

That last step is why layout is a *cost* problem and not a constraint problem. Every layout is legal; the compiler is choosing between programs that all compute the same thing, and the choice is only as good as its estimate of what a layout is worth to a kernel it has not generated yet.

## 25.3 Theory

> **Definition 25.1 (Layout).** A *layout* for a rank-`r` tensor is a permutation `σ` of `{0..r−1}` giving the order in which dimensions are laid out from slowest- to fastest-varying. The linear address of index `(i₀..i_{r−1})` is the row-major flattening of the permuted index vector.

Chapter 35 proves the flattening is a bijection; here the only consequence needed is that two layouts of the same tensor hold the same values and a *permutation of the data* converts one to the other.

> **Definition 25.2 (Layout assignment problem, stated here).** Given a DAG whose operations have layout preferences and conversion costs, assign a layout to every value minimizing the sum of conversion costs plus per-operation costs, where an operation's cost depends on whether its operands are in its preferred layout.

> **Theorem 25.3.** *(Classical.)* Layout assignment is an instance of the *multiway cut* problem and is NP-hard for three or more distinct layouts.

The reduction is the natural one: each layout is a terminal, each value is a vertex, and a conversion cost is an edge weight to be cut. With two layouts it is min-cut and polynomial; with three — NCHW, NHWC, blocked — it is not.

So, again, a heuristic. The standard one, and the one used here, is:

> **Definition 25.4 (Greedy propagation with local accept, stated here).** Propagate preferences forward, collect the required conversions, group them by (value, from, to), and accept a group only when its estimated benefit exceeds its estimated cost. Reject everything else and leave those consumers with the layout they were given.

The weakness of Definition 25.4 is that it is local: it decides each conversion against its own consumers, with no view of whether accepting two nearby conversions would have let a third become free. The strength is that it never makes the program worse *by its own model* — and §25.5 is about what happens when the model is wrong.

## 25.4 In mlfw: preference, analysis, insertion

### Who has a preference

Three operations declare one, through the `INFER_LAYOUT` op-attribute from Chapter 11 ([`ops/linalg.ts:45`](../../../src/compiler/ir/graph/ops/linalg.ts)):

```ts
    opAttrs: { [OpAttrKey.GPU_CAPABLE]: true, [OpAttrKey.LAUNCH_BOUNDARY]: 'matmul', [OpAttrKey.LAYOUT_SENSITIVITY]: 4, [OpAttrKey.INFER_LAYOUT]: dotLayout },
```

`dot` and `conv` in [`ops/linalg.ts`](../../../src/compiler/ir/graph/ops/linalg.ts), `reduce` in [`ops/reduction.ts:21`](../../../src/compiler/ir/graph/ops/reduction.ts). Everything else has none, which is the right default: an elementwise operation should inherit, not demand.

The rules themselves are short and target-dependent ([`ops/linalg.ts:17`](../../../src/compiler/ir/graph/ops/linalg.ts)):

```ts
function convLayout(op: Operation, target: LayoutTarget): LayoutPreference | null {
  const input = op.getOperand(0).type as TensorType;
  const rank = input ? input.rank : NHWC_RANK;
  if (target.preferredConvLayout) {
    const preferred = target.preferredConvLayout as unknown as Layout;
    return new LayoutPreference([preferred, null], [preferred]);
  }
  if (rank !== NHWC_RANK || !(target.isGPU() || target.isCPU())) return null;
  const nhwc = new Layout([0, 2, 3, 1]);
  return new LayoutPreference([nhwc, null], [nhwc]);
}
```

A target may name a layout outright; otherwise a rank-4 convolution on a CPU or GPU asks for NHWC. The `null` in the operand list is "no preference for the weights", which is how a rule expresses partial opinions.

`dotLayout` ([`ops/linalg.ts:29`](../../../src/compiler/ir/graph/ops/linalg.ts)) is more interesting because it is asymmetric:

```ts
  const lhsLayout = Layout.rowMajor(lhsType.rank);
  if (target.isCPU() && rhsType.rank === 2) {
    return new LayoutPreference([lhsLayout, Layout.columnMajor(rhsType.rank)], [lhsLayout]);
  }
```

On a CPU, a matrix multiply wants its *right-hand* operand column-major, so the inner product walks both operands contiguously. This is the classical reason `B` is transposed in a hand-written GEMM.

### Propagation

[`analysis/layout_analysis.ts`](../../../src/compiler/analysis/layout_analysis.ts) is Definition 25.4's first two steps. Function arguments start row-major or with whatever their type carries ([`layout_analysis.ts:58`](../../../src/compiler/analysis/layout_analysis.ts)), then the topological walk assigns ([`layout_analysis.ts:84`](../../../src/compiler/analysis/layout_analysis.ts)):

```ts
      const def = registry.get(op.opName);
      const isEW = def && def.hasTrait(OpTrait.ELEMENTWISE);

      for (let r = 0; r < op.numResults; r++) {
        const val = op.getResult(r);
        if (!(val.type instanceof TensorType)) continue;
        if (isEW) {
          assignments.set(val, resolveFromInputs(op, assignments));
        } else {
          assignments.set(val, Layout.rowMajor((val.type as TensorType).rank));
        }
      }
```

Elementwise operations inherit; everything without a preference resets to row-major. That second branch is a conservative choice with a cost: a `reshape` or a `slice` sitting between two convolutions breaks the NHWC chain and forces a conversion back.

A second walk collects the mismatches into `LayoutConversion` records — value, consumer, operand index, from, to.

### Insertion, and the bill

[`passes/layout/layout_transform.ts`](../../../src/compiler/passes/layout/layout_transform.ts) groups the conversions and prices them ([`layout_transform.ts:60`](../../../src/compiler/passes/layout/layout_transform.ts)):

```ts
      g.consumers.push({ consumer, operandIdx });
      const capable = this.target.layoutAwareOps && this.target.layoutAwareOps.has(consumer.opName);
      g.benefit += capable ? (this._policy as LayoutPolicy).estimateBenefit(consumer, value.type, 1) : 0;
```

and accepts group by group ([`layout_transform.ts:65`](../../../src/compiler/passes/layout/layout_transform.ts)):

```ts
    let totalCost = 0, totalBenefit = 0;
    const keep: ConversionGroup[] = [];
    for (const g of groups.values()) {
      if (g.benefit < g.cost) continue;
      keep.push(g);
      totalCost += g.cost;
      totalBenefit += g.benefit;
    }
    if (keep.length === 0 || totalCost > totalBenefit) return PassResult.UNCHANGED;
```

Accepted groups become `layout_transform` operations inserted right after the value's producer, with every consumer in the group rewired to the transformed value — so a value needed in NHWC by four convolutions is transposed once, not four times.

The cost and benefit models are two functions in [`layout_policy.ts`](../../../src/compiler/passes/layout/layout_policy.ts):

```ts
  estimateConversionCost(fromLayout: Layout | null, toLayout: Layout | null, tensorType: IRType): number {
    if (!(tensorType instanceof TensorType)) return 0;
    if (layoutEquals(fromLayout, toLayout)) return 0;
    const numEl = tensorType.numel();
    if (numEl < 0) return 1024;
    return numEl * 2;
  }
```

A conversion costs two units per element — a read and a write. Benefit ([`layout_policy.ts:47`](../../../src/compiler/passes/layout/layout_policy.ts)) is `numEl × LAYOUT_SENSITIVITY × useCount`, where sensitivity is 4 for `dot` and `conv` and 2 for `reduce`. So a `dot` consuming a value in its preferred layout is worth four units per element against a conversion costing two, and the accept condition passes with room to spare.

Two units and four units. Both are made up.

## 25.5 Lab — A layout that does nothing, and then does harm

```bash
node docs/part4/ch25-layout/labs/01-a-layout-that-does-nothing.mjs
```

The running example — two `Linear` layers, whose `dot` operations have a CPU layout preference — compiled three ways.

```
=== optimization.layout off (the default) ===
  target.layoutAwareOps = {}
  pass report: the pass reported nothing
  layout_transform operations in the graph: 0
  3.223 ms

=== optimization.layout on, target declares nothing ===
  target.layoutAwareOps = {}
  pass report: the pass reported nothing
  layout_transform operations in the graph: 0
  3.307 ms

=== optimization.layout on, target declares dot layout-aware ===
  target.layoutAwareOps = {dot}
  pass report: 2 conversion(s) proposed, 2 kept
  layout_transform operations in the graph: 2
  3.931 ms
```

Three findings, in increasing order of interest.

**The pass is off by default.** `optimization.layout` is `false` in the config ([`compiler.ts:147`](../../../src/compiler/pipeline/compiler.ts)), so the pass is never constructed.

**Turning it on changes nothing.** The second row is the surprising one: the pass runs, `dotLayout` proposes column-major for both weight operands, `LayoutAnalysis` records two conversions — and every one is discarded. The reason is the `capable` line above: benefit is added only when the consumer is in `target.layoutAwareOps`, and **no target in this repository populates that set** ([`target.ts:129`](../../../src/backend/target.ts) constructs it from a config field nobody sets). Benefit is therefore zero, `g.benefit < g.cost` for every group, and the pass returns UNCHANGED without emitting its `pass_detail` event.

So layout is switched off twice: once by a config flag, and once by an empty set that no configuration in the repository fills. Turning on the flag alone is a no-op, which is a bad failure mode for an optimization — it looks enabled and is not.

**And when you do enable it properly, it is slower.** The third row declares `dot` layout-aware, the two conversions are accepted, and the program goes from 3.22 ms to 3.93 ms — 22% worse.

> **What that row demonstrates, and what it does not.** It is evidence that **the mechanism works**: a preference was proposed, a conversion was priced, accepted and inserted, and the emitted program changed. That is the claim §25.5 is making, and the third row is a clean demonstration of it. It is *not* evidence about whether layout selection is a good optimization — for that you would need a case where the chosen layout is one the backend's kernels actually exploit, and §25.6 explains why no such case exists here yet. Nor is it a measurement of "the cost of layout conversion" in general: it is the cost of converting two weight tensors on every call, on one shape, on one machine. Read the 22% as *the price of this particular forced conversion under these particular conditions*, and read the row itself as a demonstration that the plumbing is connected.

That is not a mystery, and it is not the pass malfunctioning. Look at where the transforms went: the weights are *function arguments*, so a `layout_transform` on a weight runs on **every call**. The model priced the conversion at `2 × numEl` against a benefit of `4 × numEl` and concluded it was worth it — a judgement that would be right if the conversion happened once and the benefit accrued forever, and is wrong when both happen once per call. And the benefit side is speculative anyway: it assumes the CPU matmul kernel exploits a column-major right-hand operand, and Chapter 54's generated matmul does not — it indexes with whatever strides the layout gives it, so a "preferred" layout buys nothing at all.

**Try this.** Declare `layoutAwareOps = {dot}` and set the weights as compile-time constants using `foldWeights: true` (Chapter 61), so the transform can be hoisted out of the call. Then measure again.

## 25.6 Why this chapter is short, and honest

Layout is a genuinely large subject — blocked layouts, tensor-core tile shapes, per-target GEMM packing — and this compiler has the *skeleton* of all of it: a `Layout` type with permutations, a preference protocol on the registry, an analysis, a cost model, a conversion operation, and an insertion pass. What it does not have is a reason to run any of it, because the benefit model describes kernels that the backends do not generate.

That is worth stating plainly rather than dressing up, because it is the normal state of a compiler optimization at this stage of its life, and recognizing it is a skill. The diagnostic question is not "is the pass correct" — it is — but **"is there a kernel downstream that will actually be faster?"** Layout transformation pays only when some consumer has a specialized implementation keyed to the layout. Absent that, the transform is a pure cost, and the measurement in §25.5 is what a pure cost looks like.

The repository's own record agrees: an optimization gate (Chapter 61) that compiles candidate configurations and keeps the winner found layout worth 1.18× on a small CNN and nothing elsewhere, which is why it is not on by default. That gate is the right mechanism for a decision like this — measure, don't model — and Chapter 46 argues the general case.

## 25.7 Traps and limits

- **`layoutAwareOps` is empty for every target, so the pass is inert even when enabled.** [`layout_transform.ts:61`](../../../src/compiler/passes/layout/layout_transform.ts) is the only reader. There is no warning; the pass simply reports UNCHANGED, which is indistinguishable from "nothing to do".
- **`LayoutAnalysis` is never cached.** It takes a third `policy` argument that the analysis protocol has no room for, so the pass calls `LayoutAnalysis.compute` directly rather than going through the manager (Chapter 16). Every run recomputes it.
- **Anything without a preference resets to row-major** ([`layout_analysis.ts:93`](../../../src/compiler/analysis/layout_analysis.ts)). Only `ELEMENTWISE` operations inherit a layout from their inputs; every other operation's results are assigned `Layout.rowMajor(rank)` outright. So a `reshape` between two convolutions breaks an NHWC chain where a smarter propagation would carry the layout through.

  Be careful about the *cost* of that break, though, because "forces two conversions" is a tempting summary and it is not what the algorithm does. The reset assigns a layout to the reshape's result; it does not insert anything. Conversions are emitted later, in the pass over consumers, and how many appear depends entirely on what the consumers ask for and whether they are in `layoutAwareOps`. Three outcomes are all reachable from the same reset: **two** conversions if the producer's non-row-major output must be converted in and the next convolution's preference must be converted back out; **one** if only one side disagrees; and — the actual behaviour today — **none**, because with `layoutAwareOps` empty no conversion accumulates benefit and every candidate group is discarded. The reset is a real weakness in the propagation, and its price is "a conversion the chain did not need, when the pass is doing anything at all", not a fixed count of two.
- **The cost and benefit constants are unitless and unmeasured.** `numEl * 2` for a conversion, `numEl * sensitivity * useCount` for the benefit, sensitivity 4 or 2 by op. Nothing derives these from the target, and the comparison `benefit < cost` is therefore a comparison between two invented scales.
- **Conversions are priced per call, and weights are converted per call.** §25.5's measurement. A layout pass that is worth anything on inference needs the weight transform hoisted to load time; nothing here does that.
- **Blocked layouts exist in the type system and nothing produces them.** `Layout` supports an `isBlocked()` query ([`layout_analysis.ts:18`](../../../src/compiler/analysis/layout_analysis.ts)) and no rule returns one, so the NP-hardness of Theorem 25.3 is not yet reachable — the assignment problem here has two layouts, and is polynomial.

## 25.8 Read the tests

- [`tests/compiler/passes/layout/`](../../../tests/compiler/passes/layout/) — preference inference, the propagation walk, and the accept/reject arithmetic with hand-set costs.
- [`tests/compiler/pipeline/opt-gate.test.js`](../../../tests/compiler/pipeline/opt-gate.test.js) — the measurement gate that decides whether layout is worth enabling for a given model and target, which is the mechanism §25.6 recommends over the model.

---

**Next:** [Chapter 26 — Three optional pipelines](../ch26-optional-pipelines/README.md), which closes Part IV with the three graph-level transformations that are, like this one, off unless you ask: rematerialization, quantization, and partitioning.
