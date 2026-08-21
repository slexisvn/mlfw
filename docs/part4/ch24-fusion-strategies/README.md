# Chapter 24 — Fusion III: the three strategies

Chapter 22 said fusion is worth 2.55×. Chapter 23 said a merge is legal exactly when contracting it leaves a DAG. Neither says *which* legal, profitable merges to make, in what order, and that turns out to be the question with the largest spread between good and bad answers.

There are three fusion engines in this compiler and two auxiliary passes. They see the same graph, use the same legality checks and the same cost model, and on a three-operation program they used to produce three different answers, the best of which ran 2.4× faster than the default. They now produce the same answer. What separated them was not the strategies at all — it was a single `||` in the cost model, and §24.5 is the chase.

## 24.1 The problem: fusion decisions are not independent

Suppose four operations, `A → B → C` and `A → D`. Fusing `{A,B}` is legal and profitable. So is fusing `{A,D}`. But `A` can only be inside one kernel, so the two choices compete, and taking one may make the other illegal or unprofitable.

That makes fusion a *partitioning* problem: choose a partition of the operations into groups, subject to contraction leaving a DAG, maximizing total benefit. Written that way it is obviously hard — it is graph partitioning with a cost function, and the general form is NP-hard. Nobody solves it exactly.

So every fusion engine is a greedy heuristic, and the interesting differences are in three choices:

- **What is a candidate?** A single producer-consumer edge, or a whole region of the graph?
- **In what order are candidates considered?** Graph order, or best-first?
- **When is the decision final?** Immediately, or after re-evaluating what the merge changed?

## 24.2 Intuition: three ways to be greedy

**Dominator-based.** Walk the graph and, for each operation, gather the region it *dominates* — everything downstream that cannot be reached except through it. That region is a natural fusion candidate: it has one entry, so contracting it cannot create a cycle through the entry. Take the region whole or leave it. This is the classical approach and it makes big decisions cheaply.

**Priority-based.** Score every legal producer-consumer *edge* by how much fusing it would save, put them in a max-heap, and repeatedly pop the best one. After each merge, re-score the edges the merge touched. Small decisions, best first, with the ordering recomputed as the graph changes. This is XLA's priority fusion, and it is this compiler's default.

**Greedy walk.** Visit operations in order and extend the current group as long as the next operation is legal and the cost model agrees. Simplest, and most dependent on the order you happen to walk in.

The difference between them is not a matter of taste. A region-at-a-time engine evaluates the cost model on the *whole* region, so one expensive property of one member vetoes the entire group. An edge-at-a-time engine evaluates it on each pairwise merge, so it can stop just short of the member that would have vetoed — a smaller kernel rather than none. §24.5 shows all three engines under a cost model that vetoed wrongly, which is where that difference becomes visible, and then under one that does not, where it disappears.

## 24.3 Theory

> **Definition 24.1 (Fusion partition).** A *fusion partition* of a DAG `G` is a partition of its vertices into groups such that contracting every group simultaneously leaves a DAG. Its *benefit* is the sum over groups of `memorySaved + launchSaved` from Chapter 22.

> **Theorem 24.2 (Hardness).** *(Classical.)* Maximizing the benefit of a fusion partition is NP-hard in general — it contains graph partitioning with arbitrary vertex weights as a special case.

Hence heuristics, and hence the practical question is not "which is optimal" but "which fails less badly, and how would you know". Two properties distinguish greedy schemes here:

> **Definition 24.3 (Monotone candidate set, stated here).** A greedy scheme has a *monotone* candidate set if merging never makes a previously-illegal merge legal. Under monotonicity, a candidate rejected once may be discarded; without it, rejected candidates must be reconsidered after every merge.

Fusion is **not** monotone in the cost sense: merging `A` with `B` changes the group's inputs and outputs, so an edge that was unprofitable can become profitable, and vice versa. It is tempting to think it is at least monotone in the cycle sense — that contracting more can only create cycles, never remove them — and that is false in both directions:

> **Counterexample 24.4 (Legality is not monotone).** Take `A → X → B` together with `A → B`. Contracting `{A, B}` is illegal: the merged vertex reaches `X` and is reached by it. Contracting `{A, X, B}` is legal, since it leaves one vertex and no edges at all. So a merge rejected for a cycle can become legal after a later merge — here, after `X` joins the group — and a legal contraction can have an illegal sub-contraction.

The consequence is that a rejected candidate may not be discarded. The priority engine keeps a heap it re-pushes into after every merge, and validates popped entries against a version counter rather than trusting them:

> **Definition 24.5 (Stale candidate, stated here).** A heap entry naming two groups is *stale* if either group has been merged since the entry was pushed. A stale entry must be discarded rather than applied, because its recorded benefit was computed for groups that no longer exist.

This is a lazy-deletion priority queue, and it is the standard way to avoid the alternative — finding and removing every affected entry on each merge, which needs an indexed heap.

## 24.4 In mlfw: five passes

[`buildGraphPipeline`](../../../src/compiler/pipeline/graph_pipeline.ts) selects between them on a config string ([`graph_pipeline.ts:80`](../../../src/compiler/pipeline/graph_pipeline.ts)):

```ts
  if (config.fusion.enabled) {
    const fCfg = config.fusion;
    const launchOverheadUs = fCfg.launchOverheadUs ?? DEFAULT_LAUNCH_OVERHEAD_US;
    if (fCfg.strategy === 'dominator') {
      passes.push(new DominatorFusionPass({ target, ...fCfg }));
    } else if (fCfg.strategy === 'priority') {
      passes.push(new PriorityFusionPass({ target, cost: { launchOverheadUs }, ...fCfg }));
      passes.push(new MultiOutputFusionPass({ maxFusionSize: target?.maxFusionSize, ...fCfg }));
    } else {
      passes.push(new FusionPass({ target, cost: { launchOverheadUs }, ...fCfg }));
      passes.push(new FusionMergerPass({ maxFusionSize: target?.maxFusionSize, ...fCfg }));
      passes.push(new MultiOutputFusionPass({ maxFusionSize: target?.maxFusionSize, ...fCfg }));
    }
    passes.push(new DCEPass());
  }
```

Note the `else` branch has no guard: any string that is not `'dominator'` or `'priority'` selects the original `FusionPass`. The labs call that one `'greedy'`, which is descriptive and is not a name the code knows.

| Pass | Strategy | Candidate | Order |
|---|---|---|---|
| [`PriorityFusionPass`](../../../src/compiler/passes/fusion/priority_fusion.ts) | priority (default) | one edge | max-heap by benefit |
| [`DominatorFusionPass`](../../../src/compiler/passes/fusion/dominator_fusion.ts) | dominator | a post-dominated region | graph order |
| [`FusionPass`](../../../src/compiler/passes/fusion/fusion_pass.ts) | anything else | a producer's consumers | graph order |
| [`FusionMergerPass`](../../../src/compiler/passes/fusion/fusion_merger.ts) | with `FusionPass` | two existing groups | graph order |
| [`MultiOutputFusionPass`](../../../src/compiler/passes/fusion/multi_output_fusion.ts) | with `PriorityFusionPass` or `FusionPass` — **not** `dominator` | siblings sharing inputs | graph order |
| [`EpilogueFusionPass`](../../../src/compiler/passes/fusion/epilogue_fusion.ts) | target opt-in | `dot`/`conv` + elementwise tail | graph order |

### The priority engine

Three data structures ([`priority_fusion.ts:87`](../../../src/compiler/passes/fusion/priority_fusion.ts)):

```ts
    const cycles = new GraphCycles(n, edges);
    const version = new Int32Array(n);
    const groupOf = new Map<number, FusionGroup>();
```

`cycles` is Chapter 23. `version` implements Definition 24.5 — one counter per group representative, bumped on every merge. `groupOf` maps a representative to its group.

The benefit function is one line ([`fusion_cost.ts:70`](../../../src/compiler/passes/fusion/fusion_cost.ts)):

```ts
  edgeBenefit(bytes: number): number {
    const w = this.benefitWeights;
    return w.launch * this.launchOverheadUs + w.memory * bytes;
  }
```

with weights `{ memory: 1, launch: 1000 }` ([`fusion_cost.ts:38`](../../../src/compiler/passes/fusion/fusion_cost.ts)), so a saved launch is worth `1000 × 5 = 5000` "bytes". That is a unit-free trade-off between two incomparable quantities, and the number encodes an assumption about how expensive a launch is relative to a byte of bandwidth. On a CPU, where a "launch" is a function call, it is very likely too high.

The main loop is Definition 24.5 in practice ([`priority_fusion.ts:189`](../../../src/compiler/passes/fusion/priority_fusion.ts)):

```ts
    while (!heap.isEmpty()) {
      const cand = heap.pop() as MergeCandidate;
      const ra = cycles.find(cand.a);
      const rb = cycles.find(cand.b);
      if (ra === rb) continue;
      if (version[ra] !== cand.va || version[rb] !== cand.vb) continue;
```

Already merged, or either endpoint's version has moved: discard and take the next. Then the three checks in Chapter 23's order, the merge, and a re-evaluation of every edge the new group touches.

One detail in the merge is worth noticing ([`priority_fusion.ts:204`](../../../src/compiler/passes/fusion/priority_fusion.ts)):

```ts
      const big = ga.size >= gb.size ? ga : gb;
      const small = big === ga ? gb : ga;
      big.merge(small);
```

Weighted union — the smaller group's operations move into the larger. Same trick as `GraphCycles.merge`, same reason: the cost of a merge is the size of the side you move.

### The auxiliary passes

**`MultiOutputFusionPass`** runs after the priority and greedy strategies — but not after `dominator`, which gets no auxiliary pass at all. A fusion region may `yield` more than one value (Chapter 9's region can have several results), which lets two consumers of the same producer share one kernel even though neither is downstream of the other. This is *horizontal* fusion, and it is why the priority strategy's output in §24.5 has two results.

**`EpilogueFusionPass`** handles what the others cannot. `dot` and `conv` carry the `OPAQUE` trait, so Chapter 23's legality refuses to fuse them with anything — they are library-shaped operations whose loop nest the general fusion machinery cannot generate. But their *consumers* can often be folded into the tail of the matmul kernel: a bias add and a ReLU applied to the accumulator before it is written out. That is `fused_dot_epilogue`, and §24.6 is what it looks like.

## 24.5 Lab 1 — Three strategies, one graph

```bash
node docs/part4/ch24-fusion-strategies/labs/01-three-strategies.mjs
```

The program is a diamond — one value used twice:

```js
class Diamond extends Module {
  forward(x, y) {
    const s = x.add(y);
    return s.mul(x).add(s);
  }
}
```

Three operations, all elementwise, all the same shape, on 256K-element tensors. Every pair is legal by Chapter 23. And this is what the lab printed before the fix described below:

```
=== strategy 'priority' (the default) ===
  cost model: add+mul -> fused
  1 fusion region(s) holding: add, mul
  1.473 ms

=== strategy 'dominator' ===
  cost model: mul+add+add -> not-fused: shared memory 1048576 exceeds limit 49152
  0 fusion region(s) holding: (nothing)
  1.598 ms

=== strategy 'greedy' (the original FusionPass) ===
  cost model: add+mul+add -> not-fused: shared memory 1048576 exceeds limit 49152
  0 fusion region(s) holding: (nothing)
  1.565 ms

=== strategy 'dominator', shared-memory budget raised to 16 MiB ===
  cost model: mul+add+add -> fused: saves 6291456 bytes, 10us launch
  1 fusion region(s) holding: add, mul, add
  0.606 ms
```

Two of the three engines fused **nothing**. The reason is in the explanation: `s` is used twice inside the candidate group, so the cost model charges it as shared memory ([`fusion_cost.ts:203`](../../../src/compiler/passes/fusion/fusion_cost.ts)) at its full tensor size — one megabyte — against a limit of 48 KiB.

The priority engine reached a different answer for a structural reason. It merges *edges*, so it first forms `{add, mul}` — at which point `s` has one internal user and one external one, no shared-memory charge, and the merge passes. Then it tries to add the final `add`, which would make `s` internally reused, and *that* merge is refused. It stopped one step short of the group the other two engines evaluated as a whole, producing a two-output region: the kernel yields both `s` and `s*x`, and the final add reads them back.

The fourth run is the control. Same engine, same graph, one number changed — and all three operations fuse into one kernel, **2.4× faster than the default configuration**. So the 48 KiB limit cost this program 2.4×, and it is worth asking where the number came from. 48 KiB is a GPU shared-memory budget. This compilation targets a CPU, which has no shared memory at all. The plumbing explains it ([`priority_fusion.ts:60`](../../../src/compiler/passes/fusion/priority_fusion.ts)):

```ts
      maxSharedMemory: target.sharedMemoryBytes,
```

`CPUTarget` does not mention `sharedMemoryBytes` at all, and the base constructor turns that silence into a number — `config.sharedMemoryBytes || 0` ([`target.ts:114`](../../../src/backend/target.ts)). The cost model then read that number as `config.maxSharedMemory || 49152` ([`fusion_cost.ts:62`](../../../src/compiler/passes/fusion/fusion_cost.ts)). **Zero is falsy.** So "I have no shared memory" and "I did not say" collapsed to the same value twice over, and the second `||` read that value as the former having never been stated — falling back to the GPU default. `CUDATarget` sets `48 * 1024` explicitly and was unaffected; CPU and WASM, which say nothing, inherited a budget for a resource they do not have.

The obvious repair is `??` instead of `||`, and it is not enough: `??` faithfully propagates the CPU's zero, and a *limit* of zero refuses the fusion even harder — `shared memory 1048576 exceeds limit 0`. The `||` was hiding a second question the code never asked, which is what a device reporting no scratchpad *means*. It does not mean a budget of zero bytes; it means the budget does not apply, because a fused intermediate on a CPU lives in ordinary memory and nothing on-chip bounds it. So the resolution has three cases, not two ([`fusion_cost.ts:43`](../../../src/compiler/passes/fusion/fusion_cost.ts)):

```ts
function deviceLimit(stated: number | undefined, whenUnspecified: number): number {
  if (stated === undefined) return whenUnspecified;
  return stated === 0 ? Infinity : stated;
}
```

`maxRegistersPerThread` gets the same treatment, and for the same reason: `target.registersPerThread` is also `0` on CPU and WASM, and `|| 255` was handing them a GPU register file too. With both budgets resolved that way the lab reads:

```
=== strategy 'priority' (the default) ===
  cost model: add+mul+add -> fused
  1 fusion region(s) holding: add, mul, add
  0.621 ms

=== strategy 'dominator' ===
  cost model: mul+add+add -> fused: saves 6291456 bytes, 10us launch
  1 fusion region(s) holding: add, mul, add
  0.621 ms

=== strategy 'greedy' (the original FusionPass) ===
  cost model: add+mul+add -> fused: saves 6291456 bytes, 10us launch
  1 fusion region(s) holding: add, mul, add
  0.615 ms

=== strategy 'dominator', shared-memory budget raised to 16 MiB ===
  cost model: mul+add+add -> fused: saves 6291456 bytes, 10us launch
  1 fusion region(s) holding: add, mul, add
  0.622 ms
```

All four runs now agree, and the control run — the one that had to raise the limit by hand — is no longer distinguishable from the default. The default strategy went from 1.473 ms to 0.621 ms, so the finding was worth **2.4× on the path everyone actually takes**, not only on the two strategies nobody selects.

This is a one-character class of bug — `||` where `??` was meant — and it is worth dwelling on because of where it landed. It did not produce a wrong answer, it did not fail a test, and it did not appear in any trace unless you were running the non-default strategy that explains its refusals. It cost 2.4× and reported nothing. It is also worth noticing that the one-character fix alone would have made things worse, and that the difference between "no shared memory" and "48 KiB of shared memory" is a difference the *type* `number` cannot express.

**Try this.** Set `sharedMemoryBytes` on the target to `1` instead of leaving it at zero, and confirm the fusion is refused for a limit of 1 — which is the evidence that a stated budget is still honoured, and that only silence is now read as silence.

## 24.6 Lab 2 — Epilogue fusion

```bash
node docs/part4/ch24-fusion-strategies/labs/02-epilogue-fusion.mjs
```

Two `Linear` layers with ReLU. With the CPU default:

```
    %5 = dot(%0, %1) {lhs_batch = [], lhs_contracting = [1], rhs_batch = [], rhs_contracting = [1]} : tensor<64x256xf32>
    %6 = constant() {tensor_type = tensor<64x256xf32>, value = 0} : tensor<64x256xf32>
    %7 = fusion(%5, %2, %6) {fusion_kind = "kElementwise"} : tensor<64x256xf32>
    {
      ^bb(%8: tensor<64x256xf32>, %9: tensor<256xf32>, %10: tensor<64x256xf32>):
      %11 = add(%8, %9) : tensor<64x256xf32>
      %12 = maximum(%11, %10) : tensor<64x256xf32>
      yield(%12)
    }
    %13 = dot(%7, %3) {lhs_batch = [], lhs_contracting = [1], rhs_batch = [], rhs_contracting = [1]} : tensor<64x128xf32>
    %14 = constant() {tensor_type = tensor<64x128xf32>, value = 0} : tensor<64x128xf32>
    %15 = fusion(%13, %4, %14) {fusion_kind = "kElementwise"} : tensor<64x128xf32>
    {
      ^bb(%16: tensor<64x128xf32>, %17: tensor<128xf32>, %18: tensor<64x128xf32>):
      %19 = add(%16, %17) : tensor<64x128xf32>
      %20 = maximum(%19, %18) : tensor<64x128xf32>
      yield(%20)
    }
    return(%15)
```

Four kernels: matmul, bias+relu, matmul, bias+relu. Turning on `enableEpilogueFusion` — the flag CUDA sets and CPU does not — gives:

```
    %5 = fused_dot_epilogue(%0, %1, %2) {epilogue_ops = ["add", "constant", "maximum"], epilogue_tags = ["bias", "relu"], lhs_batch = [], lhs_contracting = [1], num_dot_operands = 2, num_extra_inputs = 1, rhs_batch = [], rhs_contracting = [1]} : tensor<64x256xf32>
    %6 = fused_dot_epilogue(%5, %3, %4) {epilogue_ops = ["add", "constant", "maximum"], epilogue_tags = ["bias", "relu"], lhs_batch = [], lhs_contracting = [1], num_dot_operands = 2, num_extra_inputs = 1, rhs_batch = [], rhs_contracting = [1]} : tensor<64x128xf32>
    return(%6)
```

Eleven operations become three. The first listing holds two `dot`s, two zero constants, two `fusion` regions, the four elementwise operations inside those regions and the `return`; all ten of the non-terminators collapse into two `fused_dot_epilogue`s. Each `Linear` + `ReLU` is now a single operation carrying its epilogue as an attribute, tagged `["bias", "relu"]` so a backend can recognize the shape without re-deriving it.

And the measurement: 3.444 ms with the flag off, 3.511 ms with it on. **No improvement.** That is the honest result and it explains the flag. On CPU the epilogue was *already* one fused kernel, so folding it into the matmul saves one pass over a 64×256 tensor — 64 KiB — against a matmul that moves far more than that and is compute-bound anyway. The change is real and the benefit is below the noise.

On a GPU the arithmetic is different: the matmul's output sits in registers at the end of the kernel, and writing it out, launching a second kernel, and reading it back costs a full round trip plus a launch at GPU launch latencies. That is why `enableEpilogueFusion` is `true` for CUDA and `false` for CPU ([`target.ts:220`](../../../src/backend/target.ts)) — not because the transformation is invalid on CPU, but because it is not worth the code path there.

This is the clearest example in Part IV of an optimization whose value is entirely a property of the target, and it is the reason the flag exists rather than a heuristic: the cost model has no term that would have discovered it.

## 24.7 Traps and limits

- **A per-thread resource budget of `0` now means "no such budget", which is a convention and not a type.** §24.5 is the fix; the residual trap is that a real device with genuinely zero usable shared memory cannot be expressed, because `0` is spoken for. No target in the tree is in that position, and the alternative — an explicit `null` for "not applicable" — would have to be threaded through `TargetFeatures`, three fusion passes and the cost model to be worth anything.
- **The launch weight is 1000 and is not target-derived.** [`fusion_cost.ts:38`](../../../src/compiler/passes/fusion/fusion_cost.ts) fixes `{ memory: 1, launch: 1000 }` for every target, so a saved launch is worth 5,000 bytes of traffic everywhere. A target may override it through the `fusionBenefitWeights` attribute ([`priority_fusion.ts:55`](../../../src/compiler/passes/fusion/priority_fusion.ts)); none does.
- **The strategy string has no validation.** Anything that is not `'dominator'` or `'priority'` silently selects `FusionPass` ([`graph_pipeline.ts:88`](../../../src/compiler/pipeline/graph_pipeline.ts)). A typo in a config becomes a different fusion engine rather than an error.
- **`edgeBenefit` scores the edge, not the merge.** The heap is ordered by the bytes on one dataflow edge plus a launch constant, while the accept/reject decision uses the full group cost. So the *ordering* and the *decision* use different models, and a candidate can be at the top of the heap and then rejected — which is fine, and means the heap ordering is a heuristic on a heuristic.
- **Priority fusion explains only its successes.** Chapter 18's finding, and §24.5 is why it matters: while the budget was wrong, the default strategy's refused merge produced no event at all, and the refusal was visible only by switching to a strategy you were not going to use. The finding cost 2.4× and the mechanism that should have reported it stayed silent, which is a stronger argument for explaining refusals than any of Chapter 18's.
- **`FusionMergerPass` runs only with the `greedy` strategy.** It exists to merge two already-formed groups, which is the step the priority engine performs inline. Under `dominator` neither pass runs, so a dominator-based compilation has no mechanism at all for combining two regions after the fact.

## 24.8 Read the tests

- [`tests/compiler/passes/fusion/priority.test.js`](../../../tests/compiler/passes/fusion/priority.test.js) — heap ordering, the version-based staleness of Definition 24.5, and the weighted union.
- [`tests/compiler/passes/fusion/dominator.test.js`](../../../tests/compiler/passes/fusion/dominator.test.js) — region selection through the post-dominance analysis.
- [`tests/compiler/passes/fusion/multi-output.test.js`](../../../tests/compiler/passes/fusion/multi-output.test.js) and [`epilogue.test.js`](../../../tests/compiler/passes/fusion/epilogue.test.js) — the two auxiliary passes, including the epilogue tagging §24.6 shows.
- [`tests/compiler/passes/fusion/cost.test.js`](../../../tests/compiler/passes/fusion/cost.test.js) — every veto in `shouldFuse`, which is where the 48 KiB limit is asserted as behaviour.

---

**Next:** [Chapter 25 — Layout](../ch25-layout/README.md), which changes not what is computed or in what groups, but how the numbers are arranged in memory — and which is the first optimization in this part that is switched off by default.
