# Chapter 24 — Fusion III: the three strategies

Chapter 22 said fusion is worth roughly 1.9×. Chapter 23 said a merge is legal exactly when contracting it leaves a DAG. Neither says *which* legal, profitable merges to make, in what order, and that turns out to be the question with the largest spread between good and bad answers.

There are three fusion engines in this compiler and two auxiliary passes. They see the same graph, use the same legality checks and the same cost model, and on a three-operation program they used to produce three different answers, the best of which ran about 2.3× faster than the default. They now produce the same answer. What separated them was not the strategies at all — it was a single `||` in the cost model, and §24.5 is the chase.

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

> **Definition 24.1 (Fusion partition).** **(stated here)** A *fusion partition* of a DAG `G` is a partition of its vertices into groups such that contracting every group simultaneously leaves a DAG. Its *benefit* is the sum over groups of `memorySaved + launchSaved` from Chapter 22.

> **Theorem 24.2 (Hardness).** **(classical)** Fix a DAG `G` with rational edge benefits and a bound `B`. Deciding whether `G` admits a fusion partition (Definition 24.1) of total benefit at least `B` is NP-complete.
>
> *Proof sketch.* Membership is clear: a partition is a polynomial-size certificate, and both its acyclicity after contraction and its benefit are checkable in polynomial time. For hardness, note that the acyclicity constraint is vacuous on a DAG with no edges between the vertices being grouped, so the problem restricted to such instances is exactly **maximum-benefit graph partitioning** with arbitrary weights, which is NP-hard by reduction from `MAX-CUT`. ∎

**State the problem before quoting the hardness, because "fusion is NP-hard" on its own means very little.** Several distinct problems live under that phrase and they have different answers. Deciding whether *one* merge is legal is polynomial — Chapter 23's cycle check. Finding *any* maximal legal partition is polynomial and is what a greedy engine does. Finding the partition of *maximum benefit* under a fixed additive cost function is the one Theorem 24.2 is about, and it is the one no engine here attempts. And "the partition that actually runs fastest" is not in NP at all under any obvious encoding, because the objective is a wall-clock time nobody can evaluate without running the program — which is precisely why Part VIII stops predicting and starts measuring.

So the load-bearing claim for this chapter is the third one, and its practical content is modest: **there is no reason to expect a greedy engine to be optimal, so the differences between the three engines below are expected rather than surprising.** The hardness does not tell you how far from optimal any of them is, and §24.5's finding is a reminder that in practice the gap between engines was not a search-quality gap at all.

Hence heuristics, and hence the practical question is not "which is optimal" but "which fails less badly, and how would you know". Two properties distinguish greedy schemes here:

> **Definition 24.3 (Monotone candidate set).** **(stated here)** A greedy scheme has a *monotone* candidate set if merging never makes a previously-illegal merge legal. Under monotonicity, a candidate rejected once may be discarded; without it, rejected candidates must be reconsidered after every merge.

Fusion is **not** monotone in the cost sense: merging `A` with `B` changes the group's inputs and outputs, so an edge that was unprofitable can become profitable, and vice versa. It is tempting to think it is at least monotone in the cycle sense — that contracting more can only create cycles, never remove them — and that is false in both directions:

> **Counterexample 24.4 (Legality is not monotone).** Take `A → X → B` together with `A → B`. Contracting `{A, B}` is illegal: the merged vertex reaches `X` and is reached by it. Contracting `{A, X, B}` is legal, since it leaves one vertex and no edges at all. So a merge rejected for a cycle can become legal after a later merge — here, after `X` joins the group — and a legal contraction can have an illegal sub-contraction.

The consequence is that a rejected candidate may not be discarded. The priority engine keeps a heap it re-pushes into after every merge, and validates popped entries against a version counter rather than trusting them:

> **Definition 24.5 (Stale candidate).** **(stated here)** A heap entry naming two groups is *stale* if either group has been merged since the entry was pushed. A stale entry must be discarded rather than applied, because its recorded benefit was computed for groups that no longer exist.

This is a lazy-deletion priority queue, and it is the standard way to avoid the alternative — finding and removing every affected entry on each merge, which needs an indexed heap.

## 24.4 In mlfw: five passes

[`buildGraphPipeline`](../../../src/compiler/pipeline/graph_pipeline.ts) selects between them on a config string ([`graph_pipeline.ts:93`](../../../src/compiler/pipeline/graph_pipeline.ts)):

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

Three data structures ([`priority_fusion.ts:100`](../../../src/compiler/passes/fusion/priority_fusion.ts)):

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

with weights `{ memory: 1, launch: 1000 }` ([`fusion_cost.ts:38`](../../../src/compiler/passes/fusion/fusion_cost.ts)).

**Check the dimensions, because they do not check out.** `launchOverheadUs` is a time, in microseconds. `bytes` is a count of bytes. The expression adds `1000 × 5 µs` to `1 × bytes` and returns a `number` — so the quantity being maximized has no unit at all, and the "1000" is not a conversion factor between microseconds and bytes but an unexplained scalar that happens to make a saved launch outrank about 5000 bytes of traffic. Nothing in the code names that exchange rate or says where it came from, and TypeScript is content because both sides are `number`.

A dimensionally sound version is not hard to write and is worth knowing as the target: pick one unit — time — and convert. Traffic becomes `bytes / bandwidth`, launches stay in microseconds, the sum is in microseconds, and the "weight" disappears entirely because the machine supplies the exchange rate. Chapter 4's roofline gives exactly the number needed: at the ~7.7 GB/s this eager CPU path achieves, 5000 bytes is about 0.65 µs, so the current weights value a launch at roughly eight times a 5000-byte transfer rather than exactly equal to it. Whether that is right is now a *question you can ask*, which is the whole benefit of carrying units.

Two consequences follow from the unit-free form. The weights cannot be transferred between machines, because a ratio between a time and a byte count is a property of a machine's bandwidth and launch latency, and nothing records which machine these were tuned on. And they cannot be transferred between backends: §22.3 showed `launchOverheadUs` is a target-independent constant, so on the CPU the first term contributes a large fixed bonus for a saving that is not there. Chapter 46 rebuilds this idea properly, with a model whose output is a predicted *time* and which is graded against measured times.

The main loop is Definition 24.5 in practice ([`priority_fusion.ts:191`](../../../src/compiler/passes/fusion/priority_fusion.ts)):

```ts
    while (!heap.isEmpty()) {
      const cand = heap.pop() as MergeCandidate;
      const ra = cycles.find(cand.a);
      const rb = cycles.find(cand.b);
      if (ra === rb) continue;
      if (version[ra] !== cand.va || version[rb] !== cand.vb) continue;
```

Already merged, or either endpoint's version has moved: discard and take the next. Then the three checks in Chapter 23's order, the merge, and a re-evaluation of every edge the new group touches.

One detail in the merge is worth noticing ([`priority_fusion.ts:206`](../../../src/compiler/passes/fusion/priority_fusion.ts)):

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

Three operations, all elementwise, all the same shape, on 256K-element tensors. Every pair is legal by Chapter 23. And this is what the lab printed before the fix described below (Node 24.9, 2026-08-21):

```
=== strategy 'priority' (the default) ===
  cost model: add+mul -> fused
  1 fusion region(s) holding: add, mul
  1.014 ms

=== strategy 'dominator' ===
  cost model: mul+add+add -> not-fused: shared memory 1048576 exceeds limit 49152
  0 fusion region(s) holding: (nothing)
  1.083 ms

=== strategy 'greedy' (the original FusionPass) ===
  cost model: add+mul+add -> not-fused: shared memory 1048576 exceeds limit 49152
  0 fusion region(s) holding: (nothing)
  1.109 ms

=== strategy 'dominator', shared-memory budget raised to 16 MiB ===
  cost model: mul+add+add -> fused: saves 6291456 bytes, 10us launch
  1 fusion region(s) holding: add, mul, add
  0.438 ms
```

Two of the three engines fused **nothing**. The reason is in the explanation: `s` is used twice inside the candidate group, so the cost model charges it as shared memory ([`fusion_cost.ts:203`](../../../src/compiler/passes/fusion/fusion_cost.ts)) at its full tensor size — one megabyte — against a limit of 48 KiB.

The priority engine reached a different answer for a structural reason. It merges *edges*, so it first forms `{add, mul}` — at which point `s` has one internal user and one external one, no shared-memory charge, and the merge passes. Then it tries to add the final `add`, which would make `s` internally reused, and *that* merge is refused. It stopped one step short of the group the other two engines evaluated as a whole, producing a two-output region: the kernel yields both `s` and `s*x`, and the final add reads them back.

The fourth run is the control. Same engine, same graph, one number changed — and all three operations fuse into one kernel, **about 2.3× faster than the default configuration**. So the 48 KiB limit cost this program roughly 2.3×, and it is worth asking where the number came from. 48 KiB is a GPU shared-memory budget, and this compilation targets a CPU, which has no shared memory at all. A CPU target says nothing about the quantity, and the chain of defaults underneath read that silence as the GPU number rather than as "not applicable" — a distinction the type `number` cannot express, and one §24.7 takes up. Resolving silence and zero separately, for the shared-memory budget and the per-thread register budget alike, the lab reads:

```
=== strategy 'priority' (the default) ===
  cost model: add+mul+add -> fused
  1 fusion region(s) holding: add, mul, add
  0.450 ms

=== strategy 'dominator' ===
  cost model: mul+add+add -> fused: saves 6291456 bytes, 10us launch
  1 fusion region(s) holding: add, mul, add
  0.434 ms

=== strategy 'greedy' (the original FusionPass) ===
  cost model: add+mul+add -> fused: saves 6291456 bytes, 10us launch
  1 fusion region(s) holding: add, mul, add
  0.441 ms

=== strategy 'dominator', shared-memory budget raised to 16 MiB ===
  cost model: mul+add+add -> fused: saves 6291456 bytes, 10us launch
  1 fusion region(s) holding: add, mul, add
  0.429 ms
```

All four runs now agree, and the control run — the one that had to raise the limit by hand — is no longer distinguishable from the default. The default strategy went from 1.014 ms to 0.450 ms, so the budget was worth **2.3× on the path everyone actually takes**, not only on the two strategies nobody selects.

Both columns come from `docs/part4/ch24-fusion-strategies/labs/01-three-strategies.mjs`, Node 24.9, 2026-08-21, on a `1<<18`-element chain. The same pair measured on another machine came out 1.473 ms → 0.621 ms, a ratio four percent away, which is about as much precision as this measurement deserves. What should reproduce is not the exact pair but the shape: the four rows agree with one another, and the misconfigured default is roughly twice the corrected one.

### So what does separate the three engines?

The measurement above answers the chapter's opening question in a way the chapter did not intend: on this program, *nothing* separates them once the cost model is right. That is worth taking seriously rather than filing away, because it is the most common outcome of a strategy comparison — the strategies were never the variable.

But they are not interchangeable, and the differences are structural rather than measurable on three operations. Four of them, in the order you are likely to meet them:

| | priority (default) | dominator | greedy |
|---|---|---|---|
| what a veto costs you | the *last* merge only — the group formed so far survives | the **whole region**, since the cost model is evaluated on it as a unit | the rest of the current walk |
| horizontal fusion | `MultiOutputFusionPass` runs after it | **no auxiliary pass at all** | `MultiOutputFusionPass` runs after it |
| merging two finished groups | inline, as part of the main loop | not available | `FusionMergerPass` |
| cost of deciding | a heap plus re-scoring every touched edge after each merge — §23.6's quadratic term | one pass over the post-dominance tree | one walk |

The first row is the one §24.5 accidentally demonstrated. While the shared-memory budget was wrong, priority stopped one operation short and still produced a two-output region; dominator and greedy evaluated the three-operation region as a whole, hit the same veto, and produced nothing. **An edge-at-a-time engine degrades; a region-at-a-time engine falls off a cliff.** That is a real and general property, and it is the argument for the default — not that priority finds better partitions, but that it fails more gracefully when the cost model is wrong, which it periodically will be.

The second and third rows say the opposite thing about `dominator`: it is the only strategy with no way to combine regions after the fact and no horizontal fusion, so a graph whose wins are sibling-shaped gets none of them. And the fourth row is the price of the first: §23.6 measured the re-scoring loop at close to quadratic in the size of the final group, which the other two engines do not pay.

None of that is measured here, and it would need graphs shaped to expose each row rather than one diamond. What §24.5 does establish is narrower and more useful: **before comparing search strategies, check that they are searching under the same constraints.** Three engines disagreeing by a factor of two looked like a search-quality result for as long as nobody read the constraint they shared.

## 24.6 Lab 2 — Epilogue fusion

```bash
node docs/part4/ch24-fusion-strategies/labs/02-epilogue-fusion.mjs
```

Two `Linear` layers with ReLU. With the CPU default:

```
    %5 = tera.dot %0, %1 {lhs_batch = array<i64>, lhs_contracting = array<i64: 1>, rhs_batch = array<i64>, rhs_contracting = array<i64: 1>} : (tensor<64x128xf32>, tensor<256x128xf32>) -> tensor<64x256xf32>
    %6 = tera.constant dense<0.0> : tensor<64x256xf32>
    %7 = "tera.fusion"(%5, %2, %6) ({
      ^bb0(%8: tensor<64x256xf32>, %9: tensor<256xf32>, %10: tensor<64x256xf32>):
        %11 = "tera.add"(%8, %9) : (tensor<64x256xf32>, tensor<256xf32>) -> tensor<64x256xf32>
        %12 = tera.maximum %11, %10 : tensor<64x256xf32>
        tera.yield %12 : tensor<64x256xf32>
    }) {fusion_kind = "kElementwise"} : (tensor<64x256xf32>, tensor<256xf32>, tensor<64x256xf32>) -> tensor<64x256xf32>
    %13 = tera.dot %7, %3 {lhs_batch = array<i64>, lhs_contracting = array<i64: 1>, rhs_batch = array<i64>, rhs_contracting = array<i64: 1>} : (tensor<64x256xf32>, tensor<128x256xf32>) -> tensor<64x128xf32>
    %14 = tera.constant dense<0.0> : tensor<64x128xf32>
    %15 = "tera.fusion"(%13, %4, %14) ({
      ^bb0(%16: tensor<64x128xf32>, %17: tensor<128xf32>, %18: tensor<64x128xf32>):
        %19 = "tera.add"(%16, %17) : (tensor<64x128xf32>, tensor<128xf32>) -> tensor<64x128xf32>
        %20 = tera.maximum %19, %18 : tensor<64x128xf32>
        tera.yield %20 : tensor<64x128xf32>
    }) {fusion_kind = "kElementwise"} : (tensor<64x128xf32>, tensor<128xf32>, tensor<64x128xf32>) -> tensor<64x128xf32>
    return %15 : tensor<64x128xf32>
  }
```

Four kernels: matmul, bias+relu, matmul, bias+relu. Turning on `enableEpilogueFusion` — the flag CUDA sets and CPU does not — gives:

```
    %5 = "tera.fused_dot_epilogue"(%0, %1, %2) {epilogue_ops = ["add", "constant", "maximum"], epilogue_tags = ["bias", "relu"], lhs_batch = [], lhs_contracting = [1], num_dot_operands = 2, num_extra_inputs = 1, rhs_batch = [], rhs_contracting = [1]} : (tensor<64x128xf32>, tensor<256x128xf32>, tensor<256xf32>) -> tensor<64x256xf32>
    %6 = "tera.fused_dot_epilogue"(%5, %3, %4) {epilogue_ops = ["add", "constant", "maximum"], epilogue_tags = ["bias", "relu"], lhs_batch = [], lhs_contracting = [1], num_dot_operands = 2, num_extra_inputs = 1, rhs_batch = [], rhs_contracting = [1]} : (tensor<64x256xf32>, tensor<128x256xf32>, tensor<128xf32>) -> tensor<64x128xf32>
    return %6 : tensor<64x128xf32>
```

Eleven operations become three. The first listing holds two `dot`s, two zero constants, two `fusion` regions, the four elementwise operations inside those regions and the `return`; all ten of the non-terminators collapse into two `fused_dot_epilogue`s. Each `Linear` + `ReLU` is now a single operation carrying its epilogue as an attribute, tagged `["bias", "relu"]` so a backend can recognize the shape without re-deriving it.

And the measurement: 3.444 ms with the flag off, 3.511 ms with it on. **No improvement.** That is the honest result and it explains the flag. On CPU the epilogue was *already* one fused kernel, so folding it into the matmul saves one pass over a 64×256 tensor — 64 KiB — against a matmul that moves far more than that and is compute-bound anyway. The change is real and the benefit is below the noise.

On a GPU the arithmetic is different: the matmul's output sits in registers at the end of the kernel, and writing it out, launching a second kernel, and reading it back costs a full round trip plus a launch at GPU launch latencies. That is why `enableEpilogueFusion` is `true` for CUDA and `false` for CPU ([`target.ts:230`](../../../src/compiler/support/target.ts)) — not because the transformation is invalid on CPU, but because it is not worth the code path there.

This is the clearest example in Part IV of an optimization whose value is entirely a property of the target, and it is the reason the flag exists rather than a heuristic: the cost model has no term that would have discovered it.

## 24.7 Traps and limits

**Where §24.5's 2.3× came from.** `CPUTarget` does not mention `sharedMemoryBytes`, and the base constructor turns that silence into a number — `config.sharedMemoryBytes || 0` ([`target.ts:120`](../../../src/compiler/support/target.ts)). The cost model then read that number as `config.maxSharedMemory || 49152` ([`fusion_cost.ts:62`](../../../src/compiler/passes/fusion/fusion_cost.ts)). **Zero is falsy**, so "I have no shared memory" and "I did not say" collapsed to the same value twice over, and the second `||` read it as the latter — falling back to the GPU default. `CUDATarget` sets `48 * 1024` explicitly and was unaffected; CPU and WASM, which say nothing, inherited a budget for a resource they do not have. `maxRegistersPerThread` had the same shape, `|| 255` handing them a GPU register file too.

Writing `??` for `||` is not enough on its own: `??` faithfully propagates the CPU's zero, and a *limit* of zero refuses the fusion even harder. A device reporting no scratchpad does not mean a budget of zero bytes; it means the budget does not apply, because a fused intermediate on a CPU lives in ordinary memory and nothing on-chip bounds it. So the resolution has three cases, not two ([`fusion_cost.ts:43`](../../../src/compiler/passes/fusion/fusion_cost.ts)):

```ts
function deviceLimit(stated: number | undefined, whenUnspecified: number): number {
  if (stated === undefined) return whenUnspecified;
  return stated === 0 ? Infinity : stated;
}
```

That class of defect is worth dwelling on because of where it landed: it produced no wrong answer, failed no test, and appeared in no trace unless you were running the non-default strategy that explains its refusals. It cost more than a factor of two and reported nothing. To confirm that a stated budget is still honoured, set `sharedMemoryBytes` to `1` and watch the fusion refused for a limit of 1 — only silence is now read as silence.

- **A per-thread resource budget of `0` means "no such budget", which is a convention and not a type.** The residual trap is that a real device with genuinely zero usable shared memory cannot be expressed, because `0` is spoken for. No target in the tree is in that position, and the alternative — an explicit `null` for "not applicable" — would have to be threaded through `TargetFeatures`, three fusion passes and the cost model to be worth anything.
- **The launch weight is 1000 and is not target-derived.** [`fusion_cost.ts:38`](../../../src/compiler/passes/fusion/fusion_cost.ts) fixes `{ memory: 1, launch: 1000 }` for every target, so a saved launch is worth 5,000 bytes of traffic everywhere. A target may override it through the `fusionBenefitWeights` attribute ([`priority_fusion.ts:55`](../../../src/compiler/passes/fusion/priority_fusion.ts)); none does.
- **The strategy string has no validation.** Anything that is not `'dominator'` or `'priority'` silently selects `FusionPass` ([`graph_pipeline.ts:98`](../../../src/compiler/pipeline/graph_pipeline.ts)). A typo in a config becomes a different fusion engine rather than an error.
- **`edgeBenefit` scores the edge, not the merge.** The heap is ordered by the bytes on one dataflow edge plus a launch constant, while the accept/reject decision uses the full group cost. So the *ordering* and the *decision* use different models, and a candidate can be at the top of the heap and then rejected — which is fine, and means the heap ordering is a heuristic on a heuristic.
- **Priority fusion explains only its successes.** Chapter 18's finding, and §24.5 is why it matters: while the budget was wrong, the default strategy's refused merge produced no event at all, and the refusal was visible only by switching to a strategy you were not going to use. The finding cost more than a factor of two and the mechanism that should have reported it stayed silent, which is a stronger argument for explaining refusals than any of Chapter 18's.
- **`FusionMergerPass` runs only with the `greedy` strategy.** It exists to merge two already-formed groups, which is the step the priority engine performs inline. Under `dominator` neither pass runs, so a dominator-based compilation has no mechanism at all for combining two regions after the fact.

## 24.8 Read the tests

- [`tests/compiler/passes/fusion/priority.test.js`](../../../tests/compiler/passes/fusion/priority.test.js) — heap ordering, the version-based staleness of Definition 24.5, and the weighted union.
- [`tests/compiler/passes/fusion/dominator.test.js`](../../../tests/compiler/passes/fusion/dominator.test.js) — region selection through the post-dominance analysis.
- [`tests/compiler/passes/fusion/multi-output.test.js`](../../../tests/compiler/passes/fusion/multi-output.test.js) and [`epilogue.test.js`](../../../tests/compiler/passes/fusion/epilogue.test.js) — the two auxiliary passes, including the epilogue tagging §24.6 shows.
- [`tests/compiler/passes/fusion/cost.test.js`](../../../tests/compiler/passes/fusion/cost.test.js) — every veto in `shouldFuse`, which is where the 48 KiB limit is asserted as behaviour.

---

**Next:** [Chapter 25 — Layout](../ch25-layout/README.md), which changes not what is computed or in what groups, but how the numbers are arranged in memory — and which is the first optimization in this part that is switched off by default.
