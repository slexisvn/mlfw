# Chapter 45 — Sketches

Chapter 44 counted a space without saying where it comes from. The count assumed a skeleton — four tile levels on each spatial axis, one on the reduction axis, a `parallelize` here and a `vectorize` there — and took the parameters of that skeleton as the thing being searched. This chapter is about the skeleton: what generates it, what it is allowed to assume about the block it is handed, and what happens when the assumption is wrong.

The name for it is a **sketch**, and the idea is Ansor's: rather than searching over sequences of primitives, generate a small number of *incomplete schedules* — structurally correct, parametrically open — and search the parameters. It is the move that makes the search space finite and, more importantly, makes almost all of it legal. A random sequence of twenty primitives is almost surely a program that does not typecheck. A random point of a sketch is almost surely a program that runs.

## 45.1 The problem: a random sequence of primitives is not a schedule

Take the twenty-two primitives of Chapter 38 and compose five of them at random on the matmul nest. `fuseLoops` of a loop with its grandchild: refused, the inner loop must be a direct child. `computeAt` between two blocks with different iteration domains: refused. `reorder` of loops that are not on one path: refused. `decomposeReduction` on anything the lowering rules emit: refused. Almost every sequence dies on its first or second step, and the ones that survive are dominated by no-ops.

That is Part VII working exactly as designed — each primitive refuses rather than repairs — and it is useless as a generator. What a search needs is a *distribution over legal programs*, and the primitives give a *predicate on programs*. Turning the second into the first is what a sketch does.

The refusals are also not a complete defence, which is the other reason a random walk is the wrong tool. `rfactor('matmul_1', 'ls0_6', 2)` — reassociate the matmul over its *row* axis, which is not a sum at all — is accepted, and §45.5 runs the kernel it produces. A generator that only avoids what the primitives reject is relying on a predicate that was never meant to be complete.

The second problem is that legality is not the interesting constraint. A schedule that is legal but structurally silly — nine loops with eight of extent 1, a `parallelize` on a loop that runs once — costs a compile and a measurement and teaches the search nothing. What you want generated is the shape a human expert would write, with the numbers left blank. For a reduction over two operands the expert's shape has a name: split every spatial axis into a parallel level, a cache-blocking level, a register-blocking level and a vector level; split the reduction axis into an outer and an inner; and interleave them in the order `S S R S R S`. That string is the skeleton, and this compiler has it — in a constant, unreachable.

## 45.2 Intuition: a form with typed fields

A sketch is a form. The fields are named and each has a menu; filling them in produces a schedule; the menus are chosen so that every filled-in form is legal.

Three things follow from that picture, and all three matter.

**The form knows the block.** The menus are not fixed constants — `s0`'s candidate list is the factorisations of *this* block's first spatial extent. So a sketch is generated per block, not looked up, and generating it requires reading the loop nest.

**Choosing the form is a classification problem.** Before you can fill in the fields you must decide which form. Is this block a reduction with two operands, i.e. a contraction worth tiling? A reduction with one operand, i.e. a `sum` where tiling buys nothing? Or elementwise, where the only question is the vector width? Three questions, tried in order, and the last one matches everything so the classification is total.

**The form is a program, not a description.** A sketch's payload is a function that calls schedule primitives. It is not a declarative record that some later stage interprets. That makes it powerful — the GPU matmul sketch of §45.5 exploits it to replace the function body outright — and it makes it opaque: nothing can look at a sketch and say what it will do without running it.

## 45.3 Theory

> **Definition 45.1 (Sketch, stated here).** A *sketch* is a triple `(name, V, apply)` where `V = (V₁,…,V_k)` is a finite list of search variables with finite candidate sets, and `apply : Schedule × BlockName × Target × Params → unit` is a partial function defined by a sequence of schedule-primitive calls. `apply` is *total on the space* if it does not throw for any `p ∈ C₁ × ⋯ × C_k`.

The definition deliberately does not require totality, because one of the sketches this compiler derives for a matmul is defined at no point of its space at all.

> **Definition 45.2 (Derivation, stated here).** A *sketch rule* is a pair `(matches, derive)` with a priority. `derive` maps a block to a list of sketches. The *derivation* of a block is `derive` of the first rule, in priority order, whose `matches` accepts the block's structure.

First-match, not best-match: the rules are a decision list, so their order is part of their meaning and adding a rule at a lower priority number can silently take work away from an existing one.

> **Proposition 45.3 (Derivation is total, given a supported target and `richGpu` off, stated here).** For a CPU or GPU target with `opts.richGpu` unset, `deriveSketches` returns a non-empty list for every block. It returns the empty list for a target that is neither CPU nor GPU ([`derivation.ts:78`](../../../src/compiler/autotune/derivation.ts)), and for a GPU target with `richGpu` set it returns the empty list for every block of a recognised pure matmul except the reduction block ([`gpu_matmul_sketch.ts:539`](../../../src/compiler/autotune/gpu_matmul_sketch.ts)).

*Proof.* The rule registered at priority 30 has `matches: () => true` ([`derivation.ts:69`](../../../src/compiler/autotune/derivation.ts)) and its `derive` returns a one-element list, so the scan in `deriveSketches` always terminates on a rule that produces a sketch. The two exceptions are the two early returns above the scan. ∎

The exceptions are not idle. `WasmTarget` is neither `TargetKind.CPU` nor `isGPU()`, so **on WASM the autotuner derives no sketches at all** and every block falls through to the `empty` task kind — a fact §45.7 returns to.

The core of any tiling sketch is the claim that a factorisation can be realised by repeated splitting without arithmetic loss.

> **Definition 45.4 (Tile structure, stated here).** A *tile structure* is an ordered list of `(kind, level)` pairs with `kind ∈ {S, R}`, together with a partial map from `kind·level` to a role in `{parallelize, vectorize, unroll, blockIdx, threadIdx}`. Its *level counts* `L_S`, `L_R` are one more than the largest level of each kind.

> **Proposition 45.5 (A multi-level split realises a factorisation exactly, stated here).** Let a loop have constant extent `n` and let `(f₀,…,f_{L−1})` satisfy `∏ f_i = n`. Then `multiLevelSplit` produces `L` nested loops whose extents are exactly `f₀,…,f_{L−1}`, and no guard predicate is introduced.

*Proof.* By induction on `i`. Before step `i` the loop being split has extent `∏_{j≥i} f_j`; the code splits by `inner = ∏_{j>i} f_j` ([`tiling.ts:23`](../../../src/compiler/autotune/tiling.ts)), which divides that extent exactly with quotient `f_i`. Chapter 40's `split` therefore computes `outerExtent = ⌈(∏_{j≥i} f_j)/inner⌉ = f_i` and, since `extent % factor === 0`, takes the branch that emits no `IfThenElseNode` ([`schedule.ts:282`](../../../src/compiler/schedule/schedule.ts)). The remaining loop has extent `∏_{j>i} f_j`, which is the induction hypothesis at `i+1`. After `L−1` steps the innermost extent is `f_{L−1}`. ∎

> **Corollary 45.6 (The bindings are a mixed-radix reconstruction, stated here).** After a multi-level split of an axis into `(f₀,…,f_{L−1})`, the block's binding for that axis is `Σ_i v_i · ∏_{j>i} f_j`, which by Theorem 35.3 is a bijection from `∏ [0, f_i)` onto `[0, n)`. The iteration space is therefore preserved exactly — the same points, renamed.

Proposition 45.5 is why Chapter 40's guard, which exists for the general case, never appears in a tiling sketch: the search only ever offers factor tuples that multiply back to the extent. It also explains why a factor of `1` is harmless — it produces a loop of extent 1 and a multiplier that repeats, which the simplifier folds — and why the sketch has to keep those degenerate tuples: without `[1,1,1,n]` a prime extent would have no tiling at all.

Soundness of the whole space then follows from Part VII, with two named exceptions.

> **Theorem 45.7 (A sketch space is sound if its primitives are, stated here).** Let `apply` call only primitives that are sound in the sense of Definition 38.3. Then for every parameter point on which `apply` does not throw, the resulting `PrimFunc` is semantically equivalent to the input. Consequently no point of the space is a wrong program, and a search over it needs no correctness oracle.

*Proof.* Proposition 38.4: a composition of sound partial functions is sound. A point on which `apply` throws contributes no program at all, because `BlockTuningSession` clones the function before every attempt ([`session.ts:183`](../../../src/compiler/autotune/session.ts)) and discards the clone — which is what the search needs, since `tile` is known to leave the IR modified after failing (Chapter 40, finding 33). ∎

> **Counterexample 45.8 (Two ways the hypothesis fails).** `createRfactorSketch` calls `rfactor`, which is sound only under a relaxed floating-point semantics (Theorem 41.2 and Counterexample 41.3), so the `rfactor` sketch's space contains points that change the answer. And `createMatmulRegisterBlockGPUSketch` calls no primitive at all: it assigns `schedule.func.body` a nest built from scratch ([`gpu_matmul_sketch.ts:393`](../../../src/compiler/autotune/gpu_matmul_sketch.ts)), so Theorem 45.7 says nothing whatever about it. Its correctness rests on `buildRegisterBlockedMatmul` being right, which is a 115-line hand-written kernel generator and a different kind of obligation.

## 45.4 In mlfw

### The three rules

[`derivation.ts:60`](../../../src/compiler/autotune/derivation.ts), and the whole classifier is twelve lines:

```ts
registerSketchRule({
  matches: (s: BlockStructure) => s.hasReduction && s.spatial >= 1 && s.reads >= 2,
  derive: deriveMultiLevel,
}, { priority: 10 });
registerSketchRule({
  matches: (s: BlockStructure) => s.hasReduction,
  derive: (primFunc: PrimFunc, blockName: string, target: ScheduleTarget) => [reductionSketch(target)],
}, { priority: 20 });
registerSketchRule({
  matches: () => true,
  derive: (primFunc: PrimFunc, blockName: string, target: ScheduleTarget) => [elementwiseSketch(target)],
}, { priority: 30 });
```

`BlockStructure` is four numbers ([`block_analysis.ts:38`](../../../src/compiler/autotune/block_analysis.ts)): the count of spatial loops, the count of reduction loops, the size of the *declared* read set, and whether the block has a reduction at all. Nothing else about the block is consulted, and in particular the operator is not — `matches` cannot tell a matmul from a convolution from a hand-written contraction, and does not need to.

`reads >= 2` is the compute-intensity proxy, and it is the one clause that leans on a declaration rather than on the body. A block reading one buffer twice at different subscripts declares one read and is classified as a plain reduction; a block whose declaration is stale in the other direction would be classified as a contraction and offered a tiling sketch it does not deserve. The lowering rules make the declaration accurate today (Chapter 34), and the tiling sketch would still be sound if it were not — only wasteful.

### `deriveMultiLevel`

[`derivation.ts:34`](../../../src/compiler/autotune/derivation.ts), seventeen lines, and the shape of a CPU block's whole offer:

```ts
  const tiling = createMultiLevelTilingSketch(info, getTileStructure(target));
  if (tiling) sketches.push(tiling);
  if (target.kind === TargetKind.CPU) {
    const ssrsrs = createSSRSRSTilingSketch(info, CPU_TILING_SSRSRS);
    if (ssrsrs) sketches.push(ssrsrs);
    const rf = createRfactorSketch(info);
    if (rf) sketches.push(rf);
    const consumer = dag ? findFusibleConsumer(primFunc, dag, blockName, classifyBlock) : null;
    if (consumer) sketches.push(createFusedTilingSketch(consumer));
  }
  sketches.push(reductionSketch(target));
```

Four candidate skeletons plus a fallback, and the `if (…)` guards are all *constructor-time* checks: whether the sketch can be built, not whether it can be applied. `createSSRSRSTilingSketch` builds successfully — it has spatial loops, reduction loops and constant extents — and throws on every parameter. That gap between "was constructed" and "can run" is where the chapter's two findings live.

### `tileBlock` and `applyRoles`

[`tiling.ts:60`](../../../src/compiler/autotune/tiling.ts) is Proposition 45.5 plus a permutation. It splits each spatial axis, then each reduction axis, then assembles the requested nesting order by walking `structure.order` and collecting, at each `(kind, level)`, the level-`level` loop of every axis of that kind:

```ts
  const orderedNames: string[] = [];
  for (const [kind, level] of structure.order) {
    const axes = kind === 'S' ? spatialLevelNames : reductionLevelNames;
    for (const axisNames of axes) {
      if (level < axisNames.length) orderedNames.push(axisNames[level]);
    }
  }
```

So `S0 S1 S2 S3 R0` on a two-spatial-one-reduction block becomes `m₀ n₀ m₁ n₁ m₂ n₂ m₃ n₃ k`: the levels interleave across axes rather than nesting per axis. That is the standard multi-level tiling layout and it is what puts the two innermost spatial loops adjacent, which is what a register-blocking level wants.

`applyRoles` ([`tiling.ts:32`](../../../src/compiler/autotune/tiling.ts)) then applies the annotations, and three of its four branches are worth reading closely:

```ts
    if (role === 'parallelize') {
      const loop = axes[0] && level < axes[0].length ? find(axes[0][level]) : null;
      if (loop) schedule.parallelize(loop);
    } else if (role === 'vectorize') {
      const lastAxis = axes[axes.length - 1];
```

`parallelize` is applied to the *first* axis's level-0 loop and `vectorize` to the *last* axis's level-3 loop. Neither considers the loop's extent: a factorisation of `[1,1,1,64]` gives `parallelize` a loop that runs once, and the primitive accepts it — correctly, since a one-iteration loop carries no dependence. Chapter 46 shows that the cost model rewards it exactly as much as a 64-iteration one.

Note also what `applyRoles` does *not* do: it never checks a return value. `parallelize` and `vectorize` throw on refusal, so a role that is illegal aborts the whole `apply`, and the point is dropped by the session's `catch`.

### The four small sketches

[`sketch_generators.ts`](../../../src/compiler/autotune/sketch_generators.ts), 158 lines for six generators. Two constants set the menus:

```ts
const BLOCK_SIZE_CANDIDATES = [32, 64, 128, 256, 512, 1024];
const VECTOR_CANDIDATES = [1, 2, 4, 8, 16];
```

`createElementwiseCPUSketch` ([`sketch_generators.ts:49`](../../../src/compiler/autotune/sketch_generators.ts)) branches on the loop count: a 1-D nest is split and the outer half parallelised, an `n`-D nest gets `parallelize` on the outermost and a vector split on the innermost. `createReductionCPUSketch` ([`sketch_generators.ts:117`](../../../src/compiler/autotune/sketch_generators.ts)) has no variables at all — it parallelises the outermost loop and stops, which is a one-point space and the fallback for every reduction the priority-10 rule declines.

`createRfactorSketch` ([`sketch_generators.ts:33`](../../../src/compiler/autotune/sketch_generators.ts)) requires exactly one reduction axis with a constant extent, and its menu is the proper divisors of that extent:

```ts
  return new ScheduleSketch('rfactor', [new SearchVariable('rf_factor', factors)], (schedule, blockName, target, params) => {
    schedule.rfactor(blockName, reductionVar, (params.rf_factor as number));
    const loops = schedule.getLoops(`${blockName}_rf_p`);
    if (loops.length > 0) schedule.parallelize(loops[0]);
  });
```

This is the sketch Chapter 41 warned about: it offers reassociation of a floating-point reduction with no `fastMath` gate and no dtype test, on any block with a single divisible reduction axis. On a CPU matmul it is one of four sketches the search may draw from.

### `createFusedTilingSketch` and its gate

[`sketch_generators.ts:25`](../../../src/compiler/autotune/sketch_generators.ts) is reached only when `findFusibleConsumer` returns a name. That function ([`block_dag.ts:93`](../../../src/compiler/autotune/block_dag.ts)) is forty-one lines of structural matching, and the clause that decides everything is:

```ts
  const pSpatialNames = pInfo.loops.filter(l => !pInfo.reductionLoopVars.has(l.loopVar.name)).map(l => l.loopVar.name);
  const pWriteNames = indexVarNames(pStore.indices);
  if (!pWriteNames || pWriteNames.join(',') !== pSpatialNames.join(',')) return null;
```

`pSpatialNames` are *loop* variable names; `pWriteNames` are the names of the variables in the store's subscript, which in a block are its *iteration* variables. Chapter 33's block abstraction is precisely the separation of those two: the loop is `ls0_6`, the iteration variable is `vls0_9`, and the binding relates them. The comparison is therefore between two disjoint namespaces, and §45.5 shows it failing on every block the compiler lowers and succeeding on a hand-built pair that indexes with its loop variables directly.

## 45.5 Lab — deriving sketches

```bash
node docs/part8/ch45-sketches/labs/01-deriving-sketches.mjs
```

The classifier on four programs:

```
  program              block              S  R  reads  reduction   CPU sketches                              GPU sketches
  a matmul             matmul_init_0      2  0      0  false       elementwise_cpu                           elementwise_gpu
                       matmul_1           2  1      2  true        mlt_cpu ssrsrs_cpu rfactor reduction_cpu  mlt_gpu reduction_gpu
  matmul then relu     matmul_init_0      2  0      0  false       elementwise_cpu                           elementwise_gpu
                       matmul_1           2  1      2  true        mlt_cpu ssrsrs_cpu rfactor reduction_cpu  mlt_gpu reduction_gpu
                       maximum_block_2    2  0      2  false       elementwise_cpu                           elementwise_gpu
  two elementwise ops  mul_block_0        2  0      2  false       elementwise_cpu                           elementwise_gpu
                       add_block_1        2  0      2  false       elementwise_cpu                           elementwise_gpu
  a sum over one axis  reduce_init_0      1  0      1  false       elementwise_cpu                           elementwise_gpu
                       reduce_acc_1       1  1      1  true        reduction_cpu                             reduction_gpu
```

The decision list at work. `reduce_acc_1` is a reduction with one declared read, so it fails `reads >= 2` and falls to priority 20 — a parameterless sketch that parallelises the outer loop. `matmul_1` passes and gets four. Every non-reduction block, including a `relu` that reads two buffers and the zeroing block that declares none, falls to priority 30.

The holes, on a 16×16×16 matmul:

```
  s0  35 candidates:  [1,1,1,16] [1,1,2,8] [1,1,4,4] [1,1,8,2] [1,1,16,1] ...
  s1  35 candidates:  [1,1,1,16] [1,1,2,8] [1,1,4,4] [1,1,8,2] [1,1,16,1] ...
  r0   1 candidates:  [16]
```

`F(16, 4) = C(4+3, 3) = 35`, below the cap, so this is the whole space. And two points instantiated:

```
  {"s0":[1,1,1,16],"s1":[1,1,1,16],"r0":[16]}
   for ls0_6_o_0 in 0..1 @parallel {
   for rs0_7_o_6 in 0..1 {
   for ls0_6_i_1_o_2 in 0..1 {
   for rs0_7_i_7_o_8 in 0..1 {
   for ls0_6_i_1_i_3_o_4 in 0..1 {
   for rs0_7_i_7_i_9_o_10 in 0..1 {
   for ls0_6_i_1_i_3_i_5 in 0..16 {
   for rs0_7_i_7_i_9_i_11 in 0..16 @vectorized {
   for c0_8 in 0..16 {
   bind vls0_9 = ((ls0_6_o_0 * 16) + ((ls0_6_i_1_o_2 * 16) + ((ls0_6_i_1_i_3_o_4 * 16) + ls0_6_i_1_i_3_i_5)))

  {"s0":[2,2,2,2],"s1":[1,2,4,2],"r0":[16]}
   for ls0_6_o_12 in 0..2 @parallel {
   for rs0_7_o_18 in 0..1 {
   for ls0_6_i_13_o_14 in 0..2 {
   for rs0_7_i_19_o_20 in 0..2 {
   for ls0_6_i_13_i_15_o_16 in 0..2 {
   for rs0_7_i_19_i_21_o_22 in 0..4 {
   for ls0_6_i_13_i_15_i_17 in 0..2 {
   for rs0_7_i_19_i_21_i_23 in 0..2 @vectorized {
   for c0_8 in 0..16 {
   bind vls0_9 = ((ls0_6_o_12 * 8) + ((ls0_6_i_13_o_14 * 4) + ((ls0_6_i_13_i_15_o_16 * 2) + ls0_6_i_13_i_15_i_17)))
```

Proposition 45.5 and Corollary 45.6 side by side. Nine loops in both, the extents reading off the factor tuples in order, the interleaving `m n m n m n m n k` of `S0 S1 S2 S3 R0`, and the binding a mixed-radix sum with radices `(2,2,2)` in the second case — `8 = 2·2·2`, `4 = 2·2`, `2`, `1`. No `if` appears anywhere: every factor divides.

The first point is the degenerate one and it is instructive. Six of its nine loops have extent 1, the `@parallel` loop is one of them, and the schedule is a slightly obfuscated copy of the original nest. It is a legal point of the space and the search will draw it as readily as any other.

The two nests also differ in their loop *names* — `_0` and `_1` versus `_12` and `_13` — because the fresh-variable counter is global and the lab instantiated one point after the other without resetting it. That is cosmetic here. Chapter 48 is where it stops being cosmetic.

Then the two skeletons that cannot fire:

```
=== `ssrsrs_cpu`: the standard reduction tiling, and what stops it ===

  offered with 6125 points, r0 = [1,16] [2,8] [4,4] [8,2] [16,1]
  applied: decomposeReduction: block 'matmul_1' has no initBody
```

Every one of the 6,125 points fails on the first statement of `apply`. `createSSRSRSTilingSketch` opens with `schedule.decomposeReduction(blockName)` ([`tiling.ts:131`](../../../src/compiler/autotune/tiling.ts)) and Chapter 41 established that `decomposeReduction` requires an `initBody`, which no lowering rule sets. The sketch is constructed, counted, sampled from, and refused.

The cost is not the wasted evaluations. It is that `ssrsrs_cpu` is the *only* structure in the compiler with two reduction levels, so the block-`K` dimension of the classical three-level cache blocking is unreachable — and, as Chapter 47 shows, at the default seed it is also the only sketch the search ever samples.

```
=== `fused`: producer-consumer fusion, and what stops it ===

  block DAG of `relu(a @ b)`:
    maximum_block_2    declared reads [buf_7,buf_6]  writes [buf_5]
    matmul_1           declared reads [buf_1,buf_3]  writes [buf_7]
    matmul_init_0      declared reads []  writes [buf_7]

  matmul_1: enclosing spatial loop variables [ls0_8,rs0_9]
            store subscript variables        [vls0_11,vrs0_12]
            findFusibleConsumer -> null

  the same test on a hand-built pair whose store subscript IS its loop
  variable: findFusibleConsumer('prod') -> cons
```

Two namespaces, compared for equality. Every structural precondition `findFusibleConsumer` checks before that line is satisfied — one consumer, no reduction in it, a single write, no cycle — and the comparison at [`block_dag.ts:119`](../../../src/compiler/autotune/block_dag.ts) rejects it. The hand-built pair, whose blocks bind each iteration variable to itself, passes. So the code is right about the shape it was written for and that shape is not the one Chapter 33 defines.

And the refusals, from the other side:

```
=== what the primitives refuse, and what they do not ===

  fuseLoops(m, k)  — not a direct child          fuseLoops requires inner loop to be direct child of outer loop
  decomposeReduction on a lowered block          decomposeReduction: block 'matmul_1' has no initBody
  bindThread(m, "warpIdx.x")                     Invalid thread tag: warpIdx.x. Must be one of: blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x, threadIdx.y, threadIdx.z
  rfactor over the reduction axis c0_8           ACCEPTED
  rfactor over the SPATIAL axis ls0_6            ACCEPTED  (validator: 2 errors)
```

The first three are §45.1's point: a primitive states a precondition, tests it, and names it when it fails. The last row is the exception, and it is worth being exact about what goes wrong.

`rfactor(blockName, axisName, factor)` checks four things ([`schedule.ts:633`](../../../src/compiler/schedule/schedule.ts) to [`schedule.ts:654`](../../../src/compiler/schedule/schedule.ts)): the named loop is one of the block's, its extent is constant, `1 < factor < K` with `factor ∣ K`, and the block body is a single accumulating store over an operator in `RFACTOR_ASSOCIATIVE_OPS`. Every one of those holds when you name a matmul's `m` axis, because the *body* is an accumulating store no matter which loop you point at. What is never checked is the thing Definition 41.1 actually requires: that the axis being factored is the reduction axis. The block records exactly that — `vc0_11` is `CommReduce` and `vls0_9` is `DataPar` (Chapter 33) — and `rfactor` does not look.

The result is not a slower program; it is not a program:

```
  running the spatially-rfactored kernel: ReferenceError: ls0_6_rfo_11 is not defined
```

`rfactor` builds its combine nest to write at the store subscript the original block had — `buf_5[vls0_9, vrs0_10]` — over loops for the axes it did *not* factor. When the factored axis is `c0_8` that is exactly right, because Definition 41.1 requires the accumulator's subscript `s` not to involve the reduction axes, so every variable in the subscript still has a loop. When the factored axis is `ls0_6`, `vls0_9` is *in* the subscript and the combine nest has no loop for it; `_iterVarsOver` keeps only the iteration variables of the surviving axes, so `vls0_9` is left carrying the `k_o·f + k_i` rebinding the partial nest introduced, and the emitted combine loop refers to `ls0_6_rfo_1` and `ls0_6_rfi_0` two nesting levels away and out of scope.

Definition 41.1's clause "`s` does not involve the reduction axes" is therefore load-bearing in the same way Chapter 33's `DataPar` declaration is: the construction is correct exactly when it holds, and nothing tests it.

Two things stop this mattering. `createRfactorSketch` draws its axis from `blockInfo.reductionLoopVars` ([`sketch_generators.ts:34`](../../../src/compiler/autotune/sketch_generators.ts)), so no sketch ever asks; and the searched path runs `ScheduleValidator`, which reports the malformed nest and makes the candidate score `null` ([`session.ts:186`](../../../src/compiler/autotune/session.ts)). That second protection is worth noticing, because it is the validator doing exactly what Chapter 42 said it does for searched schedules and does not do for rule-produced ones — and here it is the only thing between a legal-looking primitive call and a kernel that throws.

Finally the escape hatch:

```
=== the GPU escape hatch: a sketch that replaces the body ===

  analyzePureMatmul: block matmul_1, M=128 N=128 K=128
  sketches: matmul_register_block_gpu[32]
  and it carries an enumerate(): true, 32 configurations
  after applying config 0: 0 trace steps recorded
  the body it produced:
     allocate rb_As[1024] (shared) {
       allocate rb_Bs[1024] (shared) {
         for rb_by in 0..1 @thread_binding [blockIdx.y] {
           for rb_bx in 0..1 @thread_binding [blockIdx.x] {
             for rb_ty in 0..16 @thread_binding [threadIdx.y] {
               for rb_tx in 0..16 @thread_binding [threadIdx.x] {
```

Counterexample 45.8's second half. `richMatmulSketches` recognises a pure matrix multiply, enumerates register-blocking configurations against the target's shared memory and register budgets ([`gpu_matmul_sketch.ts:187`](../../../src/compiler/autotune/gpu_matmul_sketch.ts)), and returns a single sketch whose `apply` throws the schedule away. It also carries an `enumerate()` method that no other sketch has, which `BlockTuningSession` detects and uses to replace the search with exhaustive evaluation ([`session.ts:111`](../../../src/compiler/autotune/session.ts)) — 32 configurations is small enough to try all of them.

This is the compiler's fastest GPU matmul and it is not a schedule. Chapter 43 reached the same conclusion from the code-generation side; the version visible here is that Theorem 45.7 does not apply to it, and Chapter 48's version is that it records nothing.

## 45.6 Lab — how many distinct programs

```bash
node docs/part8/ch45-sketches/labs/02-how-many-distinct-programs.mjs
```

A sketch's advertised size is `∏|C_i|`. That is an upper bound on the number of programs, and the gap between bound and count is a direct measure of wasted search budget. Instantiating every point and comparing the printed IR:

```
=== a 16x16x16 matmul, CPU ===

  mlt_cpu                        nominal  1225   distinct  1225   refused     0   invalid   0
  ssrsrs_cpu                     nominal  6125   distinct     0   refused  6125   invalid   0
  rfactor                        nominal     3   distinct     3   refused     0   invalid   0
  reduction_cpu                  nominal     1   distinct     1   refused     0   invalid   0
  elementwise_cpu (init block)   nominal     5   distinct     5   refused     0   invalid   0
    the 5 widths produce 5 programs: 1 2 4 8 16
  elementwise_cpu (4x4 mul)      nominal     5   distinct     4   refused     0   invalid   0
    the 5 widths produce 4 programs: 1 2 4 8
```

`mlt_cpu` is injective on its whole space, which Proposition 45.5 predicts: distinct factor tuples give distinct extent sequences. Not one of its 1,225 points is refused and not one fails `ScheduleValidator` — a strong statement about the skeleton, since it means every parameter the search can draw yields a program the session will accept.

`elementwise_cpu` is injective on a 16-wide innermost loop and loses two points on a 4-wide one, where widths 8 and 16 both fail `extent >= vector_width` ([`sketch_generators.ts:71`](../../../src/compiler/autotune/sketch_generators.ts)) and produce the same bare `parallelize`.

On GPU the collapse is larger and structural:

```
=== the same matmul, WebGPU ===

  mlt_gpu                        nominal   225   distinct   225   refused     0   invalid   0
  reduction_gpu                  nominal     6   distinct     4   refused     0   invalid   0
  elementwise_gpu (4096 elts)    nominal     6   distinct     4   refused     0   invalid   0
    candidates 32 64 128 256 512 1024  ->  distinct block sizes 32 64 128 256
    target.maxThreadsPerBlock = 256, and gpuThreadCap clamps to min(that, 256)
```

`gpuThreadCap` ([`sketch_generators.ts:10`](../../../src/compiler/autotune/sketch_generators.ts)) is `Math.min((target && target.maxThreadsPerBlock) || 256, 256)` — the same hard 256 ceiling Chapter 43's finding 31 found in `bindFusedSpatialGPU`, arriving in the sketch generator. Three of the six advertised thread-block sizes are the same kernel, and on a device advertising 1024 threads per block the two largest are unreachable.

And a case where every point collapses:

```
=== an alias is not free ===

  elementwise_gpu on a 16x16 init block: 6 points, 1 distinct program.
```

The init block of a 16×16 matmul has 256 elements, and the function it belongs to contains a reduction, which triggers the shortcut at [`sketch_generators.ts:102`](../../../src/compiler/autotune/sketch_generators.ts): one thread block, whatever was asked for. Six parameter values, one kernel, and nothing notices — `EvolutionarySearch` memoises on `sketch.name + JSON.stringify(params)` ([`search.ts:117`](../../../src/compiler/autotune/search.ts)), which is parameter identity, not program identity. With a benchmark runner attached and `topKForBenchmark` at its default of 5, the same kernel is compiled and timed five times.

## 45.7 Traps and limits

- **`rfactor` does not check that the axis it is factoring is a reduction axis.** [`schedule.ts:633`](../../../src/compiler/schedule/schedule.ts) onwards tests the loop, the extent, the factor and the body's operator, and never the axis's `IterVarKind`, so `rfactor('matmul_1', 'ls0_6', 2)` is accepted and produces a nest whose combine loop references two out-of-scope variables. Definition 41.1's requirement that the accumulator subscript not involve the factored axis is the missing hypothesis. Two things keep it latent: `createRfactorSketch` only ever names an axis from `blockInfo.reductionLoopVars` ([`sketch_generators.ts:34`](../../../src/compiler/autotune/sketch_generators.ts)), and `ScheduleValidator` rejects the result on the searched path ([`session.ts:186`](../../../src/compiler/autotune/session.ts)) — which is the validator earning its place, since a rule-produced schedule would not be checked.
- **`ssrsrs_cpu` is derived, counted, sampled and always refused.** [`tiling.ts:131`](../../../src/compiler/autotune/tiling.ts) calls `decomposeReduction`, which needs an `initBody` no lowering rule sets (Chapter 33, finding 12). It advertises 6,125 of a matmul block's 7,354 points on a 16×16×16 problem — 83% of the block's space is unreachable, and the reachable part contains no schedule that tiles the reduction axis.
- **The `fused` sketch is never derived.** `findFusibleConsumer` compares store-subscript variable names against enclosing loop variable names ([`block_dag.ts:119`](../../../src/compiler/autotune/block_dag.ts), and again at `:124` and `:130` for the consumer). Those namespaces are disjoint for every block a lowering rule emits, so the comparison always fails, `createFusedTilingSketch` has no reachable caller, and `BlockTuningSession`'s `needsWholeFunc` branch ([`session.ts:96`](../../../src/compiler/autotune/session.ts)) — which exists to give the fused sketch a whole-function evaluation context — is dead with it.
- **The autotuner derives nothing for a WASM target.** `deriveSketches` returns `[]` unless the target is `TargetKind.CPU` or `isGPU()` ([`derivation.ts:78`](../../../src/compiler/autotune/derivation.ts)), and `WasmTarget` is neither. Every block becomes an `empty` task, `tune` returns no results, and `tuneAndApply` falls back to the rule policy — which is the correct output, silently obtained without tuning. WASM is the one shipped backend that acts on both `@parallel` and `@vectorized`, so it is also the target where a schedule search would have most to gain.
- **`gpuThreadCap` caps the menu at 256.** [`sketch_generators.ts:10`](../../../src/compiler/autotune/sketch_generators.ts). Two of the six `BLOCK_SIZE_CANDIDATES` are unreachable on every device and three are aliases on a 256-thread device. Same constant, same consequence as Chapter 43's finding 31, in a different file.
- **A degenerate factorisation is a first-class point.** `[1,1,1,n]` parallelises a one-iteration loop and leaves the nest as it was. The sketch cannot avoid offering degenerate tuples: for a prime extent *every* tuple is one, since `F(p, 4) = 4` is exactly the four placements of `p` among four slots with 1s elsewhere. Nothing down-weights them, so a search that samples uniformly spends a constant fraction of its budget on schedules that do nothing.
- **`applyRoles` picks the axis by position, not by extent.** `parallelize` always goes to `axes[0]` and `vectorize` always to `axes[axes.length − 1]` ([`tiling.ts:39`](../../../src/compiler/autotune/tiling.ts), [`tiling.ts:43`](../../../src/compiler/autotune/tiling.ts)). For a nest whose first spatial axis is short and second is long, the parallel loop is the short one and the search has no parameter that can swap them.
- **A sketch is a closure, so nothing can inspect it.** `ScheduleSketch` holds `_apply` as a function ([`sketch.ts:27`](../../../src/compiler/autotune/sketch.ts)); there is no declarative form. The only way to learn what a point does is to run it, which is why `ssrsrs_cpu` can be counted into the space by code that has no way to discover that it throws.

## 45.8 Read the tests

- [`tests/compiler/autotune/autotuner.test.js`](../../../tests/compiler/autotune/autotuner.test.js) — `derives multi-level-tiling + reduction sketches …` pins the exact derivation table of §45.5, and `selection is structural, not by block name` pins the property that makes Definition 45.2 honest, using an elementwise block called `matmul_in_name_only`. Two further tests sample twelve points of `mlt_cpu` and assert the validator accepts each and that the reduction loop stays innermost — the executable half of Proposition 45.5.
- [`tests/compiler/autotune/ansor.test.js`](../../../tests/compiler/autotune/ansor.test.js) — the factorisation properties Corollary 45.6 depends on, including the degenerate tuple for prime extents.
- [`tests/compiler/autotune/rfactor.test.js`](../../../tests/compiler/autotune/rfactor.test.js) — the `rfactor` sketch's primitive, compiled and compared against a scalar reference over four shapes and three factors. It is where Theorem 45.7's exception is checked in the cases where it holds exactly.

Two of the four sketch generators a CPU matmul receives have no test that applies them: `ssrsrs_cpu` because it cannot be applied, and `fused` because it is never derived. The tests above assert what the derivation *returns*, which is why neither absence shows up as a failure.

---

**Next:** [Chapter 46 — Cost models](../ch46-cost-models/README.md), which takes the space this chapter generates and asks how to order it without running it — and finds that on the largest sketch, the shipped model orders it not at all.
