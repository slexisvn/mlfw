# Chapter 38 — Separating what from how

Chapter 34 lowered `mul` into this:

```
for i0_5 in 0..4096 {
  block mul_block_0 {
    bind v0_6 = i0_5
    reads([buf_1[...], buf_4[...]])
    writes([buf_3[...]])
    buf_3[v0_6] = (buf_1[v0_6] * buf_4[])
  }
}
```

One loop, 4096 iterations, in order, on one core. That is a correct program and, on every device in this book, a bad one. A CPU with eight cores and eight-wide vectors wants 512 iterations of a vectorised body across eight workers; a GPU wants 4096 threads; a single-core WebAssembly runtime wants 1024 iterations of a four-wide body and no threads at all.

Four machines, four answers, and *the same computation in all four*. This chapter is about the design decision that follows from that observation, and Part VII is what the decision buys.

## 38.1 The problem: the loop nest is one answer to two questions

The nest above answers two questions at once.

**What is computed?** For every `v0_6` in `[0, 4096)`, `buf_3[v0_6]` is `buf_1[v0_6] * buf_4[]`.

**How is it computed?** In one loop, ascending, serially.

The lowering rule of Chapter 34 had to answer both, because a `ForNode` is the only way to say the first. But only the first is a property of the program; the second is a property of the machine, and the compiler will not know the machine's answer until it knows which of the four targets it is compiling for — and, in Part VIII, not until it has *measured* several candidates.

If the two answers stay welded together, every transformation has to re-derive the first from a nest that has been reshaped for the second. Split the loop and the `4096` is gone; tile it and there are four loops; bind it to threads and there is no loop at all in the emitted source. Something has to survive.

## 38.2 Intuition: the algorithm and the plan

The idea is Halide's, sharpened by TVM, and it is one sentence: **write the algorithm once, and write the plan for executing it separately.**

An analogy that holds up further than most. A recipe says what a dish is: these ingredients, combined this way, to this end state. A *kitchen plan* says who does what, in what order, on which burner, and how much can happen at once. Two cooks with the same recipe and different kitchens produce the same dish. A plan cannot make the dish something else; the worst a bad plan can do is make it slow.

In the loop nest, the algorithm is the **block** of Chapter 33 — the body, the read and write sets, the iteration variables and their kinds. The plan is the **loops** — how many, in what order, at what extents, with what annotations. Part VII is the language for changing the second while the first is held fixed, and Chapter 42 is where "held fixed" is made precise, because it is not quite as strong as this paragraph implies.

## 38.3 Theory

> **Definition 38.1 (Schedule).** Let `P` be a `PrimFunc`. A *schedule* on `P` is a finite sequence of *primitives* `p₁,…,p_n`, each a partial function on `PrimFunc`s. The scheduled program is `p_n(⋯p₁(P)⋯)`, undefined if any primitive is applied outside its domain.

"Partial" is the operative word, and it is how legality is expressed in this compiler: a primitive that would be illegal is not applied and returns an error, rather than being applied and repaired afterwards. Chapter 42 is entirely about the domains.

> **Definition 38.2 (Semantic equivalence for a `PrimFunc`, stated here).** Two `PrimFunc`s over the same buffer signature are *equivalent* if, for every initial contents of their input buffers, they leave identical contents in every buffer that outlives the function.

Note what this definition deliberately does not say. It does not quantify over intermediate states, so a schedule may materialise scratch buffers, and it does not quantify over the *order* of the writes, so two schedules that write the same locations in different orders are equivalent. It also says *identical*, not *close*: under floating-point arithmetic this makes reassociation inequivalent, which is why Chapter 41's `rfactor` needs a licence that Chapter 40's `split` does not.

> **Definition 38.3 (Sound primitive, stated here).** A primitive `p` is *sound* if `p(P)` is equivalent to `P` for every `P` in its domain.

> **Proposition 38.4 (Soundness composes, stated here).** If every primitive in a schedule is sound, the scheduled program is equivalent to the original.

*Proof.* Equivalence is transitive, and each step preserves it. ∎

Proposition 38.4 is trivial and it is the reason the design works: it makes "is this schedule correct?" — a question about an unbounded search space, which Part VIII will explore by machine — into "is this primitive sound?", twenty-two questions asked once. A search that only ever composes sound primitives cannot produce a wrong program, however badly it searches.

The whole force of that sentence is in its hypothesis, and the hypothesis is a per-primitive obligation rather than a given. Twenty-one of the twenty-two discharge it. One does not, and cannot:

**`rfactor` on a float reduction reassociates the accumulation**, which is an N2 transformation. Reassociating `+` changes the bits and Definition 38.2 says *identical*, so it is not sound in the sense of Definition 38.3 — and a version of `rfactor` that did not reassociate would not be `rfactor`. The primitive is sound at N2 and not at N1; the honest answer is to make the *level* explicit and let the caller choose, not to change the transform. Chapter 41 measures a case where it turns 3 into 6, and Part VIII's sketch generator offers it without asking.

The other twenty-one are worth a sentence each in Chapters 40 and 41, because their obligations are easy to state and easy to miss: `split` has to carry a loop's lower bound and refuse a thread-bound loop; `rfactor` has to derive its identity from the operator *and* the dtype and validate that the store really is an accumulation; `reorder` inherits whatever the dependence analysis gives it, which is why Chapter 36 §36.7 spends a section on normalising direction vectors.

So state the guarantee in the conditional form it actually has:

> **Proposition 38.4 buys "a search cannot produce a wrong program" only for the primitives that are sound.** It is a theorem about composition, and composition of a false premise proves nothing. A search that reaches `rfactor` on a float reduction can change a result at N1 without reporting anything. Part VIII §44 onward assumes this proposition throughout, and inherits that one exception.

The design is what makes that statement possible at all. Reducing "is this schedule correct?" to twenty-two questions asked once is what makes the exceptions *enumerable* — one of them, named, with a measured counterexample — which is not something a compiler without this structure could say about itself.

There is a third question hiding behind the second, and it is the one this chapter's second lab is about.

> **Definition 38.5 (Advisory annotation, stated here).** A loop *annotation* — `parallel`, `vectorized`, `unrolled`, `thread_binding` — is *advisory* on a backend if that backend emits the same code with and without it.

An advisory annotation is sound for free: a backend that ignores it cannot be broken by it. That is a real property and not a good one, because the primitive that sets it is still legality-checked, still recorded in the trace, still costed by Part VIII's model, and still buys nothing.

## 38.4 In mlfw

### Where scheduling happens

`SchedulePass` ([`passes/schedule/schedule_pass.ts:11`](../../../src/compiler/passes/schedule/schedule_pass.ts)) is a `PrimFuncPass` — it runs once per function, after lowering, before simplification and memory planning ([`tir_pipeline.ts:24`](../../../src/compiler/pipeline/tir_pipeline.ts)). It has three modes, chosen by configuration and nothing else:

```ts
    if (sCfg.autotune) {
      …
    } else if (sCfg.enabled || sCfg.gpuTiling) {
      if (pf.hasAttr(FuncAttr.EXTERNAL_CODEGEN) || pf.hasAttr(FuncAttr.TENSOR_INTRIN)) return;
      const ft0 = performance.now();
      const sch = new Schedule(pf);
      const handled = applyDeterministicGpuSchedule(sch, this.target as never, sCfg as never);
      if (!handled && sCfg.enabled) {
        (this._policy as SchedulePolicy).applyToAllBlocks(sch);
      }
```

Three modes, in increasing order of cost: a hand-written GPU template (`applyDeterministicGpuSchedule`, Chapter 43), a rule per block (`SchedulePolicy`, below), and a search (`Autotuner`, Part VIII). A function that already has an external kernel or a tensor intrinsic is skipped entirely — someone else has decided how it runs.

The defaults are worth reading against that code. `CompilerConfig` starts from `{ enabled: false, autotune: false, gpuTiling: false }` ([`compiler.ts:143`](../../../src/compiler/pipeline/compiler.ts)) and overlays the target's declaration. Only two targets declare anything: CUDA says `{ gpuTiling: true }` ([`target.ts:225`](../../../src/backend/target.ts)) and WebGPU says `{ enabled: true }` ([`target.ts:261`](../../../src/backend/target.ts)). **So scheduling is off by default on CPU and on WASM, and on CUDA only the template path is on.** §38.6 shows what that costs.

### The primitives

`Schedule` ([`schedule/schedule.ts:202`](../../../src/compiler/schedule/schedule.ts)) is 1,137 lines and holds four things — the function, a `ScheduleState` (Chapter 39), a `ScheduleTrace` (Chapter 48) and a `ScheduleMutator`, the 44-line helper that does the actual splicing — behind twenty-eight public members. Six are queries (`getBlock`, `getBlockSRef`, `getLoops`, `getTrace`, `verify`, and the constructor); the other twenty-two are the primitives:

| Group | Primitives | Chapter |
|---|---|---|
| Loop shape | `split`, `fuseLoops`, `reorder`, `tile` | 40 |
| Loop annotation | `vectorize`, `unroll`, `parallelize`, `annotate` | 40 |
| Thread mapping | `bindThread` | 43 |
| Reduction | `rfactor`, `decomposeReduction` | 41 |
| Memory | `cacheRead`, `cacheWrite`, `setScope`, `storageAlign` | 41 |
| Locality | `computeInline`, `computeInlineBlock`, `computeAt`, `reverseComputeAt`, `fuseConsumer` | 41 |
| Structure | `blockize`, `tensorize` | 43 |

Only thirteen of the twenty-two have a caller anywhere in `src/`, and only seven are reachable from the default rule set. §38.7 lists the other nine.

Every primitive has the same four-part shape, and `parallelize` ([`schedule.ts:599`](../../../src/compiler/schedule/schedule.ts)) is the shortest example of it:

```ts
  parallelize(loop: LoopRef): void {
    loop = this._resolveLoop(loop);
    if (loop.type !== 'ForNode') throw new Error('parallelize expects ForNode');
    const carried = loopCarriedDependence(this.state, loop, IterVarPolicy.SPATIAL);
    if (carried !== null) throw new Error(`Cannot parallelize: ${carried}`);
    loop.kind = ForKind.PARALLEL;
    this.state.invalidate();
    if (!this._replaying) {
      this.trace.record('parallelize', [loop.loopVar.name]);
    }
  }
```

Resolve the argument; **check legality and throw before mutating anything**; mutate; invalidate the cached analyses and record the step. The order of the middle four lines is the contract: a rejected primitive leaves the IR untouched, so a caller may try one and fall back to another. The contract binds each primitive on its own and does not extend to one built out of others — `tile` is two `split`s followed by a `reorder`, and a `reorder` that throws leaves the splits standing. Chapter 40's traps have the case; what to carry from here is that atomicity is a property of the individual primitive, not of the schedule step a caller wrote. Chapter 42 is about the `loopCarriedDependence` call and Chapter 48 about the `trace.record`.

### The rules

`SchedulePolicy` ([`rules.ts:525`](../../../src/compiler/schedule/rules.ts)) is a list of nine `ScheduleRule`s tried in order, first match wins ([`rules.ts:536`](../../../src/compiler/schedule/rules.ts)):

```ts
  static defaultRules(): ScheduleRule[] {
    return [
      new MatmulTiledCPURule(),
      new MatmulTiledGPURule(),
      new ReductionCPURule(),
      new ReductionGPURule(),
      new ReductionWasmRule(),
      new ElementwiseCPURule(),
      new ElementwiseGPURule(),
      new ElementwiseWasmRule(),
      new FallbackRule()
    ];
  }
```

A rule is a pair of methods, `matches(primFunc, blockName, target)` and `apply(schedule, blockName, target)`. Both consult `classifyBlock` ([`rules.ts:42`](../../../src/compiler/schedule/rules.ts)), which caches per function a five-field summary: how many loops enclose the block, whether it reduces, which loop variables are the reduction axes, and the read and write buffer names. That is the whole of what a rule sees; a rule never looks at the block body.

The ordering is a priority scheme and reads as one: the most specific shape (a matmul) before a general one (a reduction) before the most general (elementwise), each group split by target, with `FallbackRule` matching unconditionally at the end. Every rule name carries its target and `matches` opens by checking it, so the nine are eight shape-and-target pairs plus a default — and the grid has a hole in it. Reduction and elementwise each have a CPU, a GPU and a WASM rule; matmul has only CPU and GPU. A WASM matmul falls past `MatmulTiledCPURule` and `MatmulTiledGPURule` on their target checks and is picked up by `ReductionWasmRule`, which parallelises the outer spatial loop and, if SIMD is available, vectorises the contraction axis. That is not a bad schedule for a GEMM; it is just not a tiled one.

`applyToBlock` wraps `apply` in a `try` ([`rules.ts:559`](../../../src/compiler/schedule/rules.ts)):

```ts
    const rule = this.selectRule(schedule.func, blockName);
    if (rule) {
      try {
        rule.apply(schedule, blockName, this.target);
      } catch (e) {
        invalidateClassifyBlock(schedule.func, blockName);
        this._explain(blockName, 'none', `rule '${rule.name}' rejected: ${(e as Error).message}`);
        return null;
      }
```

A rule that trips a legality check is not a compilation failure — the block simply runs unscheduled, and the reason reaches the explain stream of Chapter 18 rather than the error list. This is what makes Proposition 38.4 usable in practice: rules may be optimistic, because the primitives are the ones that have to be right.

## 38.5 Lab — one program, four schedules

```bash
node docs/part7/ch38-separating-what-from-how/labs/01-one-program-four-schedules.mjs
```

`x.mul(2.0)` over 4096 elements, scheduled for four machines:

```
=== CPU — 8 cores, vector width 8 ===
  for i0_5_o_0 in 0..512 @parallel {
    for i0_5_i_1 in 0..8 @vectorized {
      block mul_block_0 {
        bind v0_6 = ((i0_5_o_0 * 8) + i0_5_i_1)

=== WASM — 1 core, SIMD width 4 ===
  for i0_5_o_0 in 0..1024 {
    for i0_5_i_1 in 0..4 @vectorized {
      block mul_block_0 {
        bind v0_6 = ((i0_5_o_0 * 4) + i0_5_i_1)

=== WASM — 4 cores, SIMD width 4 ===
  for i0_5_o_0 in 0..1024 @parallel {
    for i0_5_i_1 in 0..4 @vectorized {
      block mul_block_0 {
        bind v0_6 = ((i0_5_o_0 * 4) + i0_5_i_1)

=== CUDA — 1024 threads per block ===
  for i0_5_o_0 in 0..16 @thread_binding [blockIdx.x] {
    for i0_5_i_1 in 0..256 @thread_binding [threadIdx.x] {
      block mul_block_0 {
        bind v0_6 = ((i0_5_o_0 * 256) + i0_5_i_1)
```

Four plans. The block is byte-identical in all four but for the binding, and the binding is the joint: it is the one place the block admits that the loops have changed shape. Read the four bindings together and the pattern is `outer × factor + inner` every time, with a different factor and a different meaning attached to each half.

The two WASM lines are the sharper comparison, because the target differs in one field. `numCores: 1` and the rule declines to parallelise; `numCores: 4` and the same rule, same block, same everything else, marks the outer loop `@parallel`. The decision is `outerSize >= numCores * 4` ([`rules.ts:412`](../../../src/compiler/schedule/rules.ts)) — an arithmetic threshold, not a legality question.

**Try this.** Pass `WasmTarget({ numCores: 4, simd: false })` and the split disappears entirely — the whole 4096-iteration loop is marked `@parallel` and nothing is vectorised. `ElementwiseWasmRule` guards its one-dimensional split on `target.supportsSimd()` ([`rules.ts:413`](../../../src/compiler/schedule/rules.ts)), so with SIMD off the rule falls through to a bare `parallelize(loops[0])`.

## 38.6 Lab — what an annotation is worth

```bash
node docs/part7/ch38-separating-what-from-how/labs/02-what-the-annotation-is-worth.mjs
```

The same schedule, read at the far end of the pipeline. On CPU, with `scheduling` at its default and then on:

```
=== CPU, scheduling off (the shipped default) ===
function Object(buf_1, buf_3) {
  for (let i0_5 = 0; i0_5 < 100; i0_5++) {
    buf_3[i0_5] = (buf_1[i0_5] * 2);
  }
}

=== CPU, scheduling on ===
function Object(buf_1, buf_3) {
  for (let i0_5_o_0 = 0; i0_5_o_0 < 13; i0_5_o_0++) {
    for (let i0_5_i_1 = 0; i0_5_i_1 < 8; i0_5_i_1++) {
      if ((((i0_5_o_0 * 8) + i0_5_i_1) < 100)) {
        buf_3[((i0_5_o_0 * 8) + i0_5_i_1)] = (buf_1[((i0_5_o_0 * 8) + i0_5_i_1)] * 2);
      }
    }
  }
}
```

Neither annotation survives. `@parallel` and `@vectorized` are both advisory on the CPU backend in the sense of Definition 38.5: `backend/cpu/codegen.ts` reads exactly one loop kind, `ForKind.UNROLLED` ([`codegen.ts:231`](../../../src/backend/cpu/codegen.ts)), and JavaScript has neither threads nor an intrinsic vector type to spend the other two on. What the schedule did contribute is the split — a second loop, a guard, and the address arithmetic to reconstruct the index. The scheduled kernel does strictly more work than the unscheduled one, which is exactly why `enabled` defaults to `false` here.

WASM is the other case, on the same two annotations:

```
=== WASM, 4 cores, scheduling on ===
  76 lines of WAT, of which
  5 mention the worker pool's slice of the parallel loop:
      (local $i0_5_o_0 i32) … (local $_par_start i32) (local $_par_end i32) …
      local.set $_par_start
      local.set $_par_end
      (local.get $_par_start)
      (local.get $_par_end)
  13 are SIMD, e.g.
      f32x4.splat
      f32x4.mul
      f32x4.splat
      f32x4.replace_lane 1
```

`_par_start` and `_par_end` are the parameters the worker pool passes in ([`backend/wasm/codegen.ts:151`](../../../src/backend/wasm/codegen.ts)), so `@parallel` has become a partition of the iteration space across threads, and `@vectorized` has become thirteen SIMD lines — ten `f32x4` opcodes and three mentioning the `v128` locals they work through. Same schedule, real effect.

Then CUDA, and the finding this lab exists for:

```
=== CUDA, DEFAULT scheduling config ===
__global__ void Object(float* buf_1, float* buf_3) {
  float buf_4[1];
  buf_4[0] = 2.0f;
  for (int i0_5 = 0; i0_5 < 4096; i0_5++) {
    const int v0_6 = i0_5;
    buf_3[v0_6] = (buf_1[v0_6] * buf_4[0]);
  }
}

=== CUDA, scheduling.enabled = true ===
__global__ void Object(float* buf_1, float* buf_3) {
  const int i0_5_o_0 = blockIdx.x;
  const int i0_5_i_1 = threadIdx.x;
  …
  buf_3[v0_6] = (buf_1[v0_6] * buf_4[0]);
}
```

A `__global__` function containing a serial loop over 4096 elements is a GPU kernel that uses one thread. The chain that produces it is three links long and each is individually reasonable: CUDA's target attributes declare `{ gpuTiling: true }` and not `enabled`; `gpuTiling` alone reaches `applyDeterministicGpuSchedule`, which recognises a matmul or a convolution and returns `false` for everything else; and `SchedulePass` then reads `if (!handled && sCfg.enabled)` and skips the policy. WebGPU declares `{ enabled: true }` and does not have the problem. The difference between the two GPU targets is one key in one attribute table.

## 38.7 Traps and limits

- **Nine of the twenty-two primitives have no caller in `src/`.** `cacheRead`, `cacheWrite`, `setScope`, `storageAlign`, `computeInline`, `computeAt`, `reverseComputeAt`, `annotate` and `blockize` are reachable only from tests. They are implemented, and several are tested for numerical equality against a baseline, and no compilation runs them. Chapter 41 covers the memory group and says so for each.
- **Seven primitives are reachable from the default rule set:** `split`, `fuseLoops`, `tile`, `vectorize`, `parallelize` and `bindThread` are called by a rule directly, and `reorder` only from inside `tile`. `unroll`, `rfactor`, `decomposeReduction` and `fuseConsumer` need the autotuner (Part VIII); `computeInlineBlock` runs in `InlineReindexPass` on GPU targets only; `tensorize` needs `optimization.tensorize`.
- **`tensorize` records nothing in the trace.** Every other mutating primitive calls `this.trace.record(...)`; `tensorize` ([`schedule.ts:1093`](../../../src/compiler/schedule/schedule.ts)) sets a function attribute and returns. A trace replayed onto a fresh `PrimFunc` therefore reproduces every step except that one. `tile` also records nothing, which is correct — it is defined as a sequence of `split`s and a `reorder`, and those record themselves.
- **A rule sees a five-field summary, never the body.** `classifyBlock` is what `matches` gets, so `isMatmulShape` ([`rules.ts:99`](../../../src/compiler/schedule/rules.ts)) is "has a reduction, reads two buffers, writes one, at least three loops" — a description that a batched attention score, a bilinear form and a genuine GEMM all satisfy.
- **The classification cache is keyed on the `PrimFunc` and cleared by hand.** `_classifyCacheByFunc` is a `WeakMap` ([`rules.ts:40`](../../../src/compiler/schedule/rules.ts)) invalidated by explicit calls to `invalidateClassifyBlock` after each `apply` and `invalidateClassifyCache` at the start of `applyToAllBlocks`. This is the manual version of Chapter 16's invalidation problem, in a subsystem that does not use the analysis manager.
- **`FallbackRule` parallelises the outermost loop of anything, on CPU only.** It matches unconditionally and its `apply` is five lines ([`rules.ts:507`](../../../src/compiler/schedule/rules.ts)), of which the operative one runs for `target.isCPU()`. On WASM a block that no rule matched gets nothing at all — including the parallel annotation that WASM, unlike the CPU, would have spent.

## 38.8 Read the tests

- [`tests/compiler/schedule/rules.test.js`](../../../tests/compiler/schedule/rules.test.js) — which rule matches which block shape on which target, and the fallback.
- [`tests/compiler/schedule/primitives.test.js`](../../../tests/compiler/schedule/primitives.test.js) — the primitives one at a time, each against the IR it should produce.
- [`tests/compiler/schedule/trace.test.js`](../../../tests/compiler/schedule/trace.test.js) — that a recorded schedule replays to the same program, which is Proposition 38.4 made executable.

---

**Next:** [Chapter 39 — The sref tree and block scopes](../ch39-sref-tree-and-block-scopes/README.md), which is the data structure that makes "change the loops, keep the block" an O(subtree) edit rather than a rebuild.
