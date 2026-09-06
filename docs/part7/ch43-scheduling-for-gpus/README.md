# Chapter 43 — Scheduling for GPUs

Everything in Chapters 40 to 42 assumed a loop is a loop: something that runs its iterations in an order, which a schedule may change. On a GPU the outermost loops do not run at all. They are *replaced* by the index space the hardware hands out, and the body executes once per point of it, simultaneously, with no ordering between points and no way to impose one except a barrier.

That changes three things: what an annotation means, what a legality question is, and — the part this chapter ends on — whether the scheduling language is the thing that writes the fast kernel.

## 43.1 The problem: three memories and no order

A CUDA kernel launch has a grid of blocks, each with a block of threads. Three facts follow, and every GPU scheduling decision is downstream of one of them.

1. **There is no order between blocks.** Two blocks may run concurrently, sequentially, or on different SMs, and there is no barrier that spans them. A value written by one block and read by another is a race, full stop; the only fix is two kernels.
2. **Threads within a block can be ordered, once, cheaply.** `__syncthreads()` is a barrier over one block. A value written by one thread and read by another needs one, and needs the value to live somewhere both can see.
3. **There are three memories, and they differ by two orders of magnitude.** Registers per thread, shared memory per block, global memory per grid. Chapter 4's arithmetic-intensity argument says the kernel is fast exactly when the innermost loop reads registers and the middle loop reads shared memory.

Chapter 40's primitives can express (1) and (2) — `bindThread` chooses which loop becomes which axis — and Chapter 41's can express (3), because `cacheRead(block, buf, 'shared')` is exactly "stage this in shared memory". Whether that is *enough* to write a fast matmul is the question §43.7 answers, and the answer is no.

## 43.2 Intuition: the loops you bind stop existing

Take the elementwise nest one last time.

```
for i in 0..4096 { body(i) }
```

Split by 256 and bind: the outer loop becomes `blockIdx.x`, the inner becomes `threadIdx.x`, and the kernel source contains neither loop — just `const int i_o = blockIdx.x;` and `const int i_i = threadIdx.x;` at the top and one copy of the body. The 4096 iterations still happen; the *iteration space* survives; the loops do not.

Which is why binding is the one annotation that cannot be advisory. A backend that ignored `@parallel` runs a correct program slowly. A backend that ignored `@thread_binding` would run the body once instead of 4096 times.

And it is why the legality question changes shape. "Does this loop carry a dependence?" becomes "can two points of this index space, running at once, disagree about a memory location?" — the same question, except the answer now depends on *which* axis, because thread-space and block-space have different repair options: a shared buffer and a barrier for the first, nothing at all for the second.

## 43.3 Theory

> **Definition 43.1 (Thread binding).** **(stated here)** Binding loop `L` with extent `n` to axis `t` replaces `L` by the assertion that the enclosing kernel is launched with at least `n` points along `t`, and that the body executes once for each, with `L`'s variable equal to the point's coordinate.

> **Definition 43.2 (Launch geometry).** **(stated here)** For a `PrimFunc`, `blockDim[a]` is the maximum extent bound to `threadIdx.a` and `gridDim[a]` the maximum bound to `blockIdx.a`, each defaulting to 1.

Taking the *maximum* is what makes two sibling nests with different extents on the same axis launchable at all, and it is also the first place a schedule can be wrong: the smaller nest then runs on more threads than it was written for.

> **Definition 43.3 (Binding signature).** **(stated here)** For an access inside a kernel, its *binding signature* is the set of `tag:extent` pairs of the thread bindings enclosing it.

> **Theorem 43.4 (Cross-block RAW is unrepairable).** **(stated here)** Let a global buffer element be written by one thread and read by another, and let those two threads be able to fall in different blocks of a grid with more than one block. Then no insertion of intra-kernel barriers makes the kernel deterministic.

*Proof sketch.* A barrier synchronises the threads of one block. The two accesses may be executed by threads of different blocks; blocks have no common synchronisation point inside a kernel launch; so the write is not ordered before the read, and both orders are permitted executions of the same program. ∎

Both hypotheses are load-bearing, and neither is decidable in general — which is why the compiler tests for neither. `crossBlockRAWBuffers` ([`gpu_race.ts:239`](../../../src/compiler/analysis/gpu_race.ts)) tests something weaker and sufficient for doubt: a storage buffer with at least one store and one load whose binding signatures are not all identical. Two accesses under different signatures need not touch a common element, and need not land in different blocks — two thread-level signatures can sit inside the same block. So the pass decides "the compiler cannot prove these accesses block-local", and serialising is the conservative response to *that*, not a consequence Theorem 43.4 forces. Only the grid hypothesis is checked at all, and it is checked at the call site rather than in the analysis ([`codegen.ts:605`](../../../src/backend/cuda/codegen.ts)).

> **Proposition 43.5 (Cross-thread sharing is repairable).** **(stated here)** If a *kernel-local* buffer is written under one signature and read under another, and every access is inside one block, then promoting the buffer to shared memory and inserting a barrier after each write makes the kernel deterministic.

*Proof sketch.* Within a block, `__syncthreads()` orders every access before it against every access after it, and shared memory is visible to all threads of the block. ∎

**Four hypotheses are hiding in "inserting a barrier after each write", and the proof sketch above uses all of them.** They are worth writing down because a repair that satisfies three of them is not a repair:

- **Convergent participation.** `__syncthreads()` is only defined when *every* thread of the block reaches it. A barrier placed inside a guarded region — the predicate `split` emits when the tile does not divide the extent (Chapter 40), or a bounds check — is reached by some threads and not others, and the behaviour is undefined rather than merely unsynchronised. Any barrier this repair inserts must sit outside every conditional.
- **Unique writers.** The proposition orders writes before reads; it says nothing about two threads writing the *same* element. A WAW race is not repaired by a barrier — both writes are still unordered relative to each other — and the result is whichever landed last. The analysis in §43.4 detects "written under one signature, read under another" and does not distinguish this case.
- **Phase separation.** One barrier per write is sufficient only if the accesses fall into alternating write-phases and read-phases. A loop that writes, reads, and writes again needs a barrier on *both* sides of the read — the second write must not overtake another thread's read. "After each write" is the right rule for a single producer-consumer step and not for a loop.
- **Capacity.** The promotion moves the buffer to shared memory, which is finite (48 KiB on the targets here). A buffer that does not fit cannot be promoted, so the repair is unavailable rather than merely expensive.

The implementation's response to all four is the same one: it does not attempt the general case. It promotes only *kernel-local* buffers whose accesses it can see, and where it cannot establish what it needs it falls back to serialisation — which is Theorem 43.4's response applied to a case Proposition 43.5 might have covered. That is the right engineering choice, and it means Proposition 43.5 describes *what a correct repair would require*, not a procedure the compiler carries out in full.

Theorem 43.4 and Proposition 43.5 are why the CUDA backend has two responses to a detected race, and §43.6 shows both.

> **And neither is a proof that the emitted kernel is race-free.** The analysis in §43.4 is a *detector*, and a conservative one: it reports doubt when a buffer's accesses carry differing binding signatures, which is neither necessary nor sufficient for a genuine race. Not sufficient, because two differing signatures may still touch disjoint elements — §43.4 says so, and the cost is unnecessary serialisation. Not necessary in the strong sense either: it examines buffer accesses under thread bindings, so races arising through any other channel — an atomic used incorrectly, a device-scope buffer aliased by the runtime, or a barrier the *backend* omits when emitting a construct the analysis never saw — are outside what it inspects. What the compiler establishes is: **of the races this analysis models, none survives into the emitted kernel.** That is a useful property and it is not "the generated GPU program is race-free"; no analysis in this compiler proves the latter, and Chapter 65's differential testing against the CPU backend is what actually catches the rest.

The third piece of theory is the one the compiler does *not* use:

> **Definition 43.6 (Tensorisation).** **(stated here)** *Tensorising* a sub-nest replaces it with a call to a hardware intrinsic computing the same thing — for a tensor core, a fixed-shape `M×N×K` matrix multiply-accumulate. It is legal iff the sub-nest computes exactly the intrinsic's function on operands laid out as the intrinsic requires.

The "laid out as required" clause is what makes tensorisation a *scheduling* problem rather than a peephole: the operands must already be in the right memory at the right stride, which takes a tile, a cache stage and an alignment. In this compiler `tensorize` sets a function attribute and the backend supplies the whole kernel, so the clause is discharged by construction and never checked.

## 43.4 In mlfw

### The GPU path in `SchedulePass`

```ts
      const sch = new Schedule(pf);
      const handled = applyDeterministicGpuSchedule(sch, this.target as never, sCfg as never);
      if (!handled && sCfg.enabled) {
        (this._policy as SchedulePolicy).applyToAllBlocks(sch);
      }
```

`applyDeterministicGpuSchedule` ([`gpu_matmul_schedule.ts:126`](../../../src/compiler/schedule/gpu_matmul_schedule.ts)) tries three things in order and returns whether any of them claimed the function:

```ts
export function applyDeterministicGpuSchedule(schedule: Schedule, target: ScheduleTarget, sCfg: GpuScheduleConfig = {}): boolean {
  if (!target.isGPU() || (target.isWebGPU && target.isWebGPU())) return false;
  let handled = applyDeterministicGpuMatmul(schedule, target, sCfg);
  if (!handled) handled = applyImplicitGemmConv(schedule, target, sCfg);
  if (!handled) handled = applyDeterministicGpuConv(schedule, target);
  return handled;
}
```

WebGPU is excluded on line 2, so everything below is CUDA-only. Of the three, only the last is a schedule in the sense of Chapter 38 — it runs `SchedulePolicy` over the convolution's blocks. The first two *replace the function body*.

### `bindThread`

[`schedule.ts:611`](../../../src/compiler/schedule/schedule.ts), sixteen lines, of which nine are the tag check:

```ts
    const validTags = [
      'blockIdx.x', 'blockIdx.y', 'blockIdx.z',
      'threadIdx.x', 'threadIdx.y', 'threadIdx.z'
    ];
    if (!validTags.includes(threadTag)) {
      throw new Error(`Invalid thread tag: ${threadTag}. Must be one of: ${validTags.join(', ')}`);
    }
    loop.kind = ForKind.THREAD_BINDING;
    loop.threadTag = threadTag;
```

No dependence question, no check that the tag is unused, no check against `maxBlockDimX`. It is the least defended primitive in the file, and §43.6 explains why that is a reasonable division of labour rather than an oversight.

The rules that call it all go through one helper ([`rules.ts:186`](../../../src/compiler/schedule/rules.ts)):

```ts
function bindFusedSpatialGPU(schedule: Schedule, fused: ForNode, target: ScheduleTarget): void {
  const blockSize = Math.min(target.maxThreadsPerBlock, 256);
  const extent = fused.extent;
  if (extent.type === 'IntImmNode' && extent.value > blockSize) {
    const [outer, tx] = schedule.split(fused, blockSize);
    schedule.bindThread(tx, 'threadIdx.x');
    if (!primFuncHasRecurrence(schedule.func)) schedule.bindThread(outer, 'blockIdx.x');
  } else {
    schedule.bindThread(fused, 'threadIdx.x');
  }
}
```

Fuse the spatial loops into one, split by 256, bind the inner half to threads and the outer half to blocks. Two details carry weight. The `256` is a literal, so a device advertising 1024 threads per block gets 256. And `if (!primFuncHasRecurrence(...))` is a whole-function test: a function containing a `scan` anywhere gets *no* block binding on *any* of its nests, because the recurrence would then span blocks and Theorem 43.4 would apply. That is a conservative, global, correct decision made in a local helper.

### The dynamic-shape gate

```ts
    if (this.target.isGPU() && blockHasNonConstExtent(schedule.func, blockName)) {
      this._explain(blockName, 'none', 'block has dynamic loop extents; runs sequentially (no dynamic grid)');
      return null;
    }
```

[`rules.ts:560`](../../../src/compiler/schedule/rules.ts). A loop whose extent is a symbolic dimension is not bound, so Chapter 62's dynamic shapes cost all GPU parallelism on this compiler, and the trace says so.

**That is a limitation of this implementation, not of the hardware, and the distinction is worth insisting on** because the sentence "a GPU needs its grid at compile time" is false and gets repeated. CUDA's launch geometry is an argument to `cuLaunchKernel`, computed by the *host* immediately before the launch; a kernel written with the standard `idx = blockIdx.x * blockDim.x + threadIdx.x; if (idx < n) ...` guard runs correctly for any `n` the host supplies, and that is how essentially every hand-written CUDA kernel handles a runtime size. WebGPU's `dispatchWorkgroups` is the same shape.

What is compile-time here is Definition 43.2's *geometry*, because this compiler computes the triple during scheduling from constant loop extents and stores it on the function. Supporting a dynamic grid would mean deferring that computation to the runtime — emitting the bound-and-guard form, and having the launcher divide the symbolic extent by the block size at call time. Nothing about the hardware prevents it; the geometry is simply not currently a runtime expression. So read this gate as **"the compiler declines to bind what it cannot count"**, and note that it is one of the larger open items on the dynamic-shape side.

### Race detection

[`analysis/gpu_race.ts`](../../../src/compiler/analysis/gpu_race.ts), 302 lines, run by the CUDA backend ([`cuda/codegen.ts:602`](../../../src/backend/cuda/codegen.ts)) and the WebGPU backend ([`webgpu/codegen.ts:438`](../../../src/backend/webgpu/codegen.ts)) — not by any pass. `profileGpuAccesses` walks the kernel with a scope carrying the enclosing bindings, and records per access its binding signature, its thread multiplicity, whether it sits under a block binding, and whether its index depends on a sequential loop variable. Five predicates then read the profile:

| Predicate | Detects |
|---|---|
| `crossBlockRAWBuffers` | the conservative approximation of Theorem 43.4 — a storage buffer touched under two signatures |
| `threadSharedIntermediates` | Proposition 43.5 — a kernel-local buffer read by a thread that did not write it |
| `loopCarriedIntermediates` | a local buffer indexed by a sequential loop variable |
| `extentMismatchBuffers` | a buffer written at one multiplicity and read at another |
| `hasMultiExtentBlockBinding` | one axis bound at two different extents in one kernel |

`_analyzeSharing` ([`cuda/codegen.ts:596`](../../../src/backend/cuda/codegen.ts)) turns those into one of three outcomes per kernel: proceed; **repair**, by promoting the offending buffers to `__shared__` and enabling barriers; or **serialise**, by setting the launch to 1×1×1 and emitting the thread-bound loops as ordinary `for` loops ([`cuda/codegen.ts:562`](../../../src/backend/cuda/codegen.ts)). The diagnosis is kept on `_launchDiagnosis` with the reason and the buffer names.

Note the first two lines of `_analyzeSharing`:

```ts
    if (func.hasAttr(FuncAttr.TENSOR_INTRIN)) { this._needsBarriers = false; return; }
    if (func.getAttr(FuncAttr.GPU_REGISTER_BLOCKED)) {
      this._needsBarriers = false;
      return;
    }
```

A function the deterministic template wrote is exempt from race analysis. The template inserts its own `SyncThreadsNode`s, and it is trusted to have got them right.

### The template

`buildTiledSharedMatmul` ([`gpu_matmul_schedule.ts:45`](../../../src/compiler/schedule/gpu_matmul_schedule.ts)) is 35 lines that build a nest directly out of node constructors: two `shared` tiles, a `local` accumulator, a k-tile loop with `SyncThreadsNode` before and after the inner product, and thread bindings on four axes. `buildRegisterBlockedMatmul` (in [`schedule/matmul_tiling.ts`](../../../src/compiler/schedule/matmul_tiling.ts)) is the register-blocked version the default path actually uses, with an `M×N` micro-tile per thread. Neither calls a scheduling primitive. `applyDeterministicGpuMatmul` finishes with:

```ts
  schedule.func.body = body;
  if (schedule.func._setChild) schedule.func._setChild('body', body);
  schedule.func.setAttr(FuncAttr.GPU_REGISTER_BLOCKED, true);
  return true;
```

Assignment, not transformation. The `Schedule` object is present only to carry the function.

### `tensorize` and `blockize`

`tensorize` ([`schedule.ts:1093`](../../../src/compiler/schedule/schedule.ts)) validates that its argument has numeric `M`, `N`, `K`, sets `FuncAttr.TENSOR_INTRIN`, and returns. It records nothing in the trace. Its one caller is `AutoTensorizePass` ([`tensorize_pass.ts:77`](../../../src/compiler/passes/schedule/tensorize_pass.ts)), which runs when `optimization.tensorize` is on and the target has tensor cores, matches a 16×16×16 f16 GEMM, and hands the whole function to a backend that emits `wmma` intrinsics. Definition 43.6's layout clause is never stated as a check.

`blockize` ([`schedule.ts:1103`](../../../src/compiler/schedule/schedule.ts)) wraps a loop in a new block whose read and write sets are collected from the body. It is the primitive that would let a scheduler treat a tile as a unit and then tensorise it. It has no caller.

## 43.5 Lab — binding loops to hardware

```bash
node docs/part7/ch43-scheduling-for-gpus/labs/01-thread-bindings.mjs
```

The elementwise nest, split and bound by hand:

```
  for i0_5_o_0 in 0..16 @thread_binding [blockIdx.x] {
    for i0_5_i_1 in 0..256 @thread_binding [threadIdx.x] {
      block mul_block_0 {
        bind v0_6 = ((i0_5_o_0 * 256) + i0_5_i_1)
```

and the geometry those two annotations imply:

```
  blockIdx.x   <- i0_5_o_0     extent 16
  threadIdx.x  <- i0_5_i_1     extent 256
  blockDim [256,1,1]   gridDim [16,1,1]
  4096 threads for 4096 elements
```

Then the kernel:

```
__global__ void traced(float* buf_1, float* buf_3) {
  const int i0_5_o_0 = blockIdx.x;
  const int i0_5_i_1 = threadIdx.x;
  float buf_4[1];
  buf_4[0] = 2.0f;
  const int v0_6 = ((i0_5_o_0 * 256) + i0_5_i_1);
  buf_3[v0_6] = (buf_1[v0_6] * buf_4[0]);
}
```

Both loops are gone. The binding expression `i_o·256 + i_i` survives verbatim — it is the block's iteration-variable binding, untouched since Chapter 34 wrote it — and the grid is not in the source at all: it is metadata the runtime reads to size the launch.

What `bindThread` checks:

```
  threadIdx.x    accepted
  threadIdx.w    Invalid thread tag: threadIdx.w. Must be one of: blockIdx.x, …
  blockIdx.z     accepted
  warpIdx.x      Invalid thread tag: warpIdx.x. Must be one of: blockIdx.x, …
```

The tag, and nothing else. Binding the *reduction* axis of a matmul to `threadIdx.x` is accepted without complaint, and Chapter 42's three layers have to sort it out later.

And what the rule picks, on two GPU targets:

```
  CUDA     maxThreadsPerBlock 1024   -> blockIdx.x=16  threadIdx.x=256
  WebGPU   maxThreadsPerBlock 256    -> blockIdx.x=16  threadIdx.x=256
```

Identical, because of the literal `256` in `bindFusedSpatialGPU`. The CUDA target's advertised 1024 is read and then discarded by the `Math.min`.

## 43.6 Lab — templates, and the two answers to a race

```bash
node docs/part7/ch43-scheduling-for-gpus/labs/02-races-and-templates.mjs
```

Four CUDA compilations. The first is the elementwise kernel above, reached through `compile()` — a schedule the primitives wrote.

The second is a 128×128 matmul:

```
  allocate rb_As[512] (shared) {
    allocate rb_Bs[512] (shared) {
      for rb_by in 0..2 @thread_binding [blockIdx.y] {
        for rb_bx in 0..2 @thread_binding [blockIdx.x] {
          for rb_ty in 0..16 @thread_binding [threadIdx.y] {
            for rb_tx in 0..16 @thread_binding [threadIdx.x] {
              let rb_tid = ((rb_ty * 16) + rb_tx)
              …
  62 lines of TIR, none of which any primitive produced.
  shared-memory buffers: 2
  thread bindings      : [blockIdx.y] [blockIdx.x] [threadIdx.y] [threadIdx.x]
  __syncthreads in CUDA: 2, printed in the TIR as
      [UnknownNode: SyncThreadsNode]
      [UnknownNode: SyncThreadsNode]
```

This is the fast kernel: 2×2 blocks of 16×16 threads, each thread holding a 4×4 register tile, two shared-memory staging buffers double-buffered across the k-tile loop, barriers on both sides of the inner product. It is also the one nest in the compiler that no primitive touched. `applyDeterministicGpuMatmul` matched the shape and assigned `schedule.func.body = buildRegisterBlockedMatmul(dims, cfg, plan.epilogue)`.

The `[UnknownNode: SyncThreadsNode]` lines are the printer gap Chapter 32 noted showing through, and they are also a diagnostic: `SyncThreadsNode` has no printer visitor because no lowering rule and no primitive ever emits one. Seeing it in a printout tells you the nest came from somewhere else.

The third compilation is the interesting race:

```
__global__ void Object(float* buf_1, float* buf_3) {
  __shared__ float buf_6[8];
  …
  buf_6[siv0_8] = buf_4[0];
  __syncthreads();
  float _acc_0 = buf_6[sa0_9];
  for (int r0_11 = 0; r0_11 < 1024; r0_11++) {
    _acc_0 = (_acc_0 + buf_1[((sa0_9 * 1024) + r0_11)]);
  }
  buf_6[sa0_9] = _acc_0;
  __syncthreads();
  const int v0_14 = i0_13;
  buf_3[v0_14] = (buf_6[v0_14] * buf_5[0]);
```

`x.sum(1).mul(2.0)` over an 8×1024 input. The reduction writes an 8-element intermediate, one row per thread, and the scaling reads it — under a different binding, so a thread may read a row it did not write. `threadSharedIntermediates` flagged `buf_6`; it is not stored under a block binding, so Proposition 43.5 applies; and `_promoteCrossThreadToShared` moved it to `__shared__` and switched barriers on. Repair, not refusal, and the schedule that caused it was never told.

The fourth is the refusal:

```
__global__ void Object_p0(float* buf_1, float* buf_3) {
  for (int si0_5 = 0; si0_5 < 300; si0_5++) {
    …
  for (int sa0_7_o_4 = 0; sa0_7_o_4 < 2; sa0_7_o_4++) {
    for (int sa0_7_i_5 = 0; sa0_7_i_5 < 256; sa0_7_i_5++) {
      if ((((sa0_7_o_4 * 256) + sa0_7_i_5) < 300)) {
```

No `blockIdx`, no `threadIdx`, ordinary `for` loops with the split guard of Chapter 40 still in place — inside a `__global__` function. The schedule bound both of those loops; `crossBlockRAWBuffers` found a *storage* buffer written under one binding signature and read under another; the compiler cannot rule out the situation Theorem 43.4 describes, and cannot repair it if it holds; so `_serialize` set the launch to 1×1×1 and emitted the loops.

That is the fourth answer to Chapter 42's question, and the only one that reaches the emitted code. The primitives accepted the bindings, the validator was never asked, the WASM-style backend re-check does not apply — and the CUDA backend, the one layer that knows what a grid is, silently produced a single-threaded kernel. Correct, and 300× slower than the schedule asked for, with the reason recorded on `_launchDiagnosis` and nowhere in the trace.

## 43.7 Traps and limits

- **The compiler's fastest GPU kernel is not a schedule.** §43.6. The matmul and convolution paths assign a hand-written body and set an attribute; the scheduling language contributes nothing. Everything Part VII teaches applies to the *elementwise and reduction* kernels around them, which is where the remaining time goes only after the GEMM is already fast.
- **A serialised kernel is invisible.** `_serialize` records a reason on `_launchDiagnosis`, and nothing emits it as a trace event. A compilation whose main kernel silently became single-threaded looks, from the trace, exactly like one that did not. This is the same shape as Chapter 34's un-fused fusion group, and the fix is the same: one `trace.warn`.
- **`bindFusedSpatialGPU` caps the block at a literal 256.** [`rules.ts:187`](../../../src/compiler/schedule/rules.ts). `maxThreadsPerBlock` is read and then discarded by `Math.min(..., 256)`, so no target can ask for more, and a target advertising fewer is respected only because it is the smaller of the two.
- **Nothing checks a binding against the device's dimension limits.** `maxBlockDimY`, `maxGridDimY` and `maxGridDimZ` are declared on every GPU target ([`target.ts:79`](../../../src/compiler/support/target.ts)) and read by nothing in `src/compiler/`. A schedule binding 70,000 iterations to `blockIdx.y` would be accepted, compiled, and fail at launch.
- **No scheduling primitive ever allocates shared memory.** `cacheRead(block, buf, 'shared')` is the primitive for it, and Chapter 41 established that it has no caller and would stage the whole buffer anyway. Every `__shared__` array this compiler emits comes from one of four other places: the register-blocked matmul template ([`matmul_tiling.ts:285`](../../../src/compiler/schedule/matmul_tiling.ts)), the tiled one ([`gpu_matmul_schedule.ts:39`](../../../src/compiler/schedule/gpu_matmul_schedule.ts)), the implicit-GEMM convolution ([`conv_implicit_gemm.ts:82`](../../../src/compiler/schedule/conv_implicit_gemm.ts)), and the CUDA backend's own `_promoteCrossThreadToShared`. The fifth is the odd one: the CUDA-specific *lowering rule* for `scaled_dot_product_attention` ([`rules/attention.ts:125`](../../../src/compiler/passes/lowering/rules/attention.ts)) sizes a flash-attention tile against `target.sharedMemoryBytes` and allocates it, which means the memory scope of the busiest kernel in a transformer is chosen in Chapter 34, before Part VII gets a look.
- **Race analysis lives in two backends and not in a pass.** [`cuda/codegen.ts:602`](../../../src/backend/cuda/codegen.ts) and [`webgpu/codegen.ts:438`](../../../src/backend/webgpu/codegen.ts) call `profileGpuAccesses` independently and act on different subsets of the five predicates — CUDA on cross-block RAW, cross-thread sharing and multi-extent block bindings; WebGPU additionally on `loopCarriedIntermediates` and `extentMismatchBuffers`. A third GPU backend would have to reimplement the choice.
- **The cross-block race test never asks whether the accesses overlap.** `crossBlockRAWBuffers` groups by buffer *name* and fires on a signature mismatch ([`gpu_race.ts:239`](../../../src/compiler/analysis/gpu_race.ts)); no region is compared, so a kernel whose two signatures touch provably disjoint halves of a buffer is serialised exactly as hard as one with a genuine cross-block dependence. The region machinery to decide it exists — Chapter 35's `coverRangeOfForm`, Chapter 36's `accessDependence` — and this analysis, living in the backend rather than in a pass, reaches for none of it.
- **`tensorize` neither checks nor records.** No layout precondition (Definition 43.6), and no `trace.record`, so a tuned schedule that tensorises cannot be replayed faithfully.
- **`blockize` has no caller.** It is the primitive that would let a scheduler wrap a tile in a block and hand that block to `tensorize`; without it, tensorisation is stated at the function level and the backend supplies the whole kernel.
- **A `scan` anywhere in a function disables block binding everywhere in it.** `primFuncHasRecurrence` ([`rules.ts:182`](../../../src/compiler/schedule/rules.ts)) is a whole-function test consulted per nest, so an elementwise nest that shares a function with an unrelated recurrence is confined to one block.

## 43.8 Read the tests

- [`tests/compiler/schedule/rules.test.js`](../../../tests/compiler/schedule/rules.test.js) — the GPU rules' binding choices per target, including the recurrence case.
- [`tests/compiler/analysis/gpu-race.test.js`](../../../tests/compiler/analysis/gpu-race.test.js) — the five predicates, each on a kernel built to trip exactly one.
- [`tests/backend/cuda/`](../../../tests/backend/cuda/) — the emitted CUDA for the register-blocked template, the promotion path and the serialised path. Most of the numerical assertions here are hardware-gated and skip without a device; the source-shape assertions are not.

---

Part VII ends here. [Part VIII — Autotuning](../../part8/README.md) picks up the twenty-two primitives as a search space: how large it is, how a sketch generates a skeleton of one, how a cost model ranks candidates without running them, and how the winner is stored so that the same program compiles to the same kernel tomorrow.
