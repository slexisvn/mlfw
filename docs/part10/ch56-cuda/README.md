# Chapter 56 — CUDA

The three backends so far translated a loop nest into a loop nest. This one does something different in kind: **it deletes the outer loops**, because on a GPU they do not run. They are the index space, and the hardware supplies an index to each of tens of thousands of threads that run the body once.

That changes the backend's job twice over. It has to *decide a launch geometry* — how many threads, arranged how — which is information that lives outside the source text. And it has to *decide whether the program it was handed is safe to run that way at all*, because a loop nest that is correct when executed in order can be a race when executed all at once.

## 56.1 The problem: a loop that is not a loop

Take the smallest possible kernel, `x * 2` over 1,024 elements, and emit it without a schedule:

```cuda
__global__ void traced(float* buf_1, float* buf_3) {
  float buf_4[1];
  buf_4[0] = 2.0f;
  for (int i0_5 = 0; i0_5 < 1024; i0_5++) {
    const int v0_6 = i0_5;
    buf_3[v0_6] = (buf_1[v0_6] * buf_4[0]);
  }
}
```

That compiles and it is correct and it is a catastrophe: a `__global__` function launched with one block of one thread, running a 1,024-iteration loop on a device built to run 1,024 things at once. The GPU is being used as a very slow single-core CPU across a PCIe bus.

What the backend wants instead is:

```cuda
__global__ void traced(float* buf_1, float* buf_3) {
  const int i0_5_o_0 = blockIdx.x;
  const int i0_5_i_1 = threadIdx.x;
  float buf_4[1];
  buf_4[0] = 2.0f;
  const int v0_6 = ((i0_5_o_0 * 256) + i0_5_i_1);
  buf_3[v0_6] = (buf_1[v0_6] * buf_4[0]);
}
```

with `blockDim = (256,1,1)` and `gridDim = (4,1,1)`. The loop is gone. Its variable is a register the hardware writes. And the extents — 256 and 4 — are not in the source at all; they are numbers the runtime passes to the launch.

So this backend produces *two* things: a text, and a geometry. Chapter 43 decided which loops become which axes; this chapter is about what the backend does with that decision, and about the much harder question of what it does when the decision is unsafe.

## 56.2 Intuition: a barrier you have and a barrier you do not

A CUDA launch has two levels of grouping. Threads are gathered into **blocks**; blocks are gathered into a **grid**. Threads in one block run on one multiprocessor, share a small fast scratchpad (`__shared__`), and can wait for each other (`__syncthreads()`). Blocks are independent: they may run in any order, at any time, possibly not concurrently, and **nothing inside an ordinary launch makes one block wait for another**.

That single asymmetry explains almost every decision in this backend.

- If a value is written by one thread and read by another *in the same block*, the backend can fix it: put the value in `__shared__` and put a barrier between the write and the read.
- If a value is written by one block and read by another, the backend cannot fix it at all. The only barrier it has between blocks is the end of the launch, so the graph would have to be split into two kernels — and by codegen time it is too late to do that. (There is one other barrier, and §56.3 prices it.)

When the second case appears, the backend has exactly one safe move left: **give up on the parallel launch**. Set the geometry to one thread, turn the thread-bound loops back into ordinary `for` loops, and produce a correct, dreadfully slow kernel. §56.6 measures that happening.

## 56.3 Theory

### Geometry

> **Definition 56.1 (Launch geometry).** **(invariant)** A thread tag is one of `threadIdx.{x,y,z}` (the *thread space*) or `blockIdx.{x,y,z}` (the *block space*). A loop bound to tag *t* with extent *e* contributes `dim[axis(t)] ← max(dim[axis(t)], e)`, where `dim` is `blockDim` for the thread space and `gridDim` for the block space. Every dimension not written stays 1.
>
> The launch then runs `∏ gridDim × ∏ blockDim` threads.

The `max` is the interesting part. Two loops in the same function may both be bound to `threadIdx.x` at different extents — a 32-wide elementwise nest and an 8-wide reduction, say — and a launch has one `blockDim.x`. So the tag's extent is the largest requested, and every binding that asked for less must be *guarded*.

> **Definition 56.2 (Binding guard).** **(invariant)** A loop bound to tag *t* with static extent *e* less than *t*'s launch extent is emitted inside `if (t < e) { … }`.

> **Proposition 56.3 (Guarding preserves the loop's meaning).** **(stated here)** Under Definition 56.2 the set of (tag value, body) pairs executed is exactly `{ (k, body) : 0 ≤ k < e }`, which is the set the serial loop `for (k = 0; k < e; k++)` executes — provided the body's effect does not depend on the *order* of those executions.
>
> The proviso is not decorative; it is the whole of §56.3's next subsection.

### Races

> **Definition 56.4 (Thread-shared intermediate).** **(stated here)** A buffer local to a kernel is *thread-shared* when the set of thread-index values under which it is written differs from the set under which it is read — that is, some thread reads an element it did not write.

> **Definition 56.5 (Cross-block RAW).** **(stated here)** A global buffer has a *cross-block read-after-write* when it is written under one block-space binding signature and read under a different one.

> **Theorem 56.6 (What a barrier can and cannot repair).** **(classical)** Let a kernel contain a write to location ℓ by thread *p* and a read of ℓ by thread *q ≠ p*, with no synchronisation between them.
>
> 1. If *p* and *q* are in the same block, inserting `__syncthreads()` between the write and the read, with ℓ in `__shared__` or in global memory, makes the read see the write.
> 2. If *p* and *q* are in different blocks **under an ordinary launch**, no instruction sequence inside the kernel makes the read see the write, because block execution order is unconstrained and blocks are not required to be resident simultaneously.
>
> *Proof of (2).* Suppose such a sequence existed and consider a device that schedules the grid one block at a time to completion — which an ordinary launch permits. If *q*'s block runs first, the read precedes the write in real time; the sequence would have to make *q* wait, but a device that runs blocks to completion would then never run *p*'s block, and the kernel would not terminate. So no sequence both terminates and orders the two. ∎

**"Ordinary launch" is a hypothesis, and it can be bought off.** A *cooperative* launch — `cudaLaunchCooperativeKernel`, with `grid.sync()` from the cooperative-groups header — does give a grid-wide barrier, and it gives it by purchasing exactly what the proof assumes away: the launch *fails* unless the whole grid is co-resident, so the device that runs blocks one at a time is no longer a device the kernel has to survive. The price is that the grid may be no larger than the hardware will hold at once, which makes the launch geometry a function of the device rather than of the problem — and Definition 56.1 derives geometry from loop extents, which is the other thing entirely. This compiler never emits a cooperative launch, so part 2 holds for everything it produces. The exception is worth stating anyway, because it says what a cross-block barrier actually *costs*: not that the hardware refuses, but that the price is the grid.

Theorem 56.6 is why `_analyzeSharing` has the shape it has: cross-block first, because it is unrepairable; then thread-shared, because promotion plus a barrier may repair it; then the multi-extent cases.

> **Definition 56.7 (Serialization).** **(invariant)** To *serialize* a kernel is to set `blockDim = gridDim = (1,1,1)`, emit every thread-bound loop as an ordinary `for` loop over its extent, insert no barriers, and record a reason on `launchDiagnosis`.
>
> Serialization always produces a correct kernel and never produces a fast one.

### Reductions

A reduction to a scalar has no parallel index at all — every thread would write the same location. The standard answer is a two-stage tree:

> **Definition 56.8 (Block tree reduction).** **(classical)** With *T* threads and *n* elements: thread *j* accumulates a private partial over the strided subset `{ j, j+T, j+2T, … }`; the *T* partials are written to a shared array; then ⌈log₂ T⌉ rounds each halve the active thread count and add the upper half into the lower, with a barrier between rounds; thread 0 stores the result.
>
> The depth is *n*/*T* + log₂ *T* rather than *n*, and the association changes — level **N2**.

### Intrinsics

> **Definition 56.9 (Tensor intrinsic).** **(stated here)** A *tensor intrinsic* is a name, registered against an emitter, attached to a `PrimFunc` as an attribute together with the operand names and problem dimensions. When present, the backend does not walk the function body: it calls the emitter, which writes the whole kernel including its own launch geometry.
>
> The attribute is a *claim* that the body it replaces computes what the intrinsic computes. The backend does not verify the claim; the schedule primitive that set the attribute is responsible for it.

## 56.4 In mlfw: the decision before the text

`CUDACodegen.generate` ([`cuda/codegen.ts:95`](../../../src/backend/cuda/codegen.ts)) does four things before it emits a line: read the thread bindings (from the LIR metadata, or by scanning), find which buffers are stored into, run `_analyzeSharing`, and collect global scratch.

### The decision tree

`_analyzeSharing` ([`cuda/codegen.ts:563`](../../../src/backend/cuda/codegen.ts)) turns Theorem 56.6 into a sequence of tests ordered by severity — the unrepairable case first, so that a repairable one is never mistaken for it:

```ts
    const profile = profileGpuAccesses(func, { sharedBuffers: this._sharedBuffers, threadBindings: this._threadBindings });
    const gridThreads = this._gridDim[0] * this._gridDim[1] * this._gridDim[2];
    const crossBlock = crossBlockRAWBuffers(profile);
    if (gridThreads > 1 && crossBlock.size > 0) {
      this._serialize(GpuRaceReason.CROSS_BLOCK_RAW, crossBlock);
      return;
    }
    if (this._threadBindings.size > 0) {
      const blockThreads = this._blockDim[0] * this._blockDim[1] * this._blockDim[2];
      const crossThread = threadSharedIntermediates(profile);
      if (blockThreads * gridThreads > 1 && crossThread.size > 0) {
        if (!storedUnderBlockBinding(profile, crossThread) && this._promoteCrossThreadToShared(func, crossThread)) {
          this._needsBarriers = true;
          return;
        }
        this._serialize(GpuRaceReason.THREAD_SHARED_INTERMEDIATE, crossThread);
        return;
      }
    }
```

Cross-block is Definition 56.5 and Theorem 56.6(2): unrepairable, serialize. Thread-shared is Definition 56.4 and Theorem 56.6(1): try promotion, and serialize only if promotion fails. `storedUnderBlockBinding` is the extra condition that makes the repair sound — a buffer written under a block binding is not confined to one block, so putting it in `__shared__` would give each block its own copy.

The five reasons live in one frozen table ([`gpu_race.ts:38`](../../../src/compiler/analysis/gpu_race.ts)), shared with the WebGPU backend, which reaches three of them by a different route.

`_promoteCrossThreadToShared` ([`cuda/codegen.ts:535`](../../../src/backend/cuda/codegen.ts)) is the repair, and it is budgeted: it accumulates the bytes it would need and returns `false` — falling through to serialization — as soon as the running total exceeds `target.sharedMemoryBytes`. §56.6 sweeps that budget and watches the same program change plan.

Definition 56.7 takes five lines to carry out (`_serialize`, [`cuda/codegen.ts:529`](../../../src/backend/cuda/codegen.ts)). The `for` loops come back at `_visitForNode` ([`cuda/codegen.ts:317`](../../../src/backend/cuda/codegen.ts)), which checks `_serializeThreads` before anything else.

### The geometry, and the two limits

The `max` of Definition 56.1 lives in `_applyBindingDim` ([`cuda/codegen.ts:243`](../../../src/backend/cuda/codegen.ts)), and the guard of Definition 56.2 at [`cuda/codegen.ts:327`](../../../src/backend/cuda/codegen.ts):

```ts
      const extent = node.extent.type === 'IntImmNode' ? node.extent.value : 0;
      const tag = node.threadTag;
      const maxExtent = this._getMaxBindingExtent(tag);
      if (extent > 0 && maxExtent > 0 && extent < maxExtent) {
        this._emit(`if (${tag} < ${extent}) {`);
```

At the end, the geometry is clamped to the device's per-axis maxima and then two hard errors are thrown: more threads per block than `maxThreadsPerBlock`, or more shared bytes than `sharedMemoryBytes` ([`cuda/codegen.ts:204`](../../../src/backend/cuda/codegen.ts) and [`:210`](../../../src/backend/cuda/codegen.ts)). These are compilation failures rather than launch failures — a kernel that cannot start is something the user should hear about while compiling, not while running.

### Local memory

A `__global__` function's local arrays live in per-thread local memory, of which there is 512 KiB per thread — a large-looking budget that a serialized kernel exhausts instantly, because a serialized kernel's "per-thread" arrays are the whole tensor.

`_collectGlobalScratch` ([`cuda/codegen.ts:649`](../../../src/backend/cuda/codegen.ts)) has three branches and they are disjoint. **With thread bindings and no serialization it does nothing** — a per-thread array in a genuinely parallel kernel is per-thread-sized and small. **With no thread bindings at all** it offloads any single buffer over 32,768 elements and stops. **When serialized** it sums the candidates and, only if the total exceeds a 256 KiB budget, offloads them largest-first until it is under.

An offloaded buffer becomes an extra kernel **parameter**, which the runtime allocates in global memory. Note that the 256 KiB budget is half the 512 KiB the hardware allows a thread: the backend is leaving room for whatever NVCC decides to spill on its own, which it cannot see (§53.2).

### Reductions and intrinsics

`_matchFullReduction` ([`cuda/codegen.ts:254`](../../../src/backend/cuda/codegen.ts)) recognises a serial nest accumulating into a single-element buffer with at least 2,048 total iterations, and `_emitParallelReduction` ([`cuda/codegen.ts:285`](../../../src/backend/cuda/codegen.ts)) emits Definition 56.8 at *T* = 256. §56.6 shows what happens to it.

Intrinsics are Definition 56.9, at [`cuda/codegen.ts:180`](../../../src/backend/cuda/codegen.ts):

```ts
    const tensorIntrin = func.getAttr<TensorIntrinAttr>(FuncAttr.TENSOR_INTRIN);
    if (tensorIntrin) {
      const emit = getCudaIntrin(tensorIntrin.name);
      if (!emit) throw new Error(`CUDA codegen: unknown tensor intrinsic '${tensorIntrin.name}'`);
      emit(this, tensorIntrin.info);
    } else {
      this._visitNode(func.body);
    }
```

Two are registered ([`cuda/tensor_intrin.ts`](../../../src/backend/cuda/tensor_intrin.ts)): a WMMA 16×16×16 fragment loop, and a double-buffered pipelined SGEMM. An unknown name throws rather than falling back, and the asymmetry is deliberate: a silent fallback would turn "the tensor cores were not used" into an invisible performance regression, which is the one failure this backend has too many of already.

## 56.5 Lab — the loops that do not run

```bash
node docs/part10/ch56-cuda/labs/01-the-loops-that-do-not-run.mjs
```

The first pair is §56.1's two kernels, side by side: one thread and a 1,024-iteration loop, against 1,024 threads and no loop at all. The bindings were applied by hand with the primitives of [Chapter 43](../../part7/ch43-scheduling-for-gpus/README.md); nothing else changed.

The second section builds Definition 56.2's guard on purpose — three bindings on `threadIdx.x`, one at extent 32 and two at extent 8 — and the emitted kernel shows all of §56.3 at once:

```cuda
__global__ void traced(float* buf_1, float* buf_3) {
  __shared__ float buf_6[256];
  const int sa0_13 = threadIdx.x;
  ...
  for (int i0_7 = 0; i0_7 < 8; i0_7++) {
    buf_6[((v0_9 * 32) + v1_10)] = (buf_1[((v0_9 * 32) + v1_10)] * buf_4[0]);
    __syncthreads();
  }
  if (threadIdx.x < 8) {
    buf_3[siv0_12] = buf_5[0];
  }
  __syncthreads();
  if (threadIdx.x < 8) {
    float _acc_0 = buf_3[sa0_13];
    for (int r0_15 = 0; r0_15 < 32; r0_15++) {
      _acc_0 = (_acc_0 + buf_6[((sa0_13 * 32) + r0_15)]);
    }
    buf_3[sa0_13] = _acc_0;
  }
  __syncthreads();
}
```

`blockDim` is 32, the largest extent asked for. The two extent-8 bindings are guarded. `buf_6` — written by 32 threads in the first nest and read across all 32 columns by 8 threads in the third — was *promoted* to `__shared__`, and barriers were inserted, which is Theorem 56.6(1) applied.

Then the limit:

```
  64 x 64 threads per block: [codegen] kernel 'traced' block 64x64x1 = 4096 threads exceeds maxThreadsPerBlock 1024
```

And then the number that matters most in this chapter:

```
  --- shipped defaults ---
  graph           kernel    threads  shared  diagnosis
  elementwise     ew              1       0  -
  softmax         sm              1       0  -
  layer_norm      ln              1       0  -
  matmul          mm            256    4096  -
  sum to scalar   ra              1       0  -

  --- scheduling: { enabled: true } ---
  elementwise     ew          65536       0  -
  softmax         sm            512    6208  -
  layer_norm      ln            512    4160  -
  matmul          mm            256    4096  -
  sum to scalar   ra              1       0  -
```

**With the shipped defaults, every kernel but the matmul launches one thread.** `CUDATarget` declares `{ gpuTiling: true }` in its scheduling attributes and does not declare `enabled`; the scheduling pass applies the matmul and convolution templates first and then tests `enabled` before reaching the general rules, so a softmax, a layer norm and a plain elementwise chain all reach this backend as serial loop nests and are emitted as serial loop nests inside a `__global__` function. `WebGPUTarget` declares `{ enabled: true }`. That one key is the difference between the two GPU targets, and it is visible only here, at the launch geometry.

The `sum to scalar` row stays at one thread even with scheduling on, for a different reason that §56.6 takes apart.

**Try this.** Bind a loop whose extent is a symbolic dimension and look for the guard. There is none: `_visitForNode` reads `extent` as `node.extent.type === 'IntImmNode' ? value : 0`, and a `0` extent fails the `extent > 0` test, so a dynamic binding is emitted unguarded and every thread the launch happens to have runs the body.

## 56.6 Lab — when the backend refuses

```bash
node docs/part10/ch56-cuda/labs/02-when-the-backend-refuses.mjs
```

The first table sweeps the shared-memory budget on the guard kernel above:

```
  shared budget    blockDim __shared__  diagnosis
          49152    [32,1,1]          1  -
           1024    [32,1,1]          1  -
            512     [1,1,1]          0  kernel-local buffer read by a thread that did not write it
```

256 `f32` is exactly 1,024 bytes. At 512 the promotion cannot fit, `_promoteCrossThreadToShared` returns `false`, and Definition 56.7 takes over: one thread, no barriers, a recorded reason. The kernel still computes the right answer and is 32 times narrower. **A device parameter changed the parallelism of the emitted program by a factor of 32, with no error and no warning.**

The same decision, reached from a graph rather than by hand:

```
  kernel    threads  diagnosis
  smb_p0         64  -
  smb_p1          1  cross-block read-after-write on a global buffer
  smb_p2      16384  -
```

A 64×256 softmax splits into three kernels, and the middle one is serialized by Theorem 56.6(2). **Nothing reads `launchDiagnosis`.** It is on the kernel's metadata, and no trace event carries it, so a compilation in which one of three kernels silently became single-threaded is indistinguishable in every diagnostic from one in which nothing happened.

Then the local-memory offload:

```
  kernel parameters: buf_1, buf_3, buf_6
  offloaded to global scratch: [{"name":"buf_6","dtype":"f32","size":65536}]
```

A 65,536-element `f32` temporary is 256 KiB, which a thread's local memory would hold and a second one would not; the backend turns it into a parameter instead. Note the consequence for the calling convention: the kernel's parameter list is now longer than its buffer map, and the runtime has to know to allocate the difference.

The fourth section is the one worth the whole chapter.

```
  from the PrimFunc: blockDim [256,1,1]  __shared__ decls 1  reported sharedMemBytes 0
  from the LIRFunc : blockDim [1,1,1]  __shared__ decls 0  reported sharedMemBytes 0
```

Handed a `PrimFunc`, the backend recognises the full reduction and emits Definition 56.8 — 256 threads, a strided partial-sum loop, a shared array, a log-depth tree, a single store from thread 0. Handed the `LIRFunc` lowered from the *same function*, it emits a serial double loop on one thread.

The reason is Chapter 53. `_matchFullReduction` pattern-matches `ForNode → BlockNode → BufferStoreNode` accumulating into a single-element buffer, and `lowerToLIR` replaces exactly that shape with an `LIRAccumulatorNode`. **So the block tree reduction fires on the path the unit tests exercise and never on the path a compilation takes**, which is why `sum to scalar` stayed at one thread in §56.5's table even with scheduling enabled.

There is a second, smaller thing in the same two lines: the pre-LIR kernel declares `__shared__ float _redsh[256]` — 1,024 bytes — and reports `sharedMemBytes: 0`, because `_emitParallelReduction` writes the declaration with `_emit` instead of registering a shared buffer. Statically-declared shared memory does not need the launch parameter, so this is harmless on CUDA and it makes the reported figure wrong.

The last section puts Definition 56.9 to work. A tensor-intrinsic attribute replaces the body with fragment operations and sets its own geometry (`blockDim (32,1,1)` — one warp — and `gridDim (4,4,1)` for a 64×64 output in 16×16 tiles). An unregistered name is a hard error.

**Try this.** Attach the intrinsic to a function whose body is *not* a matmul and watch the kernel emit anyway. The attribute is Definition 56.9's unchecked claim, and the only thing standing between it and a wrong kernel is the schedule primitive that sets it.

## 56.7 Where this was verified

Everything above is codegen: the emitted text and the launch metadata, both produced and checked without a device. The tests that actually *run* these kernels are gated on hardware — the files under `tests/backend/cuda/` whose names carry a `.cuda.` segment are routed to a separate vitest project that runs serially and skips when no CUDA device is present.

Which means the honest statement is two-part. The decisions of §56.3 and §56.4 — geometry, guards, serialization, promotion, scratch — are checked on every run, because they are properties of the text and the metadata. The numerical results are checked on a machine with a device, and the last such run for this chapter's material was on a single consumer GPU.

## 56.8 Traps and limits

### A block reduction that fires only on the path a compilation does not take

`_matchFullReduction` and `_emitParallelReduction` are 60 lines implementing Definition 56.8, and no compilation reaches them, because Chapter 53's accumulator lowering removes the pattern they match. §56.6 shows the two outputs side by side. The fix is to match on `LIRAccumulatorNode` — which carries strictly more information than the shape being matched, including the operator and the flush address — rather than on the pre-LIR tree.

### The rest

- **A serialized kernel is invisible.** `_launchDiagnosis` is set by `_serialize` ([`cuda/codegen.ts:529`](../../../src/backend/cuda/codegen.ts)) and travels on the kernel metadata to the runtime, where nothing reads it and no trace event carries it. The difference between "this kernel runs on 16,384 threads" and "this kernel runs on 1" is available at compile time, recorded, and never surfaced.
- **The shipped `CUDATarget` schedules only matmul and convolution.** §56.5 measures it: with the defaults, a softmax reaches this backend as a serial loop nest. The cause is one key in one attribute table, and the symptom is a whole class of kernel launching one thread.
- **A dynamic thread extent is never guarded.** `_visitForNode` reads a non-`IntImmNode` extent as `0`, and the guard test requires `extent > 0`, so a binding over a symbolic dimension emits no `if`. Whether that is safe depends on the runtime passing a launch geometry that matches the actual extent, which is a contract nothing states.
- **`_defaultDtype` is whichever buffer came last.** It is assigned inside the parameter loop ([`cuda/codegen.ts:139`](../../../src/backend/cuda/codegen.ts)), so after the loop it holds the *last* buffer's dtype — and it is what `_emitFloatLiteral` uses to choose the `f`/nothing suffix on every float literal in the kernel, and what `_emitExternCall` falls back to when a call has no dtype. A kernel whose last parameter is `f64` gets `f64` literal suffixes throughout.
- **The parallel-reduction threshold is a bare 2,048.** `_matchFullReduction` refuses smaller reductions ([`cuda/codegen.ts:281`](../../../src/backend/cuda/codegen.ts)) and the thread count is a bare 256 ([`cuda/codegen.ts:286`](../../../src/backend/cuda/codegen.ts)). Neither is read from the target, so a device with a different warp size or occupancy profile gets the same numbers. Latent, given the finding above.
- **`alloca` in a `__global__` function.** A dynamically-sized non-shared allocation is emitted as `alloca(...)` ([`cuda/codegen.ts:380`](../../../src/backend/cuda/codegen.ts)). NVCC accepts it and places it in local memory; it interacts badly with the scratch-offload path, which sizes candidates by `_estimateBufferSize` — a product over the *static* dimensions only, so a dynamically-sized buffer is estimated as much smaller than it is and is never offloaded.
- **Promotion and scratch can disagree about the same buffer.** `_analyzeSharing` runs first and may promote a buffer to shared; `_collectGlobalScratch` runs second and skips anything already promoted. But the multi-extent branch of `_analyzeSharing` ([`cuda/codegen.ts:596`](../../../src/backend/cuda/codegen.ts)) promotes *every* non-shared allocation it can see, with no budget check at all — unlike `_promoteCrossThreadToShared`, which has one — so that path can produce a kernel that exceeds the device's shared memory and is caught only by the hard error at the end.

## 56.9 Read the tests

- [`tests/backend/cuda/codegen.test.js`](../../../tests/backend/cuda/codegen.test.js) — the emitted text and metadata without a device: geometry from bindings, the guard, the two hard errors, serialization and its reason, promotion, and the scratch offload.
- [`tests/backend/cuda/tensor-intrin.test.js`](../../../tests/backend/cuda/tensor-intrin.test.js) — the registry, the two registered emitters, and that an unknown name throws.
- [`tests/backend/cuda/compile.test.js`](../../../tests/backend/cuda/compile.test.js) — end to end through `compileGraph`, still without a device.
- [`tests/backend/cuda/kernels.cuda.test.js`](../../../tests/backend/cuda/kernels.cuda.test.js), [`matmul-shared-tiling.cuda.test.js`](../../../tests/backend/cuda/matmul-shared-tiling.cuda.test.js), [`conv-float4.cuda.test.js`](../../../tests/backend/cuda/conv-float4.cuda.test.js), [`symbolic-shape.cuda.test.js`](../../../tests/backend/cuda/symbolic-shape.cuda.test.js), [`tensorize-exec.cuda.test.js`](../../../tests/backend/cuda/tensorize-exec.cuda.test.js) — the hardware-gated half: these compile with NVRTC, launch, and compare against the CPU backend. They skip silently without a device, which is exactly why §56.7 exists.
- [`tests/backend/cuda/deep-model-quality.test.js`](../../../tests/backend/cuda/deep-model-quality.test.js) — properties of the emitted kernels for whole models, which is where a regression in the serialization decision would show up as a kernel that lost its bindings.

---

**Next:** [Chapter 57 — WebGPU and WGSL](../ch57-webgpu-and-wgsl/README.md), the same execution model under a much stricter language.
