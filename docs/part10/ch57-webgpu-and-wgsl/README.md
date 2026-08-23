# Chapter 57 — WebGPU and WGSL

A WebGPU compute shader may declare eight storage buffers. Not eight by convention — eight is the floor of the specification, so a kernel with nine tensor arguments is not slow, it is a program that will not run on a conforming device. Every array it allocates must have a size written as a literal in the source. And it takes no scalar arguments at all.

That is the whole chapter: a backend meeting an *interface* budget rather than a performance one, under a language that will not let it allocate its way out. The execution model underneath is Chapter 56's — threads in workgroups, workgroups in a dispatch, no barrier across workgroups, the same race analysis, the same last resort of serialization — and this chapter takes that as read and spends its length on the constraints CUDA does not have. The answer to all three of them turns out to be the same move, and it is a move Part IX already made twice.

## 57.1 The problem: eight slots and no pointers

CUDA hands a kernel its buffers as C pointer parameters. There is no limit worth worrying about, they can alias, and a temporary can be an `alloca` sized at run time.

A WebGPU compute shader has none of that.

**Buffers arrive through a binding table.** Each is declared at module scope with an explicit `@group(0) @binding(k)` and an access mode, and the host must build a matching bind group before it can dispatch. So the binding count is not an implementation detail the backend can leave to the text: it is a number the runtime has to reproduce, against a device limit the backend cannot exceed.

**There is no dynamic allocation.** Every array in WGSL has a size fixed in the source. A workgroup-memory array is `var<workgroup> a: array<f32, 512>` — 512 is a literal. A private array is the same. So every temporary the kernel needs must be sized at compile time, and the total must fit the device's workgroup-memory budget.

**There are no scalar kernel arguments.** A compute entry point takes builtin values — the invocation ids — and nothing else. A shape parameter, which on the other three backends is an ordinary integer argument, has to become a field of a uniform buffer in one more binding slot.

Each of those is a *shape* constraint rather than a semantic one, and each is met by the same move: pack many things into few.

## 57.2 Intuition: packing, three times

By the time this chapter's backend runs, the compiler has already solved a packing problem twice. [Chapter 50](../../part9/ch50-arena-allocation/README.md) packed buffers into one arena by their live intervals. Chapter 53's `memoryLayout` packed them again, more crudely, for WebAssembly's flat memory.

This backend does it three more times, at three granularities, for two different reasons:

- **Across the binding table**, because there are too many buffers and too few slots. Buffers of the same element type and access mode are concatenated into one storage array, each keeping a base offset. This one is not about saving memory at all; it is about fitting an interface.
- **Over workgroup memory**, and again **over private arrays**, because the scratchpad is small and every array must be sized in the source. Both are Chapter 50's interval packing run a second and a third time, on different sets of buffers, by different code. A buffer whose accesses are entirely thread-local skips both and becomes a *scalar*.

So the same problem is now solved five times in one compilation, by five pieces of code that run at different times against different information and cannot see each other's answers. Hold onto that: §57.7 opens on a number that is wrong for exactly that reason.

## 57.3 Theory

### Packing a binding table

> **Definition 57.1 (Packed binding).** **(invariant)** Let *B* be the kernel's buffers, each with an element type and an access mode (read, or read-write). A *packing* assigns each buffer *b* a group key (its type and mode) and an offset *o(b)*, such that within a group the intervals `[o(b), o(b) + size(b))` are pairwise disjoint. Each group becomes one storage binding, and every access `b[i]` is emitted as `group[o(b) + i]`.

> **Proposition 57.2 (Packing preserves accesses).** **(stated here)** If every buffer in a group is statically sized and offsets are assigned by running concatenation with each size rounded up to a multiple of 4, then the intervals are disjoint and `group[o(b) + i]` names the same element `b[i]` did, for every in-bounds *i*.
>
> *Proof.* Running concatenation gives o(b₍ₖ₊₁₎) = o(bₖ) + align₄(size(bₖ)) ≥ o(bₖ) + size(bₖ), so intervals are disjoint by construction. Within an interval, `o(b) + i` is an affine relabelling of *i* and is injective. ∎

The two hypotheses are both load-bearing. **Statically sized** is why the packing is refused wholesale when any buffer has a dynamic extent — there is no offset to compute. **Same access mode** is why a read-only and a read-write buffer never share a group: WGSL's `var<storage, read>` and `var<storage, read_write>` are different types, and a kernel that wrote through a `read` binding would not compile.

### Scalarization

> **Definition 57.3 (Thread-private buffer).** **(invariant)** A kernel-local buffer *B* is *thread-private* in a kernel that is not serialized and has thread bindings when both of these hold:
>
> 1. **No load of *B* is indexed by a sequential loop variable.** Every read is at an index built only from thread indices and loop-invariant terms. (`loopCarriedIntermediates` collects the buffers that fail this.)
> 2. **Every load's thread multiplicity is one a store used.** The number of distinct thread-index values under which an element is read matches the number under which it was written. (`extentMismatchBuffers` collects the failures.)

> **Proposition 57.4 (A thread-private buffer needs one element per thread).** **(stated here)** Under Definition 57.3, for each thread the set of elements of *B* it reads is exactly the set it wrote, and no element it wrote is read by another thread. Replacing *B* with one scalar per thread — a WGSL `var` in function scope — therefore computes the same values, *provided each thread writes at most one element*.
>
> *Proof sketch.* Clause 1 says every read index is determined by the reading thread's own indices, so a thread cannot read an element whose index only some other thread's loop produced. Clause 2 says reads and writes agree on multiplicity, so a thread does not read an element written under a broader or narrower thread signature than its own. Together the read set and the write set of a thread coincide and are disjoint across threads, which is what a private scalar models. ∎

The proviso in italics is the part Definition 57.3 does **not** give you, and it is worth being precise about because the implementation approximates it. `scalarEligible` ([`webgpu/codegen.ts:661`](../../../src/backend/webgpu/codegen.ts)) tests the two clauses and then adds `buf.numel() <= totalThreads` — at most as many elements as threads. That is a *sizing* test, not a one-element-per-thread test: a four-element buffer in a 256-thread launch passes it, and if a single thread wrote all four the collapse would lose three of them. What rules that case out is clause 1, not the count — a thread writing four elements writes them from a sequential loop, and its reads would then be sequentially indexed too.

So the count is a cheap necessary condition that keeps the transformation off large buffers, and the soundness is entirely in the two access-pattern clauses. Reading it the other way round — as if `numel ≤ totalThreads` were the licence — is the mistake to avoid.

The saving is not a constant factor. An *n*-element array in workgroup memory becomes zero workgroup memory and one register.

### Interval packing, again

The workgroup pool is Chapter 50's problem, restated:

> **Definition 57.5 (Workgroup pool).** **(invariant)** Candidates are grouped by dtype. Within a group, each buffer gets a live interval — its first and last touch in a walk of the recurrence body, or the whole program for a buffer touched outside it. Buffers are sorted by first use, then by size descending, and each is placed at the lowest offset not occupied by an interfering buffer. The group's pool is one `var<workgroup>` array sized to the peak.

That is first-fit over intervals, which [Proposition 50.5](../../part9/ch50-arena-allocation/README.md) shows is valid and [Theorem 50.4](../../part9/ch50-arena-allocation/README.md) shows is not optimal. Chapter 50's argument transfers unchanged; only the units differ.

### Serialization

The decision tree of §56.3 runs here too, over the same `GpuRaceReason` table, with two differences in the inputs.

- The cross-thread test is `loopCarriedIntermediates` rather than `threadSharedIntermediates`, because WebGPU's failure mode arrives most often through a recurrence.
- There is one reason CUDA does not have: `RECURRENCE_EXCEEDS_WORKGROUP`. A sequential recurrence — an RNN step, a scan — can be run inside *one* workgroup with barriers between steps, and cannot be run across workgroups at all. So the backend checks whether the whole recurrence fits one workgroup's threads and one workgroup's memory, and serializes when it does not.

### Literals

> **Definition 57.6 (Representable literal).** **(classical)** WGSL requires every floating-point literal to denote a finite value representable in its type. There is no `inf` or `nan` literal syntax, and a constant expression that would overflow is a shader-creation error rather than an infinity.

That is a constraint on the *text*, and the backend meets it by emitting the largest finite `f32`, `0x1.fffffep+127`, where the program said infinity. On finite inputs nothing changes; §57.5 shows the input on which something does.

## 57.4 In mlfw: 973 lines, mostly about fitting

`WebGPUCodegen.generate` ([`webgpu/codegen.ts:151`](../../../src/backend/webgpu/codegen.ts)) runs the race analysis first, exactly as CUDA does, and then spends most of its length on the three placements of §57.2.

### The binding table

```ts
    const bufCount = func.bufferMap.size + (func.shapeParams.length > 0 ? 1 : 0);
    const bufList = [...func.bufferMap.values()];
    const canPack = bufCount > 6 && bufList.every(b => b.numel() > 0);
```

([`webgpu/codegen.ts:198`](../../../src/backend/webgpu/codegen.ts).) Proposition 57.2's two hypotheses, plus a threshold of 6 that leaves headroom for the uniform binding and for whatever the device implementation wants. When it fires, buffers are grouped by `(isWrite, wgslType)` and offsets accumulate with `align4` ([`webgpu/codegen.ts:213`](../../../src/backend/webgpu/codegen.ts)).

Shape parameters become a `struct ShapeParams` in a uniform binding, and `_resolveShapeParam` ([`webgpu/codegen.ts:970`](../../../src/backend/webgpu/codegen.ts)) renders every use as `i32(_shapes.name)`.

### Four ways to name a buffer

`_packedBufAccess` ([`webgpu/codegen.ts:947`](../../../src/backend/webgpu/codegen.ts)) is where all of it lands:

```ts
  _packedBufAccess(bufName: string, indexExpr: string): string {
    if (this._wgPoolOffsets && this._wgPoolOffsets.has(bufName)) {
      const info = this._wgPoolOffsets.get(bufName) as PoolRef;
      if (info.offset === 0) return `${info.pool}[${indexExpr}]`;
      return `${info.pool}[${info.offset}u + u32(${indexExpr})]`;
    }
    if (this._packedMode && this._packedOffsets && this._packedOffsets.has(bufName)) {
      const info = this._packedOffsets.get(bufName) as PackedRef;
      if (info.offset === 0) return `${info.storage}[${indexExpr}]`;
      return `${info.storage}[${info.offset}u + u32(${indexExpr})]`;
    }
```

— and then two more branches, for a private slot and for the buffer's own name. **Four levels of indirection between a name in the IR and a name in the emitted shader, resolved at every access**, with a special case at each level for offset zero so the common access stays short.

### Private slots and scalars

Forty lines of `_assignLocalSlots` ([`webgpu/codegen.ts:643`](../../../src/backend/webgpu/codegen.ts)) do at codegen time what Chapters 49 and 50 did over the whole program:

```ts
      let heap = freeByDtype.get(buf.dtype);
      if (!heap) { heap = new MinHeap<HeapSlot>((a, b) => a.freeAt - b.freeAt); freeByDtype.set(buf.dtype, heap); }

      let slot: HeapSlot | null = null;
      const top = heap.peek();
      if (top && top.freeAt < mn) slot = heap.pop();
```

Buffers are visited in order of first use; a slot whose last use is before this buffer's first use is reused, otherwise a new one is declared. One min-heap per dtype, keyed on when each slot frees. That is the classic left-endpoint colouring of an interval graph, and it is optimal in the **number** of slots — the count it produces equals the largest number of buffers live at once, which no assignment can beat.

It is not optimal in bytes, and the line that gives that away is `if (size > slot.decl.size) slot.decl.size = size;` ([`webgpu/codegen.ts:687`](../../../src/backend/webgpu/codegen.ts)): a reused slot grows to its largest occupant, so a one-element buffer sharing with a 512-element one costs 512. Minimizing bytes rather than slots is Chapter 50's problem, and [Theorem 50.4](../../part9/ch50-arena-allocation/README.md) says it is NP-hard. This backend is solving the easy version because it can: WGSL needs a *declaration count* it can write down, not a byte budget it has to pack.

A buffer that clears Definition 57.3's two clauses skips the heap entirely: `scalarEligible` sends it to a `_s` name that is declared as a bare `var`, never as an array.

### The workgroup pool

`_packWorkgroupPool` ([`webgpu/codegen.ts:515`](../../../src/backend/webgpu/codegen.ts)) implements Definition 57.5, and the clause worth reading is the lifetime:

```ts
      const carried = persistent.has(c.name) || !minPos.has(c.name);
      const fb = carried ? 0 : minPos.get(c.name) as number;
      const lb = carried ? Number.MAX_SAFE_INTEGER : maxPos.get(c.name) as number;
```

A buffer touched *outside* the recurrence body is carried across iterations, so its interval is the whole program and it interferes with everything. A buffer touched only inside gets its real interval and can share. That is [Lemma 49.4](../../part9/ch49-buffer-lifetimes/README.md) — a loop widens an interval — applied to the one loop that matters here.

## 57.5 Lab — eight bindings and a browser

```bash
node docs/part10/ch57-webgpu-and-wgsl/labs/01-eight-bindings-and-a-browser.mjs
```

Three buffers, three bindings:

```wgsl
@group(0) @binding(0) var<storage, read> buf_1: array<f32>;
@group(0) @binding(1) var<storage, read> buf_3: array<f32>;
@group(0) @binding(2) var<storage, read_write> buf_5: array<f32>;
```

Nine buffers, two bindings:

```wgsl
@group(0) @binding(0) var<storage, read> _pr_f32: array<f32>;
@group(0) @binding(1) var<storage, read_write> _pw_f32: array<f32>;
    _pr_f32: 8 buffers, 128 elements
      buf_1@0  buf_3@16  buf_5@32  buf_7@48  buf_9@64  buf_11@80  buf_13@96  buf_15@112

  and an access reads: _pw_f32[v0_37] = (_s0 + _pr_f32[112u + u32(v0_37)]);
  every intermediate collapsed to a per-thread scalar: _s0
```

That one line contains three of the chapter's mechanisms. `_pr_f32[112u + u32(v0_37)]` is Definition 57.1 — the eighth input at base 112. `_s0` is Proposition 57.4 — seven intermediates, each one element per thread, collapsed to a single scalar rather than seven arrays. And the `u32` cast is the type discipline WGSL insists on and the other three languages do not.

Then dynamic shapes, which have nowhere to live except a uniform:

```wgsl
struct ShapeParams {
  _ds_2: u32,
  _ds_3: u32,
  _ds_6: u32,
  _ds_7: u32,
}
@group(0) @binding(2) var<uniform> _shapes: ShapeParams;

  for (var i0_10: i32 = 0; i0_10 < i32(_shapes._ds_6); i0_10 = i0_10 + 1) {
```

A dynamic shape costs a binding slot, which interacts with the six-buffer packing threshold: a dynamically-shaped kernel is exactly the kernel that cannot be packed, and it is also the one that needs one more slot.

The third section is Part IX arriving for the third and fourth time:

```
  sm_p0    private slots: _lt0
           workgroup:     (none)
           declared workgroup bytes 0, reported sharedMemBytes 0
  sm_p1    private slots: (none)
           workgroup:     buf_17[512] buf_7[512] buf_6[1] buf_12[512] buf_22[8] buf_29[512]
           declared workgroup bytes 8228, reported sharedMemBytes 0
```

Six workgroup arrays, 8,228 bytes — **and a reported `sharedMemBytes` of 0**. §57.7 opens on why the two numbers are allowed to disagree.

Then the literal:

```
  cuda     buf_4[0] = (-INFINITY);
  webgpu   _lt0[0] = f32(-0x1.fffffep+127);
```

A max-reduction's identity. On finite inputs the two agree exactly, because every finite `f32` is greater than −3.4 × 10³⁸ and less than +∞ alike. On an input that *already contains* −∞ — which is how every causal attention mask in every transformer is written — they do not: CUDA's max propagates the infinity and WebGPU's returns the clamp.

And finally the same give-up decision as Chapter 56, reached over the same reason table:

```
  graph        kernel     threads  diagnosis
  elementwise  ew            4096  -
  softmax      sm2_p0           8  -
  softmax      sm2_p1           1  kernel-local buffer read by a thread that did not write it
  layer_norm   ln_p0            8  -
  layer_norm   ln_p1            1  kernel-local buffer read by a thread that did not write it
  matmul       mm            4096  -
```

Unlike Chapter 56's table, everything here *was* scheduled — `WebGPUTarget` declares `{ enabled: true }` — so the kernels that end at one thread ended there because the backend refused, not because nothing asked. Which is the better failure of the two, and is still invisible.

**Try this.** Raise the softmax to 64 rows and watch the split change: with more work per row the middle kernel's intermediate no longer fits, and the reason changes from a thread-shared intermediate to an extent mismatch. The reason table has five entries and a given program reaches at most one; sweeping a size is the quickest way to see three of them.

## 57.6 Where this was verified

WGSL emission and the launch metadata are checked on every test run, without a device, because they are properties of the text. Execution is not: `webgpu` in Node is a Dawn binding that is not stable enough to run in the test suite, so the executing tests drive a real Chrome through Puppeteer and skip when one is not available. The browser-bundle check is gated further still — it needs a prior `npm run build`, and it verifies both that the eager and compiled paths run and that no Node builtin leaked into the bundle.

So, honestly: the packing, the slot assignment, the scalarization and the serialization decisions in this chapter are verified continuously; the numbers those shaders produce are verified in a browser, on demand.

## 57.7 Traps and limits

### The workgroup memory a kernel reports is not the workgroup memory it declares

`sharedMemBytes` is computed as `this._sharedBuffers.reduce(...)` ([`webgpu/codegen.ts:338`](../../../src/backend/webgpu/codegen.ts)) — the buffers that arrived already in shared scope — and neither `_promotedBufferDecls` nor `_wgPoolDecls` is included. §57.5 measures a kernel that declares 8,228 bytes of `var<workgroup>` and reports 0. CUDA's equivalent sums both its lists ([`cuda/codegen.ts:208`](../../../src/backend/cuda/codegen.ts)).

Nothing breaks today, because WGSL workgroup arrays are statically sized and the pipeline does not need the figure at launch. What breaks is every consumer of it: a device-limit check, an occupancy estimate, a cost model that wants to know how much scratchpad a candidate schedule uses. The number is reported, it is wrong, and it is wrong in the direction that makes a kernel look cheaper than it is.

### The rest

- **The packing threshold is a constant, not a device property.** `bufCount > 6` ([`webgpu/codegen.ts:200`](../../../src/backend/webgpu/codegen.ts)) is a bare 6. `maxStorageBuffersPerShaderStage` is a real limit a device reports, and `TargetFeatures` has no field for it, so a device that offers four is over-subscribed and one that offers sixteen is packed unnecessarily — and packing is not free: every access gains an offset and a `u32` cast.
- **A dynamically-shaped kernel cannot be packed at all.** `canPack` requires `numel() > 0` for every buffer, so one dynamic dimension anywhere refuses the packing for the whole kernel — and adds a uniform binding on top. The combination that most needs packing is exactly the one that cannot have it.
- **The workgroup pool has a budget and the promotion list does not.** The recurrence branch checks `smemUsed + pool.bytes <= smemLimit` before committing ([`webgpu/codegen.ts:441`](../../../src/backend/webgpu/codegen.ts)); the fall-through branch adds candidates one at a time while they fit ([`webgpu/codegen.ts:457`](../../../src/backend/webgpu/codegen.ts)) and **silently drops the ones that do not**, leaving those buffers unpromoted in a kernel whose barriers were enabled on the assumption that they would be. There is no error and no diagnosis for that case.
- **`_livenessWalk` counts node visits, not statements.** Positions come from a pre-order counter over the whole tree ([`webgpu/codegen.ts:703`](../../../src/backend/webgpu/codegen.ts)), so an interval's endpoints are node indices rather than program points, and a buffer whose access sits inside a deep expression gets a later position than one at the same statement in a shallow expression. It is monotone, so the ordering is sound and the reuse is valid; it is not the linearization [Definition 49.1](../../part9/ch49-buffer-lifetimes/README.md) describes, and two analyses in the same compiler now mean different things by "position".
- **The workgroup size is halved until it fits.** When the bindings imply more threads than `maxThreadsPerBlock`, the largest axis is right-shifted repeatedly ([`webgpu/codegen.ts:286`](../../../src/backend/webgpu/codegen.ts)) rather than throwing as CUDA does. The result is a launch that covers fewer indices than the loops asked for, and the guard that would make that safe is the one CUDA emits and this backend does not have an equivalent test for.
- **`f16` is enabled by scanning the buffer dtypes only.** `_checkF16Usage` ([`webgpu/codegen.ts:345`](../../../src/backend/webgpu/codegen.ts)) looks at `func.bufferMap`, so a kernel that computes in `f16` in a local without any `f16` parameter does not get the `enable f16;` directive and fails to compile in the browser.
- **`bf16` is four bytes here and two bytes everywhere else.** WGSL has no `bf16`, so the WGSL type table maps it to `f32` at width 4 ([`dtype_map.ts:252`](../../../src/util/dtype_map.ts)) while every other layer — including [Chapter 50](../../part9/ch50-arena-allocation/README.md)'s memory planner, which sized the buffer — reads 2. The widening is silent and it is the one place in Part X where a dtype changes a buffer's *size* rather than the width it computes in. Chapter 54 §54.3 has the storage-versus-compute split this falls out of.
- **`fmod` is WGSL's `%`.** Chapter 53 §53.6 covers it: on floats WGSL's `%` truncates, so this backend disagrees with eager and with the CPU backend on a negative dividend.

## 57.8 Read the tests

- [`tests/backend/webgpu/codegen.test.js`](../../../tests/backend/webgpu/codegen.test.js) — the emitted WGSL: binding declarations and access modes, the packed form and its offsets, the `ShapeParams` struct, slot assignment and scalarization, and the workgroup pool.
- [`tests/backend/webgpu/compile.test.js`](../../../tests/backend/webgpu/compile.test.js) — end to end through the pipeline, checking the metadata the runtime binds against.
- [`tests/backend/webgpu/scan-parallel.test.js`](../../../tests/backend/webgpu/scan-parallel.test.js) — the recurrence path: that a scan that fits one workgroup keeps its barriers and one that does not is serialized with `RECURRENCE_EXCEEDS_WORKGROUP`.
- [`tests/backend/webgpu/exec.test.js`](../../../tests/backend/webgpu/exec.test.js) — the executing half, driven through a real browser and skipped without one.

---

**Next:** [Chapter 58 — Calling someone else's kernel](../ch58-someone-elses-kernel/README.md), and the case where the best code a backend can generate is none.
