# Chapter 59 — The runtime module

[Part X](../../part10/README.md) ended with an object: a `CompiledKernel` holding a name, a source string, a target and a metadata record. That object is inert. Something has to turn the string into something a machine will execute, find memory for every buffer it names, put the arguments in the order it expects, and call it.

That something is 410 lines in [`src/runtime/runtime.ts`](../../../src/runtime/runtime.ts), and this chapter is about why it is only 410 lines — because the interesting part of the answer is what the runtime *refuses* to know.

## 59.1 The problem: four targets, one caller

Consider what "run this kernel" means on each of the four targets Part X produced.

On the CPU it means `new Function(source)` and call it, passing typed arrays. On WebAssembly it means assembling the text into bytes, instantiating a module, **copying every argument into the module's linear memory at the byte offsets codegen chose**, calling the export, and copying the results back out. On CUDA it means loading a module onto a device, allocating device memory, uploading, launching with the grid and block dimensions codegen chose, and synchronising. On WebGPU it means all of that plus building a bind group from a binding table, and it cannot be done synchronously at all.

Those four procedures share almost nothing. They do not take the same arguments, they do not have the same failure modes, and two of the four are asynchronous while the other two are not.

And yet the thing above them — the tracer's compiled function, the eager dispatcher's just-in-time path, the training loop — wants to say one sentence: *run the kernel called `mlp` on these tensors.* If the runtime does not absorb the difference, every caller absorbs it instead, and the framework grows a `if (target === 'cuda')` in a dozen places.

There is a second problem, and it only appears once a graph compiles to more than one kernel. A single kernel takes its buffers as arguments and the caller allocates them. A *sequence* of kernels has intermediate values that no caller ever sees: the output of step 0 that step 1 reads and nothing else ever looks at. Someone has to allocate those, and someone has to decide how many distinct allocations are actually needed — which is [Part IX](../../part9/README.md)'s question again, one level up.

## 59.2 Intuition: a dictionary of kernels and a table of slots

The runtime is two ideas.

**A runtime module is a dictionary from kernel name to compiled kernel, plus a lazily-filled dictionary from kernel name to *live instance*.** Asking for a kernel by name the first time instantiates it — compiles the JavaScript, assembles the WebAssembly, loads the CUDA module — and remembers the result. Asking again returns the same instance. Which of the four procedures runs is decided by one map lookup on a string in the metadata.

**A plan is a numbered set of slots.** Every value the compiled program touches — each input, each captured weight, each intermediate, the output — gets a slot number. A step says "read slots 3 and 1, write slot 4". The caller fills the argument slots; the runtime fills the rest; running the plan is walking the step list. Because the slots are numbered rather than named, deciding that two slots can share one allocation is just an array from slot number to buffer number.

The two ideas are independent, and that is the design: a kernel does not know whether it is part of a plan, and a plan does not know what kind of kernel a step holds.

## 59.3 Theory

> **Definition 59.1 (Runtime kernel).** **(invariant)** A *runtime kernel* is `{ name, source, target, metadata }` where `metadata.kind` is a string naming a runtime backend. Nothing else about the object is inspected outside that backend.

This is [Definition 53.7](../../part10/ch53-lir-the-third-ir/README.md)'s backend contract read from the other end. Part X promised the four fields; the runtime is the consumer that makes the promise worth making.

> **Definition 59.2 (Runtime backend).** **(invariant)** A backend is `{ instantiate(kernel), runSync(instance, tensorArgs, shapeValues), runAsync(instance, tensorArgs, shapeValues), isAsync(instance?, kernel?) }`, optionally with `runPlan(plan, slots, steps, opts)`. Backends live in a module-global map keyed by `metadata.kind`.

> **Proposition 59.3 (The runtime is target-agnostic).** **(invariant)** `runtime.ts` names no target: `cuda`, `wasm` and `webgpu` do not appear in it at all, and `js` appears only inside import specifiers. Every target-specific decision is reached through `getBackend(kernel.metadata.kind)`.
>
> *Why it holds.* Instantiation, execution and the sync/async question are all delegated ([`runtime.ts:215`](../../../src/runtime/runtime.ts), [`:267`](../../../src/runtime/runtime.ts), [`:287`](../../../src/runtime/runtime.ts)). The registry is populated by [`backend_registry.ts`](../../../src/runtime/backend_registry.ts), a *different* module, which is where the four target-specific procedures live.
>
> *Consequence.* Adding a fifth execution target is one `registerBackend` call and changes no existing file. §59.5's first lab does exactly that, in eight lines.

> **Definition 59.4 (Execution plan).** **(invariant)** A plan is `{ numSlots, argSlots, intermediates, steps, returnFixups?, buffers? }`. `argSlots[i]` is the slot the caller's *i*-th argument occupies; `intermediates` lists the slots the runtime must allocate, with a shape and a dtype; each step names a kernel and the slots it reads and writes.

The separation between `argSlots` and `intermediates` is the whole ownership rule: **a slot is either the caller's or the runtime's, never both.**

> **Definition 59.5 (Slot colouring).** **(stated here)** A *colouring* of a plan is a map `slotBuffer: slot → buffer` together with `bufferBytes: buffer → size`. It is *valid* when, for every pair of slots `s ≠ t` with `slotBuffer[s] = slotBuffer[t]`, the live intervals of `s` and `t` are disjoint, and `bufferBytes[slotBuffer[s]] ≥ bytes(s)` for every slot.

This is [Definition 49.1](../../part9/ch49-buffer-lifetimes/README.md)'s live interval and [Definition 50.1](../../part9/ch50-arena-allocation/README.md)'s assignment, with the *step index* playing the role the linearised statement index played there. The difference is granularity and nothing else: Part IX coloured buffers inside one kernel; this colours slots across kernels.

> **Theorem 59.6 (A donation is a colouring one step earlier).** **(stated here)** Let step *k* read slot *f* for the last time and define slot *t*. Under the free-list discipline of [`plan_buffer_assignment.ts:142`](../../../src/compiler/passes/memory/plan_buffer_assignment.ts) — where a slot's buffer is returned to the free list at the *start of step k+1* — slots *f* and *t* may not share a buffer. A *donation* is the additional licence to give *t* the buffer of *f* at step *k* itself.
>
> *Proof sketch.* The intervals `[def(f), k]` and `[k, lastUse(t)]` overlap at *k*, so Definition 59.5 forbids sharing. The overlap is only real if step *k* reads *f* after writing *t*; if the kernel provably reads each element of *f* before writing the corresponding element of *t*, they can share. Establishing that is exactly [Theorem 51.3](../../part9/ch51-inplace-and-donation/README.md)'s index-equality argument, lifted from elements to whole buffers. ∎
>
> *Consequence.* Element-transparency in that operand — the operand's element *i* determines the output's element *i*, and nothing else does — is enough to make a donation safe, and it is the condition the implementation tests. It is sufficient and not necessary: a kernel that reads its operand entirely before writing anything is safe too, and is refused. §59.5 measures what that buys and what it does not.

> **Definition 59.7 (Uniform plan backend).** **(invariant)** A plan has a *uniform backend* when every one of its steps has already been instantiated and all instances came from the same backend object. Only then may the runtime hand the whole plan to `backend.runPlan` instead of walking the steps itself ([`runtime.ts:331`](../../../src/runtime/runtime.ts)).

That check is what lets the CUDA backend keep every intermediate on the device for the whole plan instead of round-tripping each one through host memory — and, being a check rather than an assumption, it degrades to the step-by-step loop rather than failing when a plan is mixed.

## 59.4 In mlfw: loading, calling, and where the arguments come from

### The registry, and the one line that makes the runtime portable

```ts
  private _createInstance(name: string): KernelInstanceEntry {
    const kernel = this.kernels.get(name);
    if (!kernel) throw new Error('Kernel \'' + name + '\' not found');
    const backend = getBackend(kernel.metadata.kind) as RuntimeBackend | null;
    if (!backend) throw new Error('No runtime backend registered for kind: ' + kernel.metadata.kind);
    const entry = { backend, instance: backend.instantiate(kernel) };
    this._instances.set(name, entry);
    return entry;
  }
```

([`runtime.ts:212`](../../../src/runtime/runtime.ts).) Seven lines, and Proposition 59.3 lives in the third of them. Note that `instance` may be a `Promise` — `instantiate` is allowed to be asynchronous, and the WebGPU backend's is, because acquiring a device is.

The four registrations are the target-specific half. The CPU one is two real methods — `runAsync` repeats `runSync` verbatim and `isAsync` returns `false`:

```ts
registerBackend('js', {
  instantiate(kernel: RuntimeKernel): JsKernelFn {
    return new Function('return ' + kernel.source)() as JsKernelFn;
  },
  runSync(fn: JsKernelFn, tensorArgs: readonly NumericTypedArray[], shapeValues: readonly number[] | null): unknown {
    const callArgs = shapeValues ? [...tensorArgs, ...shapeValues] : tensorArgs;
    return fn(...callArgs);
  },
```

([`backend_registry.ts:141`](../../../src/runtime/backend_registry.ts).) The WebAssembly one is the interesting contrast, because it has to bridge two memory models:

```ts
  for (let i = 0; i < nBufs; i++) {
    const data = tensorArgs[i];
    if (ArrayBuffer.isView(data)) {
      new (data.constructor as WasmBufferViewCtor)(memory.buffer, offsets[i], data.length).set(data);
    }
  }
```

([`backend_registry.ts:117`](../../../src/runtime/backend_registry.ts).) `offsets` comes from `metadata.bufferOffsets`, which [Chapter 55](../../part10/ch55-webassembly/README.md)'s flat placement produced. The arguments are copied *in*, the export is called with the offsets as `i32` arguments, and the results are copied back *out*. A WebAssembly kernel call is three loops, only one of which is the kernel.

### Argument order is a convention, and this is where it is written down

```ts
  _prepareArgs(name: string, args: readonly RuntimeArg[]): { tensorArgs: unknown[]; shapeValues: number[] | null } {
    const tensorArgs: unknown[] = [];
    const tensorShapes = new Map<number, number[]>();
    for (let i = 0; i < args.length; i++) {
      const arg = args[i];
      if (arg instanceof RuntimeTensor) {
        tensorArgs.push(arg.data);
        tensorShapes.set(i, arg.shape);
      } else {
        tensorArgs.push(arg);
      }
    }
    const shapeParamMap = this._shapeParamMaps && this._shapeParamMaps.get(name);
    let shapeValues = null;
    if (shapeParamMap && shapeParamMap.size > 0) {
      const bufferMap = this._bufferMaps && this._bufferMaps.get(name);
      shapeValues = RuntimeModule._extractShapeParams(shapeParamMap, tensorShapes, args, bufferMap);
    }
    for (const cb of constBuffersOf(this.kernels.get(name))) tensorArgs.push(cb.data);
    return { tensorArgs, shapeValues };
  }
```

([`runtime.ts:239`](../../../src/runtime/runtime.ts).) Three kinds of argument, in a fixed order: **the caller's tensors, then the constant buffers, then the resolved symbolic dimensions.** The order is split across two places and neither states it — `_prepareArgs` appends the constant buffers to `tensorArgs` on its last line, and each backend then concatenates `shapeValues` after the whole of `tensorArgs` ([`backend_registry.ts:146`](../../../src/runtime/backend_registry.ts)). The constant buffers are [Chapter 61](../ch61-tracing/README.md)'s folded weights arriving — data that codegen chose to link into the kernel rather than take as a parameter, appended after everything the caller passed. The trailing integers are [Chapter 62](../ch62-dynamic-shapes/README.md)'s subject: a kernel compiled for a symbolic dimension takes that dimension's *value* as an extra argument, and the runtime reads it off the shape of whichever tensor the compiler said it came from. §62.5's `function Object(buf_1, buf_4, buf_6, _ds_2)` is the same convention read off an emitted signature.

### Allocating a plan

```ts
function _allocIntermediates(plan: ExecutionPlan, slots: Array<RuntimeTensor | null>): void {
  const grouping = plan.buffers ? plan.buffers.slotBuffer : null;
  const stores: Array<ArrayBuffer | null> = [];
  if (grouping) {
    const sizes: number[] = [];
    for (const it of plan.intermediates) {
      const group = grouping[it.slot];
      const need = _intermediateNumel(it) * typedArrayCtor(it.dtype).BYTES_PER_ELEMENT;
      if (!(sizes[group] >= need)) sizes[group] = need;
    }
    for (let i = 0; i < sizes.length; i++) stores[i] = sizes[i] > 0 ? new ArrayBuffer(sizes[i]) : null;
  }
  for (const it of plan.intermediates) {
    const ctor = typedArrayCtor(it.dtype);
    const numel = _intermediateNumel(it);
    const store = grouping ? stores[grouping[it.slot]] : null;
    slots[it.slot] = new RuntimeTensor(store ? new ctor(store, 0, numel) : new ctor(numel), it.shape, it.dtype);
  }
}
```

([`runtime.ts:89`](../../../src/runtime/runtime.ts).) This is where a colouring becomes bytes, and the shape of the code is the point: **with `plan.buffers`, one `ArrayBuffer` per colour and a typed-array view per slot; without it, one fresh array per slot.** [Chapter 51](../../part9/ch51-inplace-and-donation/README.md) spent its length on the gap between a plan that records a saving and a program that takes one; here the two are the same eleven lines, and the lab measures the allocation rather than the report.

### The colouring itself

[`plan_buffer_assignment.ts`](../../../src/compiler/passes/memory/plan_buffer_assignment.ts) is 263 lines and runs as the last phase of compilation, gated on there being a plan at all:

```ts
      {
        name: 'planBufferAssignment',
        when: (ctx: CompileContext) => ctx.compiler.config.memory.planReuse !== false && !!(ctx.split && ctx.split.plan),
        run: (ctx: CompileContext) => ctx.compiler._assignPlanBuffers(ctx),
      },
```

([`compiler.ts:384`](../../../src/compiler/pipeline/compiler.ts).) The allocator is a linear scan with a free list:

```ts
  const freeList: number[] = [];
  const take = (size: number, owner: number): number => {
    let fit = -1;
    let fitBytes = Infinity;
    let largest = -1;
    let largestBytes = -1;
    for (let i = 0; i < freeList.length; i++) {
      const b = bufferBytes[freeList[i]];
      if (b >= size && b < fitBytes) { fitBytes = b; fit = i; }
      if (b > largestBytes) { largestBytes = b; largest = i; }
    }
    const pick = fit >= 0 ? fit : largest;
    if (pick < 0) return newBuffer(size, owner);
```

([`plan_buffer_assignment.ts:121`](../../../src/compiler/passes/memory/plan_buffer_assignment.ts).) Best fit, falling back to *grow the largest* rather than allocate a new one — [Definition 50.6](../../part9/ch50-arena-allocation/README.md)'s two policies, with a third behaviour stitched on: when nothing fits, the allocator would rather enlarge an existing buffer than add one. That choice is why the buffer *count* in §59.5's table is so often 2 while the *bytes* differ by 1.6×.

And the donation test, which is Theorem 59.6's element-transparency made concrete:

```ts
function passThroughResults(user: Operation, operandIndex: number, valueType: TensorType): Value[] | null {
  const hasRegions = user.regions && user.regions.length > 0;
  if (isElementwiseOp(user.opName) && !hasRegions) {
```

([`plan_buffer_assignment.ts:188`](../../../src/compiler/passes/memory/plan_buffer_assignment.ts).) `computePlanDonations` walks forward from each of a step's function arguments; the walk survives elementwise operations of the same layout and stops at the terminator, and the donation is recorded only if the argument reaches return position 0. An operation that is not elementwise — a `matmul`, a reduction, a reshape — ends the walk with `null` and the donation is refused.

## 59.5 Lab — four fields, a registry, and a table of slots

```bash
node docs/part11/ch59-the-runtime-module/labs/01-four-fields-and-a-registry.mjs
```

The same three-operation graph, compiled for all four targets and handed to a `RuntimeModule`:

```
  target           kind      source async  metadata
  cpu_generic      js           626 false  paramCount
  wasm_generic     wasm        4467 false  memoryPages, bufferOffsets, imports, params, bufferMap
  cuda_generic     cuda         904 true   blockDim, gridDim, sharedMemBytes, params, outputIndices, scratch, launchDiagnosis
  webgpu_generic   webgpu      1150 true   workgroupSize, dispatchSize, sharedMemBytes, params, bindings, launchDiagnosis
```

That is [Chapter 58 §58.6](../../part10/ch58-someone-elses-kernel/README.md)'s metadata table, read back through the interface that consumes it. Nothing in `runtime.ts` knows what `blockDim` is; the CUDA backend does.

Loading and calling one by hand is four lines, and the second call is roughly four times cheaper than the first because instantiation happened once:

```
  in  [0..6]        -3 -2 -1 0 1 2
  out [0..6]        0 0 0 1 3 5
  first run 0.089ms   second run 0.023ms
```

Then the module is serialised to a plain object, revived, and run again to the same numbers — which is what makes ahead-of-time compilation possible at all: the compiler's output survives a JSON round trip, because it is a string and a record.

Finally the lab registers a **fifth backend** in about eight lines, one that wraps the CPU one and counts launches, and routes a kernel through it by changing one string:

```
  launches counted  3
  same numbers      true
```

No file under `src/` was touched.

```bash
node docs/part11/ch59-the-runtime-module/labs/02-slots-and-the-plan.mjs
```

A three-matmul chain, compiled with a graph-split threshold so it becomes a plan:

```
  7 slots, 5 steps, 4 intermediates
  argSlots [0,1,6]  (inputs, then captured weights, then the output)

  step   kernel       in -> out
  0      Object_p0    [0,1] -> [2]
  1      Object_p1    [2] -> [3]
  2      Object_p2    [3,1] -> [4]
  3      Object_p3    [4] -> [5]
  4      Object_p4    [5,1] -> [6]

  slot  bytes  buffer
     0      0       0   (argument — pinned)
     1      0       1   (argument — pinned)
     2    512       3
     3    512       3
     4    512       4
     5    512       4
     6      0       2   (argument — pinned)
```

Seven slots, four of which carry bytes, coloured with two buffers. The pinned slots have `bytes 0` because they are the caller's — the plan records their existence and not their size.

Then the three configurations, on three graph shapes:

```
  graph                 configuration                          bytes  buffers  donated
  a chain of six        planReuse + planDonation (default)      1024        2        5
  a chain of six        planReuse only                          1024        2        0
  a chain of six        neither                                 5120       10        0
  a widening chain      planReuse + planDonation (default)      2560        2        2
  a widening chain      planReuse only                          4096        2        0
  a widening chain      neither                                 5120        4        0
  two branches joined   planReuse + planDonation (default)       512        1        1
  two branches joined   planReuse only                          1024        2        0
  two branches joined   neither                                 1024        2        0
```

Every row computes the same numbers to `0e+0`, which is the property that makes the comparison meaningful.

Read the three graphs against each other and each mechanism's contribution separates cleanly.

**On the chain, reuse is worth 5× and donation is worth nothing.** Ten intermediates, two live at a time, so the free list alone reaches the minimum; the five donations it records change *which* slot lands in which buffer and not how many buffers there are. This is Theorem 59.6 being true and not binding: reuse-one-step-later is enough when the step after next needs a buffer anyway.

**On the widening chain, donation is worth 1.6× and reuse alone cannot get there.** The intermediates are not the same size — a `[8,64]` product between two `[8,16]` ones — so the free list's *grow the largest* fallback inflates a buffer that donation would have avoided.

**On the two-branch graph, donation halves it and reuse alone does nothing at all.** Both branches are live simultaneously, so nothing expires early enough for the free list to recycle; the `add` that joins them is elementwise, so its output may take one of its inputs' buffers directly. One buffer instead of two.

And the trace reports the two columns Chapter 51 insisted on:

```
   {"slotBytes":2048,"bufferBytes":1024,"buffers":5,"donated":2}  over 7 slots
```

`slotBytes` is what the slots would need with no sharing; `bufferBytes` is what the program allocates. Both are computed from the same colouring the runtime consumes, so unlike Chapter 51's case there is no gap between the two to look for.

**Try this.** Set `graphSplit` to `{ matmul: 1 }` and watch a two-matmul graph become a plan as well; then set it to `{ matmul: 99 }` and watch `executionPlan` become `undefined` and the whole graph compile to a single kernel — at which point the slot machinery does not run at all, and [Part IX](../../part9/README.md)'s intra-kernel allocator is the only one working.

## 59.6 What the runtime deliberately does not know

Three absences are worth naming, because each is a decision rather than an omission.

**It does not know shapes.** `RuntimeTensor` carries a shape and strides, but no code path in `runtime.ts` checks that a tensor's shape matches what the kernel expects. The kernel was compiled for a shape; passing a different one produces whatever the flat indexing does. The checking happens two levels up, in the guard evaluation of [Chapter 62](../ch62-dynamic-shapes/README.md), and the runtime is downstream of a decision already made.

**It does not know dtypes, except to size an allocation.** Every use of a dtype in the file ends at `typedArrayCtor`: it chooses an array constructor, and is never compared against what the kernel declared. A kernel that reads its buffer as `f32` and was handed an `i32` array is a type error nothing in this file will notice.

**It does not know about gradients, parameters, or models.** A weight and an activation are both slots. This is why the same 410 lines serve inference, training, and the eager just-in-time path without a flag distinguishing them.

The cost of all three is that a mistake upstream surfaces here as a wrong number rather than an exception. The benefit is that the runtime has no policy to get wrong, and [Part XII](../../OUTLINE.md) can test the layers above it without mocking this one.

## 59.7 Traps and limits

### The colouring is discarded whole, or kept whole

```ts
  return bufferBytes.length < plan.numSlots ? { slotBuffer, bufferBytes, donated } : null;
```

([`plan_buffer_assignment.ts:169`](../../../src/compiler/passes/memory/plan_buffer_assignment.ts).) The assignment is returned only if it produced strictly fewer buffers than slots, and `null` otherwise — and `_assignPlanBuffers` then silently leaves `plan.buffers` unset ([`compiler.ts:569`](../../../src/compiler/pipeline/compiler.ts)). That is a reasonable guard against paying for a colouring that saves nothing, and it has two consequences. A plan in which every slot is pinned computes a full assignment and throws it away with no trace event, so "the pass found nothing" and "the pass did not run" are indistinguishable downstream. And the test is on the buffer *count* rather than the byte *total*, so a colouring that merged two slots into one grown buffer larger than both — which the `take` fallback can produce — passes the test while being no better in bytes.

### `runPlan` requires uniformity *and* an absence of scratch

```ts
    if (planBackend && planBackend.runPlan && !anyScratch) {
```

([`runtime.ts:309`](../../../src/runtime/runtime.ts).) The device-resident path is refused for a plan in which *any* step declares scratch memory, because scratch is allocated per launch by the step-at-a-time path and `runPlan` has no place to put it. One step needing 4 bytes of scratch demotes an entire plan to host round trips, and nothing reports it. The condition also requires every step to be *already instantiated* ([`:334`](../../../src/runtime/runtime.ts)) — which `runPlanAsync` arranges just above — so a future caller that reaches `_uniformPlanBackend` before instantiation gets `null` and the slow path, correctly but silently.

### Shape parameters are resolved by a positional guess when the buffer map is absent

```ts
      if (resolved === null) {
        for (const [, shape] of tensorShapes) {
          if (dimIdx < shape.length && shape[dimIdx] > 0) {
            resolved = shape[dimIdx];
            break;
          }
        }
      }
      result.push(resolved !== null ? resolved : 1);
```

([`runtime.ts:367`](../../../src/runtime/runtime.ts).) The intended path looks the buffer up by name and reads the named dimension off the right argument. The fallback takes *the first tensor argument that has a dimension at that index*, which is correct only when every buffer agrees on that dimension, and the final fallback is the literal `1`. Neither reports anything. A kernel whose shape parameter came from argument 2's dimension 0 and whose `bufferMap` was not registered will silently receive argument 0's dimension 0 — and if no argument has one, will loop once.

### A backend can be replaced but not removed, and instances outlive replacement

`registerBackend` overwrites by key and there is no `unregisterBackend` — the same asymmetry [Chapter 58 §58.7](../../part10/ch58-someone-elses-kernel/README.md) found in the codegen registry, in the module next door. Worse for testing: `_instances` caches the `{ backend, instance }` pair, so a module that has already run a kernel keeps calling the *old* backend after a replacement. There is no `RuntimeModule` API to drop an instance.

### `run` and `runAsync` disagree about what "async" means

`run` throws when the *instance* is a `Promise` ([`runtime.ts:263`](../../../src/runtime/runtime.ts)) — that is, when instantiation was asynchronous. `isAsync` asks the backend whether *execution* is asynchronous ([`:277`](../../../src/runtime/runtime.ts)). The CUDA backend instantiates synchronously (it just boxes the kernel) and executes asynchronously, so `run` on a CUDA kernel does not throw; it reaches `runSync`, which throws a *different*, more useful error about the runtime not being preloaded. Two mechanisms answering nearly the same question, and the one a caller is likely to consult first is not the one that guards the call.

### `RuntimeTensor.strides` is carried and never used

The constructor computes default strides or accepts them ([`runtime.ts:130`](../../../src/runtime/runtime.ts)), and `get`/`set` use them — but nothing in the runtime's execution path calls `get` or `set`, and `_prepareArgs` passes `arg.data`, the flat array, ignoring strides entirely. Every kernel this compiler emits assumes row-major contiguity, so a `RuntimeTensor` built with non-default strides would be read as though it had default ones. The field is honest about what it holds and silent about the fact that no consumer respects it.

## 59.8 Read the tests

- [`tests/runtime/runtime.test.js`](../../../tests/runtime/runtime.test.js) — the serialisation round trip that pins Definition 59.1 (a compiled module survives `JSON.parse(JSON.stringify(...))` and runs to the same numbers), and `registerBackend` used from outside `src/` exactly as §59.5 uses it.
- [`tests/compiler/passes/memory/plan-buffer-assignment.test.js`](../../../tests/compiler/passes/memory/plan-buffer-assignment.test.js) — pinning, live ranges over steps, the free list, donation, and the "fewer buffers than slots or nothing" rule.
- [`tests/e2e/cuda/plan-buffer-assignment.cuda.test.js`](../../../tests/e2e/cuda/plan-buffer-assignment.cuda.test.js) — the same colouring executed on a device, which is where a wrong one becomes a wrong number.

---

**Next:** [Chapter 60 — A PyTorch-style dispatcher](../ch60-a-pytorch-style-dispatcher/README.md), which answers the question this chapter left open: how does a call in user code find a kernel in the first place?
