# Chapter 60 — A PyTorch-style dispatcher

[Chapter 59](../ch59-the-runtime-module/README.md) answered *how a kernel runs*. This chapter answers the question underneath it: when a user writes `a.add(b)`, how does that call find a kernel at all?

The naive answer is a function that looks at the tensors and branches. It works for one backend. It stops working the moment the same call has to mean *four* different things depending on where the data lives, and then a fifth thing when a gradient is being recorded, and a sixth when a trace is being captured — because each of those is not an alternative to the others but a *layer wrapped around* them.

## 60.1 The problem: one call, several answers, in an order that matters

Take `a.add(b)` and count the things that can be true at once.

The data can be in host memory, on a CUDA device, in WebAssembly linear memory, or on a WebGPU device. That is four different kernels, and it is a genuine choice: exactly one applies.

Either operand may require a gradient. Then, *before* the arithmetic happens, a node has to be created in the autograd graph, edges wired to the operands' producers, and inputs saved for the backward pass — and *after* it, the result has to be marked as belonging to that node. That is not an alternative to running an `add`; it is a wrapper around running an `add`.

A trace may be in progress ([Chapter 61](../ch61-tracing/README.md)). Then no arithmetic should happen at all: the call must be *recorded* into a graph and a symbolic result returned. That is a wrapper that never reaches the inner call.

Mixed precision would be another such wrapper, casting operands before the call. Batching (`vmap`) would be another, rewriting one call into a looped one. Neither is implemented here, and both are pre-declared in the key list because the mechanism has room for them.

So the requirement is not "choose a kernel" but **compose an ordered stack of kernels, where each layer may do work, delegate to the rest of the stack, and do more work with the result** — with the composition decided per call, from the arguments, at no cost when no layer applies.

And there is a scale requirement hiding underneath. This is not one operator; it is 103, times nine keys, and every one of the resulting entries has to be looked up on every single eager call in a training loop.

## 60.2 Intuition: a bit set, and a table indexed by bit number

The mechanism is two data structures and one loop.

**Every tensor carries a set of dispatch keys.** A host tensor carries `CPU`. A tensor that requires a gradient carries `CPU` *and* `AUTOGRAD_CPU`. A symbolic tensor carries `TRACING` on top of whatever else it has. The set is a bit set in two 32-bit integers, so a union is an `or` and a membership test is an `and`.

**A call's key set is the union of its tensor arguments' key sets**, adjusted by any active guard. That union is the honest statement of "everything that might want to intervene in this call".

**Keys are numbered so that higher number means higher priority**, and dispatch is: take the highest-numbered key in the set, look up the kernel registered for that key on this operator, remove that key from the set, and call the kernel with the *remaining* set. A layer that wants to delegate calls back into the dispatcher with the set it was given. When the last layer is a backend kernel, the stack unwinds.

That is the whole idea. The layering is not a data structure; it is the consequence of "highest bit first, then drop it".

## 60.3 Theory

> **Definition 60.1 (Dispatch key, key set).** **(invariant)** A *dispatch key* is an integer in `[0, NUM_KEYS)`. A *key set* is a subset, held as two 32-bit words `(lo, hi)`, with `k ∈ S` iff bit `k mod 32` of the word `⌊k/32⌋` is set. `NUM_KEYS` is 49.

The numbering is deliberately sparse — backends at 0–8, functionality layers at 20–32, autograd at 40–43, tracing at 48 — so that a new key can be inserted in a band without renumbering the ones above it. The gaps cost nothing: the set is a bit set, not an array of keys.

> **Definition 60.2 (Call key set).** **(invariant)** For a call with argument list `A` and schema `σ`, the *call key set* is
> `K(A, σ) = ⋃ { keys(A[i]) : i ∈ σ.tensorArgIndices }`, recursively unioning through arrays, then rewritten by the guard stack.
>
> When no schema is available, the union runs over *all* arguments rather than the declared tensor positions.

> **Theorem 60.3 (Dispatch order is total and layer-composable).** **(stated here)** Let `S` be a key set and let `k = max S`. Dispatch calls the kernel registered at `k` with the residual set `S \ {k}`. Since each redispatch is handed a strictly smaller set, the sequence of keys visited by repeated redispatch is strictly decreasing and therefore terminates in at most `|S|` steps.
>
> *Proof sketch.* `highestPriority()` returns `63 − clz32(hi)` when `hi ≠ 0` and `31 − clz32(lo)` otherwise, which is the maximum element ([`dispatch_key.ts:135`](../../../src/dispatcher/dispatch_key.ts)); `_dispatchInternal` computes `keySet.without(key)` before calling ([`dispatcher.ts:101`](../../../src/dispatcher/dispatcher.ts)); a set of size *n* admits at most *n* strict decreases. ∎
>
> *The hypothesis this needs.* Termination requires that a layer redispatch with the *residual* set it was handed, not the original. A layer that redispatches with its own key still present loops forever, and nothing checks for it.

> **Definition 60.4 (Kernel table).** **(invariant)** A `KernelTable` is a dense array of length `NUM_KEYS`. Lookup is `_kernels[key]` — one array index, no hashing, no search.

The whole reason the keys are small integers rather than strings is this line. An operator entry is 49 slots, most of them `null`, and dispatch is an array read.

> **Definition 60.5 (Resolution order).** **(invariant)** For key `k` on operator `op`, the kernel is the first of: `op`'s table at `k`; the dispatcher's global fallback table at `k`; `op`'s catch-all. If all three are empty, dispatch throws ([`dispatcher.ts:117`](../../../src/dispatcher/dispatcher.ts)).

> **Definition 60.6 (Boxed and unboxed).** **(invariant)** An *unboxed* kernel takes `(keySet, ...args)` with arguments as native values. A *boxed* kernel takes `(keySet, stack)` where `stack` is an array of tagged `IValue`s. A `KernelFunction` may hold either or both; the missing convention is synthesised by an adapter.
>
> *Why both exist.* An unboxed kernel is fast and operator-specific. A boxed kernel is slow and *generic*: it can be written once and registered for every operator, because it does not need to know the argument list. A fallback — "on this key, do X to whatever this call is" — is only expressible boxed.

> **Definition 60.7 (Guard).** **(invariant)** The guard stack holds frames of `(exclude, include)`. `dispatch` rewrites the computed key set as `((S \ exclude₁) ∪ include₁) … ` over the frames, outermost last. `redispatch` does **not** consult the stack.

The asymmetry in the last sentence is load-bearing: if `redispatch` re-applied the guard, an included key would be re-added after the layer removed it and Theorem 60.3's termination argument would fail.

## 60.4 In mlfw: 179 lines of dispatcher and 43 of table

### The loop

```ts
  _dispatchInternal(handle: OperatorHandle, keySet: DispatchKeySet, args: DispatchArg[]): unknown {
    const key = keySet.highestPriority();
    if (key < 0) {
      throw new Error(`No dispatch key found for op '${handle.name}'`);
    }

    let kernel = handle.lookupKernel(key as DispatchKeyValue);
    if (!kernel) {
      kernel = this._fallbacks.lookup(key as DispatchKeyValue);
    }
    if (!kernel) {
      const catchAll = handle.entry.catchAll;
      if (catchAll) {
        kernel = catchAll;
      }
    }
    if (!kernel) {
      const explained = this._explainMissingKernel
        ? this._explainMissingKernel(handle, key as DispatchKeyValue)
        : null;
      throw new Error(
        explained || `No kernel registered for op '${handle.name}' with dispatch key ${key}`
      );
    }

    const remaining = keySet.without(key as DispatchKeyValue);
    return kernel.callUnboxed(remaining, ...args);
  }
```

([`dispatcher.ts:101`](../../../src/dispatcher/dispatcher.ts).) Definitions 60.5 and Theorem 60.3, verbatim. The two public entry points differ by one line — `dispatch` applies the guard stack, `redispatch` does not ([`:92`](../../../src/dispatcher/dispatcher.ts) and [`:97`](../../../src/dispatcher/dispatcher.ts)).

The failing branch is the only part of the loop that consults anything outside the dispatcher. A missing kernel is a fact about the *operator*, not about the key set, and §60.4's coverage pass is what knows why one is missing — so the dispatcher asks it for a sentence and falls back to the key number if nobody answered.

### Where a tensor's keys come from

```ts
const _AUTOGRAD_KEY_FOR_BACKEND: Readonly<Partial<Record<DispatchKeyValue, DispatchKeyValue>>> = Object.freeze({
  [DispatchKey.CPU]: DispatchKey.AUTOGRAD_CPU,
  [DispatchKey.GPU]: DispatchKey.AUTOGRAD_GPU,
  [DispatchKey.WASM]: DispatchKey.AUTOGRAD_WASM,
});
```

([`dispatch_key.ts:45`](../../../src/dispatcher/dispatch_key.ts).) A tensor's key set is its backend key, plus the matching autograd key when it requires a gradient. There are per-backend autograd keys rather than one so that an autograd kernel *can* be backend-specific — none currently is, and the generic `AUTOGRAD` key at 40 is the fallback target for a backend with no entry in that table (WebGPU, at `CUSTOM_0`, takes it).

### An actual layer

The autograd kernel is the only wrapping layer the framework installs, and it is the pattern every other one would follow:

```ts
export function wrapWithAutograd(opName: string, handle: OperatorHandle) {
  return (keySet: unknown, ...args: unknown[]) => {
    const ks = keySet as DispatchKeySet;
    if (!GradMode.isEnabled() || !_anyRequiresGrad(args)) {
      const stripped = ks.subtract(AUTOGRAD_KEY_SET);
      return dispatcher.redispatch(handle, stripped, ...args);
    }
```

([`autograd/dispatch.ts:86`](../../../src/autograd/dispatch.ts).) Three things to notice. It **subtracts the whole autograd key set**, not just the key it was dispatched on, because more than one autograd key can be present when operands live on different devices — belt and braces over `_dispatchInternal`'s own removal. It **redispatches rather than calling a kernel directly**, so it does not need to know which backend is underneath. And its fast path — no grad mode, or no operand requiring one — is a subtract and a redispatch, which is the cost the layer imposes on every call in a `noGrad` block.

### Backend coverage is a property of the operator, not of load order

```ts
function coverOperator(handle: OperatorHandle): void {
  if (handle.entry.devices) return;
  for (const key of BACKEND_COVERAGE_KEYS) {
    if (handle.entry.hasKernel(key)) continue;
    const kernel = jitKernelFor(handle.name, key);
    if (kernel) handle.entry.registerKernel(key, KernelFunction.fromUnboxed(kernel));
  }
}

export function installBackendCoverage(): void {
  dispatcher.setMissingKernelExplainer(explainMissingKernel);
  dispatcher.onOpDefined(coverOperator);
}
```

([`backend_coverage.ts:33`](../../../src/dispatcher/backend_coverage.ts).) This is worth pausing on, because it is where this book's two halves meet. **Any operator that has no hand-written kernel for a backend gets one generated by the compiler.** `jitKernelFor` builds a one-operation graph, runs it through the pass manager, lowers it, generates code for the target and caches the result ([`jit_cache.ts:158`](../../../src/dispatcher/jit_cache.ts)) — the entire pipeline of Parts II through X, invoked from an eager call, keyed on shape. [Chapter 62 §62.5](../ch62-dynamic-shapes/README.md) measures that cache.

Three clauses in those thirteen lines carry the design.

`onOpDefined` is a subscription, not a sweep ([`dispatcher.ts:54`](../../../src/dispatcher/dispatcher.ts)): the dispatcher calls the listener once for every operator already defined *and* once for every operator defined afterwards. `installBackendCoverage` runs at the end of `registerNativeOps` ([`registration.ts:41`](../../../src/tensor/native/registration.ts)), which is the first line of [`index.ts`](../../../src/index.ts) — but the five kernel modules on the lines beneath it define operators of their own, and so does the CUDA runtime when it is preloaded, minutes later. Each of those is covered at the moment it is declared. Which operators exist on which backend is therefore decided by the operator set, not by the order the modules happen to load in.

`if (handle.entry.hasKernel(key)) continue` is what makes hand-written kernels win: `impl` overwrites whatever occupies a slot, and the pass never revisits a filled one.

`if (handle.entry.devices) return` is the exception, and it is the more interesting one.

### An operator can declare where it runs

`svd`, `qr`, `fft`, `kmeans` and the other sixteen entries of [`kernels/defs/`](../../../src/kernels/defs) are not tensor expressions. They are host algorithms — Householder reflections, Lloyd's iterations, a radix-2 butterfly with a Bluestein fallback — with no representation in the graph IR at all. Asking the compiler to generate a CUDA kernel for `qr` does not produce a slow kernel; it produces a crash three layers down, in `_inferAndBuild`, complaining that `qr` is not a registered operation.

So those operators say so at the point they are defined:

```ts
export const ensureNumericSchemas = defineHostOps({
  devices: ['cpu', 'wasm'],
  schemas: NUMERIC_SCHEMAS,
});
```

([`numeric_defs.ts:9`](../../../src/kernels/defs/numeric_defs.ts).) The declaration does two things. The coverage pass skips the operator, so no kernel is generated for a backend that cannot have one. And when a call arrives on a backend the operator does not implement, the dispatcher asks for an explanation before throwing ([`dispatcher.ts:118`](../../../src/dispatcher/dispatcher.ts)):

```
Op 'qr' has no gpu implementation; it runs on cpu, wasm only
```

rather than `No kernel registered for op 'qr' with dispatch key 1`.

The declaration is not the only guard. `jitKernelFor` also refuses any operator the compiler has no lowering for, whether or not it was declared, by asking the same question `buildMappedOp` asks when it builds one ([`ir_mapping.ts:52`](../../../src/tensor/ops/ir_mapping.ts)):

```ts
export function canBuildMappedOp(opName: string): boolean {
  if (REDUCTION_OPS[opName]) return true;
  if (IR_BUILDERS[opName]) return true;
  if (typeof (IRBuilder.prototype as unknown as Record<string, unknown>)[opName] === 'function') return true;
  const opDef = registry.get(opName);
  return opDef !== null && opDef.inferResultTypes !== undefined;
}
```

Between them, the three cases are exhaustive, which is the property worth having: an operator is compiler-lowerable and gets all four backend keys automatically; or it declares the devices it runs on and the gaps are explained; or it is neither, and the first call says so in one sentence. There is no fourth state in which an operator quietly has no kernel anywhere.

## 60.5 Lab — one call, many answers

```bash
node docs/part11/ch60-a-pytorch-style-dispatcher/labs/01-one-call-many-answers.mjs
```

The table, first:

```
  103 operators registered, first eight: add sub mul div neg pow rem maximum

  key             ops with a kernel
  CPU              103
  GPU               83
  WASM             103
  META              57
  CUSTOM_0          83
  AUTOGRAD         103
  AUTOGRAD_CPU     103
  AUTOGRAD_GPU     103
  AUTOGRAD_WASM    103

  declared with no kernel anywhere: LAZY, CUSTOM_1, CUSTOM_2, CUSTOM_3, BATCHED, VMAP, FUNCTIONALIZE, AUTOCAST, TRACING
  fallbacks registered:             (none)
```

Nine keys carry kernels and nine do not. The empty ones are not dead weight in the same way: `BATCHED`, `VMAP`, `FUNCTIONALIZE` and `AUTOCAST` are declared *positions in the priority order*, which is the part of a layered design that has to be decided early, and `CUSTOM_0` — WebGPU — shows the mechanism being used for exactly that purpose after the fact.

Three of the four filled backend rows are worth reading against each other. `CPU` and `WASM` carry every operator. `GPU` and `CUSTOM_0` carry twenty fewer, and the twenty are exactly the host algorithms of §60.4 — nine linear-algebra operators, eight classical-ML ones, and `qr`, `fft`, `ifft` — each of which declared `['cpu', 'wasm']` or `['cpu', 'wasm', 'gpu']` and is therefore absent from a row by design rather than by accident. `META`'s 57 has a different reason: the coverage pass omits it, because a tensor with no storage has nothing for a generated kernel to compute.

Then the key set arriving from the arguments:

```
  a plain tensor          DispatchKeySet(CPU)
  requiresGrad_(true)     DispatchKeySet(AUTOGRAD_CPU, CPU)

  add(plain, plain)     DispatchKeySet(CPU)                    highest = CPU
  add(tracked, plain)   DispatchKeySet(AUTOGRAD_CPU, CPU)      highest = AUTOGRAD_CPU
  add(plain, tracked)   DispatchKeySet(AUTOGRAD_CPU, CPU)      highest = AUTOGRAD_CPU
```

The union is what makes `a + b` differentiable when *either* operand is — the requirement is a property of the call, not of a distinguished argument.

The lab then wraps every kernel of `add` in a recorder and watches the stack unwind:

```
  add(plain, plain)          CPU
  add(tracked, plain)        AUTOGRAD_CPU -> CPU
  inside noGrad(...)         AUTOGRAD_CPU -> CPU
```

**The third row is the interesting one.** `noGrad` does not remove the autograd key: the autograd kernel runs, reads `GradMode.isEnabled()`, finds it false, and redispatches. The guard stack — the mechanism designed for exactly this — is not involved. §60.7 weighs the two designs.

And the priority order, shown as the sequence a call would visit:

```
  backend only          1 keys, visited in order: CPU
  backend + autograd    2 keys, visited in order: AUTOGRAD_CPU -> CPU
  + tracing             3 keys, visited in order: TRACING -> AUTOGRAD_CPU -> CPU
  + autocast            4 keys, visited in order: TRACING -> AUTOGRAD_CPU -> AUTOCAST -> CPU
```

Tracing outermost, then autograd, then casting, then the backend. That is the order the numbering encodes, and reading it off is the argument for numbering keys by priority rather than by category.

Finally the lab installs a whole layer at runtime:

```
  TRACING kernels before registerTracingDispatch(): 0
  TRACING kernels after:                            103
```

A dispatch key with no kernels is a layer that has not been installed. Installing one is a loop.

```bash
node docs/part11/ch60-a-pytorch-style-dispatcher/labs/02-boxing-schemas-and-a-new-layer.mjs
```

Schemas are parsed strings, and what the parse decides is which arguments the dispatcher will even look at:

```
  mlc::add     Tensor:self Tensor:other                             tensors at [0,1] returns 1
  mlc::sum.dim Tensor:self int[]:dim bool:keepdim=False             tensors at [0] returns 1
  mlc::topk    Tensor:self int:k int:dim=-1                         tensors at [0] returns 2
  mlc::to      Tensor:self Device:device Dtype:dtype                tensors at [0] returns 1
```

Boxing, both directions, on the same two kernels:

```
  fromUnboxed  isBoxed=false  isUnboxed=true   callUnboxed(2,3) = 5   callBoxed([2,3]) = 5
  fromBoxed    isBoxed=true   isUnboxed=false  callUnboxed(2,3) = 5   callBoxed([2,3]) = 5
```

And the tag inference, which is where the boxing layer's honesty runs out:

```
  a Tensor    -> TENSOR
  3           -> INT
  2.0         -> INT
  2.5         -> FLOAT
  ...
  [1, 2]      -> INT_LIST
  [Tensor]    -> TENSOR_LIST
```

`2.0` is an `INT` because the tag is chosen by `Number.isInteger`, and JavaScript has one number type. A boxed kernel that branched on `isFloat()` would take the wrong branch for a float that happens to be integral. The tag is a *guess about intent* recovered from a value that no longer carries it — which is a general hazard of boxing in a dynamically typed language, and the reason [`_toIValue`](../../../src/dispatcher/boxing.ts) is 15 lines of heuristics.

Then the composability claim, exercised: a new operator, a backend kernel, and a *new layer above it*, all from outside `src/`:

```
  checksum(x, 10)         100
  with a CUSTOM_1 layer   100  log: ["2 args, remaining DispatchKeySet(CPU)"]
```

The layer saw the call, saw the residual key set with its own key already removed, redispatched, and the CPU kernel produced the same answer. No file under `src/` changed.

The guard stack, both halves:

```
  withIncludedKeys: depth 1, result 100, the layer ran 1 time(s) although the call site asked for CPU alone
  withExcludedKeys: depth 1, result 100, the layer ran 0 time(s) although the call site asked for it
```

— and then the property that makes guards sharp rather than convenient:

```
  A guard applies for its whole dynamic extent, so an included key must have a
  kernel on every operator reachable inside it:
    No kernel registered for op 'mul' with dispatch key 6
```

A guard is not scoped to an operator or a tensor; it rewrites the key set of *every dispatch inside the callback*. That is exactly what tracing needs — every operation inside the traced function must be recorded — and it is why `registerTracingDispatch` installs a kernel on all 103 operators before `withIncludedKeys(TRACING)` is ever entered. A partially-installed layer is a crash, not a degradation.

**Try this.** Register the `CUSTOM_1` layer for `mul` as well and re-run the last block; the guard now succeeds and the layer fires on every multiplication in the expression, including the ones inside `randn`. Then remove the `redispatch` from the layer and watch the call return `undefined` — a layer that forgets to delegate is a silent no-op, and nothing detects it.

## 60.6 What this buys the rest of the framework

The dispatcher is the reason three separate mechanisms in this book compose without knowing about each other.

**Autograd does not know about backends.** It subtracts its keys and redispatches, so a gradient is recorded identically for a CPU add and a CUDA one.

**Tracing does not know about autograd.** It sits above it in the numbering, so a traced call never reaches the autograd layer at all — [Chapter 61](../ch61-tracing/README.md)'s tracer records the *user's* operation, not the operation plus its gradient bookkeeping, and [Part V](../../part5/README.md)'s compiler-level differentiation is what supplies gradients for a compiled graph. Two completely different differentiation mechanisms, kept apart by an integer comparison.

**The compiler does not know about eager execution.** `installBackendCoverage` inverts the usual relationship: the eager path becomes a *client* of the compiler, one operation at a time. Everything Parts II–X built is reachable from `a.add(b)`.

## 60.7 Traps and limits

### Half the guard mechanism has no caller

`withExcludedKeys` — the function that would implement `noGrad` the way the design intends — has **no call site anywhere in `src/`**. The only guard the framework uses is `withIncludedKeys`, once, by the tracer ([`compile.ts:116`](../../../src/tracing/compile.ts)). `noGrad` is instead a module-global boolean read inside the autograd kernel ([`grad_mode.ts:3`](../../../src/autograd/grad_mode.ts)), which §60.5 measures: the layer still runs.

The two designs are not equivalent. The guard version removes the key, so the autograd layer is never entered and costs nothing; the flag version enters it on every operation and pays a subtract and a redispatch. The flag version also cannot express "no gradients for CUDA tensors but yes for CPU ones", which the key set can. What the two do share is their scoping rule: both `noGrad` and all three guard functions restore through `scoped` ([`util/scoped.ts`](../../../src/util/scoped.ts)), which hands a thenable body back a `.then` that restores on settle rather than restoring in a `finally` that would fire at the body's first `await`. A scope therefore covers an asynchronous body's whole extent under either design — which is the property, not the mechanism, that an inference loop depends on.

### The fallback table is empty, and the catch-all has no caller

`registerFallback` exists, `Library.fallback` exists, and no code in `src/` calls either; `setCatchAll` is used only by tests. So two of the three resolution steps in Definition 60.5 are dead on every dispatch, and every eager call pays two failed lookups before throwing or succeeding. More importantly, the *boxing* machinery exists to make fallbacks writable — a boxed kernel is the only kind that can be registered generically — and with no fallbacks registered, **nothing in `src/` ever registers a boxed kernel.** `IValue`, its ten tags, `callBoxed`, and both adapters are exercised only by the test suite. That is a complete calling convention carried for a use case the framework has not yet needed.

### `Library` accepts a `kind` and drops it

```ts
  constructor(namespace: string, kind: string) {
    this._namespace = namespace;
    this._registrations = [];
  }
```

([`library.ts:34`](../../../src/dispatcher/library.ts).) Eleven call sites pass `'DEF'` or `'IMPL'` ([`registration.ts:26`](../../../src/tensor/native/registration.ts) and ten others), and the parameter is never assigned or read. In PyTorch the analogous flag decides whether a library may define new operators or only implement existing ones — a real safety property, since a typo in an `IMPL` library would otherwise silently define a new operator instead of failing. Here `def` and `impl` are both allowed on every library, and a misspelled name in `impl` throws (`Op '…' not registered`) while a misspelled name in `def` succeeds and creates an operator nobody will call.

`Library.replay` is the other half of the same unused design: every registration is retained in `_registrations` so a library can be replayed into a second dispatcher, and there is no second dispatcher — `dispatcher` is a module-level singleton ([`dispatcher.ts:178`](../../../src/dispatcher/dispatcher.ts)). The retained array is pure memory.

### `findOrRegisterOp` invents a schema

```ts
      const schema = parseSchema(`${name}() -> Tensor`, 'mlc');
```

([`dispatcher.ts:75`](../../../src/dispatcher/dispatcher.ts).) An unknown operator is registered with **no arguments**, hence an empty `tensorArgIndices`, hence a key set computed from nothing, hence an empty set, hence `No dispatch key found`. The method has no caller, which is the only reason this does not bite; if it acquires one, the failure will be a confusing error two frames away from the typo that caused it.

### The schema parser has no failure mode

An unrecognised type name becomes `Scalar` ([`operator_schema.ts:156`](../../../src/dispatcher/operator_schema.ts)) and an unrecognised return type becomes `Tensor` ([`:173`](../../../src/dispatcher/operator_schema.ts)). §60.5 shows `Complex z` parsing cleanly. Since `tensorArgIndices` is derived from these types and *that* is what `computeKeySet` unions over, a schema with a mistyped `Tensor` silently produces an operator whose dispatch ignores one of its tensors — and if that is the only tensor argument, every call to it throws `No dispatch key found` regardless of what it was passed.

### Priority is the enum's numbering, and nothing states that

Theorem 60.3 depends on the numeric order of `DispatchKey` being the intended layering order. That fact is expressed only by the values in the enum and a blank line between bands ([`dispatch_key.ts:1`](../../../src/dispatcher/dispatch_key.ts)). Nothing asserts that autograd sorts above every backend or that tracing sorts above autograd; a key added in the wrong band changes the semantics of every call that carries it, and the failure is a wrong answer rather than an error.

### A declared device is a promise the runtime has to arrive to keep

The nine linear-algebra operators declare `['cpu', 'wasm', 'gpu']`, and their CUDA kernels are registered by `registerCudaLinalg` ([`kernels/cuda/linalg/register.ts:22`](../../../src/kernels/cuda/linalg/register.ts)) at the module scope of the CUDA runtime — that is, only once something has preloaded it. Between `import` and that moment, `svd` has a declared GPU device and no GPU kernel, and the lab's `GPU` count reflects the second fact rather than the first.

Nothing is silently wrong here — the dispatcher distinguishes the two cases and says which one it is ("`Op 'svd' runs on gpu, but its gpu kernels have not been registered; load the gpu runtime before calling it`"). But the declaration is a claim about the operator and the count is a measurement of the process, and the two agree only after a load the count cannot see. A reader comparing the table against `kernels/defs/` will find nine operators that appear to be missing kernels they were promised, and the resolution is a runtime that had not been imported yet.

### `bestKernel` and `firstOf` implement a different resolution rule, unused

`OperatorEntry.bestKernel` walks the key set in descending order and returns the first key with a kernel, falling back to the catch-all ([`operator_entry.ts:64`](../../../src/dispatcher/operator_entry.ts)) — that is, it *skips* keys with no kernel instead of throwing. `_dispatchInternal` does not use it: it takes the maximum key and throws if that key is empty. The two rules disagree exactly when the highest key in a set has no kernel for this operator and a lower one does: `_dispatchInternal` throws, `bestKernel` quietly uses the lower one. The rule is three methods deep — `OperatorHandle.bestKernel` to `OperatorEntry.bestKernel` to `KernelTable.firstOf` ([`kernel_table.ts:36`](../../../src/dispatcher/kernel_table.ts)) — and nothing calls the top of it: `OperatorHandle.bestKernel` ([`operator_handle.ts:43`](../../../src/dispatcher/operator_handle.ts)) has no caller in `src/` or in the tests, which enter the chain one level down. So the disagreement is latent — but the chain is exported, tested, and reads as though it were the resolution rule.

## 60.8 Read the tests

- [`tests/dispatcher/dispatcher.test.js`](../../../tests/dispatcher/dispatcher.test.js) — registration, the priority loop, redispatch, the fallback table and the catch-all, including the resolution order of Definition 60.5.
- [`tests/dispatcher/dispatch-key-set.test.js`](../../../tests/dispatcher/dispatch-key-set.test.js) — the bit set: union, subtract, `highestPriority`, iteration order, and the 32-bit boundary between the two words.
- [`tests/dispatcher/boxing.test.js`](../../../tests/dispatcher/boxing.test.js) — tag inference and both adapters, which is where the boxed convention is exercised at all.
- [`tests/dispatcher/operator-entry.test.js`](../../../tests/dispatcher/operator-entry.test.js) — `bestKernel`, the catch-all, and `removeKernel`.
- [`tests/dispatcher/guard.test.js`](../../../tests/dispatcher/guard.test.js) — include, exclude, nesting, and frame restoration on a thrown error and across an `await`.
- [`tests/dispatcher/backend-coverage.test.js`](../../../tests/dispatcher/backend-coverage.test.js) — the property of §60.4: no operator is left uncovered, every lowerable one has all four backend keys, an operator defined after start-up is covered and runs, and each of the three gap explanations is the one that appears.
- [`tests/dispatcher/schema.test.js`](../../../tests/dispatcher/schema.test.js) — parsing, overloads, defaults, out-arguments and `tensorArgIndices`.
- [`tests/dispatcher/autograd-dispatch.test.js`](../../../tests/dispatcher/autograd-dispatch.test.js) — the one real layer: key stripping, redispatch, and the grad-mode fast path.

---

**Next:** [Chapter 61 — Tracing](../ch61-tracing/README.md), which is what the `TRACING` key does when it is the highest one in the set.
