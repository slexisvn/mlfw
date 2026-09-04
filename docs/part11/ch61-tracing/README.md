# Chapter 61 — Tracing

[Chapter 5](../../part1/ch05-calls-to-program/README.md) argued that a compiler needs a *program*, surveyed the four ways to get one from a language that does not hand you one, and said this framework chose tracing. Fifty-five chapters have been spent on what happens to the program afterwards. This chapter is where the program comes from.

It is also where two decisions live that belong to neither the tracer nor the compiler, and are made *between* them: what to do with a model's weights, and what to do when the compiler offers more than one way to compile the same graph.

## 61.1 The problem: the program is not written down anywhere

A user writes this:

```js
const forward = (x) => x.matmul(W1).add(b1).relu().matmul(W2).add(b2);
```

Nothing in that expression is a program in the sense Part II needs. It is a JavaScript closure. `W1` is a free variable captured lexically. `matmul` is a method that dispatches, computes and returns. By the time the last `add` returns, five kernels have run and no record of the sequence exists — the intermediate tensors have been garbage collected and the only survivor is the answer.

The compiler needs the opposite: the *shape of the computation*, with no values in it, and with the free variables made explicit, because a kernel's parameter list cannot capture a closure.

There are four ways to obtain that ([Chapter 5 §5.2](../../part1/ch05-calls-to-program/README.md)): make the user write it in a restricted DSL, parse their source, capture bytecode, or **run the function with fake tensors and record what it asks for.** The last is tracing, and its appeal is that it needs nothing from the user and nothing from the language: the function is ordinary JavaScript, and it runs.

Its cost is precise and worth stating before the mechanism: **the trace records the execution that happened, not the function that was written.** A loop is recorded unrolled. A branch is recorded taken. Whatever the host language decided at trace time is baked in, and there is no marker in the graph to say a decision was made.

## 61.2 Intuition: a tensor that only remembers being asked

Replace each input with an object that has a shape and a dtype and no data. Give it the same interface as a tensor, so the user's code runs unchanged. When an operation is called on it, do not compute: **append a node to a graph and return a new such object standing for the result.**

Two questions immediately follow, and both are answered by the mechanism already in place.

*How do the operations know to record instead of compute?* They do not. A symbolic tensor carries the `TRACING` dispatch key, which sorts above everything ([Chapter 60](../ch60-a-pytorch-style-dispatcher/README.md)), so a call with one lands on a kernel that records. No operation is modified; a layer is installed.

*What about `W1`, which is a real tensor?* When a recording kernel meets a real tensor among its arguments, it *captures* it: adds a new parameter to the function being built, remembers which tensor it stands for, and uses that parameter from then on. So the closure's free variables become the tail of the function's argument list, in first-touch order.

## 61.3 Theory

> **Definition 61.1 (Trace).** **(stated here)** Given a function `f`, example inputs `x₁…xₙ`, and an execution of `f` on symbolic stand-ins for them, the *trace* is the sequence of dispatched operations recorded during that execution, together with the map from each recorded result to the operation that produced it.

> **Definition 61.2 (Symbolic tensor).** **(invariant)** A `SymbolicTensor` is a `Tensor` subclass carrying an IR value, a shape, a dtype and a symbolic shape, allocated on the `meta` device with a zero-length storage. Its key set is its base key set unioned with `TRACING`. Every accessor that would read an element — `.data`, `.storage`, `item()`, `toArray()`, `_select`, iteration — throws.

The choice to *subclass* `Tensor` rather than duck-type is what makes user code run unmodified: `instanceof Tensor` succeeds, method lookup finds the same methods, and `shape` and `dtype` answer honestly. The choice to throw on every value read is what makes Definition 61.3's failure mode loud instead of silent.

> **Theorem 61.3 (Trace validity).** **(stated here)** A trace of `f` at `x` is a valid program for input `y` if `f`'s execution on `y` would dispatch the same sequence of operations with the same attributes as its execution on `x`. This is [Theorem 5.3](../../part1/ch05-calls-to-program/README.md) restated in the tracer's own terms, and it is a sufficient condition only: two different sequences can compute the same function, and nothing here needs the converse.
>
> *Consequence, in three cases.* If `f` branches on a *host* value — a JavaScript number, a Python-style flag, a length — the condition holds for every `y` that would take the same branch, and the trace is silently wrong for the others. If `f` branches on *tensor contents*, the branch condition needs `item()`, which throws, so the failure is at trace time and loud. If `f` branches on a *shape*, the answer depends on whether that dimension was traced symbolically, which is [Chapter 62](../ch62-dynamic-shapes/README.md).

The middle case is the one this implementation improved. `item()` on a symbolic tensor could have returned `0`; instead it throws with a three-paragraph message naming the cause and the two remedies ([`symbolic_tensor.ts:23`](../../../src/tracing/symbolic_tensor.ts)). That is a design decision about *which* failures a system should make impossible to ignore, and it costs 6 lines.

> **Definition 61.4 (Capture).** **(invariant)** When a recorded operation receives a non-symbolic tensor `t`, `captureConstant(t)` returns the symbolic tensor standing for it: the memoised one if `t` has been captured before, a `constant` operation if `t` is a scalar with data, and otherwise a freshly appended block argument, recorded in capture order.
>
> *Identity, not value.* Memoisation is keyed on the tensor *object*. Two distinct tensors holding identical data are captured twice; the same tensor read twice is captured once.

> **Proposition 61.5 (Argument layout).** **(invariant)** A traced function's parameters are the *n* user inputs followed by the captured tensors in capture order. Every consumer downstream — execution, weight folding, backward compilation — depends on this and none of them re-derives it.

> **Definition 61.6 (Weight folding).** **(stated here)** *Folding* replaces a captured parameter with a `constant` operation carrying its data, deletes the parameter, and removes the tensor from the captured list. It is a compile-time specialisation: the graph is now correct only for that value of the weight.

> **Proposition 61.7 (Folding is not free).** **(stated here)** Folding a parameter of *k* elements adds *k* elements of data to the compiled artifact and removes one argument from every call. It is profitable when the weight is small relative to the per-call argument cost, or when the target links constants into the kernel rather than passing them.
>
> *Consequence.* The predicate must depend on the target. It does: the element cap is `Infinity` when the target reports `supportsConstBuffers` or when quantization is enabled ([`compile.ts:337`](../../../src/tracing/compile.ts)), and `MAX_FOLDABLE_ELEMENTS` otherwise.

> **Definition 61.8 (Optimization gate).** **(stated here)** Given a set of candidate configurations, the *gate* compiles each, runs each on the example inputs, discards any whose output differs from the baseline's beyond a tolerance, and adopts the fastest survivor only if it beats the baseline by at least a minimum gain factor.

> **Proposition 61.9 (A gate's floor must exceed its measurement noise).** **(stated here)** Let the baseline and a candidate be the same program. Their measured ratio is a random variable with median 1. The gate adopts the candidate whenever the ratio exceeds the floor `g`. So the probability of a spurious adoption is `P(ratio > g)`, which is zero only if `g` exceeds the distribution's support.
>
> *Consequence.* A floor is a claim about the measurement apparatus, not about the optimization. §61.5 measures the claim.

## 61.4 In mlfw: the tracer, the capture, and the two things around the compiler

### The recording kernel

```ts
function _tracingKernel(opName: string): (keySet: unknown, ...args: unknown[]) => TensorOutput | TensorOutput[] {
  return (keySet: unknown, ...args: unknown[]) => {
    const tracer = getActiveTracer();
    if (!tracer) {
      throw new Error(`TRACING dispatch key active but no tracer is set for op '${opName}'`);
    }

    if (_TRACE_BY_DECOMPOSITION.has(opName)) {
      const handle = dispatcher.findOp(opName)!;
      return dispatcher.redispatch(handle, keySet as DispatchKeySetType, ...args) as TensorOutput | TensorOutput[];
    }
```

([`dispatch.ts:19`](../../../src/tracing/dispatch.ts).) One kernel factory, registered for all 103 operators. The second branch is the interesting one: eleven operations — `scatter`, `repeat`, `tile`, `split`, `chunk`, `roll`, `flip`, `cumsum`, `sort`, `argsort`, `topk` — are **not** recorded as themselves. The kernel redispatches, the layer below is the eager implementation, and whatever *that* implementation calls gets recorded instead. A `chunk` becomes several `slice`s in the graph.

That is a deliberate trade: the graph gains eleven fewer operations to lower, differentiate and schedule, at the cost that the traced graph no longer mentions an operation the user wrote. [Part V](../../part5/README.md)'s VJP registry does not need a rule for `chunk`; [Part IV](../../part4/README.md)'s fusion sees slices it can reason about. The cost lands on error messages and on anyone reading the IR looking for the call they made.

The rest of the kernel separates tensors from scalars and records:

```ts
    const isTensorArg = (a: unknown): a is TensorCandidate | SymbolicTensor => (a instanceof SymbolicTensor) || isTensorValue(a);
    const pushTensor = (arg: TensorCandidate | SymbolicTensor): void => {
      if (arg instanceof SymbolicTensor) {
        tensorArgs.push(arg);
      } else if (arg.isSymbolic) {
        tensorArgs.push(arg);
      } else {
        tensorArgs.push(tracer.captureConstant(arg));
      }
    };
```

([`dispatch.ts:36`](../../../src/tracing/dispatch.ts).) Three cases: already symbolic, claims to be symbolic, or a real tensor to capture. Scalars are matched positionally against a per-operator list of attribute names from the operation metadata ([`:52`](../../../src/tracing/dispatch.ts)) — an operation's non-tensor arguments become graph *attributes*, which is what makes them constant-foldable in Part IV and invisible to Part V.

### Capture

```ts
  captureConstant(tensor: Tensor): SymbolicTensor {
    let cached = this._capturedParams.get(tensor);
    if (cached) return cached;

    if (tensor.shape.length === 0 && tensor.data) {
      const value = tensor.data[0];
      const op = this._requireBuilder().scalarConstant(value, tensor.dtype);
      const irValue = op.getResult(0);
      const sym = new SymbolicTensor(irValue, [], tensor.dtype, this, []);
      this._capturedParams.set(tensor, sym);
      return sym;
    }

    const tt = new TensorType(tensor.shape, tensor.dtype);
    const func = this._requireFunc();
    (func.inputTypes as TensorType[]).push(tt);

    const block = func.entryBlock;
    const irValue = block.addArgument(tt);
```

([`tracer.ts:151`](../../../src/tracing/tracer.ts).) Definition 61.4 in fifteen lines. A scalar becomes a `constant` — its value is baked in, because a scalar is cheap to specialise on and expensive to pass. Everything else becomes a parameter appended to the function's signature *during* tracing, which is why the signature is mutable until `getGraphModule()` freezes it ([`:205`](../../../src/tracing/tracer.ts)).

### The layer is entered by a guard

```ts
  tracer.activate();
  const TRACING_KEYS = DispatchKeySet.fromKey(DispatchKey.TRACING);
  const result = withIncludedKeys(TRACING_KEYS, () => fn(...symbolicInputs)) as MaybePromise<TensorOutput | TensorOutput[]>;
```

([`compile.ts:114`](../../../src/tracing/compile.ts).) The only `withIncludedKeys` call in `src/`, and the reason it is needed rather than relying on the symbolic tensors' own keys: an operation whose arguments are *all* real tensors — a `zeros` created inside the traced function, a weight multiplied by a scalar — would otherwise compute eagerly and be invisible to the trace. The guard makes `TRACING` present on every dispatch inside the callback regardless of what the arguments carry.

`tracer.activate()` sets a module-global; the recording kernel reads it back. That is a second channel alongside the dispatch key, and §61.7 returns to it.

### Around the compiler: folding

```ts
  function _finalize(traced: TracedCore, inputs: readonly Tensor[]): CompiledEntry {
    const prepared = foldWeights ? foldWeightParams(traced, tensorToContiguous, foldPredicate) : traced;
    const indexBounds = userArgIndexBounds(prepared.graph as unknown as GraphModule, prepared.numUserInputs);
    const signature = inputSignatureOf(inputs);
    const build = (p: TracedCore, b: readonly ArgIndexBound[], o: Readonly<Record<string, unknown>> | null) => _entryFor(p, b, o, signature);
    const tuned = tuneOptimizations ? runOptimizationGate(prepared, indexBounds, build) : null;
    return tuned ?? build(prepared, indexBounds, null);
  }
```

([`compile.ts:365`](../../../src/tracing/compile.ts).) Six lines, and they are the whole of "what `compile()` does that the compiler does not". Fold, compute index bounds, then either run the gate or build once. Both extra steps are *off by default*.

The predicate is a closure over the target:

```ts
  const quantizing = !!(opts?.quantization as { enabled?: boolean } | undefined)?.enabled;
  const linksConstants = quantizing || !!(target as { supportsConstBuffers?: boolean })?.supportsConstBuffers;
  const foldPredicate = weightPredicate(linksConstants ? Infinity : MAX_FOLDABLE_ELEMENTS);
```

([`compile.ts:336`](../../../src/tracing/compile.ts).) Proposition 61.7's consequence, three lines. On a target that links constant buffers into the kernel there is no per-call argument to save and no cap is needed; on one that does not, a weight over 1,024 elements stays a parameter.

### Around the compiler: the gate

```ts
    for (const spec of [{ name: BASELINE, optimization: undefined }, ...candidates]) {
      let entry: CompiledEntry;
      try {
        entry = build(prepared, indexBounds, spec.name === BASELINE ? null : { optimization: spec.optimization });
      } catch (e) {
        measurements.push({ name: spec.name, ms: 0, correct: false, error: String((e as Error)?.message ?? e) });
        continue;
      }
      const timed = _timeEntry(entry, exampleInputs!);
```

([`compile.ts:402`](../../../src/tracing/compile.ts).) Baseline first, then each candidate; a candidate that fails to compile is recorded as incorrect rather than throwing. `_timeEntry` warms twice and takes the median of seven ([`:260`](../../../src/tracing/compile.ts)), and refuses asynchronous runtimes outright — so **the gate does not run on CUDA or WebGPU**, the two targets whose candidate lists are non-empty when tensor cores are present.

The selection rule is nine lines:

```ts
  let best = baseline;
  for (const m of measurements) {
    if (m.name === BASELINE || !m.correct || !(m.ms > 0)) continue;
    if (m.ms < best.ms) best = m;
  }

  const gain = best.ms > 0 ? baseline.ms / best.ms : 1;
  const winner = best.name !== BASELINE && gain >= minGain ? best : baseline;
```

([`opt_gate.ts:59`](../../../src/compiler/pipeline/opt_gate.ts).) Correctness is a filter, not a tie-break: a faster wrong answer is not a candidate at all. And the floor is `DEFAULT_MIN_GAIN = 1.05`.

## 61.5 Lab — recording a program, and the two decisions around it

```bash
node docs/part11/ch61-tracing/labs/01-recording-a-program.mjs
```

The running example, traced:

```
  user inputs      1
  captured tensors 8x16, 16, 16x4, 4
  function arity   5 = 1 user + 4 captured

  func.func @mlp(%0: tensor<2x8xf32>, %1: tensor<8x16xf32>, %2: tensor<16xf32>, %3: tensor<16x4xf32>, %4: tensor<4xf32>) -> (tensor<2x4xf32>) {
    %5 = tera.dot %0, %1 {lhs_batch = array<i64>, lhs_contracting = array<i64: 1>, rhs_batch = array<i64>, rhs_contracting = array<i64: 0>} : (tensor<2x8xf32>, tensor<8x16xf32>) -> tensor<2x16xf32>
    %6 = "tera.add"(%5, %2) : (tensor<2x16xf32>, tensor<16xf32>) -> tensor<2x16xf32>
    %7 = tera.constant dense<0.0> : tensor<f32>
    %8 = tera.broadcast_in_dim %7 {broadcast_dimensions = array<i64>} : tensor<f32> -> tensor<2x16xf32>
    %9 = tera.maximum %6, %8 : tensor<2x16xf32>
    %10 = tera.dot %9, %3 {lhs_batch = array<i64>, lhs_contracting = array<i64: 1>, rhs_batch = array<i64>, rhs_contracting = array<i64: 0>} : (tensor<2x16xf32>, tensor<16x4xf32>) -> tensor<2x4xf32>
    %11 = "tera.add"(%10, %4) : (tensor<2x4xf32>, tensor<4xf32>) -> tensor<2x4xf32>
    return %11 : tensor<2x4xf32>
  }

```

This is the picture [Chapter 1](../../part0/ch01-what-this-book-is/README.md) promised and every part since has transformed. Two things in it are the tracer's doing rather than the user's. Four closure variables became parameters `%1`–`%4`. And `relu` is not there: it decomposed into a zero constant, a broadcast and a `maximum`, because the eager `relu` is written that way and the trace records what dispatched.

A symbolic tensor, and what it refuses:

```
  class     SymbolicTensor
  shape     [2,8]   dtype f32   device meta
  isSymbolic true   keys DispatchKeySet(TRACING, META)
  .data          .data is not available on a symbolic tensor: tracing records operations …
  item()         item() is not available on a symbolic tensor: tracing records operations …
```

`META` rather than `CPU` is the base key: the tensor is allocated on the meta device, which by construction has no storage. So even if `TRACING` were somehow absent, the call would land on a meta kernel rather than silently computing on garbage.

Theorem 61.3's three cases, side by side:

```
  tensor arithmetic          4 ops: constant mul constant add
  a branch on a host value   3 ops: constant broadcast_in_dim maximum
  a JavaScript loop          6 ops: constant add constant add constant add
  where(), the tensor branch 4 ops: compare constant mul where
```

**The loop is gone and the branch is gone.** Three `add`s, not a loop; one `relu`, not a choice. Only the fourth row keeps its decision, because `where` is an *operation* and survives into the graph. That row is the remedy the error message points at, and here is the error:

```
    item() is not available on a symbolic tensor: tracing records operations instead of computing
    them, so this tensor (shape [], dtype f32) carries no value to read.
    The usual cause is data-dependent control flow: a branch or loop condition computed from
    tensor contents. …
```

Capture by identity, not by value:

```
  read twice, captured 1 time(s)
  two different tensors, captured 2 time(s)
```

And the scalar rule:

```
  captured params 0, ops: constant mul return
```

```bash
node docs/part11/ch61-tracing/labs/02-around-the-compiler.mjs
```

The folding predicate, tabulated:

```
  candidate         numel  foldable?
  f32 [8,16]          128  true
  f32 [16]             16  false
  f32 [32,32]        1024  true
  f32 [33,32]        1056  false
  i32 [8,16]          128  false
  f32 scalar            1  false
```

1,024 folds and 1,056 does not, which is the cap. `f32 [16]` does not, which is *not* the cap — **rank 1 is excluded outright**, so a bias is never folded however small. On the running example that means:

```
  all weights small  arity 5 -> 3   captured 4 -> 2   folded constants 2
  one weight large   arity 5 -> 4   captured 4 -> 3   folded constants 1
```

Two of four parameters fold; the two biases stay. §61.7 asks why.

The gate's candidate list, per target:

```
  target             tensorCore  blockedLayout  candidates
  cpu_generic        false       true           layout
  wasm_generic       false       false          (none)
  cuda_generic       false       false          (none)
  webgpu_generic     false       false          (none)
  cuda +tc +layout   true        true           layout, tensorize, layout+tensorize
```

Three of the four shipped targets offer nothing, so the gate is a no-op on them. And the selection rule:

```
  a clear win             winner layout     gain 1.667
  a win below the floor   winner baseline   gain 1.000
  fast but wrong          winner baseline   gain 1.000
  two candidates          winner tensorize  gain 2.000
  an incorrect baseline   optimization gate: the baseline configuration must be measured and correct …
```

Now Proposition 61.9, measured. The lab runs the gate eight times on a CPU target, each round with a slightly different input width so the decision cache cannot answer, and then compiles the two configurations separately and compares their emitted source:

```
      N   baseline     layout    gain  winner
    128      1.361      1.309   1.000  baseline
    129      1.263      1.238   1.000  baseline
    130      1.236      1.242   1.000  baseline
    131      1.313      1.269   1.000  baseline
    132      1.267      1.230   1.000  baseline
    133      1.475      1.270   1.161  layout
    134      1.276      1.487   1.000  baseline
    135      1.307      1.342   1.000  baseline

  the two configurations emit byte-identical source (877 vs 877 characters)
  and the gate still preferred 'layout' in 1 of 8 rounds.
```

**The two configurations compile to the same 877 characters**, because — as [Chapter 25](../../part4/ch25-layout/README.md) found — `layoutAwareOps` is empty for every shipped target, so the layout pass proposes conversions and discards all of them. The gate is therefore measuring one program against itself, and on this machine it reports a 1.16× "gain" often enough to adopt the "winner" in one round of eight. Repeated runs of the lab land between one and four.

Nothing goes wrong here, because the two programs are the same program. The finding is about the floor: **1.05 is below this apparatus's noise**, so the gate as configured cannot distinguish a 5% improvement from a quiet machine. §61.7 says what would fix it.

**Try this.** Raise the floor by passing a larger `minGain` to `selectWinner` on the recorded measurements and watch the spurious adoptions disappear; then lower `GATE_REPEAT` from 7 to 1 in [`compile.ts:232`](../../../src/tracing/compile.ts) and watch them multiply. The median of seven is doing real work, and it is not doing enough of it.

## 61.6 What tracing hands to the rest of the framework

**A `GraphModule` with one function**, whose parameters are the user's inputs followed by the captured tensors, and whose body is Part II's IR. Everything from [Chapter 8](../../part2/ch08-ssa-and-dataflow/README.md) onward operates on this object.

**A list of captured tensors**, in the order their parameters appear. [Chapter 63](../ch63-training-end-to-end/README.md)'s training loop reads gradients back out positionally against this list.

**A `ShapeEnv`** holding the symbols allocated for each input dimension and the guards accumulated about them, which is [Chapter 62](../ch62-dynamic-shapes/README.md)'s subject.

**The output types and symbolic output shapes**, which is how `compile()` knows how large a buffer to allocate for a result whose shape depends on a symbol.

## 61.7 Traps and limits

### The active tracer is a module-global, and nesting is not detected

`_activeTracer` is a single module-level variable; `activate()` overwrites it and `deactivate()` clears it only if the tracer is still the active one ([`tracer.ts:213`](../../../src/tracing/tracer.ts)). Tracing a model whose `forward` itself calls `compile()` — or `trace()` — replaces the active tracer, and the inner `deactivate()` restores `null` rather than the outer tracer. The outer trace then continues with the guard's `TRACING` key still set and no tracer, and every subsequent operation throws `TRACING dispatch key active but no tracer is set`. The failure is loud, which is the good half; the message names the operation rather than the nesting, which is the other half.

### The trace records what dispatched, and the user's operation may not be it

Two mechanisms, one consequence. Eleven operations are traced by redispatching to their eager implementations, so `chunk` never appears in a graph. And `relu` appears as `constant`/`broadcast_in_dim`/`maximum` — as §61.5 shows — because the eager implementation is written that way. Neither is recorded anywhere in the graph. A reader of the IR cannot tell a decomposed `relu` from a hand-written `maximum`, and an error message about `maximum` will confuse a user who wrote `relu`. The IR has attributes and nothing carries provenance.

### The folding predicate excludes the parameters it would most like to fold

```ts
    if (!param || !param.shape || param.shape.length < 2) return false;
```

([`fold_params.ts:15`](../../../src/tracing/fold_params.ts).) A bias vector is rank 1, so it is never folded — although it is the cheapest thing in the model to fold and, being one argument per layer, one of the more expensive things to pass. The reasonable intent behind the clause is "do not fold a scalar", and scalars are already handled a level up by `captureConstant`. Widening it to `length < 1` would fold biases; nothing in the code says why it is 2. Note too that the predicate's signature offers `(param, index, arg)` and the shipped implementation reads only `param`, so a predicate that wanted to consult the parameter's *uses* — fold it only if it feeds a `dot`, say — is expressible and unused.

### The gate's floor is below its own noise, and its cache outlives the measurement

§61.5 measures the first half. The second: `_gateDecisions` is a module-global `Map` keyed by graph signature, target name and candidate list ([`compile.ts:229`](../../../src/tracing/compile.ts)), and `clearOptimizationGateCache` exists but is not exported from [`src/index.ts`](../../../src/index.ts). So a spurious decision taken once — on a machine that was briefly busy — is reused for every subsequent compilation of a graph with the same operation names and input shapes, for the lifetime of the process, with no way for a caller to invalidate it.

The signature is `opNames.join(',') + '#' + shapes.join(';')` ([`opt_gate.ts:85`](../../../src/compiler/pipeline/opt_gate.ts)), so it does not include dtypes, attributes, or the captured weights' shapes. Two models with the same operation sequence and the same *input* shapes but different hidden widths share a cached decision.

### The gate cannot run on the targets that have candidates

`_timeEntry` returns `null` when execution is asynchronous, and the gate then abandons ([`compile.ts:412`](../../../src/tracing/compile.ts)). CUDA and WebGPU both execute asynchronously ([Chapter 59 §59.5](../ch59-the-runtime-module/README.md)), and they are the only targets that can offer `tensorize`. So the gate measures candidates on the target where the only candidate is a no-op, and declines to measure on the targets where the candidates are real. The recorded reason is honest — `'runtime is asynchronous; the gate only measures synchronous runtimes'` — and it is recorded in a measurement object the caller sees only via `tuningReport()`.

### `_bucketInputs` fabricates tensors

```ts
  function _bucketInputs(shapes: readonly (readonly number[])[]): Tensor[] {
    return shapes.map((shape, i) => ({ shape, dtype: exampleInputs![i].dtype } as Tensor));
  }
```

([`compile.ts:452`](../../../src/tracing/compile.ts).) `shapeBuckets` pre-compiles for a list of shapes by tracing with objects that have a `shape` and a `dtype` and are not tensors. That works because tracing reads only those two fields from its example inputs — but it is an undocumented dependency between two files, and any future use of an example input's `device` or `data` during tracing will fail here and nowhere else. The cast to `Tensor` is where the type system was told to stop looking.

### Errors carry a repro record that nothing reads

`_attachRepro` attaches `{ name, phase, target, inputs, config }` to any error thrown during compilation or execution ([`compile.ts:304`](../../../src/tracing/compile.ts)). It is a good idea — [Chapter 67](../../OUTLINE.md) will want exactly this — and there is no consumer: nothing in `src/` reads `error.repro`, no formatter prints it, and it does not appear in the error's `message`. A user who hits it sees the original message and has to know to inspect a non-standard property.

## 61.8 Read the tests

- [`tests/tracing/compile.test.js`](../../../tests/tracing/compile.test.js) — end-to-end tracing and compilation, the compiled-function surface (`graph()`, `source()`, `kernels()`), and argument validation.
- [`tests/tracing/dispatch.test.js`](../../../tests/tracing/dispatch.test.js) — the recording kernel: tensor/scalar separation, attribute naming, and the trace-by-decomposition set.
- [`tests/tracing/symbolic-tensor.test.js`](../../../tests/tracing/symbolic-tensor.test.js) and [`symbolic-value-read.test.js`](../../../tests/tracing/symbolic-value-read.test.js) — Definition 61.2's refusals, and that each one names the operation and the remedy.
- [`tests/tracing/trace-modules.test.js`](../../../tests/tracing/trace-modules.test.js) — capture by identity, capture order, and the parameter layout of Proposition 61.5.
- [`tests/e2e/optimization-gate.test.js`](../../../tests/e2e/optimization-gate.test.js) — candidate enumeration per target, the correctness filter, the minimum-gain floor, and the decision cache.

---

**Next:** [Chapter 62 — Dynamic shapes](../ch62-dynamic-shapes/README.md), which is what happens when the shape the trace baked in is not the shape the next call brings.
