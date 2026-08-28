# Chapter 62 — Dynamic shapes

[Chapter 61](../ch61-tracing/README.md) ended with a graph whose input type is `tensor<2x8xf32>`. Every pass since [Chapter 8](../../part2/ch08-ssa-and-dataflow/README.md) has been allowed to use those numbers: fusion counted elements, the scheduler picked tile sizes for them, [Chapter 53](../../part10/ch53-lir-the-third-ir/README.md) flattened indices with them baked in as multipliers.

Then the user calls the compiled function with a batch of 7.

## 62.1 The problem: the shape is an input, and it was compiled in as a constant

A batch dimension changes. In inference it changes with the request; in training it changes on the last batch of an epoch; in a language model the sequence length changes with every prompt. Nothing about a batch dimension is a property of the *program*, and yet by the time codegen ran it was a literal in the source.

There are exactly two things to do about it, and both are legitimate.

**Recompile.** Treat each shape as a different program. Every pass gets its constants, the generated code is as specific as it can be, and the cost is a compilation per distinct shape — which for a serving workload with a hundred sequence lengths is a hundred compilations, and for a training loop with two batch sizes is two.

**Generalise.** Compile once with the dimension left as a *variable*, passed to the kernel at launch. The generated code is more conservative — every loop bound is a runtime value, so nothing can be unrolled or vectorised against it, and [Chapter 37](../../part6/ch37-proving-things-about-indices/README.md)'s analyser has less to prove with — but one kernel serves every size.

The interesting problem is not choosing between them. It is that **choosing generalisation for one dimension means every downstream decision must remain correct for all of its values**, and *proving* that is not free either. A tiling that requires the extent to be divisible by 8 is legal for a batch of 64 and illegal for a batch of 7. Something has to record the conditions under which the compiled kernel is valid, and check them on every call.

That record is a *guard set*, and it is the whole chapter.

## 62.2 Intuition: a name instead of a number, and a list of promises

Give the dimension a name — `s0` — instead of a number. Let it flow through the type system as a symbol. Now every place that would have computed with `6` computes with `s0`, and the ones that *need* a number have to ask.

When something asks and gets an answer, that answer is a **promise about future inputs**: "this kernel is valid when `s0` is 6" or "when `s0 > 0`" or "when `s0` is divisible by 4". Collect the promises. On the next call, bind the symbols to the actual shapes and check each promise. All hold: reuse the kernel. One fails: compile a new one, with its own promises, and keep both.

A static compilation is the degenerate case of exactly this. Every dimension gets a symbol; every symbol immediately gets the promise `s = 6`; the kernel is valid only for that shape; the next shape misses and recompiles. **Static and dynamic are one mechanism at two settings**, which is why there is no second code path for them.

## 62.3 Theory

> **Definition 62.1 (Shape environment).** **(invariant)** A `ShapeEnv` holds symbols — each with a *hint* (the value it had at trace time) and its origin `(inputIdx, dimIdx)` — a list of guards, and a binding from symbol to concrete value for the current call.

The hint is what makes symbolic tracing survivable: any code that must have a number can have one, at the cost of a guard.

> **Definition 62.2 (Guard).** **(invariant)** A guard is either a relation `lhs op rhs` with `op ∈ {eq, ne, gt, ge, lt, le}` over symbols and integers, or a divisibility claim `sym % d == 0`.

> **Definition 62.3 (Specialisation).** **(invariant)** `specialize(e)` returns the hint of expression `e` and, the first time it is asked for that expression and value, records the guard `e = hint`. Asking again is free — the guard is already there.

> **Theorem 62.4 (Guard soundness).** **(stated here)** A compiled kernel produced from a trace with shape environment `E` is safe to reuse for inputs `y` if every guard in `E.guards` evaluates true under the binding induced by `y`.
>
> *Proof sketch.* Each guard was recorded at the moment some decision consumed a symbolic quantity as a concrete one. If every such guard holds, every consumed quantity has the value it had at trace time, so each decision would be taken identically; the composition of identical decisions is the same program. ∎
>
> *The hypothesis, stated as sharply as it deserves.* The theorem is conditional on **every** consumption of a symbolic quantity recording a guard. It is a claim about the *completeness* of the recording, not about the checking — the checking is nine lines and obviously correct. §62.5 exhibits a consumption that records nothing.

> **Corollary 62.5 (Static compilation is the maximal guard set).** **(stated here)** Tracing with no dynamic dimensions records `sᵢ = cᵢ` for every input dimension. That set is satisfiable by exactly one shape tuple, so the cache degenerates to a lookup by shape, and the theorem's hypothesis is trivially met: nothing symbolic is ever consumed, because nothing is symbolic.

This is why the default is safe and the option is the risk.

> **Definition 62.6 (Compiled-function cache).** **(invariant)** `compile()` keeps a list of compiled entries. A call selects the first entry whose *input signature* — the dtype and device of each argument — matches, and whose guards then pass under the call's shapes. If none does, it compiles a new entry and appends it.
>
> *Consequence.* Lookup is linear in the number of entries and each probe evaluates a guard list. The cache never evicts.

> **Definition 62.7 (Eager JIT cache).** **(invariant)** `jitCompile` keys on a string built from the operation name, each argument's shape and dtype, the scalar arguments serialised as JSON, and the target's name. There is no guard evaluation: the key *is* the guard, and it is equality on everything.

The two caches answer the same question at different granularities, and the difference is instructive: one operation on known shapes has nothing to generalise over, so equality is exactly right and cheaper than a guard list.

## 62.4 In mlfw: symbols, guards, and two caches

### Allocating a symbol, and the rule that decides whether it survives

```ts
  produceShapeSpec(inputIdx: number, concreteShape: readonly number[], dynamicDims?: Set<number> | null): { irShape: number[]; symShape: MutableSymbolicShape } {
    const irShape = new Array<number>(concreteShape.length);
    const symShape = new Array<SymbolicDim>(concreteShape.length);

    for (let i = 0; i < concreteShape.length; i++) {
      if (dynamicDims && dynamicDims.has(i) && concreteShape[i] > 1) {
        const sym = this.allocate(inputIdx, i, concreteShape[i]);
        irShape[i] = DYNAMIC;
        symShape[i] = sym;
      } else {
        const sym = this.allocate(inputIdx, i, concreteShape[i]);
        this.guardRelation(sym, 'eq', concreteShape[i]);
        irShape[i] = concreteShape[i];
        symShape[i] = concreteShape[i];
      }
    }
```

([`shape_env.ts:42`](../../../src/tracing/shape_env.ts).) Corollary 62.5, implemented as an `if`. A dimension the caller asked to keep dynamic becomes `DYNAMIC` in the IR type and a symbol name in the symbolic shape; every other dimension gets a symbol *and immediately an equality guard*, and its IR type keeps the number.

The clause `concreteShape[i] > 1` is the one to notice. A dimension whose example value is 1 is refused symbolic status even when the caller asked for it, on the reasoning that a length-1 dimension is usually a broadcast axis and specialising on it is almost always right. §62.5 measures what it costs when the example happens to be a batch of one.

### Specialisation, and where it is triggered from

```ts
  specialize(expr: SymbolicDim): number | null {
    const hint = this.hintOf(expr);
    if (hint === null || typeof expr === 'number') return hint;
    const key = `${String(expr)}=${hint}`;
    if (this._specialized.has(key)) return hint;
    this._specialized.add(key);
    this.guardRelation(expr, 'eq', hint);
    return hint;
  }
```

([`shape_env.ts:80`](../../../src/tracing/shape_env.ts).) Definition 62.3, memoised so a symbol consumed a thousand times records one guard. The caller is the place a symbolic shape becomes a concrete one:

```ts
  get shape(): readonly number[] {
    const raw = super.shape;
    const sym = this._symbolicShape;
    if (!sym) return raw;
    let specialized: number[] | null = null;
    for (let i = 0; i < raw.length; i++) {
      if (raw[i] !== DYNAMIC || i >= sym.length) continue;
      const hint = this._tracer.shapeEnv.specialize(sym[i]);
      if (hint === null) continue;
      if (specialized === null) specialized = [...raw];
      specialized[i] = hint;
    }
    return specialized || raw;
  }
```

([`symbolic_tensor.ts:59`](../../../src/tracing/symbolic_tensor.ts).) **Reading `.shape` on a symbolic tensor burns a guard.** That is the crucial design point: user code that says `x.shape[0]` gets a number, so it works, and the price is that the compiled kernel is now valid only for that batch size. The specialisation is silent, it is recorded, and it is checked — which is the whole bargain of guard-based dynamism.

### Checking

```ts
  evaluateGuards(): { passed: true; failedGuard: null } | { passed: false; failedGuard: ShapeGuard } {
    for (let i = 0; i < this._guards.length; i++) {
      const g = this._guards[i];

      if ('type' in g) {
        const val = this._resolve(g.sym);
        if (val % g.divisor !== 0) return { passed: false, failedGuard: g };
        continue;
      }

      const lVal = this._resolve(g.lhs);
      const rVal = this._resolve(g.rhs);
      if (!_GUARD_OPS[g.op](lVal, rVal)) return { passed: false, failedGuard: g };
    }

    return { passed: true, failedGuard: null };
  }
```

([`shape_env.ts:101`](../../../src/tracing/shape_env.ts).) Theorem 62.4's checking half. Linear, short-circuiting, and it returns *which* guard failed — a detail that costs nothing and is the difference between a diagnosable recompilation and a mysterious one.

### The cache

```ts
  function _findCachedEntry(inputs: readonly Tensor[]): CompiledEntry | null {
    const signature = inputSignatureOf(inputs);
    for (let i = 0; i < _cacheEntries.length; i++) {
      const entry = _cacheEntries[i];
      if (!signatureMatches(entry.inputSignature, signature)) continue;
      entry.shapeEnv.bindInputShapes(inputs);
      const { passed } = entry.shapeEnv.evaluateGuards();
      if (passed) return entry;
    }
    return null;
  }
```

([`compile.ts:456`](../../../src/tracing/compile.ts).) Definition 62.6 in ten lines. Note the order: the cheap signature test first, the guard evaluation second, and `bindInputShapes` mutating the entry's environment as a side effect of *probing* it — which is fine single-threaded and is the kind of thing worth knowing.

### Resolving a symbol at launch

The symbol has to reach the kernel. `bindInputShapes` fills the environment from the call's shapes ([`shape_env.ts:94`](../../../src/tracing/shape_env.ts)); the *output* buffer is sized by resolving the output's symbolic shape ([`compile.ts:165`](../../../src/tracing/compile.ts)); and the kernel's own trailing integer parameters are filled by [Chapter 59 §59.4](../ch59-the-runtime-module/README.md)'s `_extractShapeParams`, which reads each one off the shape of the tensor argument the compiler said it came from.

### The eager cache

```ts
function _cacheKey(opName: string, tensorArgs: readonly TensorLike[], scalarArgs: ScalarArgs, target: TargetLike): string {
  let key = opName;
  for (let i = 0; i < tensorArgs.length; i++) {
    key += '|' + tensorArgs[i].shape.join(',') + ':' + tensorArgs[i].dtype;
  }
  if (scalarArgs) {
    for (const [k, v] of Object.entries(scalarArgs)) {
      key += '|' + k + '=' + JSON.stringify(v);
    }
  }
  key += '|' + target.name;
  return key;
}
```

([`jit_cache.ts:80`](../../../src/dispatcher/jit_cache.ts).) Definition 62.7. This is the cache that stands behind every eager operation with no hand-written kernel — the compiler-generated kernels of [Chapter 60 §60.4](../ch60-a-pytorch-style-dispatcher/README.md) — so in an eager training loop it is consulted on the order of a hundred times per step and missed once per distinct shape.

## 62.5 Lab — symbols, guards, and the promise nobody made

```bash
node docs/part11/ch62-dynamic-shapes/labs/01-symbols-and-guards.mjs
```

The same function traced three ways:

```
  static (the default)
    input type   f32[6,8]      (-1 is the DYNAMIC marker)
    symbols      s0=hint 6 (arg 0, dim 0), s1=hint 8 (arg 0, dim 1)
    guards       s0 eq 6 , s1 eq 8
    output shape [[6,4]]

  dynamic_shapes: [{0}]
    input type   f32[-1,8]      (-1 is the DYNAMIC marker)
    symbols      s0=hint 6 (arg 0, dim 0), s1=hint 8 (arg 0, dim 1)
    guards       s1 eq 8 , s0 gt 0
    output shape [["s0",4]]

  dynamic_shapes: [true]
    input type   f32[-1,-1]      (-1 is the DYNAMIC marker)
    symbols      s0=hint 6 (arg 0, dim 0), s1=hint 8 (arg 0, dim 1)
    guards       s0 gt 0 , s1 gt 0
    output shape [["s0",4]]
```

Corollary 62.5, visible: the static row's guard set is two equalities, and the symbol machinery ran anyway. The output shape is where the symbol reappears — `["s0", 4]` says the result's first dimension is whatever the input's was, which is how `compile()` knows how large a buffer to allocate for a shape it does not know.

The batch-of-one rule:

```
  example batch 1: type [1,8]  guards s0 eq 1, s1 eq 8
  example batch 2: type [-1,8]  guards s1 eq 8, s0 gt 0
  example batch 7: type [-1,8]  guards s1 eq 8, s0 gt 0
```

Asking for a dynamic batch dimension and passing an example of size 1 gets a fully static kernel, with no warning. It is correct — it will recompile for every other batch — and it is the opposite of what was asked for.

Then the guard set doing its job, and then not:

```
  batch dimension only: guards s1 eq 8, s0 gt 0
    [6,8     ] -> reuse
    [1,8     ] -> reuse
    [4096,8  ] -> reuse
    [6,16    ] -> recompile (failed: s1 eq 8)
  every dimension: guards s0 gt 0, s1 gt 0
    [6,8     ] -> reuse
    [1,8     ] -> reuse
    [4096,8  ] -> reuse
    [6,16    ] -> reuse
```

The first block is Theorem 62.4 working: three shapes reuse, the fourth is caught by exactly the guard that should catch it. **The second block is the theorem's hypothesis failing.** `s1` is the dimension the `matmul` contracts against an `8×4` weight, so the program is only meaningful when `s1 = 8` — and the guard set says only `s1 > 0`, because nothing in the pipeline recorded that the contraction consumed it.

What that is worth, in compilations:

```
  static         4 compilations for 6 calls over 4 distinct shapes (1 at construction)
  dynamic dim 0  1 compilations for 6 calls over 4 distinct shapes (1 at construction)
```

Four against one, for the same six calls and the same six answers. That is the case for the feature, stated as plainly as the case against it.

And the symbol arriving in the generated code:

```
  function Object(buf_1, buf_4, buf_6, _ds_2) {
```

Three buffers and one integer. `_ds_2` is `s0`.

```bash
node docs/part11/ch62-dynamic-shapes/labs/02-the-caches-and-the-gap.mjs
```

The eager cache's key, one component at a time:

```
  request                            kernel
  add(4x4, 4x4) on cpu               add_jit_0
  add(4x4, 4x4) on cpu               add_jit_0
  add(8x4, 8x4) on cpu               add_jit_1
  add(4x4, 4x4) on wasm              add_jit_2
  mul(4x4, 4x4) on cpu               mul_jit_3
  sum(4x4) {"dim":0} on cpu          sum_jit_4
  sum(4x4) {"dim":1} on cpu          sum_jit_5
  sum(4x4) {"dim":0} on cpu          sum_jit_4

  8 requests, 6 compiled kernels.
```

A shape change misses, a target change misses, a *scalar* change misses — `sum(dim=0)` and `sum(dim=1)` are different kernels, because the reduction axis was an attribute and Part IV folded it in. And a repeat hits, twice.

```
    64x64     0.263ms  add_jit_6    compiled
    64x64     0.005ms  add_jit_6    cache hit
    65x65     0.237ms  add_jit_7    compiled
    64x64     0.006ms  add_jit_6    cache hit
    66x66     0.248ms  add_jit_8    compiled
```

A hit is roughly 50× cheaper than a miss, which is the number that makes the eager path viable at all: without this cache, every `a.add(b)` with no hand-written kernel would run the whole of Parts II–X.

Now the gap, executed:

```
  input [6,8   ] -> output [6x4]  finite
  input [12,8  ] -> output [12x4]  finite
  input [6,16  ] -> output [6x4]  NOT FINITE — 24 of 24 values are NaN
```

Under `dynamic_shapes: [true]`, an input whose contracting dimension moved from 8 to 16 **passed every guard, reused the kernel, and returned 24 NaNs.** No exception, no warning, no recompilation. The same call under the default:

```
  static  [6,16   ] -> threw: Graph verification failed (before graph passes): [Linear] op 'dot' (id=56):
          dot contracting dim size mismatch at [0]: lhs dim 1 size 16 vs rhs dim 0 size 8
```

The static path recompiles, the verifier of [Chapter 12](../../part2/ch12-valid-ir/README.md) runs on the new graph, and it says exactly what is wrong. The dynamic path never reaches a verifier, because it never compiles anything.

**This is the sharpest version of Theorem 62.4's hypothesis in the book.** The checking is right. The recording is incomplete. And the failure is silent, because a guard set that is missing a guard is indistinguishable from one that never needed it. §62.7 says where the missing guard belongs.

**Try this.** Trace the same function with `dynamic_shapes: [new Set([0])]` and call it with `[6, 16]`: the `s1 eq 8` guard fails, a second entry is compiled, and the answer is correct. The feature is safe exactly to the extent that the dimensions left symbolic are ones nothing silently consumes.

## 62.6 What this costs the generated code

Making a dimension symbolic is not free downstream, and the book has already paid the bill in three places.

[Chapter 35](../../part6/ch35-index-arithmetic/README.md)'s flattening multiplies by strides; a symbolic dimension makes a stride a runtime product rather than a constant, so the index expression grows a multiply per dynamic axis.

[Chapter 37](../../part6/ch37-proving-things-about-indices/README.md)'s interval analyser proves bounds to elide guards. A symbolic extent has the bound `[0, ∞)` unless a guard narrows it, so subscripts that would have been proved in range stay checked — and `_modBound` requires a *constant* positive divisor, so a `%` by a symbolic dimension yields no bound at all even though `0 ≤ x % n < n` holds for every positive `n`.

[Chapter 45](../../part8/ch45-sketches/README.md)'s tiling factorises a *number*. `jit_cache` refuses the scheduled GPU path outright when the function has shape parameters ([`jit_cache.ts:120`](../../../src/dispatcher/jit_cache.ts)), which is the honest response: a multi-level tiling of a symbolic extent needs either a guard on divisibility or a tail loop, and the sketch generators produce neither.

So the trade is real in both directions, and the default — static, recompile per shape — is the right one for a framework whose compilations cost single-digit milliseconds.

## 62.7 Traps and limits

### The guard set does not include the constraints that make the graph type-check

This is §62.5's finding, and it deserves to be stated as a design gap rather than a bug in a line. The guards recorded are: an equality per static dimension, `> 0` per dynamic dimension, and an equality per specialisation triggered by reading `.shape`. Nothing records the *relations between* dimensions that the operations require — that a `matmul`'s contracting extents agree, that a `cat`'s non-concatenated dimensions agree, that a broadcast is admissible.

Those relations are exactly what the operation registry's `inferResultTypes` and `verify` already know ([Chapter 11](../../part2/ch11-ops-as-a-dialect/README.md)). At trace time they run against `DYNAMIC` dimensions and succeed vacuously. A tracer that asked each recorded operation for the constraints it needs, and pushed them into the environment as guards, would close the gap with no new analysis — the knowledge exists, and nobody collects it.

Until then, `dynamic_shapes` is safe for a dimension no operation constrains — a batch axis in a feed-forward network — and unsafe for any other, with nothing in the API to say which is which.

### `evaluateGuards` reports the failure and nobody reads it

`_findCachedEntry` destructures `{ passed }` and discards `failedGuard` ([`compile.ts:462`](../../../src/tracing/compile.ts)). So the mechanism knows precisely why it is recompiling — "`s1 eq 8` failed, you passed 16" — and the information is dropped at the only call site. A single `trace.explain` there would make the difference between a recompilation a user can reason about and one they discover by watching a profiler.

### The cache is linear, unbounded, and never evicts

`_cacheEntries` is an array probed front to back, and every probe evaluates a guard list ([`compile.ts:458`](../../../src/tracing/compile.ts)). A serving workload with a hundred sequence lengths and static tracing accumulates a hundred entries, each holding a compiled runtime module, and the hundredth call evaluates ninety-nine guard lists first. There is no bound, no eviction, and no way to clear it. `shapeBuckets` exists to pre-compile a chosen set of shapes and does nothing to limit the set that accumulates afterwards.

### Divisibility guards can be recorded and are not

`guardDivisible` is implemented, `evaluateGuards` checks it, and **nothing in `src/` calls it**. It is the guard a tiling would need — "this extent is a multiple of 8, so the schedule that assumed so is valid" — and its absence is the other half of why the scheduled path refuses symbolic shapes outright rather than guarding them. The mechanism for the safe version exists; the producer does not.

### `ne`, `ge`, `lt` and `le` are unreachable

`_GUARD_OPS` implements six comparisons ([`shape_env.ts:12`](../../../src/tracing/shape_env.ts)). Only `eq` and `gt` are ever recorded — `eq` by specialisation and static dimensions, `gt` by `createInput`'s `> 0` ([`tracer.ts:63`](../../../src/tracing/tracer.ts)). The other four are complete, tested and produced by nothing, which is the signature of a guard vocabulary designed for an analysis that has not been written.

### The eager JIT cache never evicts either, and its key stringifies scalars with `JSON.stringify`

`_cache` and `_runtimeModules` are module-global `Map`s cleared only by `jitCacheClear`, which is not exported from [`src/index.ts`](../../../src/index.ts). And the scalar component of the key is `JSON.stringify(v)`, so two calls whose scalar arguments differ only in property *order* — an options object built two ways — produce two keys and two identical kernels. Harmless in the shipped call paths, where scalars are extracted positionally into a fresh object, and a hazard for any caller that builds one itself.

### A symbolic dimension in the tracer is a string; in the compiler it is a `SymInt`

`ShapeEnv._resolve` handles three representations — `number`, `string`, and `SymInt` ([`shape_env.ts:135`](../../../src/tracing/shape_env.ts)) — because the tracer names symbols with strings while the compiler's analysis layer uses `SymInt` expression trees. The two meet only in this function and in `hintOf`, which bridges them by building a fresh hint map out of `_symbols` on every call and handing it to `SymInt.evaluate` — an allocation and a full copy of the symbol table per hint query, on a path that runs once per dimension.

The bridge is by name and by name only: `namedDimToSym` promotes a string dimension to `SymInt.var(dim)` and `symToNamedDim` demotes it back ([`ops/shape.ts:8`](../../../src/compiler/ir/graph/ops/shape.ts)). A `SymInt` naming `s3` and a tracer symbol named `s3` are the same dimension because the strings match, and for nothing else. Every `ShapeEnv` starts `_nextId` at zero, so two environments both name their first dynamic dimension `s0`; an expression built against one and resolved against another finds a binding and returns a number rather than complaining. Nothing records which environment allocated a `SymInt`'s free variables, which is what a symbol table with identity, rather than a shared string namespace, would give.

## 62.8 Read the tests

- [`tests/tracing/shape-env.test.js`](../../../tests/tracing/shape-env.test.js) — symbol allocation, the `> 1` rule, specialisation and its memoisation, guard evaluation and which guard is reported as failing, and the two ways a binding can be missing: an unbound symbol, and an input list that cannot supply one.
- [`tests/e2e/dynamic-shapes.test.js`](../../../tests/e2e/dynamic-shapes.test.js) — compiling once and calling with several shapes, the recompilation count, and symbolic output shapes.
- [`tests/tracing/compile.test.js`](../../../tests/tracing/compile.test.js) — the compiled-function cache: signature matching, guard-driven recompilation, and `shapeBuckets`.
- [`tests/dispatcher/dispatcher.test.js`](../../../tests/dispatcher/dispatcher.test.js) — the eager path that stands on `jitCompile`, including the shape-keyed cache behind it.

---

**Next:** [Chapter 63 — Training end to end](../ch63-training-end-to-end/README.md), where the running example is finally compiled, differentiated, trained, and measured against the eager path it replaces.
