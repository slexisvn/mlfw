# Chapter 30 — Trading memory for recomputation

Chapter 29 treated the saved set as given. It is not given; it is chosen, and it is the largest single consumer of memory in a training run.

This is the third resource trade in the book — after Chapter 25's addresses-for-locality and Chapter 26's accuracy-for-speed — and it is the only one that is on by default, because there is no configuration in which saving *everything* is the right answer.

## 30.1 The problem: the forward pass is still alive when the backward pass starts

Run a forward pass through a 48-layer network. Every layer produces an activation, and by Definition 27.5 the backward pass may need it. So none of them can be freed until the backward pass has walked past it — which happens in reverse order, so the *first* layer's activation is the *last* one released.

Peak memory is therefore reached at the moment the forward pass ends, and it holds one activation per layer. That number does not depend on how clever the allocator is (Chapter 50); the values are genuinely live. It scales with depth and with batch size, and for large models it is the reason a training job does not fit on a device.

Three ways out. Use a smaller batch, which changes the optimization problem. Use a smaller model, which changes the answer. Or **do not keep the activation** — recompute it, from something you did keep, at the moment the backward pass asks.

The third is free of the first two's downsides and is not free: it costs arithmetic. The question this chapter answers is how much, and who decides.

## 30.2 Intuition: two granularities of the same idea

The trade appears at two scales, and the compiler implements both.

**Per operation.** For each forward value the backward pass wants, ask: is it cheaper to keep this tensor around, or to recompute it from its inputs? An `exp` costs one elementwise pass to redo and one full tensor to keep — for anything larger than cache, recomputing wins easily. A `dot` costs a whole matrix multiply to redo — keep it. This is a local decision, made once per operation, and it is the default policy.

**Per segment.** Per-operation decisions bottom out when an operation's *inputs* were themselves not saved: then recomputing it means recomputing them too, and the recursion can walk all the way back to the function's arguments. The segment view embraces that. Cut the program into `k` chunks, save only the values at the cuts, and when the backward pass reaches a chunk, replay the whole chunk forward from its saved boundary before differentiating it.

The second is *checkpointing*, and it is the one with the famous result: cutting a chain of `n` layers into `√n` segments of `√n` layers each holds `√n` boundary values and `√n` values inside the segment currently being replayed — `O(√n)` memory for one extra forward pass. Turning linear memory into square-root memory for a 33% time increase is one of the best trades in the field.

## 30.3 Theory

Let `F` be a forward function with operations `o₁…o_n` in topological order.

> **Definition 30.1 (Rematerialization).** **(classical)** A backward construction *rematerializes* a forward value `v` if, instead of receiving `v` as an argument, it emits a copy of the operations producing `v` and evaluates them from values it does have.

> **Definition 30.2 (Segmentation).** **(classical)** A *segmentation* of `F` is a partition of `o₁…o_n` into consecutive segments `S₁…S_k`. Its *boundary set* is the set of values produced in one segment and read in another, plus the function's inputs and outputs.

> **Theorem 30.3 (√n checkpointing; Chen et al., 2016).** **(classical)** For a chain of `n` equal-cost layers, storing only the boundary values of `√n` segments of `√n` layers gives peak activation memory `O(√n)` at the cost of one additional forward pass.
>
> *This is Theorem 26.2 restated.* Chapter 26 met the result from the buffer level, where a pass tries to reach a byte budget on an already-lowered function; this chapter meets it from the graph level, where a builder decides what to hand across the forward-backward boundary. Same theorem, two customers, and — §30.7 — neither of them implements it.

*Proof sketch.* At any moment the construction holds `k = √n` boundary values plus the intermediates of the single segment currently being replayed, which is at most `√n` — so peak is `O(√n)`. Each segment is replayed exactly once during the backward walk, and the segments partition the program, so the total extra work is one forward pass over the program. ∎

Two remarks the theorem does not make, and both matter here.

**It is a statement about chains, and this chapter's pass is not a chain algorithm.** A general DAG has no notion of "the first `√n` operations", and a segment cut across a wide graph can have a boundary set far larger than one value. Definition 30.2's boundary set is computed, not assumed, and §30.7 is about what that costs.

That distinction is worth making sharply, because Theorem 30.3 is the famous result and it is tempting to attach its bound to whatever code sits nearest. The shipped path is not a segmentation algorithm at all: it is a *policy* consulted once per operation — "should this value be recomputed rather than saved?" — and a resolver that recurses through operands until it hits something saved. It has no segments, no `k`, and therefore no `√n`. **Theorem 30.3 bounds a construction this compiler does not perform**, and it appears here because it is the idea the policies are groping towards and the benchmark any future segmenter should be measured against. The `√n` segmenter that *would* implement it exists in [`checkpoint_policy.ts`](../../../src/compiler/ad/checkpoint_policy.ts), is tested, and §30.7 explains why nothing can reach it.

**It says nothing about which segmentation is best.** Choosing cuts to minimize memory subject to a time budget is a discrete optimization problem; `√n` is the closed-form answer for the uniform chain and a heuristic otherwise.

> **Definition 30.4 (Rematerialization is exact).** **(stated here)** Let every operation on the recomputed subgraph be **pure** — a function of its operands alone — and **deterministic**, and let it be evaluated under the same numerical mode (Definition 1.4) as the original. Then recomputing a value from the same inputs by the same operations yields a bit-identical value, so rematerialization changes memory and time and does not change the gradient.

That is worth stating because the other two trades in this book do not have it: quantization changes the answer and layout changes the addresses. Remat is the rare trade that is invisible in the output, and §30.5 checks it.

**But the hypotheses are hypotheses, and nothing here enforces them.** Three ways the equality fails, in increasing order of how likely you are to meet them:

- **Randomness.** A `dropout` mask, or anything drawing from the RNG, is not deterministic. Recomputing it draws a *different* mask, so the backward pass differentiates a different function from the one the forward pass ran. Note this is not the same failure as Chapter 5 §5.5's captured-noise problem: there the value was frozen at trace time, here it would be re-drawn at run time. Frameworks that support remat handle this by saving and replaying the RNG state; this one has no such mechanism, and no check that would notice.
- **Side effects.** An operation that writes to a buffer executes its write twice. `hasSideEffect` exists on every `OpDef` (Chapter 19), and the remat path does not consult it — the candidate predicate in `_materialize` recurses through `definingOp` until it reaches a saved value, with a cycle guard and no purity test.
- **A changed numerical mode.** The recomputed subgraph is ordinary IR and is optimized like ordinary IR. If a rewrite fires on the copy that did not fire on the original — or fires differently because the copy sits in a different fusion group — the two evaluations are N1-equivalent at best, and Chapter 20's reassociation defect makes N2 reachable. The bit-identical claim is then false, quietly and by a small amount.

In practice the graphs this pass is pointed at are pure arithmetic, which is why none of the three has bitten. The honest status is: **Definition 30.4 is a precondition the caller must meet, not an invariant the compiler maintains.** A purity check at candidate selection would be a few lines and would convert the first two from silent wrongness into a refused rematerialization.

## 30.4 In mlfw: a predicate and four segmenters

### The per-operation policy

[`ad/remat_policy.ts`](../../../src/compiler/ad/remat_policy.ts), 57 lines, is one predicate ([`remat_policy.ts:31`](../../../src/compiler/ad/remat_policy.ts)):

```ts
  shouldRematerialize(op: Operation): boolean {
    if (this._alwaysRemat.has(op.opName)) return true;
    if (this._neverRemat.has(op.opName)) return false;

    const resultType = op.numResults > 0 ? op.getResult(0).type as TensorType : null;
    if (!resultType || !resultType.shape) return false;

    const numel = resultType.numel();
    if (numel > this._sizeThreshold) return false;

    return this._isElementwise(op);
  }
```

Four tests in order: a name list that always recomputes, a name list that never does, a size cap, and then a structural fallback — recompute it if it is elementwise. The two lists are the interesting part ([`remat_policy.ts:20`](../../../src/compiler/ad/remat_policy.ts)):

```ts
    this._alwaysRemat = new Set(opts.alwaysRemat || [
      'neg', 'abs', 'sign', 'floor', 'ceil',
      'exp', 'log', 'sqrt', 'rsqrt',
      'sin', 'cos', 'tanh',
    ]);
    this._neverRemat = new Set(opts.neverRemat || [
      'matmul', 'dot', 'conv', 'reduce',
      'custom_call', 'pool2d',
    ]);
```

Read those as a claim about arithmetic intensity (Chapter 4). Everything in the first list is one elementwise pass — cheaper to redo than to move. Everything in the second is a contraction whose output is much smaller than its work — cheaper to keep. Nothing in either list is a judgement about the *derivative*; they are judgements about the *forward* operation's cost, which is the right basis.

Note the size cap runs *before* the elementwise fallback but *after* the name lists, so a `tanh` on a 100-million-element tensor is still recomputed. That is deliberate for `tanh` and is the direction you want, since a large tensor is exactly the one you cannot afford to keep.

The builder consults it once per candidate result ([`backward_builder.ts:385`](../../../src/compiler/ad/backward_builder.ts)):

```ts
  _shouldSaveResult(op: Operation): boolean {
    if (this._rematPolicy) {
      return !this._rematPolicy.shouldRematerialize(op);
    }
    return !FALLBACK_REMAT_OPS.has(op.opName);
  }
```

and with no policy at all falls back to five operation names ([`backward_builder.ts:54`](../../../src/compiler/ad/backward_builder.ts)): `neg`, `abs`, `sign`, `floor`, `ceil`. That fallback exists for direct users of `BackwardGraphBuilder`; every path through `compileWithBackward` constructs a real policy ([`compile_backward.ts:95`](../../../src/tracing/compile_backward.ts)).

Then `_materialize` (Chapter 29) does the actual rebuilding: a value that was not saved is produced by cloning its defining operation into the backward function, recursively, until the recursion reaches something that *was* saved. Definition 30.1, implemented as a graph clone rather than as a re-execution.

### The segment policies

[`ad/checkpoint_policy.ts`](../../../src/compiler/ad/checkpoint_policy.ts), 179 lines, is Definition 30.2 and four ways to choose the cuts:

| Policy | Cuts where |
|---|---|
| `EveryKPolicy` | every `k` operations |
| `SqrtPolicy` | `⌈√n⌉` operations per segment — Theorem 30.3 ([`checkpoint_policy.ts:94`](../../../src/compiler/ad/checkpoint_policy.ts)) |
| `MemoryBudgetPolicy` | when the running total of result bytes would exceed a threshold |
| `ExplicitPolicy` | at operations a caller-supplied predicate selects |

All four end by calling the same `computeBoundaries` ([`checkpoint_policy.ts:19`](../../../src/compiler/ad/checkpoint_policy.ts)), which walks every segment and records the values crossing in or out — Definition 30.2's boundary set computed rather than assumed, which is what makes the machinery correct on graphs that are not chains.

The consuming half is a second `build` path ([`backward_builder.ts:518`](../../../src/compiler/ad/backward_builder.ts)), and its loop is the theorem ([`backward_builder.ts:518`](../../../src/compiler/ad/backward_builder.ts)):

```ts
    for (let s = segments.length - 1; s >= 0; s--) {
      const seg = segments[s];

      const recomputeMap = new Map<number, Value>();

      const resolve = (v: Value) => recomputeMap.get(v.id) || savedValueMap.get(v.id) || constantMap.get(v.id) || v;
      for (const op of seg.ops) {
        const cloned = cloneOpWithRegions(builder, op, resolve);
```

Segments in reverse; within each one, **replay forward** and then backprop over the replayed copies. `resolve` is a three-level lookup — this segment's recomputation, the saved boundary values, the cloned constants — and it is the same `resolveValue` callback the sweep in Chapter 29 takes.

**One thing is missing and it is the whole public story:** no code path in `src/tracing` constructs a `CheckpointPolicy`. `compileWithBackward` reads `opts.rematPolicy` and `opts.remat` and nothing else, so the four segmenters, the boundary computation and the entire `_buildCheckpointed` path are reachable only by constructing a `BackwardGraphBuilder` directly. Theorem 30.3 is implemented, tested, and cannot be turned on through the API this book's labs use.

## 30.5 Lab — What to save

```bash
node docs/part5/ch30-memory-for-recomputation/labs/01-what-to-save.mjs
```

`rematPolicy` is duck-typed — `_shouldSaveResult` calls exactly one method on it — so any object with `shouldRematerialize(op)` can drive the saved set from outside the compiler. That is the same trick Part III used with `passContext`, and it is the reason this chapter has a lab at all.

```
=== f(x) = sum(sqrt(tanh(exp(x)) * x)),  x is [1, 4] ===

default policy                 forward returns  1   backward:  2 saved, 17 ops
save everything                forward returns  5   backward:  6 saved, 13 ops
recompute everything           forward returns  1   backward:  1 saved, 17 ops
recompute only 'exp'           forward returns  4   backward:  5 saved, 14 ops
```

The trade in one table. **Save everything**: the forward function returns five values instead of one, the backward takes six, and the backward body is 13 operations. **Recompute everything**: the forward returns one value, the backward takes one saved tensor, and the body grows to 17. Four extra operations bought five tensors.

Note the default sits almost at the recompute end — two saved, 17 operations — because every operation in this program (`exp`, `tanh`, `sqrt`) is in the `alwaysRemat` list. For this function the shipped policy *is* "recompute nearly everything", and that is the correct call for a chain of elementwise transcendentals.

And Definition 30.4:

```
=== do they agree? ===
  default          [0.764627,0.521398,0.409601,0.353561]
  vs save-all      identical
  vs recompute-all identical
  vs exp-only      identical
```

Four different memory profiles, one gradient. That invariant is what makes the whole trade safe to make automatically — unlike quantization, there is no accuracy question to put in front of the user.

The extreme case is worth reading in full, because it shows what "recompute" actually means:

```
=== what "recompute everything" put in the backward function ===
  func @backward_Object(%0: tensor<f32>, %1: tensor<1x4xf32>) -> (tensor<1x4xf32>) {
  %2 = exp(%1) : tensor<1x4xf32>
  %3 = tanh(%2) : tensor<1x4xf32>
  %4 = mul(%3, %1) : tensor<1x4xf32>
  %5 = sqrt(%4) : tensor<1x4xf32>
  %6 = reshape(%0) {new_shape = [1, 1]} : tensor<1x1xf32>
  ...
```

The first four operations of the backward function **are the forward function**, replayed from the single saved input. Everything after `%6` is the actual derivative. This is Definition 30.1 as IR: the backward graph contains a copy of the forward graph, and the pipeline of Part IV then fuses and schedules it like any other code — which is why the recomputation is cheaper than its operation count suggests.

Finally, the reason segments exist:

```
=== how the saved set grows with depth ===
  (a stack of Linear layers: a `dot` result is never rematerialized)
  layers   forward outputs   backward saved args
       1                 3                     6
       2                 5                    10
       4                 9                    18
       8                17                    34
```

Exactly linear: four more saved tensors per layer. Per-operation remat cannot fix this, because the values it refuses to recompute are the `dot` results — and it refuses correctly, since recomputing a matmul is not cheap. **The saved set of a deep network grows linearly with depth no matter how good the per-operation policy is**, and that is precisely the situation Theorem 30.3 addresses and this compiler cannot currently reach through its public API.

**Try this.** Add `'dot'` to a policy that recomputes it — `{ shouldRematerialize: (op) => op.opName === 'dot' }` — and watch the saved column go flat and the backward operation count explode. Then decide whether you would ship it.

## 30.6 What checkpointing would look like here

Since the lab cannot reach it, it is worth being concrete about what the implemented path does, so the gap is a gap and not a mystery.

`SqrtPolicy.segment` filters out `return` and `constant`, takes `⌈√n⌉` operations per segment, and computes boundaries. For the eight-layer stack above — roughly 40 operations — that is seven segments of six. The backward construction would then save the seven boundary carriers instead of thirty-four values, and replay six operations per segment during the backward walk.

The measurement that is missing from this chapter is the one that would put a number on it, and the reason it is missing is one unwired option rather than anything conceptual: `_buildCheckpointed` is complete, [`tests/compiler/ad/checkpoint.test.js`](../../../tests/compiler/ad/checkpoint.test.js) exercises all four policies against non-checkpointed gradients, and nothing between `compileWithBackward` and `BackwardGraphBuilder` passes the option through.

## 30.7 Traps and limits

- **`CheckpointPolicy` is unreachable from `compileWithBackward`.** [`compile_backward.ts:95`](../../../src/tracing/compile_backward.ts) constructs a `RematPolicy` and never looks for a checkpoint policy; `_compileSeparate` and `_compileJoint` pass only `rematPolicy` to the builders. The four segmenters, `computeBoundaries`, and both `_buildCheckpointed` paths are live code with tests and no production caller. This is the same shape as Chapter 21's `nativeOps` and Chapter 25's `layoutAwareOps`: designed, correct, unwired.
- **`scanCheckpoint` is unreachable the same way.** `BackwardBuilderOpts` has a third field for checkpointing *inside* a `scan` ([`backward_builder.ts:43`](../../../src/compiler/ad/backward_builder.ts)), consumed by Chapter 31's scan differentiator, and no caller sets it either.
- **`MemoryBudgetPolicy` measures the wrong bytes.** It accumulates each operation's *result* size and cuts when the running sum exceeds a threshold ([`checkpoint_policy.ts:128`](../../../src/compiler/ad/checkpoint_policy.ts)). That is the total produced by a segment, not the peak *live* at any point in it, so a segment of ten operations that each free the previous one is charged for all ten. Chapter 49's liveness analysis is the thing that would answer the real question, and this policy does not use it.
- **The size threshold is in elements, and the name says otherwise.** `sizeThreshold` defaults to `1024 * 1024` and is compared against `numel()` ([`remat_policy.ts:39`](../../../src/compiler/ad/remat_policy.ts)) — so it is one million *elements*, four megabytes at f32, not one megabyte.
- **Rematerialization is unbounded in depth.** `_materialize` recurses through operands until it reaches a saved value, with a cycle guard but no depth limit. A policy that recomputes everything turns the backward graph into a copy of the whole forward graph, which is exactly what §30.5's fourth block shows at four operations and would be thousands on a real model. `maxRematDepth` exists in `RematPolicyOpts` ([`remat_policy.ts:5`](../../../src/compiler/ad/remat_policy.ts)), is assigned in the constructor, and is read by nothing.
- **Recomputation is exact but not free of numerics.** Definition 30.4 holds because the same operations run on the same inputs. It stops holding the moment recomputation happens on a different device, under a different fusion decision, or after a pass that reassociates — none of which happens today, and all of which are things Part IV is otherwise allowed to do.

## 30.8 Read the tests

- [`tests/compiler/ad/checkpoint.test.js`](../../../tests/compiler/ad/checkpoint.test.js) — all four segmenters, both builders, and the assertion that checkpointed gradients equal non-checkpointed ones, which is Definition 30.4 as an executable claim.
- [`tests/compiler/ad/backward-compile.test.js`](../../../tests/compiler/ad/backward-compile.test.js) — the saved-value signature of the built backward function under the default policy.

---

**Next:** [Chapter 31 — Differentiating control flow](../ch31-differentiating-control-flow/README.md), which is the last thing the reverse sweep cannot do with a rule per operation: an operation whose body is a program, and an operation whose derivative does not exist.
