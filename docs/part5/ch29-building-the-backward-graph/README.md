# Chapter 29 — Building the backward graph

Sixty-seven rules, each knowing the derivative of one operation and nothing about any other. This chapter is the driver that turns them into a program.

It has four jobs, and only the first is obvious. Walk the operations backwards. Decide which of them are worth walking at all. Sum the contributions when one value fed several consumers. And answer, every time a rule asks for a forward value, the question of where that value is — because the backward function is a *different function* and does not have it.

## 29.1 The problem: the backward pass is not the forward pass run in reverse

Take the smallest program that shows the difficulty:

```
t = x * x
L = sum(t)
```

Reverse mode says: start with `∂L/∂L = 1`, apply `sum`'s rule to get `∂L/∂t`, apply `mul`'s rule to get `∂L/∂x`. Three steps, and each of the three raises a question the rules cannot answer.

**`mul`'s rule returns two gradients and they are for the same value.** Its operands are `x` and `x`. Two contributions arrive at one value, and `∂L/∂x` is their sum — but neither invocation of the rule knows the other happened.

**`mul`'s rule wants `x`.** The backward program does not compute `x`; the forward program did, and finished. Somebody has to arrange for the value to cross from one function to the other, or to be computed again.

**And in a real program most operations do not need differentiating at all.** An index computation, a shape constant, a mask — these participate in the forward pass and contribute nothing to any gradient. Walking them costs graph size in the backward function, which is then optimized, lowered and code-generated like everything else.

None of these is a calculus question. All three are graph questions, and the answers are what this chapter is.

## 29.2 Intuition: a second graph, built backwards, with wires to the first

Picture the forward graph laid out left to right. The backward graph is a second graph laid out right to left underneath it, one node group per forward operation, built by visiting the forward operations in reverse.

Three things run between the two pictures.

**Cotangents flow right to left.** Each value in the forward graph has a matching cotangent in the backward one, and a rule turns the cotangents of an operation's results into the cotangents of its operands.

**A fan-out becomes a fan-in.** Where one forward value fed three operations, three cotangents arrive back at it and must be added. The forward graph's fan-out is the backward graph's sum, and this is the single most reliable way to remember what reverse mode does to a program's shape.

**Values flow left to right.** The rules need forward values. Either the forward function hands them over — which makes them extra outputs of one function and extra inputs of the other — or the backward function computes them again from something it does have. Both happen, and choosing between them is Chapter 30.

The last decision is how to package it. Two functions, or one function that does both? The compiler implements both, they produce identical numbers, and §29.6 measures which is faster and finds the answer is not the one the picture suggests.

## 29.3 Theory

Fix a forward function `F` in SSA form with inputs `x₁…x_n`, a topological order `O` of its operations, and outputs `y₁…y_m`.

> **Definition 29.1 (Gradient reachability).** A value `v` is *gradient-reachable* if `v` is a function output, or `v` is an operand of an operation `f` with a VJP rule such that some result of `f` is gradient-reachable. The reachable set is the least fixed point of that rule.

Reachability is computed by one backward walk, because an operation's results precede its operands in reverse topological order — so a single reverse pass reaches the fixed point without iterating.

> **Definition 29.2 (Cotangent accumulation).** For a value `v` with consumers `f¹…f^k` in `F`, the cotangent of `v` is
>
> `w_v = Σ_{i=1..k} (rule of f^i applied at the operand position of v)`
>
> and is complete only after every consumer of `v` has been processed.

> **Theorem 29.3 (Correctness of the reverse sweep).** Processing the operations of `F` in reverse topological order, applying each operation's rule to the cotangents of its results and accumulating into the cotangents of its operands, yields `w_v = (∂L/∂v)ᵀ` for every gradient-reachable `v`, where `L` is the value seeded at the outputs.

*Proof sketch.* By induction over reverse topological order. The outputs are seeded by hypothesis. For any other value `v`, every consumer of `v` appears strictly later than `v`'s producer in topological order, hence strictly earlier in the reverse walk — so when the walk reaches `v`'s producer, every consumer has already contributed. The multivariate chain rule states exactly that `∂L/∂v` is the sum over consumers of the local derivative times the consumer's cotangent, which is Definition 29.2. SSA is what makes "every consumer" a finite, statically known set. ∎

The proof leans on SSA twice and it is worth naming both places: once for "a value has one producer", so the walk knows where to deposit `w_v`, and once for "a value's uses are enumerable", so the sum is over a known set. In a language with mutable aliasing neither holds, which is why source-level AD tools for such languages are so much harder than this one.

> **Definition 29.4 (Saved set, stated here).** The *saved set* `S` of a backward construction is the set of forward values passed into the backward function as arguments. A construction is *well-formed* if every forward value a rule reads is either in `S` or recomputable from `S` by operations the construction replays.

> **Definition 29.5 (Separate and joint form, stated here).** The *separate* form compiles two functions: `F' : (x) → (y, S)` and `B : (w, S) → (∂L/∂x)`. The *joint* form compiles one: `J : (x, w) → (y, ∂L/∂x)`, with `S` never crossing a function boundary.

> **Corollary 29.6.** The two forms compute the same gradients. They differ in what crosses a function boundary — `|S|` tensors in the separate form, none in the joint — and in whether the caller can obtain `y` without supplying `w`.

That last clause is the one with teeth, and §29.6 measures what it costs.

## 29.4 In mlfw: 523 lines, of which the sweep is thirty

[`ad/backward_builder.ts`](../../../src/compiler/ad/backward_builder.ts). The sweep itself is Definition 29.2 and nothing else ([`backward_builder.ts:56`](../../../src/compiler/ad/backward_builder.ts)):

```ts
export function backpropOps(orderedOps: readonly Operation[], { accumulator, builder, needsGrad, resolveValue, handleRegionOp = null }: BackpropOptions): void {
  for (let i = orderedOps.length - 1; i >= 0; i--) {
    const op = orderedOps[i];
    if (op.opName === 'return' || op.opName === 'constant') continue;

    const hasGradResult = op.results.some(r => needsGrad.has(r.id));
    if (!hasGradResult) continue;

    const gradOuts: (Value | null)[] = [];
    for (let r = 0; r < op.numResults; r++) gradOuts.push(accumulator.get(op.getResult(r).id));
    if (gradOuts.every(g => g === null)) continue;
```

Three `continue`s before any calculus happens, and they are three different reasons to skip an operation. A terminator or a constant has no derivative to speak of. An operation none of whose results are gradient-reachable is not on the path (Definition 29.1). And an operation whose results are reachable but whose cotangents are all still `null` has nothing to propagate *yet* — which cannot happen in a correct reverse walk and is checked anyway.

Notice what this function is not: it is not a method. `backpropOps` is a free function taking a callback for every decision it does not own — `resolveValue` for Definition 29.4, `handleRegionOp` for Chapter 31 — and both the separate and the joint builder call it with different callbacks. One sweep, four call sites, no duplicated calculus.

### Reachability

[`backward_builder.ts:91`](../../../src/compiler/ad/backward_builder.ts) is Definition 29.1, seeded from the return operation:

```ts
export function computeGradReachability(func: GraphFunction, topoOrder: readonly Operation[]): Set<number> {
  const needsGrad = new Set<number>();
  const returnOp = func.getReturnOp() as Operation;

  for (const val of returnOp.operands) {
    needsGrad.add(val.id);
  }
```

then one reverse walk propagating operand-wards, with two escapes: an operation with no VJP rule stops the walk, and so does a declared gradient barrier ([`backward_builder.ts:112`](../../../src/compiler/ad/backward_builder.ts)):

```ts
    if (!getVJPRule(op.opName)) continue;
    if (isGradientBarrier(op.opName)) continue;
```

Those two lines are why the comparison inside a `relu` does not drag its operands into the backward graph, and Chapter 31 is about the difference between them.

### The accumulator

[`ad/grad_accumulator.ts`](../../../src/compiler/ad/grad_accumulator.ts), 57 lines, is Definition 29.2. Contributions are appended, not added ([`grad_accumulator.ts:23`](../../../src/compiler/ad/grad_accumulator.ts)):

```ts
  accumulate(valueId: number, gradValue: Value | null | undefined): void {
    if (!gradValue) return;
    let arr = this._pending.get(valueId);
    if (!arr) { arr = []; this._pending.set(valueId, arr); }
    arr.push(gradValue);
    this._reduced.delete(valueId);
  }
```

and the sum is emitted only when somebody asks ([`grad_accumulator.ts:31`](../../../src/compiler/ad/grad_accumulator.ts)):

```ts
  _treeReduce(values: readonly Value[]): Value {
    let level: readonly Value[] = values;
    while (level.length > 1) {
      const next: Value[] = [];
      for (let i = 0; i < level.length; i += 2) {
        if (i + 1 < level.length) next.push(this._builder.add(level[i], level[i + 1]).getResult(0));
        else next.push(level[i]);
      }
      level = next;
    }
    return level[0];
  }
```

Two decisions worth noticing. **Lazy**: nothing is emitted until `get`, so a value that turns out to be unused costs no `add`. And **a tree, not a chain**: `k` contributions become `⌈log₂ k⌉` levels of additions rather than `k−1` sequential ones, which matters for the critical path once `k` is large — a weight tensor shared across a batch dimension can have hundreds of contributions. §29.5 watches a five-contribution tree get built.

`get` memoizes the reduction and `accumulate` invalidates the memo, which is the shape you need when contributions can arrive after somebody has already asked.

### Where forward values come from

The separate builder declares the saved set as extra function arguments ([`backward_builder.ts:192`](../../../src/compiler/ad/backward_builder.ts)):

```ts
    const bwdFunc = new GraphFunction(
      `backward_${forwardFunc.name}`,
      inputTypes,
      gradInputTypes
    );
```

where `inputTypes` is `[...gradOutputTypes, ...savedTypes]` — the cotangents first, then the saved values. `_identifySavedValues` ([`backward_builder.ts:284`](../../../src/compiler/ad/backward_builder.ts)) picks them, and `_materialize` ([`backward_builder.ts:248`](../../../src/compiler/ad/backward_builder.ts)) is the `resolveValue` callback: given a forward value, return the backward argument holding it, or **clone the operations that produce it** into the backward function. That second half is rematerialization, and it is Chapter 30.

The joint builder does not need any of this ([`joint_builder.ts:117`](../../../src/compiler/ad/joint_builder.ts)):

```ts
    const inputTypes = [...forwardFunc.inputTypes, ...gradOutputTypes];
    const outputTypes = [...forwardFunc.outputTypes, ...forwardFunc.inputTypes];
```

One function taking the forward inputs plus one cotangent per output, returning the forward outputs plus one gradient per input. Its `resolveValue` is a plain map lookup, because it replayed the whole forward pass into the same function first ([`joint_builder.ts:132`](../../../src/compiler/ad/joint_builder.ts)) and every forward value is simply *there*. That is Definition 29.5, and the reason the joint builder is 163 lines to the separate builder's 523.

## 29.5 Lab 1 — The reverse sweep

```bash
node docs/part5/ch29-building-the-backward-graph/labs/01-the-reverse-sweep.mjs
```

Three programs chosen so the driver, not the rules, is what changes.

```
=== one value, two consumers:  (x * x) + x ===
  func @backward_Object(%0: tensor<xf32>, %1: tensor<1x2xf32>, %2: tensor<xf32>) -> (tensor<1x2xf32>) {
  %3 = reshape(%0) {new_shape = [1, 1]} : tensor<1x1xf32>
  %4 = broadcast_in_dim(%3) {broadcast_dimensions = [0, 1], result_shape = [1, 2]} : tensor<1x2xf32>
  %5 = mul(%4, %1) : tensor<1x2xf32>
  %6 = add(%4, %5) : tensor<1x2xf32>
  %7 = add(%6, %5) : tensor<1x2xf32>
  return(%7)
    [0] shape [1,2] = [[3,5]]
```

`x` is read three times — twice by the `mul` and once by the `add` — so three cotangents arrive: `%4` from the addition and `%5` twice from the multiplication. The two `add`s at `%6` and `%7` are `_treeReduce` on a three-element list: pair the first two, carry the odd one, pair again. Neither `add` was written by a rule.

The arithmetic checks: `d/dx (x² + x) = 2x + 1`, which at `[1, 2]` is `[3, 5]`. And CSE (Chapter 19) has already merged what were two identical `mul(%4, %1)` operations into one `%5`, so the graph shows the *contributions* twice and the *computation* once — the backward graph being optimized by the ordinary pipeline, exactly as Chapter 28 promised.

Push it to five contributions and the tree is visible:

```
=== one value, three consumers:  x*x + x*x + x ===
  %5 = mul(%4, %1) : tensor<1x2xf32>
  %6 = add(%4, %5) : tensor<1x2xf32>
  %7 = add(%5, %5) : tensor<1x2xf32>
  %8 = add(%6, %7) : tensor<1x2xf32>
  %9 = add(%8, %5) : tensor<1x2xf32>
    [0] shape [1,2] = [[5,9]]
```

Five contributions — four from the two multiplications, one from the addition — reduced in three levels rather than four sequential additions. `%6` and `%7` are level one, `%8` is level two, `%9` folds in the carried odd element. `d/dx (2x² + x) = 4x + 1` gives `[5, 9]`. 

The middle case is Definition 29.1:

```
=== an input the output does not depend on ===
  func @backward_Object(%0: tensor<xf32>, %1: tensor<1x2xf32>, %2: tensor<xf32>) -> (tensor<1x2xf32>) {
  %3 = reshape(%0) {new_shape = [1, 1]} : tensor<1x1xf32>
  %4 = broadcast_in_dim(%3) {broadcast_dimensions = [0, 1], result_shape = [1, 2]} : tensor<1x2xf32>
  return(%4)
  gradients returned: 2
    [0] shape [1,2] = [[1,1]]
    [1] shape [1,2] = [[0,0]]
```

Two inputs went in, and **the backward function has one result**. The second input never reached the output, so it is not in the reachable set, so it gets no output slot — the graph does not contain a single operation on its behalf. The caller still receives two gradients because the wrapper fills the missing one with zeros, which is the right API and hides a real optimization: an unreachable input costs nothing in the compiled program, not even a zero tensor.

**Try this.** Make the second input reach the output through an operation with no gradient — `a.add(b.floor())` — and compare. The input is now reachable, the backward graph does contain work for it, and the answer is still zeros.

## 29.6 Lab 2 — Separate or joint

```bash
node docs/part5/ch29-building-the-backward-graph/labs/02-separate-or-joint.mjs
```

The running two-layer MLP, compiled both ways.

```
=== mode: 'separate' ===
  compiled modules: 2
    @forward   8 ops   @Sequential(%0: tensor<2x2xf32>, %1: tensor<8x2xf32>, %2: tensor<8xf32>, %3: tensor<1x8xf32>, %4: tensor<1xf32>) -> (tensor<2x1xf32>, tensor<2x8xf32>, tensor<2x8xf32>, tensor<2x8xf32>, tensor<2x8xf32>, tensor<8x1xf32>, tensor<2x1xf32>)
    @backward  13 ops   @backward_Sequential(%0: tensor<2x1xf32>, %1: tensor<8x2xf32>, %2: tensor<2x8xf32>, %3: tensor<2x2xf32>, %4: tensor<2x8xf32>, %5: tensor<8xf32>, %6: tensor<2x8xf32>, %7: tensor<2x8xf32>, %8: tensor<1x8xf32>, %9: tensor<8x1xf32>, %10: tensor<2x1xf32>, %11: tensor<1xf32>, %12: tensor<2x1xf32>) -> (tensor<2x2xf32>, tensor<8x2xf32>, tensor<8xf32>, tensor<1x8xf32>, tensor<1xf32>)

=== mode: 'joint' ===
  compiled modules: 1
    @joint     17 ops   @joint_Sequential(%0: tensor<2x2xf32>, %1: tensor<8x2xf32>, %2: tensor<8xf32>, %3: tensor<1x8xf32>, %4: tensor<1xf32>, %5: tensor<2x1xf32>) -> (tensor<2x1xf32>, tensor<2x2xf32>, tensor<8x2xf32>, tensor<8xf32>, tensor<1x8xf32>, tensor<1xf32>)
```

Definition 29.5 as two signatures. The separate forward returns **seven** values for a model with one output: the result, plus six saved tensors. The backward takes **thirteen** arguments: one cotangent plus twelve saved. The joint function takes six and returns six, and nothing is saved anywhere because nothing crosses a boundary.

Both produce the same gradients to the last bit, which is the useful invariant to know: **mode is a packaging choice, not a numerical one**.

Then the measurement, which is where the packaging stops being free. This is what the lab printed before the change described below:

```
=== mode: 'separate' ===
  forward() alone:    0.020 ms
  backward() alone:   0.025 ms
  forward+backward:   0.047 ms

=== mode: 'joint' ===
  forward() alone:    0.028 ms
  backward() alone:   0.027 ms
  forward+backward:   0.054 ms
```

Joint was consistently the slower one, by about 15%, and the reason is in its own two rows: `forward()` and `backward()` cost **the same**. In joint mode there is only one kernel and it computes everything, so `cf(x)` ran it with a zero cotangent to get the outputs, and `cf.backward(w)` ran **the whole thing again** with the real cotangent ([`compile_backward.ts:455`](../../../src/tracing/compile_backward.ts)). The forward pass was computed twice per training step, which is the exact opposite of what packaging them together is for.

That is not a flaw in Definition 29.5; it is a mismatch between the joint *form* and the two-call API. A joint function's advantage is that it hands you outputs and gradients in one call, and the public surface has no way to ask for that: `compileWithBackward` returns something you call, and then call `.backward()` on.

The way out is to notice that the two-call API does not say *when* the kernel runs, only what each call returns. So in joint mode `cf(x)` no longer runs it. It records the inputs, allocates the output buffers, hands back tensors backed by those buffers, and leaves a pending fill on their storage. Whichever comes first — a read of a forward output, or `cf.backward(w)` — forces the single run, and `backward` forces it with the real cotangent:

```ts
    const build = () => {
      const fwdOutputs = [];
      for (let i = 0; i < compiled.numForwardOutputs; i++) {
        const out = wrapResult(outputArrays[i], outputShapes[i], jointFunc.outputTypes[i].dtype, device);
        out.storage.setPendingFill(() => { if (ctx.pending) _settleJoint(ctx, ctx.gradOutputArrays, outputArrays); });
        fwdOutputs.push(out);
      }
      return fwdOutputs.length === 1 ? fwdOutputs[0] : fwdOutputs;
    };
```

`StorageImpl` grew one nullable thunk that its `data` and `rawData` getters resolve before returning ([`storage_impl.ts:57`](../../../src/tensor/core/storage_impl.ts)), which is the same shape as the host-read hook already sitting beside it. Nothing else in the tensor stack changes, and nothing outside joint mode ever sets it.

```
=== mode: 'separate' ===
  forward() alone:    0.021 ms
  backward() alone:   0.025 ms
  forward+backward:   0.048 ms

=== mode: 'joint' ===
  forward() alone:    0.013 ms
  backward() alone:   0.030 ms
  forward+backward:   0.044 ms
```

One kernel run per step instead of two, and joint is now the faster packaging rather than the slower one — 0.044 ms against separate's 0.048. Read `forward() alone` as what it now is: bookkeeping, because the lab's timing loop never reads the output it discards.

Two things about this are worth keeping. The first is that the worst case is the old behaviour and not worse: a caller that inspects the loss before calling `backward` forces the run with a zero cotangent, and `backward` then runs a second time exactly as before. The second is that `backward` still allocates fresh buffers for the gradient outputs while writing the forward outputs back into the buffers `cf(x)` already handed out — so two `backward` calls on one forward return independent tensors, which is what separate mode does and what the contract test now pins.

**Try this.** Read one element of `cf(x)`'s output before calling `backward`, and time the pair again — the 0.044 ms should go back to roughly 0.054. That difference is the deferral, measured.

## 29.7 Traps and limits

- **The backward function receives saved values it never reads.** In §29.5's first case, `%2: tensor<xf32>` is an argument and appears nowhere in the body. `_identifySavedValues` saves an operation's results whenever they are gradient-reachable and the remat policy says not to recompute them ([`backward_builder.ts:306`](../../../src/compiler/ad/backward_builder.ts)) — it does not ask whether any rule will actually read them. For a `sum` reduction, whose rule needs only the cotangent and a shape, the saved output is dead weight that survives to the compiled signature.
- **Joint mode's single run is deferred, so *when* it happens depends on what the caller touches.** §29.6. Reading a forward output before calling `backward` forces the kernel early with a zero cotangent and costs the second run back. Nothing reports which of the two paths a given training loop is taking.
- **The deferral applies only to synchronous runtimes.** A pending fill is resolved from a property getter, which cannot await, so joint mode on an async (GPU) target runs eagerly at `cf(x)` and pays for the forward twice as before ([`compile_backward.ts:393`](../../../src/tracing/compile_backward.ts)).
- **`computeGradReachability` treats "has a VJP rule" as "propagates gradient".** An operation with a rule that returns all-`null` — `floor`, say — still pulls its operands into the reachable set, so the backward graph contains work for values whose gradient is provably zero. The zero is then computed and accumulated. Chapter 31 measures it.
- **The tree reduction is over contributions, not over magnitude.** `_treeReduce` pairs adjacent entries in arrival order, which is fine for the critical path and says nothing about floating-point accuracy; summing many small gradients into one large one is order-dependent (Chapter 20), and the order here is an implementation detail of the sweep.
- **Nothing checks Definition 29.4.** If a rule reads a forward value that is neither saved nor recomputable, `_materialize` returns the original forward value ([`backward_builder.ts:281`](../../../src/compiler/ad/backward_builder.ts)) — a value belonging to a different function. The verifier of Chapter 12 catches it as an out-of-scope operand, so it fails loudly, but it fails at verification rather than at construction, and the error names the pass rather than the rule.
- **Two builders, one sweep, and only the sweep is shared.** Saved-value selection, checkpointing and value materialization are implemented separately in [`backward_builder.ts`](../../../src/compiler/ad/backward_builder.ts) and [`joint_builder.ts`](../../../src/compiler/ad/joint_builder.ts). They agree by construction today because the joint one is so much simpler; they are not held to agreement by anything.

## 29.8 Read the tests

- [`tests/compiler/ad/backward-compile.test.js`](../../../tests/compiler/ad/backward-compile.test.js) — the built backward function compiled and run, including its signature.
- [`tests/e2e/compiled-backward-contract.test.js`](../../../tests/e2e/compiled-backward-contract.test.js) — the caller-facing contract for both modes: one cotangent per output, arity and element-count checks, multi-head models.
- [`tests/e2e/differential-backward.test.js`](../../../tests/e2e/differential-backward.test.js) — separate against joint against eager, which is what pins Corollary 29.6.

---

**Next:** [Chapter 30 — Trading memory for recomputation](../ch30-memory-for-recomputation/README.md), which takes the saved set this chapter treated as given and asks how big it has to be — and what it costs to make it smaller.
