# Chapter 5 — From a sequence of calls to a program

Chapter 4 ended with a precondition: every optimization worth having needs the whole computation visible at once. This chapter is about how you get that, and what it costs you.

The short version: you get it by running the model once and writing down what it did. That works better than it has any right to, and it fails in one specific way that you must understand before you trust it.

## 5.1 The problem: your model is not a program

You have a model:

```js
class Net extends Module {
  forward(t) { return this.l2.forward(this.r.forward(this.l1.forward(t))); }
}
```

To a compiler this is not a program — it is a *method*. Its body is JavaScript: object lookups, method calls, closures, whatever the user felt like writing. Somewhere inside, tensor operations happen, but they are interleaved with arbitrary host-language logic that a tensor compiler has no interest in and no ability to reason about.

So the first job of any machine learning compiler is extraction: **turn a method that computes into a data structure that describes the computation.**

## 5.2 Four ways to get a program

Every framework solves this, and the design space has four occupied corners.

**Write in a separate language.** The user does not write host code at all; they write in a DSL the compiler owns. This is what Halide and Triton do, and what TensorFlow 1.x did with graph construction. You get a clean program, and you pay by making the user learn a second language and by losing the host language's debugger.

**Read the source.** Parse the method's source text into an abstract syntax tree and compile that. PyTorch's TorchScript did this. It captures control flow faithfully — the `if` really is an `if` — but the compiler now has to understand a large fraction of the host language, including features nobody uses on purpose. TorchScript's later deprecation is the practical verdict on how large that fraction turns out to be.

**Run it and watch.** Execute the method with fake tensors that record operations instead of computing them. JAX does this, PyTorch's `torch.jit.trace` does this, and so does this framework. It requires no new language and no parser: the host language executes normally, and the operations fall out as a side effect.

**Watch the interpreter, not the tensors.** Intercept the host language at the bytecode level: run the function, but observe the interpreter's own execution so that when it reaches a branch the compiler cannot resolve, you *stop the graph there*, emit what you have, let the host run the branch, and start a new graph after it. This is PyTorch's TorchDynamo, the machinery behind `torch.compile`, and the discontinuity it introduces is called a **graph break**.

The fourth strategy is worth understanding even though this framework does not implement it, because it changes what failure looks like. Under pure tracing, an unresolvable branch is a hard stop — you will see one in §5.5. Under bytecode capture it is a seam: you get two smaller graphs instead of one large one, the program still runs, and the cost is that no optimization can cross the seam. Neither approach makes data-dependent control flow compilable; they differ in whether the user gets an error or a silent loss of optimization.

This framework takes the tracing corner. Tracing wins on implementation cost and loses on fidelity, and §5.5 is about exactly what it loses.

## 5.3 How tracing works here

The trick is a tensor that is not a tensor. [`src/tracing/symbolic_tensor.ts`](../../../src/tracing/symbolic_tensor.ts):

```ts
export class SymbolicTensor extends Tensor {
  private _irValue: IRValueLike;
  private _tracer: Tracer;
  private _symbolicShape: SymbolicShape;

  constructor(irValue: IRValueLike, shape: readonly number[], dtype: DType, tracer: Tracer, symbolicShape: SymbolicShape) {
    const strides = computeStrides(shape);
    const storage = Storage.allocate(0, dtype, META_DEVICE);
    ...
```

Read the second line of the constructor body carefully: `Storage.allocate(0, ...)`. **Zero bytes.** A `SymbolicTensor` has a shape, a dtype and a device, and no data whatsoever. It is a claim about a value rather than a value.

What makes it work is a getter further down the class:

```ts
  get dispatchKeySet(): DispatchKeySet {
    return super.dispatchKeySet.union(TRACING_KEY);
  }
```

Every tensor carries a set of *dispatch keys* that decide which implementation an operation resolves to (Chapter 60). A symbolic tensor adds `TRACING`, which is numerically the highest key in the set (`TRACING = 48`, against `CPU = 0` and the autograd keys at 41–43) and therefore wins the priority comparison — so `add` on a symbolic tensor does not reach the CPU kernel at all. It reaches the tracing kernel, which appends an operation to the graph under construction and returns a new symbolic tensor describing the result.

The consequence is that **the model's own code runs unmodified**. Loops loop, methods dispatch, `this.l1` is looked up exactly as always. Nothing knows it is being watched, except the tensors.

> **Definition 5.1 (Trace).** A *trace* of a function *f* at input *x* is the sequence of tensor operations that *f* performs when executed on *x*, recorded as a graph with the data dependencies between them.

That definition contains the whole problem, and it is sitting in plain sight: **at input *x***. A trace describes what happened for one input. Whether it describes what would happen for another is the subject of §5.5.

## 5.4 Lab 1 — What the whole program buys you

```bash
node docs/part1/ch05-calls-to-program/labs/01-what-the-whole-program-buys.mjs
```

The model contains a deliberate mistake — a layer whose result is computed and then ignored, which is a bug people write more often than they admit:

```js
class DeadBranch extends Module {
  constructor() {
    super();
    this.used = new Linear(2, 2);
    this.unused = new Linear(2, 8);
  }
  forward(t) {
    const wasted = this.unused.forward(t).relu().tanh();
    return this.used.forward(t);
  }
}
```

Tracing records all of it — the wasted matmul, the wasted bias, the wasted `relu`, the wasted `tanh`:

```
    %5 = transpose(%1) {permutation = [1, 0]} : tensor<2x8xf32>
    %6 = dot(%0, %5) ... : tensor<2x8xf32>
    %7 = add(%6, %2) : tensor<2x8xf32>
    %8 = constant() {tensor_type = tensor<xf32>, value = 0} : tensor<xf32>
    %9 = broadcast_in_dim(%8) ... : tensor<2x8xf32>
    %10 = maximum(%7, %9) : tensor<2x8xf32>
    %11 = tanh(%10) : tensor<2x8xf32>
    %12 = transpose(%3) {permutation = [1, 0]} : tensor<2x2xf32>
    %13 = dot(%0, %12) ... : tensor<2x2xf32>
    %14 = add(%13, %4) : tensor<2x2xf32>
    return(%14)
```

**The tracer is not clever, and it is important that it is not.** It records; it does not judge. The model performed ten tensor operations and the tracer wrote down ten, seven of which cannot affect the result.

Now look at what the compiler emitted:

```js
function DeadBranch(buf_1, buf_3, buf_5, buf_7, buf_9, buf_11) {
  const buf_12 = new Float32Array(4);
  for (let di0_19 = 0; di0_19 < 2; di0_19++) {
    for (let di1_21 = 0; di1_21 < 2; di1_21++) {
      buf_12[((di0_19 * 2) + di1_21)] = 0;
    }
  }
  for (let ls0_13 = 0; ls0_13 < 2; ls0_13++) {
    for (let rs0_14 = 0; rs0_14 < 2; rs0_14++) {
      let _acc_0 = buf_12[((ls0_13 * 2) + rs0_14)];
      for (let c0_15 = 0; c0_15 < 2; c0_15++) {
        _acc_0 = (_acc_0 + (buf_1[((ls0_13 * 2) + c0_15)] * buf_7[((rs0_14 * 2) + c0_15)]));
      }
      buf_12[((ls0_13 * 2) + rs0_14)] = _acc_0;
    }
  }
  for (let i0_23 = 0; i0_23 < 2; i0_23++) {
    for (let i1_24 = 0; i1_24 < 2; i1_24++) {
      buf_11[((i0_23 * 2) + i1_24)] = (buf_12[((i0_23 * 2) + i1_24)] + buf_9[i1_24]);
    }
  }
}
```

The 8-wide matmul, the bias, the `relu`, the `tanh` — gone. What remains is the 2 × 2 matmul and its bias.

Two details deserve attention.

**The signature still has six parameters.** `buf_3` and `buf_5` are the unused layer's weight and bias. They are passed in and never read. Dead *computation* was eliminated; the dead *interface* was not, because the function's signature is part of its contract with the caller. Chapter 19 explains where that boundary sits.

**Eager execution could not have done this.** When `this.unused.forward(t)` runs eagerly, the matmul happens. Nothing at that moment knows the result will be discarded — the discarding has not been written yet, in the sense that it has not yet executed. Seeing the future is exactly what the graph provides.

This is the pattern for every optimization in the book: *tracing produces a faithful, verbose, unoptimized program; the compiler makes it good.* Keeping those two jobs separate is what keeps the tracer simple enough to be correct.

## 5.5 Lab 2 — What tracing cannot see

```bash
node docs/part1/ch05-calls-to-program/labs/02-control-flow.mjs
```

Here is a model that decides what to do based on its input:

```js
class DataDependent extends Module {
  forward(t) {
    if (t.sum().item() > 0) return t.mul(10);
    return t.mul(-1);
  }
}
```

Trace it and you get:

```
trace failed:

item() is not available on a symbolic tensor: tracing records operations instead of
computing them, so this tensor (shape [], dtype f32) carries no value to read.
The usual cause is data-dependent control flow: a branch or loop condition computed
from tensor contents. A trace cannot capture that, because it records whichever path
ran at trace time and silently reuses it for every later input.
Rewrite the decision as tensor math (where), or express the loop with a region-carrying
op: scan (src/tracing/scan.ts) records the loop body as a graph region instead of
reading a value. If what you need is a dimension rather than an element, trace with
dynamic_shapes so the size stays symbolic.
```

Understand why it fails at all. `t.sum()` returns a symbolic tensor — a claim about a value, backed by zero bytes of storage. `.item()` asks that claim for its numeric value, and there is none: it has not been computed, and cannot be, because tracing is not execution. The JavaScript `if` then needs a boolean *now*, at trace time, to decide which branch to record. It cannot have one. Every value-reading accessor fails identically — `item()`, `.data`, `toArray()`, iteration — because none of them can invent a value the trace never computed.

That message is worth dwelling on as a piece of design, because a compiler's diagnostics are part of its user interface and this one does three separate jobs. It names **what** was refused and on which tensor. It gives the **mechanism** rather than the symptom: tracing records instead of computing. And it lists the **remedies** in the order worth trying — express the decision as tensor arithmetic with `where`; move the loop into a region-carrying operation such as `scan` (§5.7); or, if what you wanted was a dimension rather than an element, trace with `dynamic_shapes` so the size stays symbolic (§5.6).

All three remedies are concepts from this chapter. A diagnostic that teaches the model instead of merely reporting a failure is worth what it costs to write — and this one was written *because* an earlier draft of this chapter had to document the previous message, which read `Cannot read properties of null (reading '0')` and taught nothing at all.

The underlying limitation, however, is not a bug — it is inherent, and every tracing framework has it. Let us state it precisely.

> **Definition 5.2 (Control-flow path).** The *control-flow path* taken by *f* at input *x* is the sequence of host-language branches evaluated during execution of *f(x)*.

> **Theorem 5.3 (Trace validity; stated here).** Let *G* be a trace of *f* at *x*. Then *G* computes *f(y)* for every input *y* that (i) takes the same control-flow path as *x*, and (ii) has the same shapes and dtypes as *x*.
>
> *Proof sketch.* Along a fixed control-flow path, *f* performs a fixed sequence of tensor operations whose operands are determined by data dependencies alone; *G* records precisely that sequence and those dependencies. Condition (ii) holds because each recorded operation was resolved against concrete shapes and dtypes at trace time. ∎

The branch above at least failed loudly. The dangerous version of condition (i) is the one that does not, and the same lab demonstrates it:

```js
class ModeDependent extends Module {
  constructor() { super(); this.flag = true; }
  forward(t) { return this.flag ? t.mul(10) : t.mul(-1); }
}
```

Nothing here reads a tensor's value. `this.flag` is ordinary host state, available at trace time, so the branch resolves and tracing succeeds. Then:

```
flag=true    eager 10,20,30,40   compiled 10,20,30,40
flag=false   eager -1,-2,-3,-4   compiled 10,20,30,40   <-- compiled is stale
```

> **Counterexample 5.4.** For `ModeDependent`, tracing at `flag = true` produces a graph computing `mul(t, 10)`. Setting `flag = false` changes the control-flow path, so Theorem 5.3 no longer applies and the compiled artifact is wrong — but it is wrong *silently*: the recorded graph contains no evidence that a branch was ever involved, so no guard can be derived and none fires.

If that example looks artificial, substitute `this.training` for `this.flag` and it becomes the single most common way to get wrong answers out of a traced model. Dropout and batch normalization behave differently in training and evaluation, and a model traced in one mode and used in the other computes the wrong thing without complaint.

So the two conditions of Theorem 5.3 receive very different treatment in practice. Condition (ii), shapes, is enforced automatically by guards (§5.6) — the framework checks it on every call. Condition (i), the control-flow path, is enforced only when the branch happens to read a tensor value, which is a loud failure; branches on host state fall outside the mechanism entirely and are left to the user's discipline. That asymmetry is worth remembering, because it is not specific to this framework — it is a property of tracing.

## 5.6 Lab 3 — Guards, and paying for shapes

```bash
node docs/part1/ch05-calls-to-program/labs/03-guards-and-recompilation.mjs
```

Condition (ii) of Theorem 5.3 — same shapes — is handled by specializing on the shapes seen, then checking them on every call:

```
after building with a [4,3] input          : 1 compilation(s)
after another [4,3] input                  : 1
after an [8,3] input                       : 2
after going back to [4,3]                  : 2
after [16,3] and [32,3]                    : 4
```

Four distinct batch sizes, four compilations, and returning to a shape already seen costs nothing. That is a cache keyed by a *guard*.

> **Definition 5.5 (Guard; stated here).** A *guard* is a predicate over the inputs, checked before a compiled artifact is reused, that is sufficient to establish the preconditions under which the artifact was proven correct.

Guards make the specialization sound: if the guard passes, Theorem 5.3 applies and the compiled program is correct; if it fails, the framework compiles a new one rather than returning a wrong answer. Chapter 62 examines what the guard set actually contains here, and what happens when it grows large.

The alternative is to not specialize at all:

```
with dynamic_shapes: one kernel served     : 4x2, 8x2, 16x2, 32x2
```

With `dynamic_shapes: [true]`, the batch dimension becomes a *symbolic* value: the generated kernel takes the size as a runtime parameter, and one kernel serves every batch. You trade generated-code quality for compilations avoided — loop bounds are no longer constants, so bounds cannot be used to unroll, vectorize or tile as aggressively.

This trade — **specialize and recompile, or generalize and lose information** — recurs in every part of the book. It is the same trade a JIT for a dynamic language makes, for the same reasons.

## 5.7 Control flow that survives tracing

Condition (i) of Theorem 5.3 — same control-flow path — has a different remedy. If the compiler cannot see through a host-language `if`, then do not write one: express the control flow as an *operation*, so it lands in the graph as data rather than being consumed at trace time.

The same lab traces a recurrence written with `scan`:

```js
const [last, ys] = scan((carry, x_t) => {
  const next = carry.mul(0.9).add(x_t).tanh();
  return [next, next];
}, h0, xs);
```

and the loop appears in the IR:

```
module @traced {
  func @traced(%0: tensor<4x3xf32>, %1: tensor<3xf32>) -> (tensor<4x3xf32>) {
    %2, %3 = scan(%0, %1) {num_carry = 1, num_xs = 1} : tensor<3xf32>, tensor<4x3xf32>
    {
      ^bb(%4: tensor<3xf32>, %5: tensor<3xf32>):
      %6 = constant() {tensor_type = tensor<xf32>, value = 0.8999999761581421} : tensor<xf32>
      %7 = mul(%5, %6) : tensor<3xf32>
      %8 = add(%7, %4) : tensor<3xf32>
      %9 = tanh(%8) : tensor<3xf32>
      yield(%9, %9)
    }
    return(%3)
  }
}
```

Notice what did **not** happen: the loop body was not unrolled four times. There is one `scan` operation carrying a *region* — the indented block — that holds the body once. The block's two arguments are the current element of `xs` and the carry, and you can tell which is which from how they are used: `%5` is multiplied by 0.9, so `%5` is the carry.

This is worth pausing on, because it is the first real payoff of the IR design. A region lets one operation contain a program. The graph stays small regardless of sequence length; the body can be optimized once; and — the reason this matters most — the body can be *differentiated* once, which is what makes backpropagation through a recurrence tractable (Chapter 31).

The cost is that `scan` is not a JavaScript `for` loop, and users must choose it deliberately. Every framework in this space ends up with the same bargain under a different name: `jax.lax.scan`, `tf.while_loop`, `torch.cond`. When you see those APIs, this is what they are for.

## 5.8 So what did we gain?

Collecting the chapter: once the computation exists as a graph rather than a sequence of calls, the compiler can

- **delete work** that provably cannot affect the output (§5.4),
- **merge operations** that would otherwise each make a pass over memory (Chapter 4's 4.47×),
- **plan memory** for the whole program instead of allocating per operation (Part IX),
- **choose layouts** by looking at consumers, not just producers (Chapter 25),
- **differentiate the program**, producing a backward graph that is itself optimizable (Part V),
- **target a different machine entirely**, since the graph says nothing about how it will be executed (Part X).

None of these is available to an interpreter running one operation at a time, at any level of engineering effort. They are consequences of the representation, not of cleverness.

And the price is Theorem 5.3: what you compiled is correct only for inputs that resemble the one you traced, in two precise senses — same path, same shapes — with guards to enforce the resemblance.

## 5.9 Traps and limits

- **Tracing records tensor operations only.** A `console.log` in `forward` runs at trace time and never again. Python-side, or in this case JavaScript-side, effects are invisible to the graph. This is a frequent source of confusion when someone adds a print statement to a compiled model and sees nothing.
- **Mutating the model after compiling does not recompile it.** Counterexample 5.4 is the general case: changing `this.training`, swapping a submodule, or editing any host-level attribute that `forward` reads leaves the compiled artifact untouched and stale. Recompile deliberately after any such change.
- **The trace is as long as the execution.** A `for` loop over 1000 timesteps written as a host loop produces 1000 copies of the body in the graph, and compilation time grows accordingly. That is the practical argument for `scan`, independent of correctness.
- **Randomness is captured as it happened.** If `forward` samples noise, the trace records the operations that produced it — you must check whether the sampling became part of the graph or was frozen into a constant. Chapter 63 returns to this when training compiled models.
- **A value read inside `trace` is always a design question, never a bug to work around.** The error from §5.5 names the accessor and the cause, but it cannot tell you which of the two remedies you want. Reach for `where` when the decision is per-element, for `scan` when it is a loop, and for `dynamic_shapes` when what you were reaching for was a size rather than an element.

## 5.10 Read the tests

- [`tests/tracing/`](../../../tests/tracing/) — what the tracer records, including `scan` regions and symbolic shapes.
- [`tests/tracing/symbolic-value-read.test.js`](../../../tests/tracing/symbolic-value-read.test.js) — the §5.5 diagnostic itself, pinned: which accessors refuse, what the message must name, and — the test worth reading — that the remedy the message suggests actually traces, with the branch rewritten as `where`.
- [`tests/e2e/dynamic-shapes.test.js`](../../../tests/e2e/dynamic-shapes.test.js) — symbolic dimensions end to end: one kernel, many batch sizes, and the cases where a model reads its own shape.
- [`tests/e2e/compiled-backward-contract.test.js`](../../../tests/e2e/compiled-backward-contract.test.js) — differentiating through a traced `scan`, which is §5.7's payoff made concrete.

---

**Next:** [Chapter 6 — The pipeline in one picture](../ch06-the-pipeline/README.md), which asks why the journey from graph to machine code takes three intermediate representations instead of one.
