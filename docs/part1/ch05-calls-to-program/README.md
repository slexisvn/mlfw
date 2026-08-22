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

The trick is a tensor that is not a tensor — [`src/tracing/symbolic_tensor.ts:32`](../../../src/tracing/symbolic_tensor.ts):

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

What makes it work is a getter further down the class — [`symbolic_tensor.ts:74`](../../../src/tracing/symbolic_tensor.ts):

```ts
  get dispatchKeySet(): DispatchKeySet {
    return super.dispatchKeySet.union(TRACING_KEY);
  }
```

Every tensor carries a set of *dispatch keys* that decide which implementation an operation resolves to (Chapter 60). A symbolic tensor adds `TRACING`, which is numerically the highest key in the set (`TRACING = 48`, against `CPU = 0` and the autograd keys at 41–43) and therefore wins the priority comparison — so `add` on a symbolic tensor does not reach the CPU kernel at all. It reaches the tracing kernel, which appends an operation to the graph under construction and returns a new symbolic tensor describing the result.

The consequence is that **the model's own code runs unmodified**. Loops loop, methods dispatch, `this.l1` is looked up exactly as always. Nothing knows it is being watched, except the tensors.

> **Definition 5.1 (Trace).** **(stated here)** A *trace* of a function *f* at input *x* is the sequence of tensor operations that *f* performs when executed on *x*, recorded as a graph with the data dependencies between them.

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

All three remedies are concepts from this chapter. A diagnostic that teaches the model instead of merely reporting a failure is worth what it costs to write. The alternative — what you get when nobody writes one — is a null-dereference stack trace from inside the tracer, naming a field the reader has never heard of and nothing they can act on.

The underlying limitation, however, is not a bug — it is inherent, and every tracing framework has it. Let us state it precisely.

> **Definition 5.2 (Control-flow path).** **(stated here)** The *control-flow path* taken by *f* at input *x* is the sequence of host-language branches evaluated during execution of *f(x)*.

> **Theorem 5.3 (Trace validity).** **(stated here)** Let *G* be a trace of *f* at *x*, taken in host environment *E* — everything *f* reads without receiving it as an argument. Then *G* computes *f(y)* for every input *y* that (i) takes the same control-flow path as *x* and (ii) has the same shapes and dtypes as *x*, provided (iii) *f* is a function of its arguments, of the tensors *G* lifts to parameters, and of *E* alone, and the non-lifted part of *E* is unchanged since the trace.
>
> *Proof sketch.* Along a fixed control-flow path, *f* performs a fixed sequence of tensor operations whose operands are determined by data dependencies alone; *G* records precisely that sequence and those dependencies. Condition (ii) holds because each recorded operation was resolved against concrete shapes and dtypes at trace time. Condition (iii) is what makes those operations a function of *y* at all: whatever *f* reads from the non-lifted part of *E* is read once, during the trace, and enters *G* as a constant — so a change to it, or a value that differs on every read, is a change to *f* that *G* has no way to represent. ∎

**Condition (iii) needs unpacking, because "captured state" is four different things and they behave differently.** The theorem above is careful about the phrase "the non-lifted part of *E*", and that qualification is the whole content of the distinction:

| What `forward` reads | How the trace treats it | Does a later mutation show up? |
|---|---|---|
| **User inputs** — the tensors passed as arguments | function parameters | yes; they are supplied fresh on every call |
| **Lifted parameter tensors** — `this.fc.weight` and friends | *also* function parameters (`%1`–`%4` in Chapter 1 §1.1), bound to the live tensor | **yes** — the artifact reads the parameter's storage at call time |
| **Host scalars** — `this.scale`, `this.training`, a plain number | folded into the graph as a constant | no; the value is frozen at trace time |
| **Folded parameters** — weights the compiler proves constant and materializes into the artifact (Chapter 61) | baked into the emitted code | no |

The second row is the one that surprises people, and it is worth running:

```
before                        eager -2.857   compiled -2.857
after mutating fc.weight      eager 397.14   compiled 397.14   <-- tracked
after mutating this.scale=7   eager 1390.0   compiled 397.14   <-- stale
```

Same model, same call, two mutations, two different answers. Editing a **parameter tensor in place** is observed by the compiled artifact, because the parameter was lifted to an argument and the artifact reads its storage on every call — that is precisely what makes training a compiled model possible at all. Editing a **host scalar** is not observed, because it was folded to a constant. So "the compiled artifact is a snapshot of your model" is too coarse: it is a snapshot of the model's *structure and host state*, holding live references to the model's *tensors*. Which of those two a given attribute falls into is decided by whether it is a tensor, and nothing in the API announces it.

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

`ModeDependent` at least fails a condition the theorem names: there is a branch, and its path changed. The quieter failure has no branch in it at all, and the same lab prints it:

```js
class Scaled extends Module {
  constructor() { super(); this.scale = 2; }
  forward(t) { return t.mul(this.scale); }
}
```

```
scale=2      eager 2,4,6,8   compiled 2,4,6,8
scale=5      eager 5,10,15,20   compiled 2,4,6,8   <-- compiled is stale
```

`this.scale` is read, not branched on, so it is folded into `G` as a constant and the control-flow path is the same for every input and every value of `scale`. Conditions (i) and (ii) both hold. The artifact is stale anyway, and what it violates is condition (iii). A random draw taken inside `forward` fails the same way for the same reason: `randn` called during the trace produces one tensor of numbers, that tensor is what `G` contains, and the compiled artifact returns the identical "noise" on every call.

So the conditions of Theorem 5.3 are not equally protected, and the asymmetry is the practical shape of the whole chapter. Shapes and dtypes are checked for you on every call — that is §5.6. The control-flow path is caught only when the branch reads a *tensor*; Counterexample 5.4 slipped through precisely because it read host state instead. And condition (iii) is not checked at all, because nothing records what `forward` read from its environment, so nothing can notice it changed. Half the theorem is left to the user's discipline, and that is not specific to this framework — it is a property of tracing. §5.9 tabulates which half is which.

## 5.6 Lab 3 — Guards, and paying for shapes

```bash
node docs/part1/ch05-calls-to-program/labs/03-guards-and-recompilation.mjs
```

Condition (ii) of Theorem 5.3 is "same shapes and dtypes". Start with the shapes — specializing on the ones seen, then checking them on every call:

```
after building with a [4,3] input          : 1 compilation(s)
after another [4,3] input                  : 1
after an [8,3] input                       : 2
after going back to [4,3]                  : 2
after [16,3] and [32,3]                    : 4
```

Four distinct batch sizes, four compilations, and returning to a shape already seen costs nothing. That is a cache keyed by a *guard*.

> **Definition 5.5 (Guard).** **(stated here)** A *guard* is a predicate over the inputs, checked before a compiled artifact is reused, that is sufficient to establish the preconditions under which the artifact was proven correct.

Read that definition as a conditional, because the conditional is where the whole design either holds or leaks. *If* a guard establishes the preconditions, then Theorem 5.3 applies, the compiled program is correct, and a guard failure costs a recompilation rather than a wrong answer. The guards this framework records establish condition (ii) and nothing else: shapes and dtypes are covered completely, the control-flow path is covered only when the branch happens to read a tensor, and the environment is not covered at all. So the specialization on shapes really is sound; the artifact as a whole is sound only to the extent that you have upheld conditions (i) and (iii) yourself. §5.9 tabulates that split, and Chapter 62 examines what the guard set contains once it grows large.

Dtypes are guarded too, by a second key rather than by the same mechanism. Each cache entry records the dtype and device it was traced from, and a lookup skips any entry whose signature does not match before evaluating a single shape predicate ([`compile.ts:442`](../../../src/tracing/compile.ts)). A miss compiles a new entry rather than returning a wrong answer, so calling the same model on `f32` and then on `i32` produces two artifacts and two correct results.

**Why two mechanisms rather than one?** Because the two questions have different shapes, and this is the transferable part. Dtype and device are *discrete and exact*: an entry either was traced for this combination or it was not, and equality decides it. Shapes are *relational*: a dynamic trace produces constraints rather than values, and the right test is whether the incoming shapes satisfy them. That is the refinement below.

### And the shape guard is not always an equality

One more refinement, because "same shapes" is only the static story. The four-compilations table above is the specializing path, where the guard really is an exact-shape equality. Under `dynamic_shapes` the artifact is compiled against an *abstract* signature — some dimensions are symbolic rather than concrete — and the guard set becomes the set of constraints the tracer had to assume in order to get through the trace: that a dimension is divisible by a tile factor, that two dimensions are equal, that one is positive. Those are the predicates `evaluateGuards` is actually iterating. So the precise version of condition (ii) is:

> *G* is valid for any *y* whose shapes **satisfy the recorded guard set**, which for a fully static trace degenerates to exact equality, and for a dynamic trace is a strictly weaker predicate admitting many shapes.

That weaker predicate is the entire point of dynamic shapes, and it is why the same theorem covers both paths.

The alternative is to not specialize at all:

```
with dynamic_shapes: one kernel served     : 4x2, 8x2, 16x2, 32x2
```

With `dynamic_shapes: [true]`, every dimension of that input becomes *dynamic*: the generated kernel takes the sizes as runtime parameters, and one kernel serves every batch. (`true` is the shorthand for the whole input; `new Set([0])` marks one dimension, which is what you usually mean by "the batch size varies". Chapter 10 compares the two.) You trade generated-code quality for compilations avoided — loop bounds are no longer constants, so bounds cannot be used to unroll, vectorize or tile as aggressively.

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

Notice what did **not** happen: the loop body was not unrolled four times. There is one `scan` operation carrying a *region* — the indented block — that holds the body once.

**The block's arguments are (element of `xs`, carry) — the reverse of the JavaScript callback's `(carry, x_t)`, and this trips everyone once.** You can confirm it from the printout without trusting anyone: `%7 = mul(%5, %6)` with `%6 = 0.9`, and the source multiplies the *carry* by 0.9, so `%5` is the carry and `%4` is the element. The order is set by [`builder.ts:471`](../../../src/compiler/ir/graph/builder.ts), which lays the block out as `[...xtTypes, ...carryTypes]` to match the operand list `[...xs, ...initCarry]` position for position — the `scan` operation takes its data inputs first and its loop-carried state second, and the region mirrors the operation, not the user-facing API. The user-facing `scan(fn, initCarry, xs)` puts the carry first because that is how `jax.lax.scan` reads. Two conventions, one adapter between them, and `num_carry` / `num_xs` are the attributes that record where the split falls.

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

None of these is available to an interpreter that *decides each operation in isolation and then forgets it*. That is the precise claim, and it is worth resisting the stronger one — "an interpreter can never do these things" — because it is false, and the history of dynamic-language runtimes is the counterexample. A sufficiently determined interpreter can buffer operations before executing them (lazy evaluation), record what it just ran and specialize the next occurrence (tracing JITs, which is exactly how this framework's own eager path already works — §4.2), or fuse at runtime once it has seen a repeated sequence. Every one of those techniques recovers some of the list above.

What they recover it by is *building a representation of more than one operation* — which is to say, by becoming a compiler, at runtime, with a smaller window and a latency budget. So the honest statement is about what the window buys, not about what an interpreter is forbidden: **these optimizations are consequences of seeing the whole computation, and the more of it you see, the more of them you get.** An eager interpreter sees one operation and gets none. A tracing JIT sees a recent history and gets some, at the cost of doing the analysis while the user waits. Ahead-of-time tracing sees all of it and gets all of them, at the cost of Theorem 5.3.

And that price is the real trade: what you compiled is correct only for inputs resembling the one you traced — same path, same shapes, same dtypes — and only while the model is still the one you traced. Shapes and dtypes are checked for you. The path and the environment are yours to keep.

## 5.9 Traps and limits

Which of Theorem 5.3's conditions the implementation actually mechanizes, and which it leaves to you:

| Condition | Enforced by | Coverage |
|---|---|---|
| (ii) shapes | shape guards, checked on every call (§5.6) | complete for the shapes themselves |
| (ii) dtypes | the cached entry's input signature, compared on every call (§5.6) | complete, including device |
| (i) control-flow path | the tracer's refusal to read a symbolic value | only when the branch reads a *tensor*; a branch on host state is invisible |
| (iii) environment | nothing | not enforced, and not enforceable by this design — nothing records what `forward` read |

Condition (ii) is fully mechanized, in two mechanisms of different shapes. The other two are not. Condition (i)'s enforcement is a side effect rather than a check: the tracer refuses because it *cannot* produce a value, not because it is guarding anything, so a branch on host state slips through. Condition (iii) cannot be enforced by tracing at all.

Two details behind the (ii) rows. `evaluateGuards` ([`shape_env.ts:101`](../../../src/tracing/shape_env.ts)) iterates *shape* predicates only — divisibility of a symbolic dimension, or a comparison between two of them — so neither dtype nor device appears in it; those live in the entry signature instead. And `compileWithBackward` carries the same signature on its metas, so the backward artifact is keyed the same way.

> **Counterexample 5.6.** Without the dtype half of the key, condition (ii) fails silently. For `forward(t) { return t.div(2); }`, integer division truncates and float division does not: eager `i32` division by 2 gives `[0, 1, 2, 3]`, while an artifact traced at `f32` computes `[0.5, 1.5, 2.5, 3.5]`. Served for an `i32` input, it would return wrong values *and* the wrong dtype, with no guard failure and no warning.

- **Tracing records tensor operations only.** A `console.log` in `forward` runs at trace time and never again. Python-side, or in this case JavaScript-side, effects are invisible to the graph. This is a frequent source of confusion when someone adds a print statement to a compiled model and sees nothing.
- **Mutating the model's *host state* after compiling does not recompile it.** This is condition (iii) of Theorem 5.3, and Counterexample 5.4 is only its loudest case: changing `this.training`, swapping a submodule, or editing any host-level attribute that `forward` reads — whether it is branched on or merely multiplied by — leaves the compiled artifact untouched and stale. Recompile deliberately after any such change. Mutating a **parameter tensor** in place is the opposite case and is tracked, per §5.5's table of what the trace captures; do not generalize from one to the other in either direction.
- **Changing the dtype or device of an input compiles a new artifact.** Counterexample 5.6 is why. That is correct and it is not free: a call site that alternates between two dtypes keeps both artifacts alive and pays a compilation for each.
- **The trace is as long as the execution.** A `for` loop over 1000 timesteps written as a host loop produces 1000 copies of the body in the graph, and compilation time grows accordingly. That is the practical argument for `scan`, independent of correctness.
- **Randomness is captured as it happened.** If `forward` samples noise, the sampling runs once, during the trace, and the numbers it produced are what the graph contains: the compiled artifact returns the identical "noise" on every call while the eager model returns a fresh draw. That is condition (iii) again, and it is the one case where a model that looks stochastic is silently deterministic. Chapter 63 returns to it when training compiled models.
- **A value read inside `trace` is always a design question, never a bug to work around.** The error from §5.5 names the accessor and the cause, but it cannot tell you which of the two remedies you want. Reach for `where` when the decision is per-element, for `scan` when it is a loop, and for `dynamic_shapes` when what you were reaching for was a size rather than an element.

## 5.10 Read the tests

- [`tests/tracing/`](../../../tests/tracing/) — what the tracer records, including `scan` regions and symbolic shapes.
- [`tests/tracing/symbolic-value-read.test.js`](../../../tests/tracing/symbolic-value-read.test.js) — the §5.5 diagnostic itself, pinned: which accessors refuse, what the message must name, and — the test worth reading — that the remedy the message suggests actually traces, with the branch rewritten as `where`.
- [`tests/e2e/dynamic-shapes.test.js`](../../../tests/e2e/dynamic-shapes.test.js) — symbolic dimensions end to end: one kernel, many batch sizes, and the cases where a model reads its own shape.
- [`tests/e2e/compiled-backward-contract.test.js`](../../../tests/e2e/compiled-backward-contract.test.js) — differentiating through a traced `scan`, which is §5.7's payoff made concrete.

---

**Next:** [Chapter 6 — The pipeline in one picture](../ch06-the-pipeline/README.md), which asks why the journey from graph to machine code takes three intermediate representations instead of one.
