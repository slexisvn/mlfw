# Chapter 63 — Training end to end

Sixty-two chapters, one program. `relu(x @ W1 + b1) @ W2 + b2` has been a JavaScript closure, a traced graph, a fused graph, a differentiated graph, a loop nest, a scheduled loop nest, a flat-index IR, and four kinds of source text. This chapter runs it.

Not once — forty times, with an optimizer between the steps, against the eager execution it is supposed to replace, and with the difference measured rather than asserted.

## 63.1 The problem: a compiled forward pass is not a training step

Everything so far compiles a *function*. A training step is not a function; it is a loop body with state that survives between iterations, and three things about it break the assumptions the previous four chapters were built on.

**Gradients have to come back out.** The compiled function returns a loss. The optimizer needs `∂loss/∂W1` and three siblings — values that are internal to the backward graph and have to be surfaced as additional outputs, in an order the caller can match to its parameters.

**The parameters change every step, and the graph captured them.** [Chapter 61](../ch61-tracing/README.md) captured `W1` as a function parameter bound to a *tensor object*. The optimizer will overwrite that object's contents. If the compiled kernel reads the object each call, the update is seen; if anything cached the data — a device upload, a folded constant — it is not, and training silently stops working while the loss keeps being reported.

**The shape changes on the last batch.** A dataset of 256 with batch size 32 divides evenly; one of 250 does not. That is [Chapter 62](../ch62-dynamic-shapes/README.md)'s problem arriving in the one place every user meets it.

And underneath all three, the question this part exists to answer: **is the compiled path the same computation as the eager one?** Not "does it converge" — a wrong gradient can converge. The same numbers, to a stated tolerance, from the same initialisation.

## 63.2 Intuition: forward, backward, apply — and who owns each

The training step is three phases and the framework draws its boundaries at the seams.

*Forward* is the compiled function. *Backward* is a second compiled function — or a second half of the same one — whose inputs are the forward's saved values and a seed gradient, and whose outputs are one gradient per input. *Apply* is the optimizer, and it stays in eager JavaScript, because it is a handful of elementwise updates over parameters that are not part of the graph.

The choice of where to cut is the chapter's one real design decision, and there are two answers.

**Separate**: compile the forward, compile the backward as its own graph, and pass the values the backward needs between them. Two compilations, two kernels, and an explicit list of saved tensors.

**Joint**: compile one graph containing both, whose outputs are the forward's outputs *followed by* the gradients. One compilation, one kernel, and the intermediate values never leave it — so fusion and [Part IX](../../part9/README.md)'s allocator see the whole step at once.

Both are implemented; §63.5 measures both.

## 63.3 Theory

> **Definition 63.1 (Training step).** **(stated here)** Given parameters `θ`, a batch `(x, y)` and a loss `L`, a *training step* computes `L(θ; x, y)`, then `g = ∂L/∂θ`, then `θ ← U(θ, g)` for an update rule `U`. A *compiled* training step is one in which the first two are executed by generated code.

> **Definition 63.2 (Separate and joint mode).** **(invariant)** In *separate* mode `compileWithBackward` produces two graph functions — [Part II](../../part2/README.md)'s `GraphFunction`, each of which is then compiled the whole way down on its own: a forward whose outputs are the user's outputs plus the values the backward needs, and a backward taking those saved values and one seed gradient per output. In *joint* mode it produces one function whose outputs are the forward outputs followed by the gradients.

> **Proposition 63.3 (Gradient layout).** **(invariant)** `backward()` returns one gradient per *function input*, in input order: the user's inputs first, then the captured parameters in capture order. A caller matching gradients to parameters must use `capturedParams()` and the offset `grads.length − capturedParams().length`.
>
> *Why this is fragile and why it is right.* It is positional, so a change in capture order silently misassigns gradients. It is also the only layout that does not require the compiler to know what a "parameter" is — [Chapter 61](../ch61-tracing/README.md) captured tensors, not weights, and nothing below the tracer distinguishes a weight from any other closed-over tensor.

> **Definition 63.4 (Differential test).** **(stated here)** Two implementations `A` and `B` of the same mathematical function are *differentially equivalent to tolerance τ* on an input set `S` if `|A(s) − B(s)| ≤ τ` for all `s ∈ S`. A differential test is an assertion of that, with `τ` derived from the numerical model rather than chosen to make the test pass.

> **Theorem 63.5 (What a compiled step may and may not change).** **(stated here)** Let the compiled and eager paths execute the same operations on the same inputs. Then their results differ only by:
> (i) *reassociation*, where fusion or scheduling changed a reduction's summation order — level N2 by [Definition 1.4](../../part0/ch01-what-this-book-is/README.md);
> (ii) *precision of intermediates*, where one path holds a value in a wider register than the other — N1;
> (iii) *approximation*, where a transcendental comes from a different implementation — which Definition 1.4's ladder does not grade at all, because it compares rearrangements of one piece of arithmetic and this is a different piece.
> No other difference is admissible, and in particular a difference in *which elements* are combined is not a tolerance question at all.
>
> *How it is used.* A differential test that reports a small discrepancy has found (i), (ii) or (iii) and needs a tolerance. One that reports a *structured* discrepancy — a permutation, a transpose, a shifted index — has found a defect, and no tolerance is the right response. §63.5 finds one of each.

> **Corollary 63.6 (Convergence is not a correctness test).** **(stated here)** Gradient descent with a systematically wrong gradient `g' = Pg` for a permutation `P` still reduces the loss whenever `⟨g, Pg⟩ > 0`, which holds for most `P` on an over-parameterised model. So "the loss went down" and "the two paths agree" are independent claims, and only the second is a test.

## 63.4 In mlfw: the two modes, and the loop that uses them

### Building both graphs from one trace

```ts
    const finish = (t: TracedCore): BackwardMeta => {
      const func = t.graph.functions().next().value as GraphFunctionLike;
      const indexBounds = userArgIndexBounds(t.graph as unknown as GraphModule, t.numUserInputs);
      const compiled = mode === 'joint'
```

([`compile_backward.ts:117`](../../../src/tracing/compile_backward.ts).) One trace, then a branch on mode. The forward graph is [Chapter 61](../ch61-tracing/README.md)'s output verbatim; the backward is built from it by [Part V](../../part5/README.md)'s `BackwardGraphBuilder` or `JointGraphBuilder`, both of which are the same VJP sweep with different output conventions.

### The framework's training loop

```ts
    const step = model.__compiledTrainStep!;
    const params = step.capturedParams();
    const { ones } = await import('../../../tensor/factory/creation_ops.js');
    let grads = step.backward(ones((loss as { shape: readonly number[] }).shape)); if (isThenable(grads)) grads = await grads;
    const gradList = grads as unknown[];
    const off = gradList.length - params.length;
    for (let i = 0; i < params.length; i++) { const g = gradList[off + i]; if (g) params[i].grad = g; }

    if ((batchIdx + 1) % accumGrad === 0) {
      for (let i = 0; i < optimizers.length; i++) {
        this._clipGradients(model, trainer);
        optimizers[i].step();
        optimizers[i].zeroGrad();
      }
```

([`training_loop.ts:175`](../../../src/lightning/core/loops/training_loop.ts).) Proposition 63.3, applied: take the tail of the gradient list, assign each to the matching captured parameter's `.grad`, then run the ordinary eager optimizer. **The optimizer never learns that anything was compiled** — it sees parameters with gradients, exactly as the eager path leaves them. That is the whole integration, and it is why `compile: true` is one boolean rather than a second training loop.

The compilation itself is lazy and happens on the first batch:

```ts
    if (!model.__compiledTrainStep) {
      const { compileWithBackward } = await import('../../../tracing/compile_backward.js');
      const { CPUTarget, CUDATarget, WebGPUTarget } = await import('../../../compiler/support/target.js');
      const deviceType = model._device && model._device.type;
      const target = deviceType === 'webgpu' ? WebGPUTarget() : deviceType === 'gpu' ? CUDATarget() : CPUTarget();
      model.__compiledTrainStep = (compileWithBackward as unknown as (module: unknown, inputs: unknown[], options: unknown) => CompiledTrainStep)({ forward: callForward }, elems, { target, mode: trainer.compileMode });
```

([`training_loop.ts:161`](../../../src/lightning/core/loops/training_loop.ts).) The traced function is `model.trainingStep` itself, so the *loss* is inside the compiled graph — which is what makes the seed gradient a scalar `ones([])` and lets fusion see the loss reduction as part of the same graph as the forward.

### Parameters that change under a compiled graph

`_prepareExecution` re-reads every captured parameter on every call:

```ts
  for (let i = 0; i < paramArrays.length; i++) {
    const rt = new RuntimeTensor(paramArrays[i], params[i].shape, params[i].dtype);
    const impl = params[i]._impl;
    if (impl) (rt as RuntimeTensorLike).resident = { key: impl.storage.rawData, version: impl.version };
    allArgs[idx++] = rt;
  }
```

([`compile.ts:178`](../../../src/tracing/compile.ts).) The `resident` record is the answer to §63.1's second problem. On a device backend the parameter is uploaded once and cached by `(storage, version)`; the optimizer's `bumpVersion()` invalidates that cache, so a mutated weight is re-uploaded and a stale one is not. On the CPU there is nothing to invalidate — the kernel reads the same typed array — and the record is inert.

## 63.5 Lab — the running example, trained

```bash
node docs/part11/ch63-training-end-to-end/labs/01-one-step-three-ways.mjs
```

What the two modes produce:

```
  mode=separate  units: forward(Object)  backward(backward_Object)
                forward graph 7 ops, backward graph 12 ops
                loss 1.291230   gradients returned: 64x16, 64x1, 16x64, 64, 64x1, 1
                captured: 16x64, 64, 64x1, 1

  mode=joint     units: joint(joint_Object)
                forward graph 15 ops, backward graph 15 ops
                loss 1.291230   gradients returned: 64x16, 64x1, 16x64, 64, 64x1, 1
                captured: 16x64, 64, 64x1, 1
```

Two compiled units against one, and the same six gradients: two for the user's `x` and `y`, then one per captured parameter. Proposition 63.3's offset is `6 − 4 = 2`.

The gradients, checked twice:

```
  parameter     shape  max |compiled - eager|
  W1            16x64                 5.96e-8
  b1               64                 4.47e-8
  W2             64x1                 1.79e-7
  b2                1                 2.38e-7
```

That is [Part V](../../part5/README.md)'s compiler-level differentiation agreeing with [Chapter 60](../ch60-a-pytorch-style-dispatcher/README.md)'s eager autograd — two completely independent implementations of the chain rule — to two ulps of `f32`.

And against the loss surface itself, using a *directional* difference along the gradient, which is one scalar per step size rather than one per element:

```
  the slope along the gradient itself, |g|^2 = 9.659095

         h  (L(p+hg) - L(p-hg)) / 2h  relative error
      1e-1                  9.785603         1.31e-2
      1e-2                  9.658504         6.12e-5
      1e-3                  9.659111         1.74e-6
      1e-4                  9.657145         2.02e-4
```

Truncation error falls as `h²` down to `10⁻³`, then roundoff — `f32` epsilon divided by `h` — takes over and the error rises again. The minimum is a property of the arithmetic, not of the gradient, and [Chapter 65](../../OUTLINE.md) is where choosing it stops being done by inspection.

Forty steps of SGD:

```
  path                      time  loss first   loss last  max |Δloss|  max |Δparam|
  eager autograd          50.4ms    1.291230    0.163815            —             —
  compiled, separate      27.5ms    1.291230    0.163815      1.19e-7       1.19e-7
  compiled, joint         39.9ms    1.291230    0.163815      1.19e-7       1.19e-7

  speedup: separate 1.83x, joint 1.26x
```

**1.19e-7 is exactly one `f32` ulp relative.** Over forty steps of a loop in which every parameter is updated from a gradient computed by a different mechanism, the two paths do not drift: the errors are at the last bit and they stay there, because the update is contractive. That is Theorem 63.5 case (ii), and it is the strongest statement this book makes about the compiler being correct.

The crossover:

```
   batch      eager   compiled   speedup  max |Δloss|
       1     12.4ms      3.4ms     3.69x       1.5e-6
       4     16.5ms      8.7ms     1.90x       1.2e-7
      16     22.8ms     12.6ms     1.81x       2.4e-7
      64     33.8ms     27.8ms     1.22x       1.2e-7
     256    156.7ms     77.2ms     2.03x       1.8e-7
    1024    599.5ms    292.7ms     2.05x       3.0e-7
```

The speedup is roughly 2× and roughly flat, and the batch-of-one row is the largest — which is [Chapter 4](../../part1/ch04-eager-execution/README.md)'s cost model saying what it always said: at small sizes the win is per-operation overhead, not bandwidth. The `1.5e-6` on that row is the one number in the table above one ulp, and it is the mean of a single element being computed two ways.

And the price of admission:

```
  path                  first call  steady step
  eager autograd                 —      1.259ms
  compiled, separate        16.2ms      0.630ms
  compiled, joint           10.9ms      0.569ms
```

Sixteen milliseconds to compile, half a millisecond to run. Twenty-six steps, and the compilation has paid for itself.

```bash
node docs/part11/ch63-training-end-to-end/labs/02-the-whole-framework.mjs
```

The framework path, which is one option on the trainer:

```
  run                   time   mse before    mse after
  compile=false        107ms     1.348836     6.606e-4
  compile=true          84ms     1.348836     3.059e-6
```

Both converge, from an identical initialisation, and **they do not converge to the same place** — 216× apart. Lab 01 established that the compiled and eager *gradients* agree to `2.4e-7` — two `f32` ulps, across every parameter — so Theorem 63.5 says this is not a tolerance question. Something structural differs.

The lab narrows it to one step of one model and prints the update each path applied, divided by the learning rate — that is, the gradient each optimizer believed it had:

```
  compile=false    0.00000  -0.38298   0.00000   0.00000  -0.14423   0.13213   0.00000  -0.33370
  compile=true     0.00000   0.00000   0.00000   0.00000  -0.38298  -0.41671  -0.37176  -0.34480

  identical: false
```

Not a small difference: a *rearrangement*. Both rows contain `-0.38298`, in different positions. Then it prints the gradients themselves, twice each — through their strides, and from their storage:

```
  compile=false  grad 8x4 strides (1,8) contiguous=false
                logical    0.00000   0.00000   0.00000   0.00000  -0.38298  -0.41671  -0.37176  -0.34480
                storage    0.00000  -0.38298   0.00000   0.00000  -0.14423   0.13213   0.00000  -0.33370
  compile=true   grad 8x4 strides (4,1) contiguous=true
                logical    0.00000   0.00000   0.00000   0.00000  -0.38298  -0.41671  -0.37176  -0.34480
                storage    0.00000   0.00000   0.00000   0.00000  -0.38298  -0.41671  -0.37176  -0.34480
```

**The two gradients are the same tensor.** Read through their strides they agree element for element. The eager one is a *transposed view* — strides `(1,8)` on an `8×4` tensor — because `nn.Linear` computes `x @ Wᵀ` and the transpose's VJP returns a view rather than a copy. And the optimizers read `p.grad._impl.storage.data`, which is the storage, not the tensor.

The smallest case that shows it, with the arithmetic small enough to check by hand — a `Linear(3, 2)` whose gradient under a sum reduction is `[5, 7, 9]` for each of its two output units:

```
  weight  2x3  strides (3,1)
  W.grad  2x3  strides (1,2)  contiguous=false
  gradient, read through its strides: 5 7 9 5 7 9
  gradient, read from its storage:    5 5 7 7 9 9
  update SGD applied:                 5 5 7 7 9 9
```

The gradient is right. The update is the gradient's *storage* subtracted from the weight's storage, element by element, and the two layouts do not agree — so four of the six weights moved by the gradient at a different element. The compiled path is unaffected because its gradient happens to come back contiguous.

This is Corollary 63.6 in one screen: eager training converges, has converged in every test in this repository, and applies a transposed update to every `nn.Linear` weight. **A convergence test cannot see it. A differential test against a second implementation can, and did.** The fix is a line in each of the four optimizers — read `p.grad.contiguous().data` rather than `p.grad._impl.storage.data` — or a line in gradient accumulation, materialising every incoming gradient; §63.7 weighs the two.

**Try this.** Give `Tiny` a square first layer — `Linear(8, 8)` — and re-run the second block. The two paths still disagree, because a transpose of a square matrix is still a permutation; then replace `Linear` with a raw `matmul` against a weight held as `[in, out]`, and they agree exactly, because no transpose ever enters the backward graph.

## 63.6 What Part XI established

Five chapters, and the compiler became a framework.

**The runtime is a dictionary and a slot table** ([Chapter 59](../ch59-the-runtime-module/README.md)), target-agnostic to the letter, so all four of Part X's backends are reachable through one call and a fifth is one registration away.

**The dispatcher composes layers rather than choosing branches** ([Chapter 60](../ch60-a-pytorch-style-dispatcher/README.md)), which is what lets autograd, tracing and four backends coexist without any of them naming the others — and what lets the whole compiler be a *fallback implementation* for any eager operation nobody hand-wrote.

**Tracing turns a closure into a graph** ([Chapter 61](../ch61-tracing/README.md)) by installing one more layer, and captures the closure's free variables as parameters. The two decisions taken around the compiler rather than inside it — folding weights, and gating an optimization on a measurement — both live here, and both are off by default.

**Shapes are symbols with guards** ([Chapter 62](../ch62-dynamic-shapes/README.md)), and static compilation is the same mechanism with a maximal guard set. The mechanism is sound exactly to the extent that every consumption of a symbolic quantity records a guard, and one important class of consumption does not.

**And a training step is forward, backward, apply**, with the first two compiled and the third left eager — which is what makes `compile: true` a boolean rather than a fork in the framework.

For [Part XII](../../OUTLINE.md), the useful inheritance is a method rather than a fact. Every number in this chapter came from running two implementations of the same computation and subtracting. That found a one-ulp agreement over forty optimizer steps, and it found a transposed parameter update that the repository's own convergence tests pass over. The tolerance in the first case was derived from the arithmetic; in the second, no tolerance was the right answer, and knowing which case you are in is Theorem 63.5's job.

## 63.7 Traps and limits

### The optimizers read a gradient's storage, not the gradient

§63.5 measures it. All four optimizers and the gradient scaler take `p.grad._impl.storage.data` ([`sgd.ts:22`](../../../src/optim/sgd.ts), [`adam.ts:37`](../../../src/optim/adam.ts), [`adamw.ts:21`](../../../src/optim/adamw.ts), [`fused.ts:119`](../../../src/optim/fused.ts), [`grad_scaler.ts:55`](../../../src/optim/grad_scaler.ts)), and `Optimizer.zeroGrad` fills the same array ([`optimizer.ts:51`](../../../src/optim/optimizer.ts)). `GradAccumulator` stores whatever tensor arrives ([`accumulator.ts:23`](../../../src/autograd/accumulator.ts)), views included. Any operation whose VJP returns a non-contiguous view therefore feeds the optimizer a permuted gradient; `transpose` is the one that reaches every `nn.Linear`.

There are two places to fix it and they are not equivalent. Materialising in the optimizers is four lines and costs a copy per parameter per step. Materialising in `GradAccumulator` is one line, costs a copy only where the incoming gradient is actually a view, and also fixes every other consumer of `.grad` — of which `clipGradNorm_` is one. The second is the better fix, and either way the test that would have caught it is a differential one against the compiled path, which this chapter's lab now is.

This is outside the compiler proper — `optim` and `nn` are callers in this book's scope, not subjects — and it is in scope for this chapter, because "compiled training measured against eager" is precisely the comparison that exposes it.

### `_compiledStep` does not scale the loss for gradient accumulation

The eager path divides the loss by `accumGrad` before backward ([`training_loop.ts:127`](../../../src/lightning/core/loops/training_loop.ts)). The compiled path does not ([`training_loop.ts:149`](../../../src/lightning/core/loops/training_loop.ts)), so with `accumulateGradBatches > 1` it accumulates *sums* where the eager path accumulates *means* — an effective learning rate multiplied by the accumulation factor. With the default of 1 the two agree, which is why no test notices.

### The compiled path skips every backward-related callback

`onBeforeBackward`, `onAfterBackward` and `onBeforeZeroGrad` are dispatched by the eager step and by neither branch of the compiled one. A callback that inspects gradients — a norm logger, a NaN detector — silently stops running when `compile: true` is set, and there is no warning. `onBeforeOptimizerStep` is dropped too.

### One compiled step per model, keyed on nothing

`model.__compiledTrainStep` is a single slot ([`training_loop.ts:161`](../../../src/lightning/core/loops/training_loop.ts)). It is created from the first batch and reused for every subsequent one — so [Chapter 62](../ch62-dynamic-shapes/README.md)'s cache is the thing that has to notice a shape change, which it does, because `compileWithBackward` keeps its own entry list. But the slot also survives `fit()` returning: a second `fit` on the same model with a different dataset reuses the step compiled for the first, and a model moved to another device after training reuses a step compiled for the old target. Neither is checked.

### Gradient-to-parameter matching is positional across two modules

`off = gradList.length - params.length` ([`training_loop.ts:180`](../../../src/lightning/core/loops/training_loop.ts)) assumes exactly Proposition 63.3's layout, computed in a different file from the one that establishes it. Nothing asserts the lengths are compatible; if `capturedParams()` returned more entries than there are gradients, `off` goes negative and the loop reads `gradList[-1]`, which is `undefined`, and the guard `if (g)` then silently leaves that parameter's gradient at its previous value — a stale gradient, applied forever, with no error.

### `capturedParams()` is not `model.parameters()`

The compiled step's parameter list is what the *trace* captured, which is every closed-over tensor the forward pass read. A parameter the model owns but does not use in `trainingStep` is not captured, gets no gradient assigned, and is stepped by the optimizer using whatever `.grad` it last had — usually `null`, so it is skipped, but not necessarily. Conversely a non-parameter tensor read by the forward pass *is* captured and does receive a gradient, which is then discarded. Neither case is reported.

### The joint mode's advantage is not visible at this size

§63.5 measures joint at 0.569ms against separate's 0.630ms per step but a *worse* whole-loop time, because the loop measurement includes allocation the steady-state one does not. The structural argument for joint — one graph, so fusion and [Part IX](../../part9/README.md)'s allocator see the forward and backward together — predicts a widening gap as the model grows, and this chapter does not demonstrate it. The honest statement is that at this size the two are within measurement noise of each other and the choice does not matter; anyone who needs the answer at scale should take [Chapter 47](../../part8/ch47-search-and-measurement/README.md)'s benchmark runner to a real model.

### The trained model is not saved, and nothing in this chapter checks that it could be

`compile: true` mutates the model's parameters in place, so `stateDict()` afterwards is correct by construction. But the compiled step itself — a `RuntimeModule` and two graphs — is discarded when the process exits, and [Chapter 59](../ch59-the-runtime-module/README.md)'s serialisation is not wired into any checkpoint format. Training compiled and serving compiled are two separate compilations of the same graph today.

## 63.8 Read the tests

- [`tests/e2e/compiled-training.test.js`](../../../tests/e2e/compiled-training.test.js) — `Trainer({ compile: true })` in both modes, converging and compared against the eager path. Note what it asserts: convergence, and a loss gap under `5e-3` after 40 epochs. Corollary 63.6 explains why that passes over the finding in §63.5.
- [`tests/e2e/compiled-backward-contract.test.js`](../../../tests/e2e/compiled-backward-contract.test.js) — the argument contract of `backward()`: one gradient per output, matching element counts, and named errors for each violation, in both modes.
- [`tests/e2e/differential-backward.test.js`](../../../tests/e2e/differential-backward.test.js) — compiled gradients against eager gradients across the operator set, which is the test shape §63.5 uses.
- [`tests/compiler/ad/model-gradcheck.test.js`](../../../tests/compiler/ad/model-gradcheck.test.js) — finite-difference gradient checks over twelve architectures, which is the third leg of the comparison this chapter draws twice.
- [`tests/e2e/dynamic-shapes.test.js`](../../../tests/e2e/dynamic-shapes.test.js) — the ragged-last-batch case, from the training side.

---

**Part XI ends here.** [Part XII](../../OUTLINE.md) takes the method rather than the machinery: what a verifier catches that a test does not, how to pick a tolerance that is neither vacuous nor flaky, how to assert that an algorithm is not accidentally quadratic, and how to bisect from a wrong answer down to the pass that caused it.
