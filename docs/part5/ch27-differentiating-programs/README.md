# Chapter 27 — Differentiating programs

Everything in Parts III and IV rewrote a program into a faster program that computes the same thing. This part does something different: it reads a program and writes **a second program**, one the user never wrote, computing a quantity the first one does not.

That quantity is the gradient, and the reason a compiler book has four chapters about it is that the second program is built out of the same IR, by the same builder, and is then optimized by the same passes you have just spent two parts on. Differentiation here is not a numerical technique bolted onto the side. It is a graph-to-graph transformation, and it is the last one in this book that *adds* work rather than removing it.

## 27.1 The problem: a million derivatives of one number

Training minimizes a scalar. A loss `L` is one number, and improving it means knowing `∂L/∂θ` for every parameter `θ` — for a small network, tens of thousands of numbers; for a large one, billions.

The obvious method is the definition. Perturb one parameter, recompute the loss, divide:

```
∂L/∂θₖ ≈ (L(θ + ε·eₖ) − L(θ − ε·eₖ)) / 2ε
```

This works, and §27.6 uses it as ground truth. As a training method it is hopeless, and it is worth being precise about why, because the reason is not "it is approximate".

**It costs two forward passes per parameter.** A model with 10⁷ parameters needs 2×10⁷ evaluations of the whole network to take *one* optimization step. Multiply by the number of steps in a training run and you are past the age of the universe.

**And the approximation has a floor.** Too large an `ε` and you measure curvature instead of slope; too small and the two losses differ in bits that f32 does not have, so the subtraction cancels into noise. There is a best `ε` and it still leaves you several digits short.

The second problem is annoying. The first is disqualifying. So the question is whether the gradient can be obtained for a cost that does *not* scale with the number of parameters — and the answer, which is the foundation of every deep learning framework, is yes.

## 27.2 Intuition: the chain rule associates two ways

Take a program as a composition of steps:

```
x ──f₁──> a ──f₂──> b ──f₃──> L
```

Each step has a derivative at the point it was evaluated. For tensors that derivative is a matrix — the Jacobian — and the chain rule says the derivative of the whole is the product of the parts:

```
J = J₃ · J₂ · J₁
```

Matrix multiplication is associative, so you may bracket that product however you like, and **the two bracketings are the two modes of automatic differentiation**.

**Bracketed from the input end**, `J₃ · (J₂ · J₁)`, the two factors nearest the input are combined first. Applied to a direction `v` in input space that reads `J₃·(J₂·(J₁·v))`: push `v` through `f₁` to get a direction in `a`-space, push that through `f₂`, and so on. One sweep, one input direction. This is *forward mode*, and it computes a **Jacobian-vector product**, `J·v`.

**Bracketed from the output end**, `(J₃ · J₂) · J₁`, the two factors nearest the output are combined first. Applied to a direction `w` in output space that reads `((wᵀ·J₃)·J₂)·J₁`: pull `w` back through `f₃` to get a direction in `b`-space, pull that back through `f₂`, and so on. One sweep, one output direction. This is *reverse mode*, and it computes a **vector-Jacobian product**, `wᵀ·J`.

> **A jvp is a column of `J` only when `v` is a basis vector, and likewise for rows.** The shorthand is convenient and it is worth not internalizing it wrongly, because §27.6 depends on the distinction. `J·v` is a *linear combination of the columns of `J`*, weighted by the entries of `v`; it coincides with the `i`-th column exactly when `v = eᵢ`. Symmetrically `wᵀJ` is a combination of the rows, equal to the `j`-th row exactly when `w = eⱼ`. In general a jvp is a directional derivative along `v`, and it is a full-rank object in its own right — not a piece of a matrix.
>
> This is why extracting a Jacobian takes `n` or `m` sweeps rather than one, and it is why the sweep count in Theorem 27.3 is what it is: you recover the matrix by feeding basis vectors one at a time. It also explains what training actually does, which is the case that matters most — a training step pushes back the cotangent `w = 1` on a *scalar* loss, and `1ᵀJ` is the gradient. That is not "a row of the Jacobian of the model"; it is the single row of the `1 × n` Jacobian of the loss, which is why one sweep suffices however many outputs the model has.

Which factor is combined first is the whole distinction, and it is easy to state backwards: the mode that starts at the *input* brackets `J₂·J₁`, the pair furthest from the output.

Now count. A loss has one output and a million inputs. `J` is a `1 × 10⁶` matrix: one row, a million columns. Getting it column by column takes a million sweeps. Getting it row by row takes **one**.

That asymmetry is the whole reason training is possible, and it is not a fact about neural networks. It is a fact about the shape of the function being differentiated.

The bill for it arrives in the next three chapters. Reverse mode needs the derivative of `f₂` *at the point `a` took during the forward pass* — so it cannot start until the forward pass has finished, and the values it needs have to survive until it asks. Forward mode carries its derivative alongside the value, so it never has to *retain* anything past the operation that produced it. The cheap mode is the one that remembers everything.

Say that carefully, though: "forward mode needs no memory" is false, and the accurate statement is about *retention*, not about consumption. Forward mode carries a tangent alongside every intermediate, so at any instant it holds roughly twice the live data an ordinary forward pass does — and `k` simultaneous directions cost `k` tangents per intermediate, which is how you would batch a Jacobian extraction. What it does not do is keep those tangents alive after the operation consuming them, so its peak is proportional to the *width* of the program rather than to its *length*. Reverse mode's peak is proportional to the length, because everything the backward sweep will need must survive the whole forward pass. That difference — width versus length — is the real distinction, and it is what makes Chapter 30 a chapter about reverse mode.

## 27.3 Theory

Write `f : ℝⁿ → ℝᵐ` for the function a program computes, and `J(x) ∈ ℝ^{m×n}` for its Jacobian at `x`.

> **Definition 27.1 (Tangent and cotangent maps).** The *jvp* (Jacobian-vector product) of `f` at `x` is the linear map `v ↦ J(x)·v` from ℝⁿ to ℝᵐ. The *vjp* (vector-Jacobian product) is the linear map `w ↦ Jᵀ(x)·w` from ℝᵐ to ℝⁿ, usually written `w ↦ wᵀJ(x)`.

A jvp answers "if the input moves this way, how does the output move?". A vjp answers "to move the output this way, which input direction is responsible?". Neither ever forms `J`.

> **Definition 27.2 (Forward and reverse mode).** *Forward mode* evaluates `f` and its jvp in one traversal of the program in dataflow order, carrying a tangent alongside every value. *Reverse mode* evaluates `f` in dataflow order, then evaluates its vjp in one traversal in reverse dataflow order, carrying a cotangent alongside every value.

> **Theorem 27.3 (Cost of the two modes).** *(Baur and Strassen, 1983; Griewank, 2008.)* Let `T` be the cost of evaluating `f`. Then:
> - the full Jacobian by forward mode costs `Θ(n·T)` — one sweep per input;
> - the full Jacobian by reverse mode costs `Θ(m·T)` — one sweep per output;
>
> and in particular, for `m = 1`, the complete gradient costs `Θ(T)`: a constant multiple of one forward evaluation, **independent of `n`**.

*Proof sketch.* Each sweep of either mode visits every operation once and, per operation, does work proportional to that operation's own cost — a linear map applied to one vector, never a matrix formed. So one sweep is `Θ(T)`. Forward mode's sweep is parameterized by an input direction, and `n` independent directions are needed to span the domain; reverse mode's sweep is parameterized by an output direction, and `m` span the codomain. ∎

**The hypotheses are doing real work, and the theorem is narrower than the slogan.** "Reverse mode gives you the gradient for the price of a forward pass" is repeated everywhere, this book included, and it holds under conditions worth naming:

- **`T` is a count of primitive operations, not a runtime.** The proof charges each operation its own cost and says nothing about memory traffic — and by Chapter 22 these kernels are bandwidth-bound. §27.5 measures forward-plus-backward at 1.2–4.7× a forward pass where the operation count predicts about 2×, and §27.7 attributes the spread. The `Θ` hides a constant that is not small and is dominated by re-reading saved tensors.
- **Every primitive's vjp must cost `Θ` of the primitive.** True for the arithmetic and contractions in this registry, and the reason it is true is that each vjp is itself expressible with the same primitives. It is not a law: an operation whose derivative is genuinely harder than the operation breaks the per-operation bound, and the theorem with it.
- **"The full Jacobian" means forming an `m × n` dense matrix.** Both `Θ(n·T)` and `Θ(m·T)` count sweeps needed to *fill a dense array*, and both are pessimistic when `J` is sparse or structured — a Jacobian with known sparsity can be recovered in far fewer sweeps by colouring its columns, which is a substantial literature this book does not use. Where the book quotes these bounds it means the dense case.
- **Memory is not in the model at all.** `Θ(T)` is a *time* bound. Reverse mode's memory is `Θ` of the number of retained intermediates, which is why Chapter 30 exists and why the time bound alone is not a reason to prefer reverse mode on a deep model with a tight memory budget.

The `m = 1` corollary survives all four caveats intact, and it is the one training relies on.

The constant hidden in `Θ(T)` matters in practice and is not large: the backward sweep does roughly the same operations as the forward one, sometimes two per forward operation, so two to four forward passes is the usual figure. §27.5 measures it here.

> **Corollary 27.4.** For a scalar loss, reverse mode obtains `∂L/∂θ` for every parameter at once. Finite differences obtain the same thing for `2n` forward passes. The ratio is `Θ(n)`.

Two consequences of Definition 27.2 are worth pulling out, because Chapters 29 and 30 are about them.

> **Definition 27.5 (The linearization point, stated here).** The vjp of an operation is a linear map that depends on the *values* its forward evaluation saw. Reverse mode must therefore have access, at the time it processes an operation, to some sufficient set of that operation's forward operands or results.

That is the sentence that costs memory. Forward mode never needs to *retain* it, because it processes each operation while those values are in hand — the tangent is consumed as soon as it is produced.

> **Corollary 27.6 (Reverse mode is not a rewrite of the forward pass).** The backward program is a *new* program whose inputs include values produced by the forward one. It cannot be obtained by editing the forward graph in place.

Which is why Chapter 29 builds a second `GraphFunction` rather than a pass.

## 27.4 In mlfw: two builders, and a rule count that tells you which one is real

Both modes exist. They are separate files, and their relative investment is visible in one number each.

| Mode | Entry point | Rules registered |
|---|---|---|
| Reverse (vjp) | [`BackwardGraphBuilder`](../../../src/compiler/ad/backward_builder.ts), [`JointGraphBuilder`](../../../src/compiler/ad/joint_builder.ts) | **67** |
| Forward (jvp) | [`buildForwardDiff`](../../../src/compiler/ad/jvp.ts) ([`jvp.ts:31`](../../../src/compiler/ad/jvp.ts)) | **43** |

*(Counts measured 2026-08-20 by asking the two registries.)*

The forward-mode builder is 214 lines and complete enough to differentiate the arithmetic core; it is used by tests and by nothing in the pipeline. The reverse-mode path is what `compileWithBackward` calls, what training uses, and what the remaining four chapters are about. That is the correct investment for a framework whose job is training, and the rule counts are the honest way to see it.

Forward mode is worth one look anyway, because it is the shorter program and it makes Definition 27.2 concrete ([`jvp.ts:52`](../../../src/compiler/ad/jvp.ts)):

```ts
  for (const op of topo) {
    if (op.opName === 'return') continue;

    const cloned = op.clone(fwdMap);
    builder.block.pushOp(cloned);
```

One traversal, in topological order, cloning the forward operation and then asking a rule for its tangent. The primal and the tangent are built *interleaved*, in one pass, which is exactly why forward mode saves nothing: the value and its derivative are alive at the same moment.

Reverse mode's sweep is the same loop with the index reversed ([`backward_builder.ts:56`](../../../src/compiler/ad/backward_builder.ts)):

```ts
export function backpropOps(orderedOps: readonly Operation[], { accumulator, builder, needsGrad, resolveValue, handleRegionOp = null }: BackpropOptions): void {
  for (let i = orderedOps.length - 1; i >= 0; i--) {
```

`for (let i = orderedOps.length - 1; i >= 0; i--)` is Definition 27.2's second clause, and `resolveValue` is Definition 27.5's obligation — the callback that answers "where do I find the forward value of this?". Chapter 29 is that callback.

One more thing both files share and neither inherited from the other: **when a rule is missing, they throw** ([`jvp.ts:65`](../../../src/compiler/ad/jvp.ts)):

```ts
      throw new Error(`buildForwardDiff: no JVP rule for op '${op.opName}' (forward-mode AD would otherwise emit a silently-wrong zero tangent)`);
```

The parenthetical is the design argument. A differentiator that quietly returns zero for an operation it does not understand produces a gradient that is wrong in a direction no test will notice, because zero is a plausible derivative. Chapter 31 is about the three ways this compiler can lose a gradient, and which of them are declared.

## 27.5 Lab 1 — One pass for every input

```bash
node docs/part5/ch27-differentiating-programs/labs/01-one-pass-for-every-input.mjs
```

An eight-input scalar function, differentiated by the compiler and by central differences.

```
=== the same eight partial derivatives, two ways ===
  k   reverse mode        central differences   rel. error
  0      0.109416448         0.109415501   8.5e-7
  1      0.160724565         0.160720199   3.8e-6
  2      0.088943124         0.088948756   5.2e-6
  3     -0.019394794        -0.019397587   2.7e-6
  4      0.052954156         0.052951276   2.7e-6
  5     -0.011402324        -0.011406839   4.5e-6
  6      0.243114963         0.243116170   9.7e-7
  7      0.018408928         0.018410385   1.4e-6

=== what each one cost ===
  reverse mode:         1 forward + 1 backward
  central differences:  16 forward evaluations
```

Agreement to five or six digits, which is what a central difference in f32 with `ε = 10⁻³` is worth and is the tolerance every gradient test in this repository uses. **The compiled gradient is the accurate one here**, not the reference: finite differences are the approximation, and this table is measuring *their* error.

Then the scaling, which is Corollary 27.4:

```
=== how the cost scales with the number of inputs (medians of 15 rounds) ===
  inputs   1 forward   reverse   differences   ratio   rev/fwd
       8       0.019     0.054         0.277     5.1x      2.9x
      32       0.020     0.094         0.487     5.2x      4.7x
     128       0.045     0.053         1.510    28.3x      1.2x
```

**Every column here is measured, including `differences`.** The tempting shortcut is to extrapolate that column as `oneForward × 2n` — one forward pass times the number of evaluations a central-difference sweep needs — and on this model it overstates the real cost about sevenfold at `n = 128`, because a sweep calls one already-compiled kernel `2n` times in a tight loop and amortizes almost everything the first call paid for. If you take one methodological lesson from this part, let it be that one: **a cost you multiplied is not a cost you measured.**

Read the columns separately. The `differences` column grows superlinearly — both the number of evaluations and the cost of each grow with input width. The `reverse` column barely moves, because there is still exactly one backward pass and this model is small enough that the per-call overhead of Chapter 4's `α` dominates its arithmetic at every size shown. The ratio therefore grows, which is Corollary 27.4, and it grows for the reason the corollary gives.

The `rev/fwd` column is the constant in Theorem 27.3, now measured directly rather than derived: forward-plus-backward costs between 1.2× and 4.7× a forward pass across these three sizes. That spread is itself informative and it is not the theorem wobbling — at `n = 128` the reverse column has stopped scaling at all, so the ratio is measuring overhead against overhead. The theorem's `Θ(T)` is an asymptotic statement about a program whose cost is its arithmetic, and none of these three rows is in that regime. A model large enough to be arithmetic-bound is where the usual two-to-four lives.

**Try this.** Push `n` to 512 and predict the ratio before running. Then change the model to have two outputs instead of one and watch what happens to the reverse column.

## 27.6 Lab 2 — The Jacobian, both ways

```bash
node docs/part5/ch27-differentiating-programs/labs/02-the-jacobian-both-ways.mjs
```

A function `ℝ⁴ → ℝ³`, so the Jacobian is a `3 × 4` matrix and neither mode is obviously better. Reverse mode fills it a row at a time; differences fill it a column at a time.

```
=== J by reverse mode: one row per backward pass ===
  [ -0.147704  -0.172364  -0.033161   0.113694 ]
  [  0.084836   0.057931   0.075426   0.023331 ]
  [  0.137787   0.010754   0.052656  -0.006642 ]
  3 backward pass(es)

=== J by central differences: one column per input ===
  [ -0.147715  -0.172354  -0.033155   0.113681 ]
  [  0.084832   0.057921   0.075430   0.023320 ]
  [  0.137791   0.010766   0.052664  -0.006624 ]
  8 forward evaluation(s)

largest relative disagreement: 1.8e-5
```

The same matrix, assembled in two different directions. The row-at-a-time version obtained row `i` by running the backward pass with a cotangent that is `1` at output `i` and zero elsewhere — which is Definition 27.1 with `w = eᵢ`, and `eᵢᵀJ` is exactly row `i`.

That is the practical form of the whole chapter: **the thing you pass into a backward pass is a direction in output space, not a scalar**. Frameworks hide this by defaulting it to `1` for a scalar loss, and `cf.backward(ones(...))` in every other lab in this part is that default written out. Pass something else and you get a different row.

And the rule the lab prints at the end:

```
  outputs = 3, inputs = 4
  reverse mode costs ~3 sweep(s); a per-input mode costs ~4
  training has outputs = 1 and inputs = every parameter, so reverse wins by that ratio
```

Three against four is nearly a tie. One against ten million is not, and that is the only reason this book has a reverse-mode differentiator and a forward-mode one that nothing calls.

**Try this.** Change `N_OUT` to 1 and `N_IN` to 8 and re-read both counts. Then swap them — `N_OUT = 8`, `N_IN = 1` — and decide which mode you would implement if that were the shape of your problem.

## 27.7 Traps and limits

- **Forward mode exists, is tested, and is called by nothing in the pipeline.** [`buildForwardDiff`](../../../src/compiler/ad/jvp.ts) has no caller in `src/` outside its own file. It is the right thing to keep — a Hessian-vector product is a jvp of a vjp, and Chapter 46's cost models would want one — but today it is a designed extension point with 43 rules and no users, and its rule set has already drifted 24 rules behind the reverse-mode one.
- **Forward mode refuses more than reverse mode does.** `reduce` in forward mode supports only `sum` and `mean` and throws for the rest ([`jvp.ts:167`](../../../src/compiler/ad/jvp.ts)), where the reverse rule handles `max`, `min` and `prod` as well (Chapter 28). Two differentiators for one IR means two coverage surfaces, and they are not the same shape.
- **Finite differences are the reference and are the less accurate side.** Every gradient test in this repository compares compiled gradients against central differences with `ε = 2×10⁻³` and a tolerance around `10⁻²`. That loose tolerance is not slack in the compiler; it is the noise floor of the reference.
- **`Θ(T)` is per output, and multi-output models pay per output.** A model with two heads needs two backward passes for the full Jacobian, and `cf.backward` takes one cotangent per forward output ([`tests/e2e/compiled-backward-contract.test.js`](../../../tests/e2e/compiled-backward-contract.test.js) pins the arity check). What training does instead is push a *single* cotangent through — the gradient of the scalar loss — which is one sweep no matter how many heads there are.
- **The cost model in Theorem 27.3 counts operations, not memory traffic.** By Chapter 22 the runtime of these kernels is bandwidth, and the backward pass re-reads saved tensors the forward pass wrote. The `rev/fwd` column in §27.5 reaches 4.7× where the operation count suggests about 2×, and Chapter 30 is where that gap gets a name. The same column also drops to 1.2× on the largest model there, which is overhead rather than efficiency — read the ratio together with the absolute times, never alone.

## 27.8 Read the tests

- [`tests/compiler/ad/backward-numerical.test.js`](../../../tests/compiler/ad/backward-numerical.test.js) — the per-operation gradient checks against finite differences; this is the executable form of §27.5.
- [`tests/compiler/ad/model-gradcheck.test.js`](../../../tests/compiler/ad/model-gradcheck.test.js) — the same check on whole models, including a CNN and a ResNet block, which is where composition of rules gets exercised.
- [`tests/compiler/ad/jvp.test.js`](../../../tests/compiler/ad/jvp.test.js) — forward mode, including the cases it refuses.

---

**Next:** [Chapter 28 — Writing a VJP rule](../ch28-writing-a-vjp-rule/README.md), which takes the `Jᵀ·w` of one operation and asks what it looks like as a registry entry — and what a rule has to be given in order to be writable at all.
