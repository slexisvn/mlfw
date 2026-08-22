# Chapter 31 — Differentiating control flow

Everything so far assumed an operation is a leaf: it has operands, it has results, and a fifteen-line rule relates their derivatives. Two operations in this IR are not leaves. A `scan` holds a loop body; an `if` holds two branches. Their derivative is not a formula, it is *another program*, and a rule taking `(operands, results, gradOutputs)` has nowhere to put it.

There is a second thing this chapter is about, and it is the more important one. Some operations have no derivative at all — a comparison, an `argmax`, a cast to integer — and every autodiff system has to decide what to do when the sweep reaches one. There are exactly three answers, this compiler uses all three, and the difference between them is the difference between a model that trains and a model that quietly does not.

## 31.1 The problem: an operation whose body is a program

A recurrent layer traces to one operation:

```
%h = scan(%xs, %h0) {num_carry = 1, num_xs = 1} : ...
{
  ^bb(%x_t, %c):
    ... the cell, twenty operations ...
    yield(%c_next)
}
```

One `scan`, whose region holds the loop body, and whose size on the page is independent of how many timesteps it will run. That independence is the whole point of having the operation (Chapter 9).

Now differentiate it. The chain rule over `T` steps says the cotangent entering step `T−1` must be pulled back through the body, then the result pulled back through step `T−2`, and so on to step 0 — with each step's pullback linearized at *that step's* values. So the backward of a loop is a loop, running the other way, over values the forward loop produced and did not keep.

A `Pattern`-shaped rule cannot express that. It would have to build a region, decide what the reversed loop carries, and arrange for `T` sets of forward values to be available. So the registry needs a second kind of entry, and the sweep needs a hook to reach it.

And then there is the other problem. Write `relu` as `where(x > 0, x, 0)`. The `>` produces booleans. There is no derivative of a boolean with respect to a float — not zero, *undefined* — and if the sweep walks into it and asks for a rule, something has to happen.

## 31.2 Intuition: a hook, and three honest answers

**For regions**, add a second registry keyed by operation name, holding functions that take the whole operation and the sweep's state — accumulator, builder, value resolver — rather than a fixed argument list. The sweep checks that registry first, and if an entry exists it hands over control entirely. The entry does whatever it likes: build a loop, unroll one, emit two branches and select between them.

**For missing derivatives**, the three answers are:

1. **Return a zero.** The rule exists and produces a tensor of zeros. Gradient flow stops, silently, and everything downstream of that operation still gets walked.
2. **Declare a barrier.** No rule; the operation is registered as one whose operands are not on the gradient path. The reachability analysis stops there, so the operands are never differentiated at all.
3. **Refuse.** No rule and no declaration: throw, naming the operation and telling the user which of the two declarations to add.

The first two both produce zero gradients and are not the same. The first is a claim that the derivative *is* zero — true for `floor`, whose derivative is genuinely zero almost everywhere. The second is a claim that the question does not arise — true for a comparison, whose output is not a float. The third is what happens when nobody has made either claim, and it is the only one that cannot cause a silently wrong training run.

## 31.3 Theory

> **Definition 31.1 (Region VJP).** A *region VJP* for an operation `f` with regions is a function that, given `f` and the reverse sweep's state, accumulates cotangents into `f`'s operands and into the free variables of `f`'s regions, and returns nothing.

The clause about free variables is the one that is easy to miss. A loop body may read values from the enclosing function — a weight matrix, say — without them being operands of the `scan`. Those are *free variables* of the region, they receive gradient, and nothing in the operation's operand list mentions them.

> **Theorem 31.2 (Reverse of a scan).** Let a scan step be `(c_{t+1}, y_t) = g(x_t, c_t)`, run for `t = 0 … T−1`. Then with `w^c_T` the cotangent of the final carry and `w^y_t` that of the output emitted at step `t`,
>
> `(w^x_t, w^c_t) = vjp_g(x_t, c_t)·(w^c_{t+1}, w^y_t)`
>
> evaluated for `t = T−1 … 0`, and the cotangent of any free variable is `Σ_t` its per-step contribution.

*Proof sketch.* The scan is the composition `g_{T−1} ∘ … ∘ g_0` with outputs tapped at each stage. Theorem 29.3 applied to that composition gives the reverse walk; the free-variable sum is Definition 29.2 applied to a value with `T` consumers, one per step. ∎

The cotangent handed to each step is a *pair*, and it is worth insisting on that. `g` has two results, so its vjp takes one cotangent per result; `w^c_{t+1}` lives in carry space and `w^y_t` in output space, and adding them together is meaningful only in the special case where the scan emits exactly the value it carries. That shape is common enough — every `scan` whose body ends `return [next, next]` has it — that a formula written as a sum can pass every gradient check you throw at it and still be the wrong statement. §31.4 concatenates, and never adds.

The theorem says what to compute, not how to hold it. It needs `(x_t, c_t)` for every `t` — the linearization points of all `T` steps — and that is Chapter 30's problem in its sharpest form, because `T` can be thousands.

> **Definition 31.3 (Gradient barrier, stated here).** A *gradient barrier* is an operation declared to terminate gradient propagation: the reverse sweep neither requires a rule for it nor propagates cotangents to its operands.

> **Corollary 31.4 (The three outcomes).** For an operation on the gradient path, an autodiff system may: apply a rule; stop at a declared barrier; or fail. A system with only the first two options cannot distinguish "the derivative is zero" from "nobody has said what the derivative is", and the second is a bug that presents as the first.

That corollary is why the third option earns its keep. A zero gradient is indistinguishable from a correct gradient that happens to be zero, so it is the perfect disguise for a missing rule — the model trains, slightly wrongly, forever.

## 31.4 In mlfw: a second registry and one throw

### The region hook

[`ad/vjp_registry.ts`](../../../src/compiler/ad/vjp_registry.ts) holds a second map alongside the first, and [`ad/scan_backward.ts`](../../../src/compiler/ad/scan_backward.ts) fills it with exactly two entries ([`scan_backward.ts:33`](../../../src/compiler/ad/scan_backward.ts)):

```ts
registerRegionVJP('scan', ((op: Operation, ctx: RegionVJPCtx) => buildScanBackward(op, ctx.accumulator, ctx.builder, ctx.materialize, ctx.needsGrad, ctx.scanCheckpoint)) as never);
registerRegionVJP('if', ((op: Operation, ctx: RegionVJPCtx) => buildCondBackward(op, ctx.accumulator, ctx.builder, ctx.materialize, ctx.needsGrad)) as never);
```

The sweep reaches them through the callback Chapter 29 left open ([`backward_builder.ts:68`](../../../src/compiler/ad/backward_builder.ts)):

```ts
    if (handleRegionOp && handleRegionOp(op)) continue;
```

placed *before* the rule lookup, so a region operation never gets as far as `requireVJPRuleOrBarrier`. And reachability has a matching special case ([`backward_builder.ts:106`](../../../src/compiler/ad/backward_builder.ts)):

```ts
    if (REGION_CONTROL_FLOW.has(op.opName)) {
      for (let o = 0; o < op.numOperands; o++) needsGrad.add(op.getOperand(o).id);
      for (const fv of regionControlFlowFreeVars(op)) needsGrad.add(fv.id);
      continue;
    }
```

Both operands *and* region free variables enter the reachable set — Definition 31.1's second clause, in the analysis rather than in the rule. `regionFreeVars` ([`scan_backward.ts:36`](../../../src/compiler/ad/scan_backward.ts)) is the walk that finds them: collect every value defined inside the region, then every operand referring to something not in that set, skipping constants.

### What `buildScanBackward` actually does

[`scan_backward.ts:232`](../../../src/compiler/ad/scan_backward.ts), and the honest summary is in the shape of its main loop ([`scan_backward.ts:285`](../../../src/compiler/ad/scan_backward.ts)):

```ts
  if (!segLen) {
    let carry: Value[] = initCarryB;
    const carriesAtT: Value[][] = [carry];
    const xsSlices: Value[][] = [];
    for (let t = 0; t < T; t++) {
      const xt = sliceX(t);
      xsSlices.push(xt);
      carry = stepForward(xt, carry);
      carriesAtT.push(carry);
    }
    for (let t = T - 1; t >= 0; t--) backwardStep(t, xsSlices[t], carriesAtT[t]);
  }
```

**This is a loop in the compiler, not in the emitted program.** `T` is read from the input tensor's leading dimension at build time, the forward body is replayed into the graph `T` times to recover every `(x_t, c_t)`, and then the backward body is emitted `T` times. Theorem 31.2 is satisfied by unrolling it.

`backwardStep` is where the theorem's cotangent pair is assembled ([`scan_backward.ts:265`](../../../src/compiler/ad/scan_backward.ts)):

```ts
    const gY_t = gYs.map(g => (g === null ? null : sliceStep(builder, g, t)));
    const gradYields = [...gCarry, ...gY_t];
```

One entry per result of the body, in the body's own order: the carry cotangents this step inherited from the step after it, then this step's slice of each emitted output's cotangent. `diffBodyStep` returns `gradArgs` split the same way, `gXsSteps` collecting the per-step input cotangents and `gCarry` becoming the pair handed to step `t−1`.

The consequence is that the backward graph is `Θ(T)` operations where the forward graph is `Θ(1)`, and §31.5 measures the constant. The `else` branch is the mitigation: with `scanCheckpoint` set, `resolveSegmentLength` ([`scan_backward.ts:222`](../../../src/compiler/ad/scan_backward.ts)) picks `⌈√T⌉` and the loop replays segment by segment instead of all at once — Theorem 30.3 applied inside the scan. Nothing sets that option (Chapter 30).

### What `buildCondBackward` does

[`scan_backward.ts:189`](../../../src/compiler/ad/scan_backward.ts) differentiates **both branches** and then selects ([`scan_backward.ts:211`](../../../src/compiler/ad/scan_backward.ts)):

```ts
  for (const [id, val] of allVars) {
    if (!needsGrad.has(id)) continue;
    const gt = gThenMap.get(id);
    const ge = gElseMap.get(id);
    if (!gt && !ge) continue;
    const zero = zeroLike(builder, val);
    const predBr = builder.broadcast(pred, (val.type as TensorType).shape, []).getResult(0);
    accumulator.accumulate(id, builder.select(predBr, gt ?? zero, ge ?? zero).getResult(0));
  }
```

The backward of a conditional is not a conditional. Both branch bodies are emitted, both are differentiated, and a `select` on the predicate picks the answer — so a two-branch `if` costs both branches in the backward pass no matter which one ran forward. That is a defensible choice for a tensor compiler, where a `select` is cheap and a data-dependent branch is not, and it is a real cost worth knowing.

### The three outcomes

Corollary 31.4 is three lines of one function ([`vjp_registry.ts:32`](../../../src/compiler/ad/vjp_registry.ts)):

```ts
export function requireVJPRuleOrBarrier(opName: string): VJPRule | null {
  const rule = _rules.get(opName);
  if (rule) return rule;
  if (_barriers.has(opName)) return null;
  throw new Error(`autodiff: op '${opName}' is on the gradient path but has no VJP rule and is not a registered gradient barrier. Register one with registerVJPRule('${opName}', ...) or registerGradientBarrier('${opName}').`);
}
```

A rule, a `null`, or an error that names the operation and both remedies. The barrier list is nine names ([`vjp_rules/control.ts:3`](../../../src/compiler/ad/vjp_rules/control.ts)):

```ts
registerGradientBarrier('stop_gradient');
registerVJPRule('stop_gradient', () => [null]);

for (const op of ['compare', 'logical_not', 'logical_and', 'logical_or', 'argmax', 'argmin', 'iota', 'one_hot']) {
  registerGradientBarrier(op);
}
```

It is tempting to summarize that list as "every one of them produces a boolean or an integer", and seven of the nine do. **`one_hot` does not**: its result dtype defaults to `f32` ([`ops/data.ts:97`](../../../src/compiler/ir/graph/ops/data.ts): `|| ScalarType.F32`), and a one-hot encoding is an ordinary float tensor that flows into ordinary float arithmetic. So the output-type criterion is not the one this set actually satisfies, and stating it that way would leave you unable to explain why `one_hot` belongs.

The criterion that does hold is about the *other* end:

> **A barrier is an operation across which no cotangent can meaningfully flow** — either because its result is not a float, so nothing downstream can hand it one, or because its differentiable-looking result depends only on operands that are *indices*, for which a derivative is undefined.

`compare`, `logical_*`, `argmax` and `argmin` are the first kind. `one_hot` and `iota` are the second: `one_hot`'s single operand is an integer index tensor, `iota`'s output depends on no operand at all. In both cases the gradient with respect to the operand does not exist — not "is zero" — and stopping is the right answer, but the reason is the operand, not the result.

That distinction has a practical consequence. An operation whose *output* is a float is one a downstream rule will happily send a cotangent to, so a barrier of the second kind is invisible until the sweep reaches it. If you add an operation with float output and index inputs and forget to declare it, the failure is Corollary 31.4's third outcome — an error naming the operation — which is the good case, and the reason §31.4's registry throws rather than defaulting.

`stop_gradient` is the exception to both kinds and appears twice — a barrier *and* a rule returning `[null]` — because it is the one operation whose whole purpose is to be a barrier the user asked for.

The first outcome, the numeric zero, is four operations ([`vjp_rules/unary.ts:199`](../../../src/compiler/ad/vjp_rules/unary.ts)):

```ts
for (const op of ['floor', 'ceil', 'round', 'sign']) {
  registerVJPRule(op, (ctx) => [ctx.full(0, ctx.operands[0].type)]);
}
```

Mathematically right — these are piecewise constant, so their derivative is zero wherever it exists — and structurally the opposite of a barrier: the operand stays on the gradient path and a full tensor of zeros gets emitted into the graph. §31.6 measures what that costs, and the answer is more interesting than it looks.

## 31.5 Lab 1 — The loop, unrolled

```bash
node docs/part5/ch31-differentiating-control-flow/labs/01-the-loop-unrolled.mjs
```

A single-layer LSTM, which traces to one `scan`, differentiated at four sequence lengths.

```
=== forward and backward graph size against sequence length ===
  T    forward ops  scan?   backward ops  scan?   bwd/T
   2            30  true             152  false   76.0
   4            30  true             304  false   76.0
   8            30  true             604  false   75.5
  16            30  true            1204  false   75.3
```

Three columns, and each one is a sentence.

**The forward graph does not grow.** Thirty operations at `T = 2` and thirty at `T = 16`, because the loop is one `scan` operation and the sequence length is a dimension of its input. This is what the region was for.

**The backward graph grows linearly, at about 76 operations per timestep.** Exactly linear — `bwd/T` moves by less than 1% across an 8× span, and the drift is the four operations of loop preamble amortized over more steps — because `buildScanBackward` emitted the forward body once and the backward body once for each of `T` steps.

**And there is no `scan` in the backward graph at all.** The differentiator did not produce a reversed loop; it produced `T` copies. For this sixteen-step toy that is 1,204 operations. For a 512-step sequence it would be about 39,000, all of which then go through canonicalization, CSE, fusion, lowering and code generation — every pass in Parts III, IV and VI, on a graph two orders of magnitude larger than the one the user wrote.

That is the single most consequential limit in Part V, and it is a design decision rather than an oversight: unrolling makes every VJP rule usable unchanged inside a loop body, and building a genuine reversed `scan` would need a second implementation of the sweep that emits into a region. The scan-level checkpointing hook exists to bound the *memory* of the unrolled form, and nothing sets it.

The gradient itself is fine:

```
=== the gradient is still right ===
  6 partials checked against central differences
  largest relative error: 1.7e-5
```

**Try this.** Push `T` to 64 and time the compile rather than the run. Then compare against a two-layer LSTM at `T = 16` and work out which of depth and length costs more.

## 31.6 Lab 2 — Three ways to lose a gradient

```bash
node docs/part5/ch31-differentiating-control-flow/labs/02-three-ways-to-lose-a-gradient.mjs
```

```
a rule exists and the derivative is real:
  sum(x)                              2 bwd ops   [1,1,1,1]
  sum(x * x)                          4 bwd ops   [3.4,-4.6,0.8,7.8]

a rule exists and returns a numeric zero:
  sum(floor(x))                       1 bwd ops   [0,0,0,0]
  sum(sign(x))                        1 bwd ops   [0,0,0,0]
  sum(x * floor(x))                   6 bwd ops   [1,-3,0,3]

the operation is a declared barrier (compare), so the
sweep never reaches its operands at all:
  sum(maximum(x, 0))   [= relu]       5 bwd ops   [1,0,1,1]
  sum(where(x > 0, x, -x))            4 bwd ops   [1,-1,1,1]
```

Read the middle group carefully, because it is not what the source predicts. `floor`'s rule emits `full(0, …)` — a scalar constant and a broadcast to the operand's full shape — and `sum(floor(x))` should therefore contain that machinery. It contains **one operation**: a folded constant.

Switch the simplification passes off and the emitted form appears:

```
=== the same programs with canonicalize/simplify disabled ===
  sum(floor(x))         6 bwd ops as the rules emitted them
  sum(x * floor(x))    10 bwd ops as the rules emitted them
```

Six and ten, against one and six. **Part IV cleans up after Part V**, and the two rows show how far it gets. In `sum(floor(x))` the zero is the *only* contribution, so constant folding (Chapter 19) turns the broadcast into a dense constant and DCE sweeps everything that fed it; six operations become one. In `sum(x·floor(x))` the zero is added to a real contribution, and there the cleanup stops:

```
    %6 = mul(%5, %3) : tensor<1x4xf32>
    %7 = constant() {tensor_type = tensor<1x4xf32>, value = 0} : tensor<1x4xf32>
    %8 = add(%6, %7) : tensor<1x4xf32>
```

`x + 0` is not an identity on floats — it maps `−0` to `+0` — so `AddZero` declines without a fast-math licence (Chapter 20), and a constant and an addition the compiler can prove are useless reach the backend anyway. This is the cost of Chapter 20's soundness gate stated as a number: two operations per zero-valued gradient contribution, on every program that has one.

It is also the reason to prefer *not emitting* the zero to *removing* it afterwards. `computeGradReachability` puts `floor`'s operand on the gradient path because `floor` has a rule, not because the rule returns anything (Chapter 29). A sweep that asked what the rule returns would emit nothing here, and nothing is cheaper than a rewrite that is only allowed to fire under a flag nobody sets.

The last group is the barrier. `relu` traces to `maximum(x, zeros)`, whose rule emits a `compare` — and `compare` is a barrier, so the sweep stops there and the comparison's operands are never differentiated. The gradient `[1, 0, 1, 1]` is correct, and the backward graph contains no attempt to differentiate a boolean.

The third outcome does not appear in the table because no traced user program can reach it: every operation a traced model can produce has either a rule or a barrier. That is the point of the check — it is a guard on *adding an operation*, and the message it would print names both remedies.

**Try this.** Replace `floor` with `round` and confirm the counts are identical, then with `abs` — which has a real derivative — and watch the middle group's numbers change.

## 31.7 Traps and limits

- **`scan` backward is fully unrolled and there is no bound on `T`.** §31.5. A long sequence produces a graph proportional to its length, at compile time, with no warning and no cap. The `scanCheckpoint` option bounds the memory of the unrolled form and not its size, and nothing sets it ([`scan_backward.ts:222`](../../../src/compiler/ad/scan_backward.ts)).
- **`T` must be a static dimension.** [`scan_backward.ts:242`](../../../src/compiler/ad/scan_backward.ts) reads `shape[0]` of the first scanned input as a number. A `scan` over a symbolic sequence length (Chapter 10) cannot be differentiated, because the unroll count is not known.
- **`if` costs both branches in the backward pass.** §31.4. The forward `if` executes one branch; its backward executes both and selects. For branches of very different cost that is a real asymmetry between the forward and backward passes.
- **Inside a region body, everything is treated as needing gradient.** `ALWAYS_NEEDS_GRAD` ([`scan_backward.ts:31`](../../../src/compiler/ad/scan_backward.ts)) is a set whose `has` returns `true` unconditionally, and it is what a nested region's backward is given. So Definition 29.1's pruning does not apply inside a loop body: every operation in the body is differentiated, whether or not its result reaches anything. On an unrolled `T`-step scan that overhead is multiplied by `T`.
- **`findUnsupportedGradOps` has no callers.** [`vjp_registry.ts:39`](../../../src/compiler/ad/vjp_registry.ts) exists to answer "which operations in this graph would fail to differentiate" *before* the sweep starts, which is exactly the diagnostic a user wants. Nothing in `src/` or `tests/` calls it, so the failure arrives from the middle of the sweep instead of up front. It also checks only the rule and barrier maps, so it would report `scan` and `if` as unsupported.
- **The barrier list is a list, not a property — and it could not be one property.** §31.4 works through this: seven of the nine barriers produce booleans or integers, and *that* part a type check could enforce automatically. The other two, `one_hot` and `iota`, produce `f32` and are barriers because of their operands rather than their results, so no single dtype rule covers the set. What that means practically is that adding an integer-valued operation without declaring it is a missing declaration rather than an automatic barrier, and adding a float-valued index consumer without declaring it is the same. Both fail as the throw rather than the right answer.

- **The sweep never revisits the operations it emits, which is why the backward graph may contain a `compare` that was never a barrier question.** The reverse walk runs over the *forward* function's operations in reverse topological order, and the graph it builds is output, not input. So when an `if` rule emits a `select` guarded by the original condition, or a `max` rule emits a `compare` to route the cotangent, those newly created comparison and control operations are simply appended — they are never handed to `requireVJPRuleOrBarrier`, never checked for a rule, and never reached by `computeGradReachability`. That is correct: they are part of the derivative, not part of the function being differentiated, and differentiating them again would be meaningless. But it is an asymmetry worth naming, because it means "every `compare` in the program is a declared barrier" is true of the forward graph and not of the compiled artifact. If you are reading a backward graph and find a `compare` in it, do not go looking for its barrier declaration.
- **There is no stated contract for mutating an input between `cf(x)` and `cf.backward(w)`.** **(invariant — undefined.)** Chapter 29 §29.6 exhibits the joint mode's version of this: input buffers are aliased rather than copied, the joint kernel is deferred, and `backward` re-runs it — so one call can return a forward output computed from the call-time input and a gradient computed from a later one. The wider point belongs here, because it applies to both modes and to the whole two-call API:

  | Question | Answer today |
  |---|---|
  | May I mutate an input after `cf(x)` returns? | unspecified |
  | May I mutate a *parameter* between forward and backward? | unspecified; the aliasing in §29.6 says the gradient sees the new value |
  | Are the forward output and the gradient guaranteed to come from the same input state? | **no** — §29.6 has the counterexample |
  | Does reading the forward output change what `backward` computes? | in joint mode, yes: it settles the kernel early |

  Every row of that table should have a documented answer and none does. This is not an exotic scenario for a training loop: an optimizer that updates parameters in place, a data loader reusing one staging buffer, or a gradient-accumulation loop that overwrites its inputs will all reach it. Until the contract is written, the safe discipline is: **do not mutate anything the compiled function was handed until you have both the outputs and the gradients you intend to use.** The framework will not warn you, and the failure is a silently wrong gradient rather than an error.

- **A numeric zero keeps its operand on the gradient path.** `computeGradReachability` tests for the *presence* of a rule, not for what the rule returns (Chapter 29), so `floor` pulls its operand into the reachable set and the zeros are emitted. Part IV removes them only where the zero is the whole contribution; where it is added to a real one, the `constant` and the `add` survive to the backend, because `x + 0` is unsound on floats.

## 31.8 Read the tests

- [`tests/compiler/ad/scan-vjp-gradcheck.test.js`](../../../tests/compiler/ad/scan-vjp-gradcheck.test.js) — LSTM and GRU gradients against finite differences, in both packaging modes, which is what pins Theorem 31.2.
- [`tests/compiler/ad/nested-control-flow.test.js`](../../../tests/compiler/ad/nested-control-flow.test.js) — a `scan` inside an `if` and the free-variable accounting that goes with it.
- [`tests/compiler/ad/checkpoint.test.js`](../../../tests/compiler/ad/checkpoint.test.js) — including the scan-level checkpointing that this chapter's option cannot reach from the public API.

---

**Part V ends here.** A forward graph goes in; a second graph comes out, built by walking the first backwards, applying one small rule per operation, summing where the forward pass forked, and either saving or recomputing every value those rules needed. It is then handed to Parts III and IV as an ordinary graph — verified, canonicalized, folded, fused — which is why this part could afford rules that emit three operations where one would do.

What has not happened yet is any of it becoming a loop. Every graph in this book so far, forward or backward, is still whole-tensor operations with no indices and no buffers.

**Next:** [Part VI — Lowering to loops](../../part6/README.md), which introduces the second IR and turns one `fusion` region into one loop nest.
