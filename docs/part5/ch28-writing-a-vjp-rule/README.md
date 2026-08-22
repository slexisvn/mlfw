# Chapter 28 — Writing a VJP rule

Chapter 27 established that the gradient is a backward sweep applying one linear map per operation. This chapter is about that map: where it lives, what it is allowed to look at, and what it is obliged to produce.

**Sixty-seven** of these rules are registered across seven files under [`ad/vjp_rules/`](../../../src/compiler/ad/vjp_rules/), and **sixty-five** of the sixty-seven name an operation the op registry actually contains (measured 2026-08-21, by asking the registry rather than by grepping). §28.7 is about the other two. Each rule is written once by whoever added the operation, and each is a small piece of calculus that has to be right — because nothing downstream will catch it if it is not.

**Most of them are short; thirteen are not, and the long ones are the interesting ones.** The impression this chapter's examples give — a derivative is three or four lines — holds for the large majority. Where it fails, it fails for one reason, and the four longest make it visible:

| Rule | Lines | Why it is long |
|---|---:|---|
| `reduce` | 70 | one branch per reduction kind — `sum`, `max`, `min`, `prod` |
| `layer_norm` | 49 | three gradients through a normalization with two statistics |
| `dot` | 45 | general contracting- and batch-dimension bookkeeping |
| `scaled_dot_product_attention` | 43 | four operands, a mask, and a softmax's derivative inlined |

The pattern is that **length tracks the arity of the operation's *attribute* space, not the difficulty of its calculus.** The cleanest demonstration is a pair. `matmul`'s derivative is two matrix multiplies and a couple of transposes, and its rule is exactly that: **fifteen lines**, of which five are a helper that swaps the last two axes. `dot` computes the same mathematics and takes forty-five, because `dot` carries `lhs_contracting`, `rhs_contracting`, `lhs_batch` and `rhs_batch` as attributes and the rule has to work out which axes the transposes touch for an arbitrary setting of all four. Same calculus, three times the code, and every extra line is index bookkeeping — exactly the kind of code Chapter 34's shared skeletons remove from the lowering rules. No equivalent factoring exists on the AD side yet, which is why the long rules are long.

A wrong pass produces invalid IR and Chapter 15 names it. A wrong derivative produces valid IR that trains a model slightly worse than it should, and the only thing that catches *that* is a finite-difference test.

## 28.1 The problem: where does `d/dx` live?

The naive version is a function with a `switch`:

```js
function gradientOf(opName, x, gradOut) {
  switch (opName) {
    case 'exp': return mul(gradOut, exp(x));
    case 'log': return div(gradOut, x);
    // ... 65 more
  }
}
```

Three things go wrong, and they are the same three as Chapter 11's.

**The knowledge is in the wrong place.** The person who adds `gelu` to the IR knows its derivative. They should not have to find and edit a 900-line `switch` in a different directory to record it.

**The signature is wrong for half the rules.** `exp`'s derivative needs the *result*, `log`'s needs the *operand*, `dot`'s needs both operands and the operation's contracting-dimension attributes, and `maximum`'s needs both operands so it can decide which one won. A single `(x, gradOut)` parameter list serves none of them.

**And the output shape is not the input shape.** An operation with two operands has two gradients. An operation whose operand was broadcast has a gradient bigger than the operand. An operation with an integer operand has no gradient for it at all — which is different from having a zero gradient, and the difference matters.

## 28.2 Intuition: a rule is a function that writes IR

The move is Chapter 17's move. Make the rule an object — here, a function — registered against the operation name, and give it everything it could want as one context argument.

A rule is handed:

- the operation, so it can read attributes;
- the **operands**, as they were in the forward pass;
- the **results**, likewise;
- the **incoming gradient**, one per result;
- a builder, positioned in the backward function.

and it returns one gradient per operand, or `null` for operands that have none.

The important word is *builder*. A rule does not compute numbers. It **emits operations** into the backward graph, and those operations are then fused, folded and lowered by every pass in Part IV, exactly like the forward ones. `exp`'s rule does not multiply anything; it writes a `mul` into a graph that will not run until later. This is what makes the whole design work: the backward pass is not a separate execution engine, it is more IR.

## 28.3 Theory

> **Definition 28.1 (VJP rule).** **(stated here)** Let `f` be an operation with operands `x₁…x_k` and results `y₁…y_p`. A *VJP rule* for `f` is a function that, given `(x₁…x_k, y₁…y_p, w₁…w_p)` and a builder, emits a subgraph computing `g₁…g_k` where
>
> `g_j = Σ_i (∂y_i / ∂x_j)ᵀ · w_i`
>
> and returns `g_j`, or `null` where `∂y/∂x_j` is identically zero or `x_j` is not differentiable.

> **Theorem 28.2 (What makes a rule correct).** **(stated here)** A rule for `f` is correct iff, for every point at which `f` is differentiable and every cotangent `w`, the subgraph it emits evaluates to `wᵀJ_f`.

*Proof sketch.* This is the definition of vjp restated as an obligation on emitted code. It is worth stating because of what it does *not* require: nothing about efficiency, nothing about the shape of the emitted subgraph, and nothing about which of `x` and `y` the rule reads. Any expression equal to `wᵀJ` is a correct rule, and §28.4 shows the same derivative written two ways for two different operations, one reading its input and one reading its output. ∎

Two distinctions carry most of the practical weight.

> **Definition 28.3 (Structural zero).** **(stated here)** A rule returns a *structural zero* — `null` — for an operand when the derivative with respect to it does not exist or is identically zero for every input. This differs from a *numeric zero*, a tensor of zeros returned as a value.

The difference is not cosmetic. A structural zero tells the driver to propagate nothing, so the entire subgraph that produced that operand becomes unreachable from the gradient and is never differentiated. A numeric zero is a tensor that gets accumulated, multiplied and carried through the whole backward graph, doing nothing at a cost proportional to its size. `clamp` returns `[null, gradX, null]` — no gradient for its bounds — and that `null` is what keeps the bounds' producers out of the backward graph entirely.

> **Definition 28.4 (Linearization dependency).** **(stated here)** A rule *depends on* the forward values it reads. A rule reading only operands can be evaluated from the forward inputs; a rule reading results requires either the forward results or their recomputation.

This is Definition 27.5 specialized to one operation, and it is the fact Chapter 30 spends a whole chapter trading against. `exp`'s rule reads `y` and `log`'s reads `x`, so the two have different memory profiles for the same one-line derivative.

Finally, the obligation that is easy to forget:

> **Note (a gradient has the operand's shape).** `wᵀJ` has, by construction, the shape of the operand. But an operation whose operand was implicitly broadcast has a *result* larger than that operand, so a rule written as `g = w · something` returns something too large. The missing step is a sum over the broadcast axes — and because it is the same step for every such rule, it belongs in the driver, not in each rule.

## 28.4 In mlfw: seven fields and a return array

[`ad/vjp_registry.ts`](../../../src/compiler/ad/vjp_registry.ts) is 71 lines and holds no calculus at all. The rule type ([`vjp_registry.ts:18`](../../../src/compiler/ad/vjp_registry.ts)):

```ts
export type VJPRule = (ctx: VJPContext) => (Value | null)[] | null | undefined;
```

One argument in, an array of gradients-or-`null` out. Definition 28.3 is the `| null` inside the array; the `| null | undefined` on the *outside* is a rule declining to produce anything at all, which the driver treats as "no gradients from this operation".

The context is Definition 28.1's parameter list ([`vjp_registry.ts:8`](../../../src/compiler/ad/vjp_registry.ts)):

```ts
export type VJPContext = {
  builder: IRBuilder;
  op: Operation;
  operands: TensorValue[];
  results: TensorValue[];
  gradOutputs: (TensorValue | null)[];
  attrs: ReadonlyMap<string, AttrValue>;
  full: (value: number, type: TensorType) => TensorValue;
};
```

Six of the seven are the theory. The seventh, `full`, is a convenience that exists because almost every rule needs a constant of the operand's shape and writing it out is three builder calls ([`backward_builder.ts:78`](../../../src/compiler/ad/backward_builder.ts)):

```ts
    const full = (value: number, type: TensorType) => builder.broadcast(builder.scalarConstant(value, type.dtype).getResult(0), type.shape, []).getResult(0) as TensorValue;
```

Registration is a map ([`vjp_registry.ts:24`](../../../src/compiler/ad/vjp_registry.ts)), and the rules are grouped by the same families the operations are ([`vjp_rules/`](../../../src/compiler/ad/vjp_rules/)):

| File | Rules | Contents |
|---|---|---|
| [`unary.ts`](../../../src/compiler/ad/vjp_rules/unary.ts) | 26 | `exp`, `log`, `tanh`, `sigmoid`, `relu`, `gelu`, the special functions |
| [`arithmetic.ts`](../../../src/compiler/ad/vjp_rules/arithmetic.ts) | 13 | `add`, `mul`, `div`, `pow`, `maximum`, `where`, `clamp` |
| [`composite.ts`](../../../src/compiler/ad/vjp_rules/composite.ts) | 13 | `softmax`, `layer_norm`, `batch_norm`, attention |
| [`shape.ts`](../../../src/compiler/ad/vjp_rules/shape.ts) | 10 | `reshape`, `transpose`, `broadcast_in_dim`, `slice`, `pad` |
| [`linalg.ts`](../../../src/compiler/ad/vjp_rules/linalg.ts) | 3 | `dot`, `matmul`, `conv` |
| [`reduction.ts`](../../../src/compiler/ad/vjp_rules/reduction.ts) | 1 | `reduce`, all five reduction kinds |
| [`control.ts`](../../../src/compiler/ad/vjp_rules/control.ts) | 1 | `stop_gradient`, plus eight barrier declarations (Chapter 31) |

*(67 registered operation names, measured 2026-08-21. The per-file counts are of names, not of call sites: `unary.ts` reaches 26 from 23 calls because one of them is a loop over `floor`, `ceil`, `round` and `sign`, and `control.ts` registers `stop_gradient` as a rule and eight further names as barriers — Chapter 31.)*

### The three shapes a rule comes in

**Read nothing.** `add` is the whole rule ([`arithmetic.ts:5`](../../../src/compiler/ad/vjp_rules/arithmetic.ts)):

```ts
registerVJPRule('add', (ctx) => {
  const grad = ctx.gradOutputs[0]!;
  return [grad, grad];
});
```

Addition is linear, so its Jacobian is the identity in both arguments and the incoming gradient passes straight through — to *both* operands, the same value, which is the graph-level statement that `∂(a+b)/∂a = ∂(a+b)/∂b = 1`.

**Read the operands.** `mul` ([`arithmetic.ts:52`](../../../src/compiler/ad/vjp_rules/arithmetic.ts)):

```ts
registerVJPRule('mul', (ctx) => {
  const grad = ctx.gradOutputs[0]!;
  const [lhs, rhs] = ctx.operands;
  const gradLhs = ctx.builder.mul(grad, rhs).getResult(0);
  const gradRhs = ctx.builder.mul(grad, lhs).getResult(0);
  return [gradLhs, gradRhs];
});
```

The product rule, and note the crossing: `lhs`'s gradient needs `rhs`. Both operands must be available in the backward pass.

**Read the result.** `exp` ([`unary.ts:5`](../../../src/compiler/ad/vjp_rules/unary.ts)):

```ts
registerVJPRule('exp', (ctx) => {
  const grad = ctx.gradOutputs[0]!;
  const result = ctx.results[0];
  return [ctx.builder.mul(grad, result).getResult(0)];
});
```

`d(eˣ)/dx = eˣ = y`, so the rule reads `results[0]` rather than recomputing `exp(x)`. Compare `log` ([`unary.ts:11`](../../../src/compiler/ad/vjp_rules/unary.ts)), whose `d(log x)/dx = 1/x` reads `operands[0]` instead. Same length, same complexity, **different memory obligations** — Definition 28.4 in two adjacent functions.

### The step no rule performs

`add`'s rule returns the incoming gradient unchanged for both operands. When the operands had different shapes — a bias broadcast across a batch — that gradient is the wrong shape for the smaller one. Fixing it in each rule would mean sixty-seven copies of the same code, so the driver does it once, on every gradient every rule returns ([`backward_builder.ts:86`](../../../src/compiler/ad/backward_builder.ts)):

```ts
      accumulator.accumulate(operandVal.id, reduceGradToOperandShape(builder, gradInputs[o] as Value, (operandVal.type as TensorType).shape));
```

and `reduceGradToOperandShape` ([`backward_builder.ts:131`](../../../src/compiler/ad/backward_builder.ts)) is the sum:

```ts
export function reduceGradToOperandShape(builder: IRBuilder, grad: Value, targetShape: Shape): Value {
  const gradShape = (grad.type as TensorType).shape;
  if (gradShape.length === targetShape.length && gradShape.every((d, i) => d === targetShape[i])) {
    return grad;
  }
  const nExtra = gradShape.length - targetShape.length;
```

Leading axes the operand does not have are summed away; axes where the operand has extent 1 and the gradient does not are summed and kept. It is the transpose of broadcasting, which is exactly what it should be: **the adjoint of "copy along an axis" is "sum along that axis"**, and §28.6 watches it happen.

## 28.5 Lab 1 — A rule is a subgraph

```bash
node docs/part5/ch28-writing-a-vjp-rule/labs/01-a-rule-is-a-subgraph.mjs
```

Six operations, each differentiated on its own with fusion switched off so the emitted subgraph stays legible.

```
=== d/dx  exp ===
  forward  @Object(%0: tensor<1x2xf32>) -> (tensor<xf32>) {
  backward @backward_Object(%0: tensor<xf32>, %1: tensor<1x2xf32>, %2: tensor<xf32>) -> (tensor<1x2xf32>) {
    %3 = exp(%1) : tensor<1x2xf32>
    %4 = reshape(%0) {new_shape = [1, 1]} : tensor<1x1xf32>
    %5 = broadcast_in_dim(%4) {broadcast_dimensions = [0, 1], result_shape = [1, 2]} : tensor<1x2xf32>
    %6 = mul(%5, %3) : tensor<1x2xf32>
    return(%6)
  grad at x = [0.5, 2.0]: [1.6487212181091309,7.389056205749512]

=== d/dx  log ===
  backward @backward_Object(%0: tensor<xf32>, %1: tensor<1x2xf32>, %2: tensor<xf32>) -> (tensor<1x2xf32>) {
    %3 = reshape(%0) {new_shape = [1, 1]} : tensor<1x1xf32>
    %4 = broadcast_in_dim(%3) {broadcast_dimensions = [0, 1], result_shape = [1, 2]} : tensor<1x2xf32>
    %5 = div(%4, %1) : tensor<1x2xf32>
    return(%5)
  grad at x = [0.5, 2.0]: [2,0.5]
```

Read `exp` first. The rule asked for `ctx.results[0]` — and the backward function *does not receive it*. Instead the first line of the body is `%3 = exp(%1)`: the value was **recomputed from the saved input** rather than saved. That is the remat policy of Chapter 30 making a decision on the reader's behalf, and it is why `exp` and `log` end up with the same three-argument signature despite reading different things.

> **Which means `ctx` is not a request, and reading it is not what decides the saved set.** The natural reading of the three rule shapes above is that a rule *declares its dependencies* — `add` touches neither list so nothing is saved, `mul` touches `operands` so the operands are saved, `exp` touches `results` so the result is saved. That is a good way to *write* a rule and it is not how the builder works. Every operand and every result is resolved before the rule is called at all ([`backward_builder.ts:73`](../../../src/compiler/ad/backward_builder.ts)):
>
> ```ts
>     const operandValues = new Array<Value>(op.numOperands);
>     for (let o = 0; o < op.numOperands; o++) operandValues[o] = resolveValue(op.getOperand(o));
>     const resultValues = new Array<Value>(op.numResults);
>     for (let r = 0; r < op.numResults; r++) resultValues[r] = resolveValue(op.getResult(r));
> ```
>
> `ctx.operands` and `ctx.results` are fully-populated arrays handed to the rule, not lazy accessors that record which entries were touched. A rule that ignores both, like `add`, still caused every one of its operands and results to be resolved — which for a saved value means it entered the backward function's signature, and for a rematerialized one means its producing subgraph was rebuilt in the backward graph.
>
> So the saved set is decided in **two** places, neither of which is the rule:
>
> 1. **The resolver's policy**, `_materialize` ([`:248`](../../../src/compiler/ad/backward_builder.ts)), which for each value either returns a saved argument or reconstructs the operation that produced it. Chapter 30 is about that policy.
> 2. **Dead-code elimination afterwards**, which removes the reconstructed subgraphs nothing consumed. This is why the emitted backward graph looks tidy despite the eager resolution — and why Chapter 31 §31.6 measures the simplification passes deleting a third of what the rules emit.
>
> The consequence worth carrying: a rule cannot make a value cheaper by not reading it, and there is no per-rule mechanism for fine-grained saving. If you are chasing training memory, the levers are in Chapter 30, not here. A lazier `ctx` — getters that record access and let the builder resolve only what was touched — is the obvious improvement and is not implemented.
>
> **So keep the three shapes and drop the mechanism you were about to infer from them.** "Reads nothing / reads the operands / reads the result" remains the right way to *classify* a derivative, because it is a fact about the mathematics: `d(a+b)/da` needs neither argument, `d(ab)/da` needs the other one, `d(eˣ)/dx` happens to equal the output. It predicts which values are *load-bearing* in the backward pass, and therefore what a better builder could exploit. What it does not predict is what this builder saves, because this builder resolves everything and lets DCE sort it out. Definition 28.4 is the mathematical fact; §30.4's policy is the mechanism; they are related by intent and not by code.

`log`'s body is three operations to `exp`'s four, and the missing one is exactly that recomputation. Two of the three in each are the same: `reshape` and `broadcast_in_dim` turning the scalar loss gradient into something the elementwise rule can multiply — which is the `sum` reduction's own VJP rule, fired one step earlier in the sweep.

`tanh` shows a rule that builds a small expression rather than one operation:

```
=== d/dx  tanh ===
    %3 = tanh(%1) : tensor<1x2xf32>
    ...
    %6 = mul(%3, %3) : tensor<1x2xf32>
    %7 = constant() {tensor_type = tensor<1x2xf32>, value = 1} : tensor<1x2xf32>
    %8 = sub(%7, %6) : tensor<1x2xf32>
    %9 = mul(%5, %8) : tensor<1x2xf32>
```

`1 − y²`, written with the `full(1, …)` helper, then multiplied by the incoming gradient. Four operations for one derivative, all of them elementwise and all of them fusible — which is why the chapter can afford to switch fusion off for legibility and the real pipeline can afford the rule.

Two more are worth reading carefully.

```
=== d/dx  relu  (traces to maximum) ===
  forward  @Object(%0: tensor<1x2xf32>) -> (tensor<xf32>, tensor<1x2xf32>) {
  backward @backward_Object(%0: tensor<xf32>, %1: tensor<1x2xf32>, %2: tensor<1x2xf32>, %3: tensor<xf32>) -> (tensor<1x2xf32>) {
    ...
    %7 = compare(%2, %1) {direction = "ge"} : tensor<1x2xbool>
    %8 = select(%7, %5, %6) : tensor<1x2xf32>
```

**The rule that fired is not the one named `relu`.** `x.relu()` traces to `maximum(x, zeros)` (Part II), so the differentiator reached `maximum`'s rule ([`arithmetic.ts:26`](../../../src/compiler/ad/vjp_rules/arithmetic.ts)) — a comparison and a `select`, routing the gradient to whichever operand won. `relu` has a rule of its own ([`unary.ts:41`](../../../src/compiler/ad/vjp_rules/unary.ts)) and it never fires on this path. This is Chapter 21's lesson arriving from the other side: **a rule is keyed to the operation in the graph, not to the method the user called**, and the set of rules that ever run is decided by what tracing and decomposition leave behind.

Note also the forward signature: `-> (tensor<xf32>, tensor<1x2xf32>)`. The forward function grew a second output. That is the zero tensor the `maximum` needed, promoted to a forward result so the backward can be handed it — Chapter 29's plumbing, visible here as a shape.

```
=== d/dx  mul(x, x) ===
    %5 = mul(%4, %1) : tensor<1x2xf32>
    %6 = add(%5, %5) : tensor<1x2xf32>
  grad at x = [0.5, 2.0]: [1,4]
```

`mul`'s rule returned two gradients, one per operand, and both operands are the same value. The `add(%5, %5)` is not in any rule — it is the driver summing two contributions to one value, which is Chapter 29. And `d(x²)/dx = 2x` gives `[1, 4]` at `[0.5, 2]`, which is the arithmetic checking out.

**Try this.** Add `['sqrt', (a) => a.sqrt().sum()]` to the case list and predict whether its rule reads the operand or the result before you look. Then add `['pow', (a) => a.pow(tensor(3.0)).sum()]` and count the operations.

## 28.6 Lab 2 — The shape fix-up

```bash
node docs/part5/ch28-writing-a-vjp-rule/labs/02-the-shape-fix-up.mjs
```

`x + b` where `x` is `[2, 3]` and `b` is `[3]`. The `add` rule returns the incoming gradient for both operands, and the incoming gradient is `[2, 3]`.

```
=== backward ===
  func @backward_Object(%0: tensor<xf32>, %1: tensor<2x3xf32>, %2: tensor<3xf32>, %3: tensor<2x3xf32>, %4: tensor<xf32>) -> (tensor<2x3xf32>, tensor<3xf32>) {
  %5 = constant() {tensor_type = tensor<xf32>, value = 0} : tensor<xf32>
  %6 = reshape(%0) {new_shape = [1, 1]} : tensor<1x1xf32>
  %7 = broadcast_in_dim(%6) {broadcast_dimensions = [0, 1], result_shape = [2, 3]} : tensor<2x3xf32>
  %8 = reduce(%7, %5) {dimensions = [0], reduce_type = "sum"} : tensor<3xf32>
  return(%7, %8)

=== the two gradients ===
  d/dx shape [2,3]  = [[1,1,1],[1,1,1]]
  d/db shape [3]  = [2,2,2]
```

One value, `%7`, returned twice — once as itself and once through `%8 = reduce(%7, dimensions = [0], sum)`. That `reduce` appears in no rule. `add`'s rule is two lines and returns `[grad, grad]`; the sum was inserted by `reduceGradToOperandShape` because the second operand's shape is `[3]` and the gradient's is `[2, 3]`.

And the numbers say what the sum means. Each element of `b` was copied into two rows of the result, so each element of `b` influences the loss twice as much as each element of `x` does — `d/db = 2` against `d/dx = 1`. **The adjoint of a copy is a sum**, and if the driver did not insert it, `d/db` would be the right shape only by accident and the wrong magnitude always.

**Try this.** Change `b` to shape `[1, 3]` and re-read the `reduce`'s `dimensions` attribute. Then make `b` the same shape as `x` and confirm the `reduce` disappears entirely.

## 28.7 Traps and limits

- **Two rules are registered for operations that do not exist.** `matmul` ([`linalg.ts:51`](../../../src/compiler/ad/vjp_rules/linalg.ts)) and `relu` ([`unary.ts:41`](../../../src/compiler/ad/vjp_rules/unary.ts)) name operations the op registry does not contain: `x.matmul(y)` traces to `dot` and `x.relu()` traces to `maximum` (§28.5). Both rules are correct, neither can ever fire, and nothing reports it — registration takes a string and the registry is never consulted. That is 67 registered names against 65 live ones, and it is the same species as Chapter 34 §34.8's `broadcast` lowering rule. The check that would catch it is one line in `registerVJPRule`, and it is not there because the AD registry deliberately does not import the op registry.
- **A rule is trusted completely.** Nothing verifies Theorem 28.2. There is no symbolic check, no automatic finite-difference comparison at registration time, and no assertion that the returned array has one entry per operand — a rule returning too few gradients silently drops the ones it omitted, because [`backward_builder.ts:83`](../../../src/compiler/ad/backward_builder.ts) skips indices past the end of the array. The whole guarantee is the test suite.
- **`pow`'s rule differentiates the exponent, and that needs `log(base)`.** [`arithmetic.ts:76`](../../../src/compiler/ad/vjp_rules/arithmetic.ts) always emits `log(base)` for the exponent gradient, even when the exponent is a constant and the gradient will be discarded. For a negative base that is a NaN computed and thrown away, and by Chapter 20 nothing will remove it, because `log` of a runtime value is not foldable.
- **`reduce` throws for reduction kinds it does not cover** ([`reduction.ts:71`](../../../src/compiler/ad/vjp_rules/reduction.ts)): `sum`, `mean`, `max`, `min` and `prod` are handled and anything else is an error naming the reduce type. That is the right choice — the message says *"would silently drop the gradient"* — and it is the same policy Chapter 31 generalizes.
- **`convert` returns a structural zero for integer inputs** ([`arithmetic.ts:101`](../../../src/compiler/ad/vjp_rules/arithmetic.ts)), which is correct and is also the mechanism by which an index computation stops being differentiated. It is worth knowing that a `null` here is doing load-bearing work rather than expressing ignorance.
- **`full` allocates.** Every call emits a `scalar_constant` and a `broadcast_in_dim` of the operand's full shape. Constant folding (Chapter 19) then materializes that broadcast into a dense constant, so a `tanh` backward on a large activation puts a tensor of ones into the compiled artifact. §28.5's `%7 = constant() {tensor_type = tensor<1x2xf32>, value = 1}` is that, at `1×2`.
- **Rules may not read the graph outside their operation.** The context has no access to the surrounding function, which is deliberate and is what makes rules composable — but it also means a rule cannot notice that its result will be multiplied by zero, and no peephole runs on the backward graph before it is optimized as an ordinary graph.

## 28.8 Read the tests

- [`tests/compiler/ad/backward-numerical.test.js`](../../../tests/compiler/ad/backward-numerical.test.js) — each rule against central differences, which is the only enforcement Theorem 28.2 has.
- [`tests/compiler/ad/conv-vjp-gradcheck.test.js`](../../../tests/compiler/ad/conv-vjp-gradcheck.test.js) — the hardest rule in the set, including strides, padding, dilation and groups.
- [`tests/compiler/ad/control-norm-vjp.test.js`](../../../tests/compiler/ad/control-norm-vjp.test.js) — the composite rules, where a single rule stands in for a subgraph the decomposition pass could also have produced.

---

**Next:** [Chapter 29 — Building the backward graph](../ch29-building-the-backward-graph/README.md), which takes those 67 rules and the reverse sweep of Chapter 27 and asks what the driver around them has to do: which values need gradients at all, what happens when one value has two consumers, and where the forward values come from.
