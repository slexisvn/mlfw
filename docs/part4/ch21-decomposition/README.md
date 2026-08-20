# Chapter 21 — Decomposition

Every optimization in Chapters 19, 20 and 22 through 24 makes the graph smaller. This one makes it eight times bigger, on purpose, and it runs second in the pipeline — before any of them.

The reason is a claim about optimization in general: **a pass can only exploit structure it can see.** An operation named `softmax` is a wall. Nothing inside it can be fused with a neighbour, no algebraic identity can reach it, no constant can propagate through it. Break it into ten primitives and every one of those becomes possible — at the cost of throwing away the one thing the name was carrying, which is that somebody may have written a very good `softmax` kernel.

## 21.1 The problem: 96 operations, and every pass has to handle all of them

The IR has 96 operations. Chapter 34 will write a lowering rule for each one that reaches TIR; Part V will write a gradient rule for each one that is differentiable; the fusion engine has to classify each one; every backend has to emit code for each one. That is four separate obligations per operation, and it grows as the framework does.

But the 96 are not equally fundamental. `add` is a primitive: there is nothing to say about it except how to add. `gelu` is not — it is a formula, `0.5x(1 + tanh(√(2/π)(x + 0.044715x³)))`, made of operations the IR already has. Writing a lowering rule, a gradient rule, a fusion classification and four backend emitters for `gelu` is writing them for a thing that could have been spelled with parts you already support.

There is a second, sharper cost. A composite operation is opaque to the optimizer *even when its parts are not*. Suppose a network computes `gelu(x) * mask`. If `gelu` is one operation, the multiply is a separate kernel and the intermediate is a full tensor round-trip. If `gelu` is eleven elementwise operations, the multiply is the twelfth and the whole thing is one loop.

## 21.2 Intuition: expand, optimize, re-collapse

Decomposition is the first half of a three-step trade the rest of Part IV completes.

1. **Expand.** Rewrite composite operations into primitives. The graph gets much bigger and the number of *distinct* operation kinds gets much smaller.
2. **Optimize.** Every pass downstream now works on a vocabulary it fully understands. Fusion in particular can group across what used to be an operation boundary.
3. **Re-collapse.** Fusion puts the pieces back into single kernels — not the ones you started with, but ones chosen by a cost model that could see the whole neighbourhood.

Step 3 is what makes step 1 affordable, and it is why this chapter sits where it does. Without a good fusion engine, decomposition is a machine for turning one kernel into ten.

The information you lose in step 1 is real, though, and it is worth naming precisely: after decomposition, **there is no `softmax` in the graph**. A backend holding a hand-tuned fused softmax kernel has nothing to match against. Chapter 58's external-codegen interface exists partly to buy that back, and §21.7 shows the escape hatch that exists for it here.

## 21.3 Theory

> **Definition 21.1 (Decomposition rule).** A *decomposition rule* for operation `f` is a function that, given an instance of `f`, emits a subgraph of other operations computing the same result, and replaces all uses of `f`'s results with the subgraph's outputs.

> **Definition 21.2 (Primitive set, stated here).** Given a set of decomposition rules `R`, the *primitive set* is the set of operations with no rule in `R`. Decomposition to fixed point rewrites any graph into one containing only primitives.

Termination is not automatic — a rule that emits the operation it decomposes, directly or through a cycle, does not terminate. Here it is obtained structurally rather than by a bound: rules are collected into a worklist *before* any of them runs ([`decomposition_pass.ts:43`](../../../src/compiler/passes/decompose/decomposition_pass.ts)), so a single pass expands each original composite exactly once and never revisits what a rule produced. A rule emitting a composite therefore leaves it in place, which is a silent under-decomposition rather than a hang. Chapter 15's fixed-point group does not contain this pass, so nothing re-runs it.

> **Definition 21.3 (Decomposition is not free, stated here).** A decomposition is *neutral* if the compiler can recover, from the emitted subgraph, every decision it could have made from the original operation. It is *lossy* otherwise.

Every decomposition in this compiler is lossy in one specific way: it destroys the name. Whether that matters depends entirely on whether anything downstream wanted the name — a backend with a library kernel, a quantization pass with a per-operation policy, an autotuner with a workload key. §21.7 makes the loss concrete by asking the compiler to keep an operation whole.

One decomposition property is worth stating because it is easy to get wrong:

> **Note (numerical form matters).** A decomposition may not be the textbook formula. `softmax(x)ᵢ = eˣⁱ / Σeˣʲ` overflows for `x` above about 88 in f32. The rule below emits `exp(x − max(x))` instead, which is mathematically equal and numerically defined for every finite input. The decomposition rule is where that choice is made once, for every backend.

## 21.4 In mlfw: a registry of 21 rules

[`passes/decompose/decomposition_pass.ts`](../../../src/compiler/passes/decompose/decomposition_pass.ts) is a map from operation name to a rewriting function ([`decomposition_pass.ts:15`](../../../src/compiler/passes/decompose/decomposition_pass.ts)):

```ts
const decompositionRules = new Map<string, DecompositionRule>();

export function registerDecomposition(opName: string, ruleFn: DecompositionRule): void {
  decompositionRules.set(opName, ruleFn);
}
```

Twenty-one rules are registered in the same file, and they fall into four families:

| Family | Operations |
|---|---|
| Activations | `sigmoid`, `gelu`, `silu`, `elu`, `leaky_relu`, `celu`, `selu`, `mish`, `hardswish`, `hardsigmoid` |
| Normalization | `layer_norm`, `batch_norm` |
| Softmax family | `softmax`, `log_softmax` |
| Structural | `stop_gradient`, `where`, `split`, `one_hot`, `embedding`, `all_reduce`, `all_gather` |

The shortest is one line of intent ([`decomposition_pass.ts:76`](../../../src/compiler/passes/decompose/decomposition_pass.ts)):

```ts
registerDecomposition('stop_gradient', (op) => {
  op.replaceAllResultsWith([op.getOperand(0)]);
  op.erase();
});
```

`stop_gradient` is a marker for automatic differentiation (Part V). By the time the graph is being optimized, differentiation has happened, and the marker is the identity function. Decomposing it to nothing is how a construct that exists only for one phase stops costing anything in the phases after it.

The most instructive is `softmax` ([`decomposition_pass.ts:124`](../../../src/compiler/passes/decompose/decomposition_pass.ts)):

```ts
registerDecomposition('softmax', (op, b) => {
  const input = op.getOperand(0);
  const axis = op.getAttr<number>('axis') as number;
  const rank = (input.type as TensorType).rank;
  const dtype = (input.type as TensorType).dtype;
  const shape = (input.type as TensorType).shape;
  const bcastDims = broadcastDimsExcluding(rank, axis);

  const maxVal = b.reduce(input, b.scalarConstant(-Infinity, dtype).getResult(0), [axis], 'max');
  const bcastMax = b.broadcast(maxVal.getResult(0), shape, bcastDims);
  const shifted = b.sub(input, bcastMax.getResult(0));
  const exps = b.exp(shifted.getResult(0));
  const sumVal = b.reduce(exps.getResult(0), b.scalarConstant(0, dtype).getResult(0), [axis], 'sum');
  const bcastSum = b.broadcast(sumVal.getResult(0), shape, bcastDims);
  const result = b.div(exps.getResult(0), bcastSum.getResult(0));

  op.replaceAllResultsWith([result.getResult(0)]);
  op.erase();
});
```

Seven operations, and the `sub` of the broadcast max is the numerical-stability shift the note in §21.3 described. Every backend gets it, for free, because it was decided here.

Note also the shape of the API: the rule receives an `IRBuilder` positioned at the operation and emits into it, then rewires and erases. There is no separate "replacement subgraph" data structure — the rule writes IR directly, which is why a rule is fifteen lines rather than fifty.

### The escape hatch

`_shouldDecompose` gives a target a veto ([`decomposition_pass.ts:37`](../../../src/compiler/passes/decompose/decomposition_pass.ts)):

```ts
  _shouldDecompose(op: Operation): boolean {
    if (!this.target) return true;
    const native = this.target.getAttr ? this.target.getAttr<ReadonlySet<string>>('nativeOps') : null;
    return !(native && native.has(op.opName));
  }
```

A target that implements `softmax` natively — a GPU with a library kernel — declares it and keeps the operation whole. **No target in this repository sets `nativeOps`.** The hatch is designed, reachable, and unused, and §21.7 is what happens when you use it.

## 21.5 Lab 1 — One operation becomes ten, and then one kernel

```bash
node docs/part4/ch21-decomposition/labs/01-one-op-becomes-seven.mjs
```

The user writes one operation:

```
=== traced: what the user wrote ===
module @traced {
  func @traced(%0: tensor<2x3xf32>) -> (tensor<2x3xf32>) {
    %1 = softmax(%0) {axis = 1} : tensor<2x3xf32>
    return(%1)
  }
}
```

The pass reports what it did, and the graph grows fivefold:

```
=== decomposition report ===
  {"softmax":1}  total 1
  DecompositionPass: 2 -> 10 ops
```

And here is what the *rest of Part IV* does with those ten:

```
module @Softmax {
  func @Softmax(%0: tensor<2x3xf32>) -> (tensor<2x3xf32>) {
    %1 = constant() {tensor_type = tensor<xf32>, value = -inf} : tensor<xf32>
    %2 = reduce(%0, %1) {dimensions = [1], reduce_type = "max"} : tensor<2xf32>
    %5 = constant() {tensor_type = tensor<xf32>, value = 0} : tensor<xf32>
    %6 = fusion(%2, %0, %5) {fusion_kind = "kReduction"} : tensor<2x3xf32>
    {
      ^bb(%7: tensor<2xf32>, %8: tensor<2x3xf32>, %9: tensor<xf32>):
      %10 = broadcast_in_dim(%7) {broadcast_dimensions = [0], result_shape = [2, 3]} : tensor<2x3xf32>
      %11 = sub(%8, %10) : tensor<2x3xf32>
      %12 = exp(%11) : tensor<2x3xf32>
      %13 = reduce(%12, %9) {dimensions = [1], reduce_type = "sum"} : tensor<2xf32>
      %16 = broadcast_in_dim(%13) {broadcast_dimensions = [0], result_shape = [2, 3]} : tensor<2x3xf32>
      %17 = div(%12, %16) : tensor<2x3xf32>
      yield(%17)
    }
    return(%6)
  }
}
kernels emitted: 1
```

Six of the seven emitted operations are inside a single `kReduction` fusion region — including the *second* reduction, the sum, which is fused together with the broadcast, the subtract, the exponential and the divide. Only the max reduction stayed outside. One kernel comes out, and it computes a numerically stable softmax.

That is the three-step trade completed inside one compile: expand to ten, optimize, re-collapse to one. The user never sees any of it, and the reason the middle step was possible is that the composite was gone.

**Try this.** Compile `a.softmax(1).mul(a)` and look at where the multiply ends up. Then try `a.softmax(1).softmax(1)` and count the reductions.

## 21.6 Lab 2 — The catalogue

```bash
node docs/part4/ch21-decomposition/labs/02-the-catalogue.mjs
```

Eight operations through the whole pipeline, counting at three points:

```
op            traced  after decomposition  after all passes  kernels
sigmoid            2                    7                 3        1
silu               2                    8                 3        1
gelu               2                   11                 4        1
elu                6                    6                 2        1
softmax            2                   10                 5        1
log_softmax        2                   11                 5        1
layer_norm         2                   18                 5        1
tanh               2                    2                 2        1
```

The column that matters is the last one: **every case ends as a single kernel.** A `layer_norm` that expands ninefold, a `gelu` that expands fivefold, all of them come back to one. Whatever decomposition costs in graph size, it is not costing kernels.

Two rows are the control group. `tanh` is a primitive — no rule, no expansion, and it was already one kernel. And `elu` traced to six operations rather than one, because `nn.ELU` builds itself out of primitives in the *module*, before tracing ever reaches an `elu` node; its decomposition rule exists and never fires on this path. Two ways of spelling the same policy, one in the framework and one in the compiler, and the graph cannot tell them apart afterwards.

Note also that "after all passes" is well above "traced" for every decomposed case — 5 operations for a softmax that arrived as 1. The graph did not return to its original size, and it did not need to: what matters is how many *kernels* it becomes, and the operations left at graph level are fusion regions and the reductions that could not join them.

## 21.7 Lab 3 — Keeping an operation whole

```bash
node docs/part4/ch21-decomposition/labs/03-keeping-an-op-whole.mjs
```

Definition 21.3 says decomposition loses the name. Here is what it costs to keep it. The lab builds a target that claims `softmax` is native:

```js
function targetClaiming(...nativeOps) {
  const target = CPUTarget();
  const inherited = target.getAttr.bind(target);
  target.getAttr = (key) => key === 'nativeOps' ? new Set(nativeOps) : inherited(key);
  return target;
}
```

```
=== default target ===
  decomposed: {"softmax":1}
  kernels: 1
=== target claims softmax is native ===
  decomposed: null
  compile failed: [lowering] Softmax: No lowering rule defined for op: softmax
```

The trade-off in one error message. Keeping the operation whole means the pipeline reaches Chapter 34 holding a `softmax` and no rule for it — because a compiler that decomposes an operation has, by that fact, stopped needing to know how to lower it.

So the two ends are: **decompose and you owe nothing, but you also get nothing back**; or **keep it whole and you owe a lowering rule, a fusion classification, a gradient rule, and a backend emitter — one per target that claims it.** The `nativeOps` hatch is the switch between them, and it is unused here because no target in this repository has a hand-written softmax kernel to justify the other four obligations.

That is also the honest answer to "should a compiler decompose?" It should decompose everything it does not have a better kernel for, and the set of operations it has a better kernel for is a property of the *backend*, not of the IR. Which is why the hatch is a target attribute.

## 21.8 Traps and limits

- **`nativeOps` has no writers.** [`decomposition_pass.ts:39`](../../../src/compiler/passes/decompose/decomposition_pass.ts) is the only reader of the attribute in `src/`, and no target sets it. The mechanism is real — §21.7 exercises it — and currently decorative.
- **Decomposition runs once, not to fixed point.** The worklist is built before any rule runs ([`decomposition_pass.ts:43`](../../../src/compiler/passes/decompose/decomposition_pass.ts)), so a rule that emits another composite leaves that composite in the graph. None of the 21 rules does today; nothing checks that a new one will not.
- **The pass always returns CHANGED when the worklist is non-empty** ([`decomposition_pass.ts:72`](../../../src/compiler/passes/decompose/decomposition_pass.ts)), without checking whether any rule actually altered anything. Harmless — it costs one extra invalidation of the analysis cache (Chapter 16) — and it is the "cheap to over-report" direction of Definition 14.1.
- **A decomposition can defeat a later pass silently.** Quantization (Chapter 26) works on named operations; a `layer_norm` decomposed into eighteen primitives is eighteen chances for the quantizer to make a different decision than it would have made about the whole. Ordering here is a policy that is not written down anywhere except in the order of `buildGraphPipeline`.
- **Two of the rules are not really decompositions.** `all_reduce` and `all_gather` rewrite collective operations into local ones — which is correct for a single device and wrong for the distributed case the operations exist for. The rule is a single-device *lowering* wearing a decomposition's clothes.
- **The numerical form is a decision with no test naming it as one.** Nothing asserts that `softmax`'s decomposition subtracts the max. Someone simplifying the rule to the textbook formula would break f32 overflow behaviour, and the failure would show up as a NaN in a large model rather than as a red test.

## 21.9 Read the tests

- [`tests/compiler/passes/decompose/`](../../../tests/compiler/passes/decompose/) — the rules individually, and the registry's register/unregister behaviour that makes a rule testable in isolation.
- [`tests/e2e/`](../../../tests/e2e/) — the decomposed operations compared against eager execution, which is what pins the *numerics* of each rule rather than its shape.

---

**Next:** [Chapter 22 — Fusion I: why it is the single most valuable optimization](../ch22-fusion-why/README.md), which is the pass that makes this chapter affordable, and the one with the largest measured effect in the book.
