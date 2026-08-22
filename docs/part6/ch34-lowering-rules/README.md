# Chapter 34 — Lowering rules

Chapter 32 left one line unexplained:

```ts
    const rule = getLoweringRule(op.opName, target, context);
    if (!rule) throw new Error(`No lowering rule defined for op: ${op.opName}`);
```

This chapter is what is on the other side of that lookup: 68 rules covering 96 registered operations, a selection mechanism with two override points, and a handful of shared builders that mean those 68 rules are much less than 68 independent programs.

## 34.1 The problem: 96 operations, one loop language

Every operation has to become loops. Doing that once per operation per backend is 384 pieces of code, most of them the same loop nest with a different scalar in the middle. Doing it once per operation, into an IR every backend reads, is 96 — and the real number is smaller still, because most operations are the same shape of loop.

But the operations are not uniform. Compare four of them:

- `add` writes one output element per iteration of a nest over the output shape.
- `reduce` writes one output element per iteration of a nest over *part* of the input shape, and reads a whole axis for each.
- `transpose` writes one output element per iteration but reads a permuted subscript.
- `conv` has seven nested loops, a guard for padding, and a channel-group division in one subscript.

Any design that makes `add` cheap to write must not make `conv` impossible to write. That is the tension the rule interface resolves.

## 34.2 Intuition: a rule is a small program-writer

A lowering rule is not a translation table. It is a function that *constructs IR*: given the operation and the buffers, it returns a statement. Two consequences follow immediately.

**A rule can look at attributes.** `slice` reads `starts` and `strides` and builds a different index expression for each. `conv` reads `padding`, `dilation` and `groups` and emits a guard only for the axes that were padded. There is no table that could hold that; there is a function that computes it.

**A rule can be parameterised by the target.** The same operation can have more than one rule, and the one used depends on which backend is being compiled for. That is how a CUDA-specific tiled attention kernel coexists with a portable naive one.

What a rule cannot do is look at the rest of the program (Definition 32.6). It sees one operation, its input buffers and its output buffers. Everything context-dependent — fusion, the broadcast alias, the return-value copy — is the driver's job.

## 34.3 Theory

> **Definition 34.1 (Op strategy).** An *op strategy* for an operation name is a set of *implementations*, each with a name, a compute function, a priority level, and an optional target kind.

> **Definition 34.2 (Implementation selection).** Given a target of kind `k`, the *candidate set* for an operation is the implementations whose target kind is `null` or `k`. The *selected* implementation is a candidate of maximum priority.

Two things are worth noticing about Definition 34.2. Selection is by priority alone, so an implementation is chosen without the compiler ever looking at the operation's shapes — a shape-dependent choice has to be made *inside* the rule, which is what the CUDA attention rule does. And ties are broken by registration order, since `best()` keeps the first maximum it sees.

> **Definition 34.3 (Coverage).** A pipeline is *covered* if every operation that can appear in a graph function at the start of the lowering phase has a selected implementation for the target being compiled.

Coverage is not "every registered operation has a rule". It is a joint property of the registry and the graph pipeline: an operation with no rule is fine as long as some earlier pass is guaranteed to have removed it.

> **Proposition 34.4 (Coverage is a joint property, stated here).** Let `R` be the set of operations with a selected implementation, and `E` the set of operations that the graph pipeline can leave in a module at the moment lowering begins. Lowering cannot fail for a missing rule if and only if `E ⊆ R`.

*Argument.* The driver reaches its rule lookup for exactly the operations in `E`, minus terminators and constants, which it handles itself; it throws exactly when that lookup returns nothing. So a failure is possible precisely when some member of `E` is outside `R`. ∎

The statement is nearly a tautology and is worth writing down anyway, because of which two things it puts on the same line. `R` is a property of the *registry* — 68 names, all in `passes/lowering/`. `E` is a property of the *pipeline* — which passes are enabled, in what order, for this target and this configuration. Neither file knows about the other, and the correctness condition mentions both.

The consequence is the practical one: **turning off a pass can make lowering fail.** Disable decomposition and `softmax` reaches a phase with no rule for it. Nothing in the decomposition pass says "lowering depends on me", and nothing in the lowering registry says "I assume decomposition ran". That coupling is the reason §34.6's table has two halves.

> **Definition 34.5 (Rule skeleton, stated here).** A *skeleton* is a shared constructor that builds the nest, the block and the store for a family of operations, taking a callback for the part that differs. A family with a skeleton has one nest-building implementation and `n` leaf callbacks rather than `n` nest builders.

## 34.4 In mlfw: registration, selection, and two overrides

### Registration

[`op_strategy.ts`](../../../src/compiler/passes/lowering/op_strategy.ts), 93 lines, is Definitions 34.1 and 34.2 verbatim. `registerLoweringRule` and `registerTargetLoweringRule` are thin wrappers that pick the priority ([`lowering_registry.ts:73`](../../../src/compiler/passes/lowering/lowering_registry.ts)):

```ts
export function registerLoweringRule(opName: string, ruleFunc: LoweringRuleFn, plevel = GENERIC_PLEVEL): void {
  registerOpStrategy(opName, { name: `${opName}.generic`, compute: ruleFunc, plevel, targetKind: null });
}

export function registerTargetLoweringRule(opName: string, targetKind: string, ruleFunc: LoweringRuleFn, plevel = TARGET_PLEVEL): void {
  registerOpStrategy(opName, { name: `${opName}.${targetKind}`, compute: ruleFunc, plevel, targetKind });
}
```

with `GENERIC_PLEVEL = 10` and `TARGET_PLEVEL = 20` ([`lowering_registry.ts:61`](../../../src/compiler/passes/lowering/lowering_registry.ts)). A target-specific rule outranks a generic one by construction, and the two-level scheme is the whole priority policy.

Registration happens once, at module load, from eleven `register()` functions called at the top of [`graph_to_tensor.ts:57`](../../../src/compiler/passes/lowering/graph_to_tensor.ts). Importing the driver registers the rules; there is no explicit initialisation step and no way to compile without them.

### Selection

`getOpStrategy` builds a filtered view per target kind and caches it ([`op_strategy.ts:69`](../../../src/compiler/passes/lowering/op_strategy.ts)):

```ts
  const view = new OpStrategy(opName);
  for (const impl of full.implementations) {
    if (impl.targetKind === null || impl.targetKind === kind) {
      view.implementations.push(impl);
    }
  }
```

and `best()` is a linear scan for the maximum priority ([`op_strategy.ts:40`](../../../src/compiler/passes/lowering/op_strategy.ts)). The cache is cleared on every registration, which matters only for tests that register and unregister rules.

### The two overrides

`getLoweringRule` consults the compiler context before the registry ([`lowering_registry.ts:81`](../../../src/compiler/passes/lowering/lowering_registry.ts)):

```ts
export function getLoweringRule(opName: string, target?: OpStrategyTarget, context: CompilerContext | null = null): LoweringRuleFn | undefined {
  if (context) {
    const override = context.getLoweringRule(opName);
    if (override) return override;
  }
  const impl = selectImplementation(opName, target);
  return impl ? impl.compute as LoweringRuleFn : undefined;
}
```

So there are two ways to change how an operation lowers without touching the registry: register a target-specific implementation, or pass `loweringRules` in the compiler configuration. The second is the same shape as Chapter 14's `PassContext` — a hook that lets a caller reach into a phase from outside — and it is what a hardware vendor would use to route one operation to a library call.

### The skeletons

Five shared constructors carry most of those rules.

| Skeleton | Line | Used by |
|---|---|---|
| `makeLoopNest` + `wrapInLoops` | [`lowering_registry.ts:222`](../../../src/compiler/passes/lowering/lowering_registry.ts) | every rule with a nest over the output shape |
| `lowerPointwise` | [`lowering_registry.ts:445`](../../../src/compiler/passes/lowering/lowering_registry.ts) | all 38 elementwise-shaped rules |
| `buildSpatialNest` | [`lowering_registry.ts:384`](../../../src/compiler/passes/lowering/lowering_registry.ts) | reductions, contractions, pooling |
| `buildDotGeometry` + `emitMatmulInitAcc` | [`lowering_registry.ts:529`](../../../src/compiler/passes/lowering/lowering_registry.ts) | `dot`, `fused_dot_epilogue`, `quantized_dot` |
| `buildConvNest` | [`lowering_registry.ts:280`](../../../src/compiler/passes/lowering/lowering_registry.ts) | `conv`, `quantized_conv` |

`lowerPointwise` is Definition 34.5 at its purest ([`lowering_registry.ts:445`](../../../src/compiler/passes/lowering/lowering_registry.ts)):

```ts
export function lowerPointwise(ctx: LoweringContext, op: Operation, inputs: readonly Buffer[], outputs: readonly Buffer[], exprBuilder: PointwiseExprBuilder): TirNode {
  const outBuf = outputs[0];
  const { loopVars, loopBinds, indices, extentNodes } = makeLoopNest(ctx, outBuf.shape, outBuf);
  const loads: BufferLoadNode[] = new Array(inputs.length);
  for (let i = 0; i < inputs.length; i++) {
    const inIndices = computeBroadcastIndices(inputs[i], outBuf, indices);
    loads[i] = new BufferLoadNode(inputs[i], inIndices);
  }
  const expr = exprBuilder(op, loads, outBuf.dtype);
  const store = new BufferStoreNode(outBuf, indices, expr);
  const block = new BlockNode(ctx.blockName(`${op.opName}_block`), loopBinds, bufRefs(inputs), [{ buffer: outBuf }], store);
  return wrapInLoops(block, loopVars, outBuf.shape, extentNodes);
}
```

Eleven lines, and every elementwise operation in the compiler is that function plus one callback. Note `computeBroadcastIndices` ([`lowering_registry.ts:402`](../../../src/compiler/passes/lowering/lowering_registry.ts)): broadcasting is handled here, in the skeleton, for every operation at once, by reading a size-1 axis with a literal `0`.

The callbacks themselves come from a table ([`rules/elementwise.ts:22`](../../../src/compiler/passes/lowering/rules/elementwise.ts)):

```ts
const ELEMENTWISE_SCALAR_OPS: Record<string, string> = {
  'add': '+', 'sub': '-', 'mul': '*', 'div': '/',
  'max': 'max', 'min': 'min', 'exp': 'exp', 'log': 'log',
  ...
```

33 entries, written onto the op registry as an attribute ([`rules/elementwise.ts:35`](../../../src/compiler/passes/lowering/rules/elementwise.ts)) so the fusion path can read the same table. `buildElementwiseExpr` then decides between an infix `MathOpNode` and a `CallExternNode` by looking up the operator string in three small sets ([`rules/elementwise.ts:52`](../../../src/compiler/passes/lowering/rules/elementwise.ts)).

### The fusion path

`fusion` is not in the strategy registry at all; the driver handles it before the lookup (§32.4). It has two modes.

`canLowerAsElementwiseFusion` ([`rules/fusion.ts:219`](../../../src/compiler/passes/lowering/rules/fusion.ts)) asks whether every operation inside the region has an *inline builder* — a function producing an expression rather than a statement — and whether all results share a shape. If so, `lowerFusion` emits **one** nest whose leaf is the whole region's expression tree. If not, `lowerFusionAsIndividualOps` walks the region and lowers each inner operation with its ordinary rule, materialising every intermediate.

41 registered operations have an inline builder ([`rules/fusion.ts:32`](../../../src/compiler/passes/lowering/rules/fusion.ts)): the 33 elementwise ones, plus `compare`, `select`, `clamp`, `convert`, `broadcast_in_dim`, `iota`, `quantize` and `dequantize`. The bound on what Chapter 24 can profitably fuse is this list, not the fusion pass's own rules.

`iota`'s entry is a placeholder that throws if it is ever called ([`rules/fusion.ts:58`](../../../src/compiler/passes/lowering/rules/fusion.ts)) — it exists so that `canLowerAsElementwiseFusion` says yes, and `lowerFusion`'s index-aware path substitutes the loop variable directly instead of building an expression from operands. An `iota` inside a fusion is the one inner operation whose value depends on *where* it is, not on what it read.

## 34.5 Where the rules live

| File | Rules | Operations |
|---|---|---|
| [`rules/elementwise.ts`](../../../src/compiler/passes/lowering/rules/elementwise.ts) | 38 | 33 from the scalar-op table, plus `compare`, `select`, `clamp`, `convert`, `copy_to_device` |
| [`rules/shape.ts`](../../../src/compiler/passes/lowering/rules/shape.ts) | 8 | `broadcast_in_dim`, `transpose`, `reverse`, `reshape`, `slice`, `pad`, `concat`, `iota` |
| [`rules/linalg.ts`](../../../src/compiler/passes/lowering/rules/linalg.ts) | 6 | `dot`, `conv`, `gather`, `scatter`, `fused_dot_epilogue`, `cublas_gemm` |
| [`rules/quantization.ts`](../../../src/compiler/passes/lowering/rules/quantization.ts) | 4 | `quantize`, `dequantize`, `quantized_dot`, `quantized_conv` |
| [`rules/control_flow.ts`](../../../src/compiler/passes/lowering/rules/control_flow.ts) | 3 | `if`, `while`, `scan` |
| [`rules/reduction.ts`](../../../src/compiler/passes/lowering/rules/reduction.ts) | 3 | `reduce`, `argmax`, `argmin` |
| [`rules/pooling.ts`](../../../src/compiler/passes/lowering/rules/pooling.ts) | 1 | `pool2d` |
| [`rules/resize.ts`](../../../src/compiler/passes/lowering/rules/resize.ts) | 1 | `resize` |
| [`rules/layout.ts`](../../../src/compiler/passes/lowering/rules/layout.ts) | 1 | `layout_transform` |
| [`rules/attention.ts`](../../../src/compiler/passes/lowering/rules/attention.ts) | 1 + 1 | `scaled_dot_product_attention`, generic and CUDA |

### The taxonomy, stated once

Several parts of this book quote counts from this table, and they have drifted against each other in the past, so here is the partition and the predicate that decides each class. Every registered operation falls in exactly one:

> **Definition 34.6 (Lowering coverage classes).** **(stated here)** For each operation `n` in the registry:
> - `n` is **ruled** if `hasLoweringRule(n)` — the lowering registry can produce a loop nest for it.
> - `n` is **decomposed** if it is not ruled and `hasDecomposition(n)` — Chapter 21 rewrites it into primitives before lowering sees it.
> - `n` is **structural** otherwise — the driver, the region-lowering path or external codegen handles it, and it never reaches a rule lookup.
>
> Coverage (Proposition 34.4) is the claim that no fourth class exists: no operation is unruled, undecomposed and unhandled.

Both predicates are exported, so the partition is checkable rather than asserted. Measured on 2026-08-21:

| Class | Count | Members |
|---|---:|---|
| **ruled** | 68 | the table above |
| **decomposed** | 21 | `softmax`, `log_softmax`, `sigmoid`, `silu`, `gelu`, `celu`, `elu`, `selu`, `mish`, `leaky_relu`, `hardsigmoid`, `hardswish`, `layer_norm`, `batch_norm`, `one_hot`, `where`, `split`, `embedding`, `stop_gradient`, `all_reduce`, `all_gather` |
| **structural** | 7 | `return`, `yield`, `tuple`, `get_tuple_element`, `call`, `fusion`, `custom_call` |

**68 + 21 + 7 = 96, with nothing left over.** That is Proposition 34.4 checked by counting rather than by trusting: the set of operations with no rule and no removal path is empty.

> **These numbers move, and the sum does not.** An operation migrating between classes — `constant` and `scalar_constant` gaining lowering rules, say — changes *two* of the three counts in step while the total stays pinned by the registry. That is the useful property of a partition: the arithmetic closing is a permanent check, while any individual class count is a fact about the implementation on a date. Quote a class count with its date, or derive it from the predicates.

## 34.6 Lab — the catalogue

```bash
node docs/part6/ch34-lowering-rules/labs/01-the-catalogue.mjs
```

The left column is what the user wrote; the middle is what actually reached the lowering phase; the right is what came out.

```
  user code                ops reaching lowering            blocks emitted                 loops

  x.add(y)                 add                              add_block                      2
  x.exp()                  exp                              exp_block                      2
  ops.where(x>y, x, y)     compare select                   compare_block select_block     4
  ops.clamp(x, 0, 1)       clamp                            clamp_block                    2
  x.sum(1)                 reduce                           reduce_init reduce_acc         3
  x.mean(1)                reduce                           reduce_init reduce_acc         4
                                                            mean_div
  x.argmax(1)              argmax                           arg_init arg_acc               2
  x.matmul(y)              dot                              matmul_init matmul             5
  x.transpose(1,0)         transpose                        transpose_block                2
  x.reshape([3,8])         reshape                          reshape_block                  2
  x.flip(0)                gather                           gather_block                   2
  ops.pad(x,[1,0],[1,0])   pad                              pad_block                      2
  ops.cat([x, x], 0)       concat                           concat concat                  4
  ops.pool2d max 2x2       reshape pool2d                   reshape_block pool_init        14
                                                            pool_acc
  ops.one_hot(argmax, 6)   argmax iota broadcast_in_dim     arg_init arg_acc iota_block    10
                           convert compare select           convert_block compare_block
                                                            select_block
  ops.conv2d 1x1x4x6       reshape conv                     reshape_block conv_init        15
                                                            conv_acc
```

Four patterns, and every rule in the compiler is one of them.

**One block, one nest over the output.** Everything elementwise, plus `transpose`, `reshape`, `pad`, `gather`. The whole difference between them is the subscript on the read.

**Two blocks: initialise, then accumulate.** `reduce`, `argmax`, `dot`, `conv`, `pool2d`. The output buffer *is* the accumulator, which is why Chapter 33's declared read sets are wrong for exactly these.

**Three blocks, when the operator is not the whole reduction.** `mean` adds a division pass after the sum ([`rules/reduction.ts:70`](../../../src/compiler/passes/lowering/rules/reduction.ts)). The effect is that the accumulation block stays a pure `CommReduce` over `+` — the form `rfactor` (Chapter 41) requires — at the cost of one more pass over an output that is usually small.

**`n` blocks, one per input.** `concat` emits one nest per operand, writing into a different offset of the same output. Two operands, two `concat` blocks, four loops.

The `x.flip(0)` row is a small surprise: reversing an axis arrives at lowering as `gather`, not as `reverse`. `reverse` has a rule ([`rules/shape.ts:29`](../../../src/compiler/passes/lowering/rules/shape.ts)) and no producer in the tracing layer at all — but it is not dead. The lab's last section shows where it comes from:

```
  function             ops reaching lowering                          blocks emitted

  Object               conv reshape add                               conv_init conv_acc reshape_block add_block
  backward_Object      reduce reshape transpose reverse conv          reduce_init reduce_acc reshape_block transpose_block reverse_block conv_init conv_acc
```

The VJP of a convolution flips the kernel ([`ad/vjp_rules/linalg.ts:114`](../../../src/compiler/ad/vjp_rules/linalg.ts)), and that flip is a `reverse`. So the rule runs on every model that backpropagates through a convolution and on no other program in the book — a reminder that Part V's output is ordinary IR arriving at this phase through the same door.

And the second half of the lab is Proposition 34.4:

```
=== ops that never reach lowering ===

  x.softmax(1)             reduce broadcast_in_dim sub exp  reduce_init reduce_acc         12
                           reduce broadcast_in_dim div      sub_block exp_block
                                                            reduce_init reduce_acc
                                                            div_block
  x.sigmoid()              neg exp add div                  neg_block exp_block add_block  8
                                                            div_block
```

No `softmax` rule exists, and the phase never sees a `softmax`. That is Chapter 21's decomposition holding up the other end of the contract.

## 34.7 Lab — one rule, many operations

```bash
node docs/part6/ch34-lowering-rules/labs/02-one-rule-many-ops.mjs
```

```
  op                     innermost two loops              store

  add  (binary infix)    for i0 in 0..2 / for i1 in 0..3  b2[i0, i1] = (b0[i0, i1] + b1[i0, i1])
  div  (binary infix)    for i0 in 0..2 / for i1 in 0..3  b2[i0, i1] = (b0[i0, i1] / b1[i0, i1])
  neg  (unary infix)     for i0 in 0..2 / for i1 in 0..3  b1[i0, i1] = (b0[i0, i1])
  exp  (extern call)     for i0 in 0..2 / for i1 in 0..3  b1[i0, i1] = exp(b0[i0, i1])
  maximum (extern call)  for i0 in 0..2 / for i1 in 0..3  b2[i0, i1] = max(b0[i0, i1], b1[i0, i1])
  where (select)         for i0 in 0..2 / for i1 in 0..3  b3[i0, i1] = if (b4[i0, i1]) { b1[i0, i1] } else { b2[i0, i1] }
```

Identical nests, identical blocks, identical stores; six different leaves. The nest was written once.

The `neg` row is where the printer lies (§32.8): the operator is missing from the text because `visitMathOpNode` prints it only for binary nodes. The emitted kernel has it:

```
    buf_3[((i0_4 * 3) + i1_5)] = (-buf_1[((i0_4 * 3) + i1_5)]);
```

Broadcasting, in the same skeleton:

```
  [2,3] + [2,3]          for i0 in 0..2 / for i1 in 0..3  b2[i0, i1] = (b0[i0, i1] + b1[i0, i1])
  [2,3] + [1,3]          for i0 in 0..2 / for i1 in 0..3  b2[i0, i1] = (b0[i0, i1] + b1[0, i1])
  [2,3] + [2,1]          for i0 in 0..2 / for i1 in 0..3  b2[i0, i1] = (b0[i0, i1] + b1[i0, 0])
  [2,3] + scalar         for i0 in 0..2 / for i1 in 0..3  b1[i0, i1] = (b0[i0, i1] + b2[])
```

A literal `0` in the broadcast axis, and a rank-0 subscript for the scalar. Chapter 4's cost model has nothing to say about a broadcast because there is nothing there to cost.

**Try this.** Add a row for `x.pow(y)` and one for `x.rem(y)`. Both are in the scalar-op table, both map to extern calls (`pow`, `fmod`), and both take the same eleven-line path — which is the point.

## 34.8 Traps and limits

- **A rule exists for an operation that does not.** `registerLoweringRule('broadcast', ...)` ([`rules/shape.ts:15`](../../../src/compiler/passes/lowering/rules/shape.ts)) registers a strategy for the name `broadcast`, which the op registry does not contain. It is dead weight: reachable only if some future dialect adds that name, and until then a rule nothing can select. The inline fusion builder registered for the same name ([`rules/fusion.ts:56`](../../../src/compiler/passes/lowering/rules/fusion.ts)) is unreachable for exactly the same reason.
- **A rule can be reachable only from the backward pass.** `reverse` (§34.6) has no forward producer — `flip` traces to `gather` — and runs on every convolution gradient, because the conv VJP flips the kernel. That is not a defect, but it does mean Proposition 34.4's set `E` has to be taken over the *joint* graph as well as the forward one: turning differentiation on enlarges the set of operations lowering must cover.
- **Selection ignores shape.** Definition 34.2 picks by priority alone. An implementation that is better for large inputs and worse for small ones cannot be expressed as two implementations; it has to be one rule with a branch inside, which is what the CUDA attention rule does ([`rules/attention.ts:204`](../../../src/compiler/passes/lowering/rules/attention.ts)) — it computes a threads-per-block figure from the target's shared-memory budget and falls back to the naive builder when it comes out below eight.
- **There is exactly one target-specific rule in the whole compiler.** The mechanism is general; its single user is attention on CUDA. Everything else that varies by backend varies in codegen (Part X) rather than in lowering.
- **Priority is a two-value scale.** `GENERIC_PLEVEL` and `TARGET_PLEVEL`, 10 and 20. Nothing uses an intermediate value, so "several generic rules ranked by quality" is expressible and unused, and a tie falls to registration order silently.
- **Registration is a module side effect.** The eleven `register()` calls run at import time of `graph_to_tensor.ts` ([`graph_to_tensor.ts:57`](../../../src/compiler/passes/lowering/graph_to_tensor.ts)), and `elementwise.ts` also writes an attribute onto the shared op registry at module scope ([`rules/elementwise.ts:35`](../../../src/compiler/passes/lowering/rules/elementwise.ts)). Import order is therefore load-bearing, and a test that imports the registry without the driver sees an op registry with no `elementwiseScalarOp` attributes on it.
- **A fusion region containing one unsupported operation loses the whole fusion.** `canLowerAsElementwiseFusion` is all-or-nothing ([`rules/fusion.ts:219`](../../../src/compiler/passes/lowering/rules/fusion.ts)): one inner operation without an inline builder sends the entire region down `lowerFusionAsIndividualOps`, materialising every intermediate. The fusion pass that formed the group does not consult the inline-builder list, so a group can be formed that lowering will then take apart, and nothing reports it.
- **Non-elementwise fusion is a lowering-time decision with no trace event.** The choice between the two modes emits nothing to the trace stream of Chapter 18, so a program whose fusion silently degraded looks, in the trace, exactly like one whose fusion worked.

## 34.9 Read the tests

- [`tests/compiler/passes/lowering/registry.test.js`](../../../tests/compiler/passes/lowering/registry.test.js) and [`op-strategy.test.js`](../../../tests/compiler/passes/lowering/op-strategy.test.js) — registration, the priority scheme, target filtering, and the cache invalidation on re-registration.
- [`tests/compiler/passes/lowering/coverage.test.js`](../../../tests/compiler/passes/lowering/coverage.test.js) — the executable form of §34.5.
- [`tests/compiler/passes/lowering/elementwise.test.js`](../../../tests/compiler/passes/lowering/elementwise.test.js), [`reduction.test.js`](../../../tests/compiler/passes/lowering/reduction.test.js), [`shape.test.js`](../../../tests/compiler/passes/lowering/shape.test.js), [`linalg.test.js`](../../../tests/compiler/passes/lowering/linalg.test.js), [`pooling.test.js`](../../../tests/compiler/passes/lowering/pooling.test.js), [`resize.test.js`](../../../tests/compiler/passes/lowering/resize.test.js), [`control-flow.test.js`](../../../tests/compiler/passes/lowering/control-flow.test.js) — one file per rule family, each asserting the emitted nest rather than the numbers.
- [`tests/compiler/passes/lowering/scan.test.js`](../../../tests/compiler/passes/lowering/scan.test.js) — the recurrence rule, which is the one that emits a loop kind rather than a plain `for`.

---

**Next:** [Chapter 35 — Index arithmetic](../ch35-index-arithmetic/README.md). Every rule in this chapter built a subscript. The next one is about what those subscripts are made of, and what the compiler can prove about them.
