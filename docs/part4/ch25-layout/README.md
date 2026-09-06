# Chapter 25 — Layout

Every optimization so far changed *what* is computed or *in what groups*. This one changes neither. The same operations run on the same values in the same order, and the only difference is which memory address each element lives at.

That sounds like it should be free, and it is not, in either direction. A layout costs a pass over memory to establish and pays nothing at all unless some kernel downstream is written to exploit it. Layout is therefore the chapter where the gap between "the compiler can do this" and "the compiler should do this" is widest, and where the second question has a sharp answer: a layout is worth exactly as much as the kernel that reads it.

## 25.1 The problem: a shape is not an address

A tensor's type says `tensor<1x16x28x28xf32>` — four dimensions, and nothing about memory. Chapter 10 introduced `Layout` as the missing half: the rule that turns an index into an address.

For a 4-D activation with dimensions (batch, channel, height, width), three conventions matter here:

- **NCHW** — row-major over `[N, C, H, W]`. Element `(n, c, h, w)` sits at `((n·C + c)·H + h)·W + w`. Consecutive `w` are adjacent; consecutive `c` are `H·W` apart.
- **NHWC** — the permutation `[0, 2, 3, 1]`. Element `(n, c, h, w)` sits at `((n·H + h)·W + w)·C + c`. Consecutive `c` are adjacent.
- **NCHW8c** — channels cut into blocks of eight. Element `(n, c, h, w)` sits at `(((n·(C/8) + ⌊c/8⌋)·H + h)·W + w)·8 + (c mod 8)`. Consecutive `w` are eight apart; the eight channels of one block are adjacent.

All three hold identical numbers. They differ only in which loop of a kernel walks memory sequentially, and that decides everything about cache behaviour. A convolution accumulating over input channels reads `C` values per output element: under NCHW those are `H·W` apart, which on a 28×28 feature map is a 3-KiB stride between consecutive reads — a cache miss per channel.

Which layout is better is not a property of the operation. It is a property of the *kernel*, and therefore of the target: a CPU with 64-byte cache lines wants the reduction axis contiguous; a GPU wants the axis that varies across threads contiguous; a tensor-core kernel wants a tile of a 16×16 patch contiguous.

The third entry in that list is also the one a permutation cannot express. NCHW and NHWC are reorderings of four dimensions; NCHW8c is not a reordering of anything, because it turns one logical dimension into two physical ones. A layout system that stops at permutations can describe the first two and has no way to say the third — which is a problem, because the third is the one that pays here.

So a compiler cannot pick one layout and be done. It needs a language for layouts that is wider than permutation, it has to decide per operation, and then it has to reconcile decisions that disagree.

## 25.2 Intuition: preferences, propagation, and a bill

Four steps, and the fourth is the one that is easy to forget.

**Preference.** Each operation states which layout it would like for each operand and each result. A convolution on this CPU says "I want my input and my weights blocked by eight along channels". An elementwise add says nothing — it is equally happy either way.

**Propagation.** Walk the graph in topological order assigning each value a layout: from the preference if the producer stated one, otherwise inherited from the operation's inputs. Operations that treat their operands as bags of elements pass a layout straight through, which is what makes long chains agree without anybody negotiating. This is the step that decides whether a five-layer network converts once or ten times.

**Reconciliation.** Wherever a value's assigned layout differs from what its consumer needs, insert a conversion — a physical repack. That costs a read and a write of the whole tensor. So the question becomes arithmetic: does the consumer save more from the layout it wants than the conversion costs?

**A kernel that cares.** None of the first three steps produces a single instruction of difference unless the code generator emits *different code* for the preferred layout. A layout is a promise about addresses; a kernel keyed to that layout is what collects on it. Where no such kernel exists, the first three steps are pure cost, and the honest description of the pass is that it is a transpose generator.

That last step is why layout is a *cost* problem and not a constraint problem. Every layout is legal; the compiler is choosing between programs that all compute the same thing.

## 25.3 Theory

> **Definition 25.1 (Layout function).** **(stated here)** A *layout* for a tensor of shape `S` is a function from the logical index space of `S` to a set of addresses. Where that function is injective — one address per element, which is every layout in this chapter — two layouts of the same tensor hold the same values, and a permutation of the data converts one into the other.

Chapter 35 proves the row-major flattening is a bijection; that is the special case where the function is `((i₀·S₁ + i₁)·S₂ + i₂)·…`, and it is the only case a permutation-based layout can reach.

### The algebra: shard, replica, offset

The abstraction this compiler uses for Definition 25.1 is taken from a published one rather than invented.

> **Definition 25.2 (Axe layout).** **(classical)** *Axe: A Simple Unified Layout Abstraction for ML Compilers*, arXiv:2601.19092. A layout is a triple `(D, R, O)`. `D` — the *shard* — is an ordered list of *iters*, each a triple `(extent, stride, axis)` with a non-zero stride; the address of a logical index is obtained by writing it in mixed radix over the extents, most significant digit first, and summing digit × stride per axis. `R` — the *replica* — is a multiset of iters describing further positions the same element also occupies. `O` — the *offset* — is one constant per axis. The layout maps a logical index to the *set* of addresses `O + D(i) + R`.
>
> Three details separate this from a hierarchical shape-and-stride notation. `D` forbids a zero stride, so a broadcast can only live in `R`. Canonicalization *generates* offsets — normalizing a negative stride emits one — so a pair without `O` is not closed under its own normal form. And the canonical form is unique only under saturation and a gap condition on `R`.

> **Proposition 25.3 (Permutations and blocking are the same kind of object).** **(invariant)** In this compiler every `Layout` is an Axe layout with `R` empty and `O` zero, over a single memory axis. A rank-`r` permutation is `r` iters; a blocked layout is `r + 1` iters, two of which carry the same logical dimension. Bound to the shape `1×16×28×28`, the three layouts of §25.1 are
>
> ```
>   NCHW     (1:12544, 16:784,        28:28,   28:1)
>   NHWC     (1:12544, 16:1,          28:448,  28:16)
>   NCHW8c   (1:12544, 2:6272, 8:1,   28:224,  28:8)
> ```
>
> where each iter is written `extent:stride`. Read the third line: the channel dimension has become *two* iters, `2:6272` and `8:1`, and the eight-wide one has stride 1. That is the whole of blocking, and it is why a permutation cannot say it — a permutation has exactly one iter per dimension.

The construction is [`types.ts:214`](../../../src/compiler/ir/graph/types.ts), which builds the iters from an `order` and an optional `block` and hands them to `AxeLayout` ([`axe.ts:167`](../../../src/compiler/ir/layout/axe.ts)). Extents are placeholders `_d0.._dn` until `bind(shape)` substitutes real ones ([`types.ts:283`](../../../src/compiler/ir/graph/types.ts)), so one `Layout` object describes a family of shapes rather than one.

The algebra is wider than what the graph tier admits, and the surplus is refused rather than ignored: `validateGraphProfile` ([`profiles.ts:30`](../../../src/compiler/ir/layout/profiles.ts)) rejects a non-empty `R`, on the grounds that nothing below the graph IR consumes a set-valued layout. The paper needs `R` for distributed and per-lane layouts; this compiler is single-device, and says so with a check rather than with an absence.

### The assignment problem

> **Definition 25.4 (Layout assignment problem).** **(stated here)** Given a DAG whose operations have layout preferences and conversion costs, assign a layout to every value minimizing the sum of conversion costs plus per-operation costs, where an operation's cost depends on whether its operands are in its preferred layout.

> **Theorem 25.5 (Layout assignment is NP-hard).** **(classical)** Layout assignment is an instance of the *multiway cut* problem and is NP-hard for three or more distinct layouts.

The reduction is the natural one: each layout is a terminal, each value is a vertex, and a conversion cost is an edge weight to be cut. With two layouts it is min-cut and polynomial; with three it is not. Three is not hypothetical here — §25.8 shows a single compilation carrying row-major, column-major and NCHW8c at once.

So, a heuristic. The standard one, and the one used here, is:

> **Definition 25.6 (Greedy propagation with local accept).** **(stated here)** Propagate preferences forward, collect the required conversions, group them by (value, from, to), and accept a group only when its estimated benefit exceeds its estimated cost. Reject everything else and leave those consumers with the layout they were given.

The weakness of Definition 25.6 is that it is local: it decides each conversion against its own consumers, with no view of whether accepting two nearby conversions would have let a third become free. The strength is that it never makes the program worse *by its own model* — and §25.7 is about the distance between that model and a stopwatch.

## 25.4 In mlfw: preference, propagation, insertion, materialization

### Who has a preference

Three operations declare one, through the `INFER_LAYOUT` op-attribute from Chapter 11 ([`linalg.ts:71`](../../../src/compiler/ir/graph/ops/linalg.ts)):

```ts
    opAttrs: { [OpAttrKey.GPU_CAPABLE]: true, [OpAttrKey.LAUNCH_BOUNDARY]: 'matmul', [OpAttrKey.LAYOUT_SENSITIVITY]: 4, [OpAttrKey.INFER_LAYOUT]: dotLayout },
```

`dot` and `conv` in [`linalg.ts`](../../../src/compiler/ir/graph/ops/linalg.ts), `reduce` in [`reduction.ts:29`](../../../src/compiler/ir/graph/ops/reduction.ts). Everything else has none, which is the right default: an elementwise operation should inherit, not demand.

`convLayout` ([`linalg.ts:25`](../../../src/compiler/ir/graph/ops/linalg.ts)) reads the target and answers in one of three ways:

```ts
    const block = { dim: kernelChannelDim, factor: spec.block.factor };
    if ((op.getAttr<number>('groups') || 1) === 1
      && channelDim === spec.block.dim && kernelChannelDim >= 0
      && blockDivides(input, spec.block) && blockDivides(kernel, block)) {
      const preferred = Layout.blocked(spec.order, spec.block.dim, spec.block.factor);
      const kernelLayout = Layout.blocked(spec.order, block.dim, block.factor);
```

If the target names a blocked convolution layout and the block genuinely fits — one group, the block sitting on the channel dimension, the factor dividing both the activation's `C` and the kernel's `I` — it asks for *both* operands blocked. The two blocks are not the same object: the activation is blocked on its `C` position and the kernel on its `I` position, each read out of the operation's `input_layout` and `kernel_layout` attributes rather than assumed equal. If the fit fails, a rank-4 convolution on a CPU or GPU asks for NHWC, and everything else asks for nothing. (A target that names a convolution layout with no block at all gets that order and stops there; no shipped target does.)

Padding is not modelled anywhere in this layout system, and this is where that shows: a channel count that is not a multiple of eight is not padded up to one, it drops out of the blocked case entirely. The layout the rule returns is also checked against Definition 25.2's graph profile before it is handed back, so a preference that could not be materialized is never proposed.

`dotLayout` ([`linalg.ts:55`](../../../src/compiler/ir/graph/ops/linalg.ts)) is short and asymmetric:

```ts
  const lhsLayout = Layout.rowMajor(lhsType.rank);
  if (target.isCPU() && rhsType.rank === 2) {
    return new LayoutPreference([lhsLayout, Layout.columnMajor(rhsType.rank)], [lhsLayout]);
  }
```

On a CPU, a matrix multiply wants its *right-hand* operand column-major, so the inner product walks both operands contiguously. This is the classical reason `B` is transposed in a hand-written GEMM.

### What the target asks for

A preference is a question addressed to the target, and the CPU target answers it twice ([`target.ts:203`](../../../src/compiler/support/target.ts)):

```ts
  preferredConvLayout: { order: [0, 1, 2, 3], block: { dim: 1, factor: 8 } },
  layoutAwareOps: LAYOUT_AWARE_OPS,
```

`preferredConvLayout` names NCHW8c: keep the dimension order, split dimension 1 by 8. `layoutAwareOps` is `{dot, conv}` ([`target.ts:13`](../../../src/compiler/support/target.ts)), and it is the target's assertion that its code generator emits different code for those two operations depending on the layout it is handed. It is §25.2's fourth step written down as data, and the last subsection here is where the assertion gets redeemed.

### Two propagation walks

[`analysis/layout_analysis.ts`](../../../src/compiler/analysis/layout_analysis.ts) is Definition 25.6's first two steps. Function arguments start with whatever their type carries ([`layout_analysis.ts:60`](../../../src/compiler/analysis/layout_analysis.ts)), then the topological walk assigns ([`layout_analysis.ts:85`](../../../src/compiler/analysis/layout_analysis.ts)):

```ts
      const def = registry.get(op.opName);
      const isEW = def && def.hasTrait(OpTrait.ELEMENTWISE);

      for (let r = 0; r < op.numResults; r++) {
        const val = op.getResult(r);
        if (!(val.type instanceof TensorType)) continue;
        if (isEW) {
          assignments.set(val, resolveFromInputs(op, assignments));
        } else {
          assignments.set(val, Layout.rowMajor((val.type as TensorType).rank));
        }
      }
```

Elementwise operations inherit; everything without a preference resets to row-major. A second walk over the same order collects the mismatches into `LayoutConversion` records — value, consumer, operand index, from, to.

That analysis assigns layouts to values *in a table*; it does not change a single type. Retyping is the job of a separate walk, `propagateBlockedLayouts` ([`layout_transform.ts:204`](../../../src/compiler/passes/layout/layout_transform.ts)), which runs after the conversions have been inserted and is what actually builds a chain. It has two halves. Forward: an operation whose preference is satisfied by its operands has its *results* retyped into the preferred layout, and an operation that is layout-agnostic adopts its operands' layout — dragging with it any operand that is a splat, since a tensor whose elements are all equal can be relabelled into any layout for free. Backward: every operand that is still blocked when its consumer cannot accept blocked gets one `layout_transform` back to plain, inserted immediately before that consumer.

"Layout-agnostic" is a narrower predicate than "elementwise" ([`layout_transform.ts:176`](../../../src/compiler/passes/layout/layout_transform.ts)):

```ts
function isLayoutAgnostic(op: Operation): boolean {
  const def = registry.get(op.opName);
  if (!def || !def.hasTrait(OpTrait.ELEMENTWISE) || op.numResults !== 1) return false;
  const result = op.getResult(0).type;
  if (!(result instanceof TensorType)) return false;
  const operands = tensorOperands(op);
  if (operands.length === 0) return false;
  return operands.every(v => (v.type as TensorType).shapeEquals(result) || isScalarOperand(v));
}
```

An operation qualifies only if every tensor operand has the result's shape, or is a single scalar. That restriction is doing real work. A ReLU passes: it is `maximum(x, zero)` with the zero broadcast to full shape, so element `i` of the result depends on element `i` of each operand and nothing else — reordering all of them the same way is invisible. A bias add does not pass: its second operand is `1×16×1×1`, a different shape, and the element of the bias that pairs with output element `i` is found by broadcasting rules over the *logical* index, which blocking the activation alone would misalign. §25.5 measures what that exclusion costs.

The consumer side is the mirror image ([`layout_transform.ts:186`](../../../src/compiler/passes/layout/layout_transform.ts)): a terminator never accepts a blocked operand, a layout-agnostic operation accepts one when all its operands agree, and anything else accepts one only if its own preference asked for exactly that layout. So the unblocking transform lands at the frontier of the blocked region and nowhere else — one per exit, not one per operation.

### Insertion, and the bill

[`passes/layout/layout_transform.ts`](../../../src/compiler/passes/layout/layout_transform.ts) groups the conversions and prices them ([`layout_transform.ts:73`](../../../src/compiler/passes/layout/layout_transform.ts)):

```ts
      g.consumers.push({ consumer, operandIdx });
      const capable = this.target.layoutAwareOps && this.target.layoutAwareOps.has(consumer.opName);
      g.benefit += capable ? (this._policy as LayoutPolicy).estimateBenefit(consumer, value.type, 1) : 0;
```

and accepts group by group ([`layout_transform.ts:81`](../../../src/compiler/passes/layout/layout_transform.ts)): a group whose benefit falls below its cost is dropped, and if nothing survives, or the groups that did survive cost more in total than they gain, the pass returns UNCHANGED. Accepted groups become `layout_transform` operations inserted right after the value's producer, with every consumer in the group rewired to the transformed value — so a value needed blocked by four convolutions is repacked once, not four times. The operation carries `src_layout` and `dst_layout`, and `src_block`/`dst_block` when either side is blocked; without those a canonicalization comparing only the permutations would see NCHW and NCHW8c as the same layout and erase the transform.

The cost and benefit models are two functions in [`layout_policy.ts`](../../../src/compiler/passes/layout/layout_policy.ts):

```ts
  estimateConversionCost(fromLayout: Layout | null, toLayout: Layout | null, tensorType: IRType): number {
    if (!(tensorType instanceof TensorType)) return 0;
    if (layoutEquals(fromLayout, toLayout)) return 0;
    const numEl = tensorType.numel();
    if (numEl < 0) return 1024;
    if (!fromLayout || !toLayout) return numEl * 2;
    return numEl * (contiguousDimOf(fromLayout, tensorType) === contiguousDimOf(toLayout, tensorType) ? 1 : 2);
  }
```

A conversion costs two units per element when it moves the contiguous dimension and one when it does not; `contiguousDimOf` reports `-1` for any blocked layout, so a repack into NCHW8c always costs two. Benefit ([`layout_policy.ts:54`](../../../src/compiler/passes/layout/layout_policy.ts)) is `numEl × LAYOUT_SENSITIVITY × useCount`, where sensitivity is 4 for `dot` and `conv` and 2 for `reduce`. So a convolution consuming a value in its preferred layout is worth four units per element against a conversion costing two, and the accept condition passes with room to spare.

Two units and four units. Both are made up, and §25.7 is about what to do about that.

### What a blocked tensor becomes in memory

A blocked layout would be a decoration if nothing below the graph could store one. The mechanism is a single idea: **a blocked rank-`N` tensor is a dense rank-`(N+1)` buffer, and consumers split the index.** It is how TVM handles NCHW4c, and it means no part of the backend has to learn about layouts.

`Layout.storage(shape)` ([`types.ts:299`](../../../src/compiler/ir/graph/types.ts)) returns the buffer's shape and strides — rank `N+1` when the layout is blocked — and the lowering context feeds it straight into the `Buffer` ([`lowering_registry.ts:148`](../../../src/compiler/passes/lowering/lowering_registry.ts)):

```ts
    const storage = t.layout ? t.layout.storage(shape) : null;
```

From that point the value's type is rank 4 and its buffer is rank 5, and everything generic downstream — the memory planner of Chapter 50, the byte counters, the code generators of Part X — sees an ordinary dense buffer with no layout attached to it at all. The two ranks are reconciled at every index site by one function ([`lowering_registry.ts:266`](../../../src/compiler/passes/lowering/lowering_registry.ts)):

```ts
export function storageIndices(indices: readonly TirNode[], layout: Layout | null): TirNode[] {
  const block = layout ? layout.block : null;
  if (!block) return [...indices];
  const logical = indices[block.dim];
  const factor = new IntImmNode(block.factor);
  const out = [...indices];
  out[block.dim] = mathOp('//', logical, factor);
  out.push(mathOp('%', logical, factor));
  return out;
}
```

The layout it consults comes from the *value's* `TensorType` ([`lowering_registry.ts:239`](../../../src/compiler/passes/lowering/lowering_registry.ts)), never from the buffer. That is deliberate: a `layout` field on `Buffer` would be a second place for the truth to live, and the buffer already carries one field written by a scheduling primitive and read by nobody.

Only two lowering rules ever see a blocked buffer, because conversions are inserted only for operands of operations that declared a preference: the `layout_transform` rule ([`rules/layout.ts:13`](../../../src/compiler/passes/lowering/rules/layout.ts)), which is a copy loop splitting the index on whichever side is blocked, and the convolution nest.

### The kernel that collects

Everything above is address arithmetic, and address arithmetic on its own is a pessimization. The default convolution nest runs `kw` innermost, and under a blocked input the address stride of `kw` goes from 1 to 8 — the same number of loads touching eight times as many cache lines.

What makes the blocked layout pay is that `buildConvNest` emits a *different nest* when it sees one ([`lowering_registry.ts:250`](../../../src/compiler/passes/lowering/lowering_registry.ts)):

```ts
  if (groups !== 1 || !inLayout || !kerLayout || !inLayout.block || !kerLayout.block) return 0;
  if (inLayout.block.dim !== inChannelDim || kerLayout.block.dim !== kerChannelDim) return 0;
  const factor = inLayout.block.factor;
  if (factor !== kerLayout.block.factor) return 0;
  if (typeof channelsPerGroup !== 'number' || channelsPerGroup % factor !== 0) return 0;
  return factor;
```

When all of that holds — one group, both operands blocked on their own contraction dimension, the same factor on each, and a channel count the factor divides — the input-channel loop is split into an outer loop over blocks and an inner loop of `factor` iterations, and the inner one is emitted *innermost* ([`lowering_registry.ts:434`](../../../src/compiler/passes/lowering/lowering_registry.ts)):

```ts
  if (icInnerVar) accBody = new ForNode(icInnerVar, new IntImmNode(0), new IntImmNode(channelBlock), ForKind.SERIAL, accBody);
```

Both operands then walk memory one element at a time in the innermost loop, and no division or remainder appears inside it at all — the split index *is* the loop variable. Every one of the conditions above is load-bearing; fail any of them and the old nest runs unchanged. That is the right fallback — an unblocked convolution should compile the way it always did — and it is also the trap, because a convolution can reach it with one operand blocked, which §25.5 measures as slower than not blocking at all.

## 25.5 Lab 1 — The layout a target asks for

```bash
node docs/part4/ch25-layout/labs/01-the-layout-a-target-asks-for.mjs
```

A chain of 3×3 convolutions over a `1×16×28×28` input, compiled with `optimization.layout` off and on, at three depths. The middle columns read the graph: how many `layout_transform` operations it contains, and — for each convolution — whether the pass gave it both operands blocked, only the kernel, or neither.

```
=== what the shipped CPU target asks for ===

  layoutAwareOps       {dot, conv}
  preferredConvLayout  order [0, 1, 2, 3], dimension 1 split by 8   (NCHW8c)

=== a chain of 3x3 convolutions over 1x16x28x28 ===

  depth  transforms  convs blocked on both operands  on the kernel only    off      on   ratio
      1           3                              1                   0   3.77   3.03  1.25x   (worst of 4 rounds +0.33 ms)
      2           4                              2                   0   7.49   5.97  1.26x   (worst of 4 rounds +0.37 ms)
      3           5                              3                   0  11.26   8.91  1.26x   (worst of 4 rounds +0.50 ms)

=== the same chain with a bias on every convolution ===

  depth  transforms  convs blocked on both operands  on the kernel only    off      on   ratio
      3           5                              1                   2  11.57  12.34  0.94x   (worst of 4 rounds +0.73 ms)
```

> **The timings.** **(measured)** One machine, 2026-09-06, one revision of the source, CPU/JavaScript backend, weights folded to compile-time constants. Each figure is the minimum of four rounds of best-of-four; the parenthesis is how far the worst round strayed above that minimum. They are evidence about this implementation on this machine, not a fact about blocked layouts.

Four things to read out of that table.

**The transform count grows by one per layer, not by two.** At depth 3 there are five transforms for three convolutions: one to block the input, three to block the three weight tensors, and one to unblock the final result. Nothing round-trips in the middle. That is `propagateBlockedLayouts` doing its job — each convolution's result is retyped blocked, the ReLU between layers adopts it, and the next convolution finds its operand already in the layout it wanted.

**The ratio is flat across depth.** 1.25, 1.26, 1.26. Flatness is the signature you want from a propagating layout pass. If each layer were repacked in and back out, the conversions would grow with depth while the win stayed per-layer, and the ratio would decay towards 1 and below. It does not, so the chain is genuinely being carried.

**A bias breaks the chain, and the third column says exactly where.** With a bias on every convolution the depth-3 chain has *one* convolution with both operands blocked and *two* with only the kernel blocked. The bias add is not layout-agnostic by §25.4's test — its second operand is `1×16×1×1`, not the result's shape — so the retyping walk stops there, unblocks, and the two later convolutions meet a row-major activation.

**And the half-blocked case is worse than no blocking at all.** 0.94×. This is not the pass failing to help; it is the pass hurting. The second and third convolutions get a blocked *kernel* against a row-major *input*, which fails the "both operands, same factor" test in §25.4, so the reordered nest is not emitted. What they get instead is the old nest with `kw` innermost and a `⌊i/8⌋`, `i mod 8` pair evaluated on the kernel index inside it — all of the address arithmetic and none of the locality. The first convolution's 1.25× is real, and is being spent paying for the two behind it.

**Try this.** Drop `foldWeights` from the lab's compile options and measure again. The weight repacks are then function arguments transformed on *every call* rather than constants transformed once, and the arithmetic in §25.4 — which prices a conversion once and its benefit forever — is being asked a question it does not model.

## 25.6 Lab 2 — What a blocked tensor becomes

```bash
node docs/part4/ch25-layout/labs/02-what-a-blocked-tensor-becomes.mjs
```

One convolution, followed all the way down: the graph after the layout pass, the lowered loop nest, and the emitted JavaScript.

```
=== the graph, once the layout pass has run ===

  %2 = "tera.layout_transform"(%1) : (tensor<16x16x3x3xf32>) -> tensor<16x16x3x3xf32, [0, 1, 2, 3]:1/8>
  %3 = "tera.layout_transform"(%0) : (tensor<1x16x28x28xf32>) -> tensor<1x16x28x28xf32, [0, 1, 2, 3]:1/8>
  %4 = tera.conv %3, %2 : (tensor<1x16x28x28xf32, [0, 1, 2, 3]:1/8>, tensor<16x16x3x3xf32, [0, 1, 2, 3]:1/8>) -> tensor<1x16x28x28xf32, [0, 1, 2, 3]:1/8>
  %5 = "tera.layout_transform"(%4) : (tensor<1x16x28x28xf32, [0, 1, 2, 3]:1/8>) -> tensor<1x16x28x28xf32>

  every type is still rank 4; the ":1/8" says dimension 1 is stored in blocks of 8

=== the buffer underneath is rank 5 ===

  buf_5[v0_10, (v1_11 // 8), v2_12, v3_13, (v1_11 % 8)] = buf_4[v0_10, v1_11, v2_12, v3_13]

  four loop indices, five subscripts: the channel index was split into a quotient and a remainder

=== the loop nest the convolution lowers to ===

  for cn_32 in 0..1
    for coc_33 in 0..16
      for co0_35 in 0..28
        for co1_36 in 0..28
          for cic_34 in 0..2
            for ck0_37 in 0..3
              for ck1_38 in 0..3
                for cici_39 in 0..8   <- the channel block, innermost

  (buf_14[cv0_40, cv4_44, (cv2_42 + (cv5_45 + -1)), (cv3_43 + (cv6_46 + -1)), cv7_47] * buf_5[cv1_41, cv4_44, cv5_45, cv6_46, cv7_47])

  both operands end in that innermost index, so both walk memory one element at a time

=== and in the emitted JavaScript ===

  buf_5[(((((i0_6 * 144) + (((i1_7 / 8) | 0) * 72)) + (i2_8 * 24)) + (i3_9 * 8)) + (i1_7 % 8))] = buf_4[((((i0_6 * 144) + (i1_7 * 9)) + (i2_8 * 3)) + i3_9)];
```

The four sections are one fact seen at four altitudes.

**In the type**, `tensor<1x16x28x28xf32, [0, 1, 2, 3]:1/8>` is Proposition 25.3's five iters printed compactly: the permutation, then "dimension 1, factor 8". The shape is unchanged. Nothing about the *tensor* has changed — sixteen channels are still sixteen channels — and that is what makes a layout a compiler-internal decision rather than a semantic one.

**In the buffer**, the rank went up. Four loop variables produce five subscripts, and the channel index appears twice: once divided by eight and once modulo eight. Chapter 33 introduces buffers as shape plus strides, and this is the one case where a value's rank and its buffer's rank legitimately differ.

**In the loop nest**, the input-channel loop has become two: `cic_34` counting the two blocks, and `cici_39` counting the eight channels within a block. `cici_39` is innermost — inside the two kernel-window loops, which is not where a textbook convolution puts the channel reduction — and both operands are indexed by it directly. No division survives in the inner loop; the split happened when the loop was built.

**In the emitted source**, the arithmetic is exactly what §25.1's NCHW8c formula says: `⌊c/8⌋·72 + h·24 + w·8 + (c mod 8)` for a `16×16×3×3` kernel, against `c·9 + h·3 + w` for the row-major original. The layout was never anything but a way of writing that expression down once, in a place where a preference could be attached to it.

## 25.7 The kernel is what pays, and only measurement knows how much

§25.4 ended with two invented constants, 2 and 4. §25.5 measured 1.26 and 0.94. Neither pair derives from the other, and it is worth being precise about what each is for.

The cost model's job is to stop the pass doing something obviously silly — transposing a tensor read once, converting a value whose consumers do not care. It is a filter, not a predictor, and it does not know the difference between §25.4's blocked convolution nest and the old one. That is why the same model happily proposes the half-blocked convolutions that make the program slower: by its arithmetic they are worth `4 × numel` against a cost of `2 × numel`, and it has no term at all for "the nest this was supposed to enable will not be emitted".

The diagnostic question for a layout decision, then, is not "is the pass correct" — it is — but **"is there a kernel downstream that will actually be faster, and did it get emitted?"** Layout transformation pays only when some consumer has a specialized implementation keyed to the layout. Here one exists, and the conditions in `contractionBlockFactor` are the exact statement of when it fires. Away from them, the transform is pure cost.

Which is why `optimization.layout` still defaults to `false` ([`compiler.ts:134`](../../../src/compiler/pipeline/compiler.ts)), and why the pass, when enabled, runs early in the graph pipeline ([`graph_pipeline.ts:59`](../../../src/compiler/pipeline/graph_pipeline.ts)) before fusion has hidden the elementwise operations it needs to walk through. What turns it on is the optimization gate of Chapter 61: it compiles the candidate configurations, times them, and adopts the winner only if it clears a floor. Measure, don't model. Chapter 46 argues the general case, and this chapter is the book's strongest evidence for it — a pass whose own cost model cannot tell its best case from its worst.

With one caution, which is why §25.5 measures by hand rather than quoting the gate. A gate is only as good as its floor, and this one's floor is 1.05 while its apparatus's round-to-round spread is several times that ([Chapter 61 §61.5](../../part11/ch61-tracing/README.md)). A gate that reports a layout win is evidence that layout was worth adopting *for that graph, on that machine, in that minute*; it is not by itself evidence that a mechanism exists. The thing that makes §25.5's numbers mean something is not their size, it is that the third column of the table says why they are that size.

## 25.8 Traps and limits

- **The two propagation walks use different rules, and where they disagree the program gets slower.** `resolveFromInputs` ([`layout_analysis.ts:137`](../../../src/compiler/analysis/layout_analysis.ts)) gives an elementwise result the most common layout among its operands, counting a bias add's `1×16×1×1` operand as an ordinary vote and so concluding that the sum is blocked; `isLayoutAgnostic` ([`layout_transform.ts:176`](../../../src/compiler/passes/layout/layout_transform.ts)) refuses that same operation. The analysis therefore believes a later convolution's activation is already blocked and proposes no conversion for it, while the retyping walk leaves it row-major — and that convolution ends up with a blocked kernel against a plain input, the one combination `contractionBlockFactor` cannot exploit. §25.5 measures 0.94× on a three-layer biased chain. Nothing detects the mismatch: there is no rule saying "a blocked kernel is only worth having when the activation is blocked too", and no diagnostic when a preference is half-satisfied.
- **`LayoutAnalysis` is never cached.** It takes a third `policy` argument that the analysis protocol has no room for, so the pass calls `LayoutAnalysis.compute` directly ([`layout_transform.ts:59`](../../../src/compiler/passes/layout/layout_transform.ts)) rather than going through the manager (Chapter 16). Every run recomputes it.
- **Anything without a preference resets to row-major** ([`layout_analysis.ts:93`](../../../src/compiler/analysis/layout_analysis.ts)), and the retyping walk agrees ([`layout_transform.ts:176`](../../../src/compiler/passes/layout/layout_transform.ts)). Blocked layouts now survive elementwise chains, so the reset no longer breaks a ReLU sequence — but it still breaks everything else. A `reshape`, a `slice`, a `transpose` or a pooling operation between two convolutions ends the blocked region and forces an unblock there, where a shape-aware propagation would map the layout through and continue. `pool2d` is the sharpest case, since a pooling layer between convolutions is the most ordinary thing in a CNN and its result is a perfectly good candidate for the same blocking as its input.

  Be careful about the *cost* of that break, though, because "forces two conversions" is a tempting summary and is not what the algorithm does. The reset assigns a layout; it does not insert anything. Conversions are emitted later, and how many appear depends on what the consumers ask for. Three outcomes are reachable from the same reset: **two** conversions if the producer's blocked output must be undone and the next convolution's preference re-established; **one** if only one side disagrees; and **none**, if the consumer declared no preference at all. The price of the reset is "a conversion the chain did not need", not a fixed count of two.
- **The cost and benefit constants are unitless and unmeasured.** `numEl * 2` or `numEl * 1` for a conversion, `numEl * sensitivity * useCount` for the benefit, sensitivity 4 or 2 by operation. Nothing derives these from the target — not from `cacheLineSizeBytes`, not from `memoryBandwidthGBs`, both of which the target already carries — so `benefit < cost` compares two invented scales. Chapter 61's measuring gate is the thing standing between that arithmetic and a shipped regression.
- **Conversions are priced per call, and weights are converted per call unless they are folded.** The benefit model prices a conversion once and its benefit forever, which is right for a constant and wrong for an argument. §25.5's measurements fold the weights; its *Try this* is the same measurement without folding.
- **Three layouts now coexist in one compilation, so Theorem 25.5's regime is reachable.** **(invariant)** A network with a convolution and a final `Linear` compiles, on `CPUTarget()`, to a graph carrying row-major, `[1, 0]` column-major (for the `dot`'s right-hand operand) and `[0, 1, 2, 3]:1/8`. Layout assignment here is therefore no longer the two-terminal min-cut that could be solved exactly; what runs is still Definition 25.6's greedy, which does not attempt an optimum and has no way to know it missed one.
- **A layout gain recorded before this mechanism existed does not mean what it says.** Until `CPUTarget` declared `layoutAwareOps`, the pass proposed conversions and rejected every one of them, so the gate's two configurations compiled to byte-identical source and it was timing one program against itself; whatever "gain" it reported was the machine. [Chapter 61 §61.5](../../part11/ch61-tracing/README.md) measures that same apparatus now that the two configurations genuinely differ, and the round-to-round spread it reports is the reason a number from the earlier period could have been anything. Do not carry a layout figure across that boundary.
- **The blocked path does not reach the vectorizer.** The channel-block loop is a reduction, and the WASM vectorizer bails on a loop-carried dependence, so the eight-wide contraction never becomes an `f32x4` pair. The layout chosen to make eight channels contiguous is, on that backend, eight scalar loads that happen to be adjacent.
- **Padding is not modelled.** `Layout.blocked` divides exactly; a factor that does not divide the extent fails the graph profile rather than padding the tensor up. `convLayout` handles that by checking divisibility before it proposes, and falling through to NHWC rather than to plain NCHW — so a 20-channel convolution silently gets a different layout from a 16-channel one on the same target. Real padding needs the logical tensor to become a slice of a larger admitted shape, and `TensorType.footprint()` is where that would attach.

## 25.9 Read the tests

- [`tests/compiler/ir/layout/`](../../../tests/compiler/ir/layout/) — the algebra of Definition 25.2 on its own: canonicalization, tiling, slicing, and the text form.
- [`tests/compiler/passes/layout/`](../../../tests/compiler/passes/layout/) — preference inference, the propagation walk, and the accept/reject arithmetic with hand-set costs. Two files are metamorphic, compiling the same graph with the layout pass off and on and comparing outputs: `transform.test.js` covers strided, padded, grouped and pooled convolutions, `live-on-cpu.test.js` covers chains through the shipped `CPUTarget()`. `live-on-cpu.test.js` also asserts on the emitted source, that the shipped target really produces the split index — so a metamorphic pass cannot be two identical unblocked compilations agreeing with each other.
- [`tests/compiler/pipeline/opt-gate.test.js`](../../../tests/compiler/pipeline/opt-gate.test.js) — the measurement gate that decides whether layout is worth enabling for a given model and target, which is the mechanism §25.7 recommends over the model.

---

**Next:** [Chapter 26 — Three optional pipelines](../ch26-optional-pipelines/README.md), which closes Part IV with the three graph-level transformations that are, like this one, off unless you ask: rematerialization, quantization, and partitioning.
