# Chapter 41 — Memory and reduction primitives

Chapter 40's four primitives are bijections on the iteration space: the same points, renamed and revisited in a different order. None of them changes *what* is computed at a point, so the only things they can get wrong are which points exist — `split`'s guard, and that is arithmetic — and in what order they are visited, which is `reorder`'s dependence question, deferred to Chapter 42.

The primitives in this chapter are not like that. `rfactor` changes the order in which values are combined, which for floating point changes the answer. `cacheRead` and `cacheWrite` introduce a buffer that did not exist. `computeInline` deletes a buffer and recomputes its contents. Each of them needs a precondition that the compiler must check or the caller must promise, and this chapter is about which is which.

## 41.1 The problem: a reduction is a chain, and a chain has no parallelism

```
for sa0_7 in 0..1 {
  for r0_9 in 0..8 {
    block reduce_acc_1 {
      buf_3[sav0_8] = (buf_3[sav0_8] + buf_1[sav0_8, rv0_10])
    }
  }
}
```

Eight iterations, each reading what the last one wrote. Chapter 36's dependence analysis reports a loop-carried dependence on `r0_9` at distance 1, Chapter 40's `parallelize` refuses it, and both are right: the eight additions form a chain, and a chain of length eight takes eight steps whatever hardware you have.

Except that it does not, because addition is associative. `((((a+b)+c)+d)` and `((a+b)+(c+d))` are the same value, and the second is two steps deep instead of three. A reduction over `K` elements can be done in `log K` depth, or — the version that matters for real hardware — in `K/f` steps on `f` independent accumulators followed by one combine over `f`.

That transformation cannot be a loop primitive, because it is not a renaming of the iteration space: it needs a new buffer of `f` partial results, and the loop that fills it computes different sums than the original loop did. And it is only correct because of a property of `+` that nothing in the IR records.

Meanwhile the other three primitives answer a plainer question. Chapter 4 measured the cost of a memory access and Chapter 22 the cost of not fusing. A schedule needs to be able to say *where* a value lives — in a register, in shared memory, in a cache-resident tile — and that means introducing buffers, which the lowering rules chose and the schedule must be able to change.

## 41.2 Intuition

**`rfactor`.** Read the reduction as a sum over a range, and split the range as Chapter 40 splits a loop: `Σ_{k<K} x_k = Σ_{i<f} ( Σ_{j<K/f} x_{j·f+i} )`. The inner sums are independent — `f` of them, each over `K/f` terms — so they go in a buffer of `f` slots, and a second loop adds the slots up. It is `split` applied to a reduction axis, plus the observation that the outer half can be *reassociated* rather than merely reordered.

**`cacheRead` / `cacheWrite`.** Interpose a buffer. A read becomes "copy the region into scratch, then read the scratch"; a write becomes "write into scratch, then copy the scratch out". The point is the scope of the scratch: a GPU shared-memory tile read by 256 threads is one global read per element instead of 256.

**`computeInline`.** The opposite move: delete a buffer and substitute the expression that filled it into every place it was read. Fewer bytes, more arithmetic. This is Chapter 22's fusion decision, arriving again at loop level, and it is the same trade-off with the same answer — it wins when the arithmetic is cheap relative to the traffic and loses when the value is read many times.

**`computeAt`.** Move a producer block *inside* one of the consumer's loops, so it computes only the slice the current iteration needs. This is what turns `cacheRead`'s whole-buffer copy into a per-tile stage.

## 41.3 Theory

> **Definition 41.1 (Reduction block).** A block is a *reduction over `⊕`* if its body is a single store `A[s] = A[s] ⊕ e`, where `s` does not involve the reduction axes and `e` does not read `A`.

The pattern-match in Definition 41.1 is doing real work: it names the accumulator (the buffer that appears on both sides at the same subscript), the operator, and the update expression. `rfactor` is exactly this decomposition, and everything it builds is assembled from those three parts.

> **Theorem 41.2 (rfactor is sound when `⊕` is associative and commutative with identity `e₀`, and unsound for some input when it is not associative).** Let `B` be a reduction over `⊕` with axis `k ∈ [0, K)` and initial value `e₀`, and let `f ∣ K`. The rfactored program — `P[i] = e₀ ⊕ ⨁_{j<K/f} x_{j·f+i}` for each `i < f`, then `A[s] = e₀ ⊕ ⨁_{i<f} P[i]` — computes the same value as `B` for all inputs if `⊕` is associative and commutative with identity `e₀`. If `⊕` is not associative, there are inputs for which it does not.

*Proof.* The original computes `e₀ ⊕ x₀ ⊕ ⋯ ⊕ x_{K−1}` left to right. The rfactored form computes the same `K` terms grouped by residue mod `f`, with `e₀` inserted once per group and once at the combine. Associativity lets the parentheses be moved, commutativity lets the terms be permuted from index order into residue order, and identity lets the `f` extra copies of `e₀` be discarded. Conversely, if `⊕` is non-associative, pick `a, b, c` with `(a⊕b)⊕c ≠ a⊕(b⊕c)`; with `K = 4`, `f = 2` the two forms bracket differently and differ. ∎

**Note that the theorem is not an "iff", and calling it one would claim more than the proof gives.** The forward direction is a genuine sufficiency result: associativity, commutativity and an identity are enough. The converse only says that a *non-associative* operator fails on *some* input — which leaves out two cases the phrase "iff" would wrongly settle. An operator that is associative but not commutative is still fine for this particular transform *if* the partition preserves index order, which the strided partition below does not; and a non-associative operator is perfectly correct on every input that does not exercise the difference, which is why float rfactor passes almost every test anyone writes. Soundness in the sense of Definition 38.3 is a claim about all inputs, so the second case is still a failure — but "unsound" and "always wrong" are not the same statement, and §41.5's counterexample had to be constructed rather than found.

The gap between Theorem 41.2 and the code is the whole of §41.5:

> **Counterexample 41.3 (Floating-point addition is not associative).** In IEEE-754 binary64, `(10¹⁶ + (−10¹⁶)) + 1 = 1` while `10¹⁶ + ((−10¹⁶) + 1) = 0`: on the left the cancellation happens first and the `1` survives; on the right `−10¹⁶ + 1` rounds straight back to `−10¹⁶`, because the ulp at `10¹⁶` is `2` and the tie breaks to even. So `f32`/`f64` addition satisfies Theorem 41.2's hypothesis only as an approximation, and rfactor changes the answer on inputs that exercise the difference. §41.5 exhibits a sum of eight `f32` values for which the serial order gives 3 and rfactor by 4 gives 6.

This is Chapter 20's fast-math question again, and the compiler answers it the same way — by treating reassociation as licensed rather than proved. What is different here is that the licence is implicit: there is no `fastMath` flag on `rfactor`, and Part VIII's search will apply it whenever the sketch generator offers it.

The memory primitives need a weaker and purely structural condition.

> **Proposition 41.4 (Interposing a buffer is sound, stated here).** Let `B` read buffer `b`, let `C` be a fresh buffer of the same shape and type, and let the program be transformed by (i) inserting a copy `C[x] = b[x]` over the whole shape immediately before `B`'s nest, and (ii) redirecting `B`'s reads of `b` to `C`. The result is equivalent, provided nothing between the copy and `B` writes `b`, and nothing writes `C`.

*Proof sketch.* After the copy, `C` and `b` agree everywhere; no intervening write makes them disagree; so every load `B` performs returns what it would have returned. `C` is fresh, so no other statement is affected. ∎

Both provisos are discharged in this compiler by *construction* rather than by a check: the copy is placed immediately above the block's outermost loop, and `C` is a buffer nobody else has a reference to. Neither is verified, and neither survives a later `computeAt` that moves the block away from its copy.

`computeInline` is where the analysis is real:

> **Definition 41.5 (Inlinable producer, stated here).** A block `P` writing `A[φ(v₁,…,v_d)]` is *inlinable* if `φ` is an invertible affine map of `P`'s own iteration variables, `P` has no `initBody`, its store value reads no buffer `P` itself writes, `A` is not used inside any index expression anywhere, and every variable the store value reads is an iteration variable of `P`.

> **Proposition 41.6 (Inlining is sound for an inlinable producer, stated here).** Replacing every load `A[ψ]` in the program by `P`'s store value with `v_i := φ⁻¹(ψ)_i`, and deleting `P`, preserves semantics.

*Proof sketch.* Invertibility of `φ` is what makes `φ⁻¹(ψ)` well defined, so each load can be rewritten to the expression that produced that element. "No `initBody`" excludes reductions, whose element is produced by a loop rather than an expression. "Reads no co-produced buffer" and "every free variable is an iteration variable" ensure the substituted expression means the same thing at the consumer's site as it did at the producer's. `A` not appearing in an index expression rules out indirect access, where the set of elements read is not statically known. ∎

Every clause of Definition 41.5 is a `throw` in `_inlinePlan`, and §41.6 lists them.

## 41.4 In mlfw

### `rfactor`

[`schedule.ts:629`](../../../src/compiler/schedule/schedule.ts), 86 lines, the longest primitive in the file. It opens with three guards — the named reduction loop exists, its extent is constant, and `1 < f < K` with `f ∣ K` — and then pattern-matches Definition 41.1:

```ts
    const store = block.body;
    if (!store || store.type !== 'BufferStoreNode' || !store.value || store.value.type !== 'MathOpNode') {
      throw new Error(`rfactor: block '${blockName}' body is not a single accumulating store`);
    }
    const acc = store.buffer;
    const spatialIdx = store.indices;
    const storeMath = store.value as MathOpNode;
    const op = storeMath.op;
    const isAccLoad = (node: TirNode | null | undefined): boolean =>
      !!node && node.type === 'BufferLoadNode'
      && (node as BufferLoadNode).buffer === acc
      && sameIndices((node as BufferLoadNode).indices, spatialIdx);
    let update: TirNode | undefined;
    if (isAccLoad(storeMath.a)) update = storeMath.b as TirNode;
    else if (isAccLoad(storeMath.b)) update = storeMath.a;
    else throw new Error(`rfactor: block '${blockName}' body is not an accumulation into '${acc.name}' at the stored subscript`);
    if (readsBuffer(update, acc)) {
      throw new Error(`rfactor: update expression in block '${blockName}' reads accumulator '${acc.name}'; cannot factor reduction`);
    }
```

Theorem 41.2's hypothesis is a four-element table ([`schedule.ts:45`](../../../src/compiler/schedule/schedule.ts)):

```ts
const RFACTOR_REDUCE_TYPE: Record<string, string> = { '+': 'sum', '*': 'prod', 'min': 'min', 'max': 'max' };
```

which doubles as the operator test and as the route to each operator's identity (§41.6). `min` and `max` select an operand rather than computing a new value, so no rounding happens and they are associative and commutative exactly; `+` and `*` are the approximate cases of Counterexample 41.3. The table does not distinguish the two situations, and there is no dtype test — an integer sum, which is exact, and a float sum, which is not, take the same path.

The rest builds two nests. The partial one carries the split of the reduction axis, the new `[factor, ...acc.shape]` buffer, and — this is the only place in the compiler that does it — an `initBody`:

```ts
    const partialStore = new BufferStoreNode(partialBuf, cfIdx(kiIter.iterVar),
      new MathOpNode(op, new BufferLoadNode(partialBuf, cfIdx(kiIter.iterVar)), partialUpdate));
    const partialInit = new BufferStoreNode(partialBuf, cfIdx(kiIter.iterVar), cloneExprTree(initVal));
    const partialBlock = new BlockNode(`${blockName}_rf_p`, partialIterVars,
      block.reads.map(r => ({ buffer: r.buffer })), [{ buffer: partialBuf }], partialStore, partialInit);
```

and the initial value it uses is `block.initBody`'s value if there is one and `IntImmNode(0)` otherwise ([`schedule.ts:656`](../../../src/compiler/schedule/schedule.ts)). Since no lowering rule sets `initBody` (Chapter 33), **every rfactor in this compiler defaults its identity element to integer zero** — correct for `+`, wrong for `*` (identity 1), `min` (identity `+∞`) and `max` (identity `−∞`).

**And the identity is where a plausible-sounding argument goes wrong.** It is tempting to reason: the reduce lowering rule emits its identity into a *separate* init block rather than into `initBody`, so `rfactor` only ever meets `+` accumulations in practice. Read it again — a separate init block is exactly what leaves `block.initBody` null, which is exactly what makes the fallback fire. The arrangement that was supposed to make a wrong default unreachable is the arrangement that reaches it.

> **Counterexample 41.7 (rfactor on a product).** A block computing `C = ∏ A[k]` over `k ∈ [0, 4)` with no `initBody` is accepted — `*` is associative and commutative. Initialise its partial buffer to `0` and every partial product is `0 × something = 0`, and the combine multiplies zeros: the product of `[2, 3, 4, 5]` becomes `0` instead of `120`. Not a rounding difference; every product in the program becoming zero.

So the identity comes from the operator and the accumulator's dtype, through the same `reduceInitValue` the lowering rules use — a reduction initialised by `rfactor` and one initialised by `lowerReduce` cannot disagree:

```ts
const RFACTOR_REDUCE_TYPE: Record<string, string> = { '+': 'sum', '*': 'prod', 'min': 'min', 'max': 'max' };

function rfactorIdentity(op: string, dtype: string): TirNode {
  const value = reduceInitValue(RFACTOR_REDUCE_TYPE[op], dtype);
  return isDtypeInt(dtype) ? new IntImmNode(value) : new FloatImmNode(value);
}
```

The dtype matters as much as the operator: `max` on a float accumulator initialises to `−∞` and on an integer one to that type's minimum. An explicit `initBody` still wins, because a reduction with a non-identity initial value is a legitimate program and rfactoring it must preserve that value.

**The accumulator pattern needs checking too, and Definition 41.1 says what to check.** Comparing buffers alone is not enough: `C[i] = C[j] ⊕ x` with `i ≠ j` would pass, and so would a store whose "update" half also reads `acc` — in which case the partial and combine nests both read a buffer they are concurrently rewriting. So the load's subscript must match the store's, compared as affine forms via `toLinearForm` rather than by node identity so that `C[0]` and `C[0+0]` are the same place, and the update expression must not read `acc` at all. Both refuse with a message naming the accumulator.

**What none of this touches is Counterexample 41.3**, and nothing can: reassociating a float sum is what `rfactor` is *for*. It remains an N2 transformation offered without asking, and §41.5 measures serial `3` against rfactored `6`.

Two derived facts about the emitted nests:

- The **partial nest** iterates `for k_i in [0, f) { for k_o in [0, K/f) }` with the binding `k = k_o·f + k_i`. The *inner* index is the partial slot, so slot `i` accumulates elements `i, i+f, i+2f, …` — a strided partition, not a contiguous one. That is what makes §41.5's counterexample pair `10¹⁷` with `−10¹⁷`.
- The **combine nest** is itself a reduction over `f`, with the same `initBody`, and the compiler makes no attempt to rfactor it in turn.

### `decomposeReduction`

[`schedule.ts:716`](../../../src/compiler/schedule/schedule.ts). One block with an `initBody` becomes two sibling blocks — an init nest over the spatial loops and an update nest over all of them. Its first line is the whole story:

```ts
    if (!block.initBody) throw new Error(`decomposeReduction: block '${blockName}' has no initBody`);
```

which, since no lowering rule sets `initBody` (Chapter 33), means it throws on every block the lowering rules produce. It is the inverse of the form those rules already emit. §41.6 follows the consequence into Part VIII.

### `cacheRead` and `cacheWrite`

[`schedule.ts:1005`](../../../src/compiler/schedule/schedule.ts) and [`schedule.ts:752`](../../../src/compiler/schedule/schedule.ts), 31 and 30 lines, mirror images. `cacheRead`:

```ts
    const cache = new Buffer(`${bufferName}_${blockName}_cache`, [...buf.shape], buf.dtype, scope);

    const idxVars = buf.shape.map((_, d) => new VariableNode(`${cache.name}_i${d}`, 'int32'));
    const copyStore = new BufferStoreNode(cache, idxVars, new BufferLoadNode(buf, idxVars));
    const copyBlock = new BlockNode(`${cache.name}_fill`, idxVars.map(v => new BlockRealizeNode(v, v)),
      [{ buffer: buf }], [{ buffer: cache }], copyStore);
```

`[...buf.shape]` is Proposition 41.4's "over the whole shape", and it is the primitive's main limitation: the cache is as large as the buffer, so `cacheRead(block, A, 'shared')` on a 4096×4096 input asks for 64 MiB of shared memory. The useful version stages a tile, and to get one you would `cacheRead` and then `computeAt` the fill block into the tile loop — which is why the two primitives are usually described together and why neither being called is one fact rather than two.

`new BlockRealizeNode(v, v)` binds each iteration variable to *itself*: the same `VariableNode` object is the loop variable and the block's iteration variable. §41.6 shows what the CUDA backend makes of that.

### `computeInline` and `computeInlineBlock`

[`schedule.ts:836`](../../../src/compiler/schedule/schedule.ts) is `_inlinePlan`, which is Definition 41.5 as a sequence of guards, and it is the most carefully written function in the file. The invertibility clause is delegated:

```ts
      const decompositions = invertWriteIndices(access);
      if (!decompositions) {
        throw new Error(`${primitive}: producer write index of '${store.buffer.name}' is not an invertible affine map of its loop variables`);
      }
```

`invertWriteIndices` ([`schedule.ts:105`](../../../src/compiler/schedule/schedule.ts)) asks Chapter 35's `mixedRadixDecomposition` for each write subscript — the same machinery that decides whether a reshape's index is a clean digit split. When it succeeds, `recoverIterVar` rebuilds each producer iteration variable from the consumer's subscript by dividing and taking the remainder. Inlining `A[i*4+j] = e` into a consumer reading `A[t]` therefore substitutes `i := t // 4`, `j := t % 4`.

The two public entry points differ by one line: `computeInline` requires the producer to write exactly one buffer, `computeInlineBlock` allows several. Only the second has a caller, `InlineReindexPass` ([`inline_reindex_pass.ts:85`](../../../src/compiler/passes/schedule/inline_reindex_pass.ts)), which runs on GPU targets only, when scheduling is on, and wraps the call in a `catch (_) {}` that discards the reason.

### `computeAt` and `reverseComputeAt`

[`schedule.ts:995`](../../../src/compiler/schedule/schedule.ts) and [`schedule.ts:1000`](../../../src/compiler/schedule/schedule.ts) are three lines each on top of `_relocateBlockToLoop`, which does three things: check that the moved block's loops have the *same iteration domains* as the target's innermost loops (`_alignedLoopPairs`, [`schedule.ts:933`](../../../src/compiler/schedule/schedule.ts)), check Proposition 39.5 (`_checkRelocationDependences`), and then splice, renaming the moved block's loop variables to the target's.

The domain check is strict equality of `min` and `extent`, which is why this pair cannot express the interesting case: staging a *tile* of a producer inside a consumer's tile loop requires the producer's domain to shrink, and here it must match exactly. What the primitives do express is "these two nests have the same shape; run them interleaved instead of one after the other" — loop fusion at the block level.

### `setScope` and `storageAlign`

[`schedule.ts:783`](../../../src/compiler/schedule/schedule.ts) and [`schedule.ts:792`](../../../src/compiler/schedule/schedule.ts), eight and nine lines, both one field assignment on a `Buffer`. `storageAlign` sets the `{axis, factor, offset}` record that Chapter 50's allocator reads to pad a row and break shared-memory bank conflicts. Neither has a caller.

## 41.5 Lab — rfactor, and the price of associativity

```bash
node docs/part7/ch41-memory-and-reduction-primitives/labs/01-rfactor.mjs
```

An 8-element sum, rfactored by 4:

```
=== rfactor(k, 4) — partial buffer buf_3_rf[4,1] ===
  for sa0_7 in 0..1 {
    for r0_9_rfi_0 in 0..4 {
      for r0_9_rfo_1 in 0..2 {
        block reduce_acc_1_rf_p {
          bind rv0_10 = ((r0_9_rfo_1 * 4) + r0_9_rfi_0)
          bind r0_9_rfvi_3 = r0_9_rfi_0
          init {
            buf_3_rf[r0_9_rfvi_3, sav0_8] = 0
          }
          buf_3_rf[r0_9_rfvi_3, sav0_8] = (buf_3_rf[r0_9_rfvi_3, sav0_8] + buf_1[sav0_8, rv0_10])
```

followed by a combine nest over `r0_9_rfp_2 in 0..4`. One block became two, `buf_3_rf[4,1]` appeared, and both new blocks carry an `init {}` — the field Chapter 33 found implemented on every side and set by no rule. `rfactor` is the one thing that sets it.

Then the two nests are compiled and run on the same eight numbers:

```
=== does the answer change? ===

  input                              serial     rfactor(4)
  1..8, exact in every order         36         36
  1e17 and its negation, four apart  3          6
```

Counterexample 41.3, executed. The eight values are `[1e17, 1, 1, 1, -1e17, 1, 1, 1]`. Serial order adds `1e17` first, so the next three `1`s are below the ulp of the running total and vanish; then `−1e17` brings it back to zero, and the last three `1`s survive: **3**. rfactor by 4 puts element `i` and element `i+4` in the same slot, so `1e17` meets `−1e17` in slot 0 and cancels exactly, leaving six `1`s: **6**.

Both are correct sums of the same eight `f32` values in different orders. The exact answer is 6, so here reassociation is *more* accurate — which is the honest form of the caveat. Reassociation does not make a reduction worse; it makes it different, and neither order is the one the user asked for, because the user asked for a sum and IEEE-754 does not have one.

That is a defence of `rfactor` as an engineering decision, and it must not be mistaken for a defence of its soundness. Definition 38.2 says *identical*; 3 and 6 are not identical; so `rfactor` on a float reduction is not a sound primitive in the sense of Definition 38.3, and Proposition 38.4 does not cover a schedule containing one. Said cleanly, there are two semantics and the primitive is sound under exactly one of them: under a relaxed semantics in which `+` is associative it is sound for every reduction; under IEEE-754 it is sound for the integer cases and for `min`/`max`, which select an operand rather than round one, and not for float `+` or `*`. The compiler runs it under the relaxed one, and unlike Chapter 20 it does not make the user ask.

The preconditions:

```
=== what rfactor refuses ===

  factor 1   rfactor: factor 1 must divide reduction extent 8 with 1 < factor < 8
  factor 3   rfactor: factor 3 must divide reduction extent 8 with 1 < factor < 8
  factor 8   rfactor: factor 8 must divide reduction extent 8 with 1 < factor < 8
  factor 16  rfactor: factor 16 must divide reduction extent 8 with 1 < factor < 8
```

`1 < f < K` and `f ∣ K`. Unlike Chapter 40's `split`, `rfactor` will not round up and guard — a partial buffer with a ragged last group would need the guard *and* an identity element in the unused slots, and the second is exactly what §41.4 showed the primitive not to have.

Finally, `argmax`:

```
  rfactor on argmax: rfactor: block 'arg_acc_1' body is not a single accumulating store
```

An argmax reduction is associative and commutative — the operator is "take the pair with the larger value" — and it is refused, because its block writes two buffers with two conditional stores and does not match Definition 41.1. The algebraic test is never reached. That is the right refusal for the wrong reason, and it is why the four-element operator set has never had to grow.

## 41.6 Lab — caches, inlining, and a primitive that cannot fire

```bash
node docs/part7/ch41-memory-and-reduction-primitives/labs/02-cache-inline-decompose.mjs
```

`x.mul(x).add(1.0)` lowers to two blocks and an intermediate. `computeInline('mul_block_0')`:

```
=== computeInline("mul_block_0") ===
  for i0_10 in 0..2 {
    for i1_11 in 0..3 {
      block add_block_1 {
        reads([buf_4[...], buf_1[...]])
        writes([buf_3[...]])
        buf_3[v0_12, v1_13] = ((buf_1[v0_12, v1_13] * buf_1[v0_12, v1_13]) + buf_4[])
```

The producer nest is gone, its store value is now a subexpression, and — worth noting for Chapter 33 — the *declared read set* was rewritten rather than merely pruned: `{buf_5, buf_4}` became `{buf_4, buf_1}`. Both halves are needed and only one is obvious. Dropping `buf_5` is the visible half; inheriting `buf_1` is the half that is easy to forget, and forgetting it is worse than leaving the declaration stale — it makes the declaration *wrong*, because the block now loads a buffer it does not declare, and inlining is precisely the primitive that introduces such a load. `retargetBufferReads` ([`schedule.ts:161`](../../../src/compiler/schedule/schedule.ts)) does both, and is one of the very few places in the compiler that maintains that declaration at all.

Also visible: one load of `buf_5` became two loads of `buf_1`. Inlining traded traffic for recomputation, and nothing in the primitive counts the trade — `InlineReindexPass` bounds it separately, refusing to inline a block whose output is loaded more than once unless the block is a pure reindex ([`inline_reindex_pass.ts:45`](../../../src/compiler/passes/schedule/inline_reindex_pass.ts)).

`cacheRead` on a matmul stages the whole of `buf_1`:

```
    for buf_1_matmul_1_cache_i0 in 0..4 {
      for buf_1_matmul_1_cache_i1 in 0..6 {
        block buf_1_matmul_1_cache_fill {
          buf_1_matmul_1_cache[…] = buf_1[…]
    for ls0_6 in 0..4 {
        …
            buf_5[…] = (buf_5[…] + (buf_1_matmul_1_cache[vls0_9, vc0_11] * buf_3[vc0_11, vrs0_10]))
```

and `cacheWrite` the mirror image. On CPU all three programs agree to the bit:

```
  baseline   -42 -21 0 21 42 -114 -57 0 57 114 -186 -93 0 93 186 -258 -129 0 129 258
  cacheRead  -42 -21 0 21 42 -114 -57 0 57 114 -186 -93 0 93 186 -258 -129 0 129 258
  cacheWrite -42 -21 0 21 42 -114 -57 0 57 114 -186 -93 0 93 186 -258 -129 0 129 258
```

The same `cacheWrite` schedule compiled for CUDA does not:

```
      float buf_5_matmul_1_cachew[20];
      buf_5_matmul_1_cachew[…] = (buf_5_matmul_1_cachew[…] + (buf_1[…] * buf_3[…]));
      const int buf_5_matmul_1_cachew_o0 = buf_5_matmul_1_cachew_o0;
```

Two defects in three lines. The cache is declared and not zeroed, and the block accumulates into it — the zeroing that `matmul_init_0` performs still targets `buf_5`, which the flush then overwrites, so the initialisation is dead and the accumulator starts at whatever the stack held. It works on CPU only because a fresh `Float32Array` is zero-filled. And the flush block emits `const int x = x;`, because `cacheWrite` reuses one `VariableNode` as both the loop variable and the block's iteration variable ([`schedule.ts:767`](../../../src/compiler/schedule/schedule.ts)), and the CUDA backend declares block iteration variables as `const` locals.

Neither is caught by anything and neither matters today, for the reason Chapter 38 gave: `cacheWrite` has no caller in `src/`. The primitives most likely to be wrong are exactly the ones nothing runs.

Last, `decomposeReduction`:

```
  decomposeReduction: block 'reduce_acc_1' has no initBody
  after rfactor, on 'reduce_acc_1_rf_p': accepted
  blocks now: reduce_init_0, reduce_acc_1_rf_c, reduce_acc_1_rf_p_init, reduce_acc_1_rf_p_upd
```

It cannot fire on a block the lowering rules produced, and it can on a block `rfactor` produced, because `rfactor` is the only writer of `initBody` in the compiler. That is not a curiosity: `createSSRSRSTilingSketch` ([`tiling.ts:131`](../../../src/compiler/autotune/tiling.ts)) opens with `schedule.decomposeReduction(blockName)`, so **the SSRSRS tiling structure — the standard multi-level tiling shape for a reduction, and the one Part VIII's search space nominally offers — cannot be applied to any block this compiler lowers.** The sketch throws, the autotuner catches, and the search proceeds without it.

## 41.7 Traps and limits

- **`rfactor`'s identity element comes from the operator and the dtype, not from the block.** [`schedule.ts:656`](../../../src/compiler/schedule/schedule.ts) reads `block.initBody` when there is one and otherwise derives the identity from `reduceInitValue` (§41.6). No lowering rule sets `initBody`, so the derived path is the one every rfactor in a real compilation takes — and since those are all `+` accumulations, a wrong default there would be invisible until someone called `rfactor` on a product by hand. `rfactor` is public, so that is not a defence.
- **`rfactor` reassociates without a licence.** There is no `fastMath` argument and no dtype check, unlike Chapter 20's algebraic patterns, which take both. Part VIII's `createRfactorSketch` will offer it for any single-axis reduction with a divisor.
- **`decomposeReduction` cannot be applied to lowered TIR.** §41.6, and it takes the SSRSRS tiling sketch with it.
- **`cacheWrite` leaves the cache uninitialised when the accumulator's init is in a sibling block.** §41.6. Correct on CPU by accident, wrong on CUDA.
- **`cacheRead`/`cacheWrite` bind an iteration variable to itself.** `new BlockRealizeNode(v, v)` with the same object on both sides ([`schedule.ts:767`](../../../src/compiler/schedule/schedule.ts), [`schedule.ts:1016`](../../../src/compiler/schedule/schedule.ts)) produces `const int x = x;` in CUDA. The TIR verifier's scoping check (Chapter 33) passes it because the name *is* in scope — it binds itself.
- **The cache is always the whole buffer.** No region inference, so `cacheRead` is only useful with a `computeAt` that shrinks it, and `computeAt` requires domains to match exactly, so the combination that would be useful does not typecheck. Neither primitive has a caller.
- **`_relocateBlockToLoop` mutates before it can fail cleanly.** The two checks run first, which is right, but the splice itself renames variables in a *clone* and then removes the original nest ([`schedule.ts:962`](../../../src/compiler/schedule/schedule.ts)); if the target loop's body is neither a `SeqNode` nor replaceable, the original is already gone.
- **`computeInline`'s guard list is the best legality argument in the file, and it is guarded by a bare `catch`.** Its one production caller swallows every one of the seven distinct refusals ([`inline_reindex_pass.ts:87`](../../../src/compiler/passes/schedule/inline_reindex_pass.ts)), so a block that could not be inlined and a block that was never a candidate are indistinguishable in the trace.

## 41.8 Read the tests

- [`tests/compiler/autotune/rfactor.test.js`](../../../tests/compiler/autotune/rfactor.test.js) — rfactor over four shapes and three factors, each compiled and compared against a scalar reference, plus `cacheRead`, `cacheWrite` and `fuseConsumer`. This is the file that pins Theorem 41.2 in the cases where it holds exactly.
- [`tests/compiler/schedule/compute-inline.test.js`](../../../tests/compiler/schedule/compute-inline.test.js) — one test per clause of Definition 41.5, each asserting the refusal, plus the invertible-affine case that succeeds. It is the executable form of Proposition 41.6.
- [`tests/compiler/schedule/trace.test.js`](../../../tests/compiler/schedule/trace.test.js) — `setScope` and the other field-setting primitives, recorded and replayed.

---

**Next:** [Chapter 42 — Legality](../ch42-legality/README.md), which collects the checks the last three chapters have been deferring, and finds three of them disagreeing about the same loop.
