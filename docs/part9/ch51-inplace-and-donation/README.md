# Chapter 51 — In-place reuse and donation

Chapter 50 let two buffers share an address when their lifetimes did not overlap. That is reuse *across* time: one buffer finishes, another begins. This chapter is about the harder and more valuable case — two buffers sharing an address while a single operation is reading one and writing the other.

## 51.1 The problem: the staircase still allocates twice

Look again at Chapter 49's staircase. `buf_8` is live over `[0, 1]` and `buf_13` over `[1, 2]`. They overlap at index 1, which is the block that reads `buf_8` and writes `buf_13`, so Chapter 50 must give them different bytes. Every adjacent pair in a chain is in this position.

So an *n*-operation elementwise chain, however cleverly packed, needs two full-sized buffers: the one being read and the one being written. That is the floor Chapter 50 can reach and cannot go below.

But look at what the block actually does:

```
buf_13[v0, v1] = (buf_8[v0, v1] + buf_5[])
```

It reads element `(v0, v1)` of one buffer and writes element `(v0, v1)` of another. Once that element is read it is never needed again. There is no reason the result could not go straight back where the input came from — and if it did, the chain would need *one* buffer instead of two.

That is in-place reuse, and it halves the floor. Getting it wrong silently corrupts data, which is why it has the most careful legality condition in this part.

## 51.2 Intuition: repainting a wall

Chapter 50's reuse is a hotel room changing occupants: the first guest leaves, the room is free, the second guest moves in. In-place reuse is repainting a wall while you are still reading what is written on it.

That works if you read each patch immediately before you paint it, and never need to look at a patch you have already painted. It fails the moment some later patch needs to read something you painted over.

So the question is never "is this buffer dead?" — it is "*in what order* does this operation touch the elements, and does it ever look back?" An operation that reads element `i` and writes element `i` never looks back. An operation that reads element `0` for every output — a broadcast — looks back at every step but the first.

## 51.3 Theory

Fix a block that reads a buffer `S` and writes a buffer `D`, and consider replacing every mention of `D` with `S`.

> **Definition 51.1 (In-place candidate).** `(S, D)` is an *in-place candidate* for a block `B` if `S` is read by `B`, `D` is written by `B`, `S ≠ D`, `S` and `D` have the same shape, dtype and scope, and neither is a function parameter.

The shape and dtype clauses are not fussiness: the bytes have to line up, and a buffer's size is its shape times its element width. The parameter clause is Chapter 49 §49.4 — a parameter's storage belongs to the caller, and §51.4 is about the one way the caller can waive that.

> **Definition 51.2 (Aliasing a candidate).** To *alias* `(S, D)` is to produce the program in which `D` names the same bytes as `S`.

> **Theorem 51.3 (Index equality suffices, stated here).** Let block `B` contain exactly one store to `D`, at index expression `w`, and let every load of `S` anywhere in `B` be at an index expression equal to `w`. Suppose further that no statement after `B` reads `S`. Then aliasing `(S, D)` preserves the results of the whole program.
>
> *Proof.* Consider one execution of `B`'s body at some point of its iteration space, where `w` evaluates to the index `k`. The body reads `S[k]` — possibly several times, all at the same `k` by hypothesis — and performs one store, to `D[k]`. Under aliasing both name the same location, so the execution reads location `k`, computes, and writes location `k`. No other location is touched. Therefore the execution at `k` cannot affect the value read by the execution at any `k' ≠ k`, and each execution reads the same value it read before aliasing, so each writes the same value. `D` therefore ends with the contents it had in the unaliased program. `S`'s contents are destroyed, and by hypothesis nothing after `B` reads them. ∎

The hypothesis is doing all the work, and it is worth seeing it fail.

> **Counterexample 51.4.** Drop the index-equality clause and let the block be `D[i] = S[i] * S[0]` over `i ∈ [0, n)`. Every element of the output needs `S[0]`. Aliased, the iteration at `i = 0` writes `S[0]`, and every later iteration multiplies by the value just written rather than the original. For `S = [2, 3, 4]` the unaliased result is `[4, 6, 8]`; aliased, `S[0]` becomes `4` and the result is `[4, 12, 16]`. Nothing about the *lifetimes* changed — `S` is still dead after this block — so no liveness argument could have caught it. The failure is entirely in the access pattern.

Two remarks about what Theorem 51.3 does *not* say, because the gap between sufficient and necessary is where a compiler's conservatism lives.

**Index equality is sufficient, not necessary.** Consider `D[i] = S[i+1]` executed with `i` ascending. Aliased, iteration `i` writes location `i` after location `i+1` has been read, and location `i` was already consumed at iteration `i−1`. It is safe — but only because the loop runs upward, and a block does not record its execution order. A compiler that wanted to exploit this would need Chapter 36's dependence direction vectors, not a syntactic index comparison. Refusing it is the cheap and sound choice.

**"Nothing after `B` reads `S`" is a liveness condition and it is separate.** Theorem 51.3 needs both: the access pattern makes the aliasing safe *within* the block, and liveness makes it safe *after*. Neither implies the other, and the implementation checks them in different places.

## 51.4 In mlfw: three filters and a walk

[`inplace_analysis.ts:21`](../../../src/compiler/passes/memory/inplace_analysis.ts) is Definition 51.1 followed by Theorem 51.3's two hypotheses, in that order — cheap tests first.

### Definition 51.1

[`inplace_analysis.ts:50`](../../../src/compiler/passes/memory/inplace_analysis.ts) onwards, for each write of the block against each read:

```ts
        if (livenessResult.isParam(dstBuf) && !allowedDonationParams.has(dstBuf)) continue;
        ...
          if (!shapesMatch(srcBuf, dstBuf)) continue;
          if (srcBuf.dtype !== dstBuf.dtype) continue;
          if (srcBuf.scope !== dstBuf.scope) continue;
```

`allowedDonationParams` is **donation**: the caller's promise that it will not look at this argument again, which lifts the parameter exclusion for exactly the buffers named. It is the mechanism behind every framework's `donate_argnums`-shaped API, and it is the only way an optimizer can overwrite something it does not own. Nothing in the default compile path passes a non-empty set — [`memory_planning.ts:100`](../../../src/compiler/passes/memory/memory_planning.ts) calls `InplaceAnalysis.analyze` with two arguments, so the third defaults to the empty set — and the SPMD path is where it is exercised.

### The liveness hypothesis

[`inplace_analysis.ts:66`](../../../src/compiler/passes/memory/inplace_analysis.ts):

```ts
          if (srcInterval.lastUse <= dstInterval.firstUse) {
            const lastRead = bufferLastReadIdx.get(srcBuf);
            if (lastRead === undefined || lastRead <= currentIdx) {
```

Two conditions, and they are not redundant. The first is about intervals: `S` must not be live past the point `D` starts. The second is sharper — a table built at [`inplace_analysis.ts:32`](../../../src/compiler/passes/memory/inplace_analysis.ts) of the last block that *reads* each buffer, checked against the current block. Writes do not count, and that is the point: a buffer written after this block is being redefined, not consumed, so it does not stop the aliasing.

### The access-pattern hypothesis

`isInplaceComputeSafe` ([`inplace_analysis.ts:149`](../../../src/compiler/passes/memory/inplace_analysis.ts)) is Theorem 51.3's other clause. It collects every store to `D` and every load of `S` in the block's body and init body, then:

```ts
  if (dstStores.length !== 1) return false;
  if (srcLoads.length === 0) return false;

  const store = dstStores[0];
  const writeIndex = store.indices;
  for (const load of srcLoads) {
    if (!indexListEqual(load.indices, writeIndex)) return false;
  }
```

Exactly one store, at least one load, and every load at the same index as the store. Counterexample 51.4 is refused at the loop: the load of `S[0]` has index `0`, the store has index `i`, and `indexListEqual` says no.

The comparison is not syntactic equality. `exprEqual` ([`inplace_analysis.ts:107`](../../../src/compiler/passes/memory/inplace_analysis.ts)) first tries `toLinearForm` on both sides and compares the affine forms — same offset, same coefficient on each variable — so `i` and `i + 0`, or `2*i` and `i + i`, are recognised as the same index. Only when neither side is affine does it fall back to structural recursion. That matters because lowering emits index arithmetic in whatever shape the rule found convenient, and a syntactic comparison would refuse safe candidates at random.

There is one deliberate escape hatch, at [`inplace_analysis.ts:183`](../../../src/compiler/passes/memory/inplace_analysis.ts): if some load of `S` sits outside the store's own expression tree, the candidate is still accepted when the store is a *pure copy* `D[w] = S[w]`. A copy cannot be corrupted by aliasing, whatever else the block mentions.

### One source, one destination

[`inplace_analysis.ts:55`](../../../src/compiler/passes/memory/inplace_analysis.ts) keeps an `alreadyAliased` set and `break`s out of the read loop on the first accepted candidate. So a buffer is donated to at most one destination, and a block writing two outputs cannot alias both onto the same input. That is necessary rather than conservative: two destinations aliased onto one source would be two different values in one place.

## 51.5 Lab — overwriting your own input

```bash
node docs/part9/ch51-inplace-and-donation/labs/01-overwriting-your-own-input.mjs
```

The first half asks which programs offer a candidate at all:

```
  elementwise chain            temporaries= 5   in-place candidates=1
  elementwise, two inputs      temporaries= 4   in-place candidates=1
  matmul then relu             temporaries= 2   in-place candidates=0
  an intermediate read twice   temporaries= 2   in-place candidates=0
```

The two zeros are the theory, each failing a different hypothesis. **`matmul then relu`** fails the access pattern: a contraction reads `A[i, k]` and `B[k, j]` while writing `C[i, j]`, and no index comparison can make those equal. **`an intermediate read twice`** — `const u = a.mul(2); return u.add(u)` — fails liveness: `u` is read by a block after the one that produced it, so `bufferLastReadIdx` exceeds the current index and the candidate is refused. Chapter 49's register would have shown the same thing as an interval that reaches too far.

The second half is where this chapter stops being a description of a mechanism and starts being about this implementation.

```
  configuration                                 plan says  candidates  allocates   max err
  poolAllocation=false inplaceReuse=true             4416           4      20480      0e+0
  poolAllocation=false inplaceReuse=false            8448           0       8192      0e+0
  poolAllocation=true  inplaceReuse=true             4416           4      20740      0e+0
  poolAllocation=true  inplaceReuse=false            8448           0       8388      0e+0
```

Read the first two rows together. The first is the shipped default. It finds four in-place candidates, and the reported peak falls from 8,448 bytes to 4,416 — a clean halving, exactly what §51.1 promised. And the program it emits allocates **20,480 bytes**, against 8,192 for the same program with the feature switched off.

```
  the shipped default plans for 4416 bytes and allocates 20480
  turning the feature off plans for 8448 bytes and allocates 8192
  so the reported peak falls by 1.91x while the real allocation rises by 2.50x
  every configuration computes the same numbers: max error 0
```

**The two numbers move in opposite directions.** Nothing here is a wrong answer — every configuration is bit-identical — so this is a performance finding rather than a correctness one. But it is the only place in this book where switching an optimization *on* makes the thing it optimizes 2.5× worse, and §51.6 traces why.

**Try this.** Lengthen the chain and watch the gap widen: each additional operation adds a candidate the plan credits and an allocation the program pays for. Then switch fusion back on and watch both columns collapse, which is Chapter 49 §49.5's point arriving again.

## 51.6 Traps and limits

**Why the two columns disagree.** The analysis in §51.4 is correct and its candidates are real. What is missing is the step that would make them true of the program. Trace one candidate through:

1. `BufferAssignment.assign` records it in `inplaceMap` and gives the destination the source's offset ([`buffer_assignment.ts:169`](../../../src/compiler/passes/memory/buffer_assignment.ts)). **The reported peak drops here** — two buffers, one address, counted once.
2. `_insertAllocations` then skips the destination entirely: `if (assignment.inplaceOf) continue` ([`memory_planning.ts:144`](../../../src/compiler/passes/memory/memory_planning.ts)). No `AllocateNode` is emitted for it.
3. On the default path — `poolAllocation` is `false` — the mechanism that actually materializes sharing is `_buildReuseAliases`, which rewrites the IR so two buffers become one name ([`memory_planning.ts:123`](../../../src/compiler/passes/memory/memory_planning.ts)). It excludes in-place destinations *and* in-place sources ([`memory_planning.ts:178`](../../../src/compiler/passes/memory/memory_planning.ts) and [`:179`](../../../src/compiler/passes/memory/memory_planning.ts)).
4. With `poolAllocation` true, `_assignPoolOffsets` excludes them too ([`memory_planning.ts:162`](../../../src/compiler/passes/memory/memory_planning.ts)).

So an in-place candidate receives no allocation, no pool offset, and no alias — and the backend, meeting a buffer name with none of those, allocates it fresh. Meanwhile the pair has been removed from the one mechanism that would have shared their storage. **The feature does not materialize its own reuse, and it suppresses the reuse that would otherwise have happened.** That is the whole of the 2.5×.

The exclusions in steps 3 and 4 are not mistakes on their own terms — a buffer whose storage is already decided by an in-place assignment should not also be aliased somewhere else. They are correct given that step 2 does what its name says. The gap is that nothing between step 2 and the backend rewrites the destination to *be* the source.

- **The reported `peakMemory` is a claim about a plan, not about the program.** This is the general form of the finding above, and it is worth carrying past this chapter. `MemoryPlan.getReport` sums the pools of an assignment ([`memory_planning.ts:71`](../../../src/compiler/passes/memory/memory_planning.ts)); whether the emitted code honours that assignment depends on which materialization path ran. Chapter 50 §50.5 showed a configuration where the two agree to within alignment padding. This chapter shows one where they differ by 4.6×. Neither number is wrong; they measure different things, and only one of them is what the machine allocates.
- **`totalInplace` counts candidates, not savings.** [`memory_planning.ts:68`](../../../src/compiler/passes/memory/memory_planning.ts) reports `inplaceCandidates.length`. A candidate that `BufferAssignment` later declines to honour — because its source had not been placed yet, Chapter 50 §50.6 — is still counted here. The trace event cannot be read as "this many buffers were saved".
- **Donation is available and unreachable from `compile()`.** `allowedDonationParams` is a real parameter with real semantics, and the default compile path passes nothing for it ([`memory_planning.ts:100`](../../../src/compiler/passes/memory/memory_planning.ts)). There is no option on `CompilerConfig` that reaches it. A user who knows an input is dead after the call has no way to say so, and the buffer that is most often dead after a call — the one holding the previous layer's activations — is exactly a parameter.
- **The analysis is per-block, so it never crosses a fusion boundary in the direction that matters.** Candidates are drawn from one block's own read and write sets. Two adjacent blocks where the first's output is the second's only input are the commonest in-place opportunity in a chain, and they are invisible here; only fusing them into one block (Chapter 24) exposes the opportunity, at which point fusion has already removed the buffer.
- **Equal shapes is stricter than equal sizes.** `shapesMatch` ([`inplace_analysis.ts:86`](../../../src/compiler/passes/memory/inplace_analysis.ts)) compares dimension by dimension, so a `[4, 6]` buffer and a `[24]` buffer of the same dtype — identical byte counts, identical layouts under row-major — are not candidates. Sound, and it refuses the reshape-shaped case that in-place reuse would help most.
- **Under a dynamic shape, `shapesMatch` compares two unknowns and calls them equal.** A dynamic dimension is the sentinel `DYNAMIC = -1` ([`types.ts:63`](../../../src/compiler/ir/graph/types.ts)), and `-1 !== -1` is false, so two buffers declared `[?, ?]` match on shape whatever their extents turn out to be. Candidates are duly produced — the lab's chain finds the same one with `dynamic_shapes` on as off. Nothing acts on them: a buffer of unknown size takes the `size < 0` path in the assignment (Chapter 50 §50.6), which gives it offset 0, size 0 and exclusion from the pool, and the reported peak then describes only the statically-sized scalars. The shape test is the weakest link in Definition 51.1 and it is currently load-bearing for nothing.

## 51.7 Read the tests

- [`tests/compiler/passes/memory/inplace.test.js`](../../../tests/compiler/passes/memory/inplace.test.js) — the candidate conditions one at a time: shape and dtype agreement, the liveness test, and the access-pattern test that refuses Counterexample 51.4's shape.
- [`tests/compiler/passes/memory/spmd-donation.test.js`](../../../tests/compiler/passes/memory/spmd-donation.test.js) — `allowedDonationParams` exercised, which is the only place in the tree where a parameter is legally overwritten.
- [`tests/compiler/passes/memory/planning.test.js`](../../../tests/compiler/passes/memory/planning.test.js) — the planner end to end, including the alias materialization that §51.6 found in-place buffers excluded from.

---

**Next:** [Chapter 52 — Scheduling to lower peak memory](../ch52-scheduling-for-peak/README.md), which stops taking the program's order as given.
