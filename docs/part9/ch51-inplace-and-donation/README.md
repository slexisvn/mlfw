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

> **Definition 51.1 (In-place candidate).** **(stated here)** `(S, D)` is an *in-place candidate* for a block `B` if `S` is read by `B`, `D` is written by `B`, `S ≠ D`, `S` and `D` have the same shape, strides, dtype and scope, and neither is a function parameter.

The layout and dtype clauses are not fussiness: the bytes have to line up, and where element `(i, j)` lands is the shape and the strides together — two buffers of the same shape under different strides put `S[i, j]` and `D[i, j]` at different addresses, and Theorem 51.3's proof needs them at the same one. The parameter clause is Chapter 49 §49.4 — a parameter's storage belongs to the caller, and §51.4 is about the one way the caller can waive that.

> **Definition 51.2 (Aliasing a candidate).** **(stated here)** To *alias* `(S, D)` is to produce the program in which `D` names the same bytes as `S`.

> **Theorem 51.3 (Index equality suffices).** **(stated here)** Let block `B` contain exactly one store to `D`, at index expression `w`, and let every load of `S` anywhere in `B` be at an index expression equal to `w`. Suppose further that no statement after `B` reads `S`. Then aliasing `(S, D)` preserves the results of the whole program.
>
> *Proof.* Consider one execution of `B`'s body at some point of its iteration space, where `w` evaluates to the index `k`. The body reads `S[k]` — possibly several times, all at the same `k` by hypothesis — and performs one store, to `D[k]`. Under aliasing both name the same location, so the execution reads location `k`, computes, and writes location `k`. No other location is touched. Therefore the execution at `k` cannot affect the value read by the execution at any `k' ≠ k`, and each execution reads the same value it read before aliasing, so each writes the same value. `D` therefore ends with the contents it had in the unaliased program. `S`'s contents are destroyed, and by hypothesis nothing after `B` reads them. ∎

The hypothesis is doing all the work, and it is worth seeing it fail.

> **Counterexample 51.4.** Drop the index-equality clause and let the block be `D[i] = S[i] * S[0]` over `i ∈ [0, n)`. Every element of the output needs `S[0]`. Aliased, the iteration at `i = 0` writes `S[0]`, and every later iteration multiplies by the value just written rather than the original. For `S = [2, 3, 4]` the unaliased result is `[4, 6, 8]`; aliased, `S[0]` becomes `4` and the result is `[4, 12, 16]`. Nothing about the *lifetimes* changed — `S` is still dead after this block — so no liveness argument could have caught it. The failure is entirely in the access pattern.

Two remarks about what Theorem 51.3 does *not* say, because the gap between sufficient and necessary is where a compiler's conservatism lives.

**Index equality is sufficient, not necessary.** Consider `D[i] = S[i+1]` executed with `i` ascending. Aliased, iteration `i` writes location `i` after location `i+1` has been read, and location `i` was already consumed at iteration `i−1`. It is safe — but only because the loop runs upward, and a block does not record its execution order. A compiler that wanted to exploit this would need Chapter 36's dependence direction vectors, not a syntactic index comparison. Refusing it is the cheap and sound choice.

**"Nothing after `B` reads `S`" is a liveness condition and it is separate.** Theorem 51.3 needs both: the access pattern makes the aliasing safe *within* the block, and liveness makes it safe *after*. Neither implies the other, and the implementation checks them in different places.

## 51.4 In mlfw: three filters and a walk

[`inplace_analysis.ts:22`](../../../src/compiler/passes/memory/inplace_analysis.ts) is Definition 51.1 followed by Theorem 51.3's two hypotheses, in that order — cheap tests first.

### Definition 51.1

[`inplace_analysis.ts:51`](../../../src/compiler/passes/memory/inplace_analysis.ts) onwards, for each write of the block against each read:

```ts
        if (livenessResult.isParam(dstBuf) && !allowedDonationParams.has(dstBuf)) continue;
        ...
          if (!layoutsMatch(srcBuf, dstBuf)) continue;
          if (srcBuf.dtype !== dstBuf.dtype) continue;
          if (srcBuf.scope !== dstBuf.scope) continue;
```

`allowedDonationParams` is **donation**: the caller's promise that it will not look at this argument again, which lifts the parameter exclusion for exactly the buffers named. It is the mechanism behind every framework's `donate_argnums`-shaped API, and it is the only way an optimizer can overwrite something it does not own. Nothing in `src/` ever passes a non-empty set: the only caller, [`memory_planning.ts:112`](../../../src/compiler/passes/memory/memory_planning.ts), calls `InplaceAnalysis.analyze` with two arguments, so the third defaults to empty. The parameter is exercised in exactly one place in the tree, a unit test that builds the donated set by hand ([`spmd-donation.test.js:60`](../../../tests/compiler/passes/memory/spmd-donation.test.js)). §51.6 returns to what that costs.

### The liveness hypothesis

[`inplace_analysis.ts:67`](../../../src/compiler/passes/memory/inplace_analysis.ts):

```ts
          if (srcInterval.lastUse <= dstInterval.firstUse) {
            const lastRead = bufferLastReadIdx.get(srcBuf);
            if (lastRead === undefined || lastRead <= currentIdx) {
```

Two conditions, and they are not redundant. The first is about intervals: `S` must not be live past the point `D` starts. The second is sharper — a table built at [`inplace_analysis.ts:33`](../../../src/compiler/passes/memory/inplace_analysis.ts) of the last block that *reads* each buffer, checked against the current block. Writes do not count, and that is the point: a buffer written after this block is being redefined, not consumed, so it does not stop the aliasing.

### The access-pattern hypothesis

`isInplaceComputeSafe` ([`inplace_analysis.ts:154`](../../../src/compiler/passes/memory/inplace_analysis.ts)) is Theorem 51.3's other clause. It collects every store to `D` and every load of `S` in the block's body and init body, then:

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

The comparison is not syntactic equality. `exprEqual` ([`inplace_analysis.ts:112`](../../../src/compiler/passes/memory/inplace_analysis.ts)) first tries `toLinearForm` on both sides and compares the affine forms — same offset, same coefficient on each variable — so `i` and `i + 0`, or `2*i` and `i + i`, are recognised as the same index. Only when neither side is affine does it fall back to structural recursion. That matters because lowering emits index arithmetic in whatever shape the rule found convenient, and a syntactic comparison would refuse safe candidates at random.

There is one deliberate escape hatch, at [`inplace_analysis.ts:188`](../../../src/compiler/passes/memory/inplace_analysis.ts): if some load of `S` sits outside the store's own expression tree, the candidate is still accepted when the store is a *pure copy* `D[w] = S[w]`. A copy cannot be corrupted by aliasing, whatever else the block mentions.

### One source, one destination

[`inplace_analysis.ts:56`](../../../src/compiler/passes/memory/inplace_analysis.ts) keeps an `alreadyAliased` set and `break`s out of the read loop on the first accepted candidate. So a buffer is donated to at most one destination, and a block writing two outputs cannot alias both onto the same input. That is necessary rather than conservative: two destinations aliased onto one source would be two different values in one place.

### From a candidate to an address

An accepted candidate is a *claim* that two buffers may be one. Three things then have to agree, and they live in [`memory_planning.ts`](../../../src/compiler/passes/memory/memory_planning.ts).

**The planner filters the analysis before anything acts on it** ([`memory_planning.ts:112`](../../../src/compiler/passes/memory/memory_planning.ts)):

```ts
      inplaceCandidates = InplaceAnalysis.analyze(primFunc, livenessResult)
        .filter((c) => donatable.has(c.srcBuffer) && shareable.has(c.dstBuffer));
```

The two sets come from `storageRoles` ([`memory_planning.ts:224`](../../../src/compiler/passes/memory/memory_planning.ts)) and they are deliberately not the same set. A buffer is *donatable* — its bytes may be taken over once it is dead — if it is a sized temporary the pass owns. A buffer is *shareable* — it may move into bytes that were somebody else's — only if it additionally does not need **defined storage**, which [`buffer_dataflow.ts:28`](../../../src/compiler/analysis/buffer_dataflow.ts) reports for a buffer whose store reads what it writes, which is read before any write, or which no unconditional write is proven to cover before its first read. The asymmetry is the whole point. A matmul accumulates into its output, so that buffer needs defined storage and can never be *moved*; but once the contraction has finished and the `relu` that consumes it is the last reader, nothing stops the `relu`'s output from being computed into it. Refusing to donate an accumulator would refuse the commonest in-place opportunity a real network has.

**The assignment gives the destination the source's address** ([`buffer_assignment.ts:169`](../../../src/compiler/passes/memory/buffer_assignment.ts)), and extends the source's effective last use to cover the destination's, so the packer of Chapter 50 keeps everything else out of those bytes.

**The rewrite makes them one buffer.** On the default path the destination is added to the same `aliasMap` that carries the planner's ordinary slot reuse — the non-pool way an assignment is materialized, one buffer taking over the name of another whose lifetime has ended ([`memory_planning.ts:181`](../../../src/compiler/passes/memory/memory_planning.ts)) — `resolveAliasChains` flattens `D → S → R` where the source is itself reusing somebody's slot, and `rewriteBufferAliases` replaces every mention of `D` in the IR with `S`. With `poolAllocation` on there is nothing to rewrite: both buffers are given the same `poolByteOffset` ([`memory_planning.ts:171`](../../../src/compiler/passes/memory/memory_planning.ts)) and become two views on one range of the arena. Either way the destination is no longer allocated, which is where the bytes are actually saved.

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

**The first row is not the number §51.1 predicts, and the gap is the whole of §51.6's donation trap arriving early.** `t.mul(2).add(1).relu()` is three operations, so it has two adjacent producer–consumer pairs, and §51.1 argued every such pair is an in-place opportunity. One candidate is found, not two. The missing one is the *last* pair: the `relu` writes the function's return buffer, and a return buffer is a **parameter** — excluded by Definition 51.1's final clause, because its storage belongs to the caller. Count it through on the second half's six-operation chain and the arithmetic closes: five adjacent pairs, minus the one that writes the return buffer, is the four candidates the table below reports.

So the shape of the loss is: **a chain of `n` operations offers `n−2` candidates, not `n−1`, and the one it never offers is the last.** That is exactly the buffer donation exists for, and §51.6 is where the fact that nothing can request it becomes the trap.

The two zeros are the theory, each failing a different hypothesis. **`matmul then relu`** fails the access pattern: a contraction reads `A[i, k]` and `B[k, j]` while writing `C[i, j]`, and no index comparison can make those equal. **`an intermediate read twice`** — `const u = a.mul(2); return u.add(u)` — fails liveness: `u` is read by a block after the one that produced it, so `bufferLastReadIdx` exceeds the current index and the candidate is refused. Chapter 49's register would have shown the same thing as an interval that reaches too far.

The second half measures both of Part IX's numbers at once — what the plan claims and what the emitted program allocates — across the two materialization paths.

```
  configuration                                 plan says  candidates  allocates   max err
  poolAllocation=false inplaceReuse=true             4416           4       4096      0e+0
  poolAllocation=false inplaceReuse=false            8448           0       8192      0e+0
  poolAllocation=true  inplaceReuse=true             4416           4       4356      0e+0
  poolAllocation=true  inplaceReuse=false            8448           0       8388      0e+0
```

Read the first two rows together. The first is the shipped default. It finds four in-place candidates, the reported peak falls from 8,448 bytes to 4,416 — a clean halving, exactly what §51.1 promised — and the program it emits allocates 4,096 bytes against 8,192 with the feature switched off. Six operations, and the generated source declares exactly one full-size array where it used to declare two.

```
  the shipped default plans for 4416 bytes and allocates 4096
  turning the feature off plans for 8448 bytes and allocates 8192
  so the reported peak falls by 1.91x and the bytes the program allocates by 2.00x
  every configuration computes the same numbers: max error 0
```

**The two columns move together, and the emitted one moves further.** The plan is quoted over aligned slots and counts the chain's five scalar constants at 64 bytes each; the emitted program folds those constants into the expressions that use them and allocates only the two full-size buffers, so it comes in under the plan rather than over it. The pooled rows are the same story with the arena's own bookkeeping added. Every configuration is bit-identical, which is Theorem 51.3 checked rather than assumed.

Both numbers are printed because they can disagree, and not hypothetically — §51.6 has the version of this pass in which they moved in opposite directions. The lab is the regression test for it: `plan says` and `allocates` are read against each other, and against the same program with the feature off.

**Try this.** Lengthen the chain and watch both columns stay flat while `candidates` grows: an *n*-operation chain needs two buffers without this pass and one with it, whatever *n* is. Then switch fusion back on and watch the whole table collapse to the constants, which is Chapter 49 §49.5's point arriving again — there are no intermediates left to overwrite.

## 51.6 Traps and limits

### A candidate is not a saving until something rewrites the program

The analysis of §51.4 proves a pair may share; three separate pieces of code then have to act on that, and each of them can silently decline. It is worth knowing which does what, because the failure mode is not a wrong answer but a plan that describes a program nobody emitted:

1. `BufferAssignment.assign` records the pair and gives the destination the source's offset ([`buffer_assignment.ts:169`](../../../src/compiler/passes/memory/buffer_assignment.ts)). **The reported peak drops here** — two buffers, one address, counted once. Nothing has been emitted yet.
2. The destination is aliased onto the source, either by the IR rewrite or by a shared pool offset (§51.4). **The bytes are saved here**, and only here.
3. `_insertAllocations` allocates every temporary that is not an alias ([`memory_planning.ts:150`](../../../src/compiler/passes/memory/memory_planning.ts)), so a destination that step 2 declined still gets storage of its own rather than none.

An earlier version of this pass had step 1 and step 3 keyed on the same flag and no step 2: the destination was recorded as sharing, skipped by the allocator, materialized by nobody, and excluded from the slot reuse that would otherwise have covered it — so the plan halved while the emitted program allocated 2.5× more. The lesson generalizes past this chapter and is the reason §51.5 prints both numbers: **an optimization that is recorded in a plan and not applied to a program is worse than one that never ran**, because the plan is what the trace reports.

### The rest

- **The reported `peakMemory` is still a claim about a plan, not about the program.** `MemoryPlan.getReport` sums the pools of an assignment ([`memory_planning.ts:81`](../../../src/compiler/passes/memory/memory_planning.ts)); it is what the trace event carries and what every figure in this part is quoted from unless a lab says otherwise. The two agree across every configuration §51.5 measures, and they are still different functions: the plan pads each buffer to the alignment and counts constants the backend folds into expressions, and on a GPU it sums pools that are not one address space (Chapter 50 §50.6). Read it as an upper bound that the emitted program is expected to meet, not as the allocation itself.
- **`totalInplace` counts candidates, not bytes.** [`memory_planning.ts:78`](../../../src/compiler/passes/memory/memory_planning.ts) reports the length of the filtered candidate list. On the default path every one of them is materialized; under `poolAllocation` a destination outside `global` scope is not, because the pool only places global buffers ([`memory_planning.ts:167`](../../../src/compiler/passes/memory/memory_planning.ts)). And a candidate saves its own size only when the source is not already sharing a slot with a third buffer, in which case the two savings are the same bytes counted once. The number is a count of decisions, and the bytes are in `peakMemory`.
- **Donation at this level is available and unreachable from `compile()`.** `allowedDonationParams` is a real parameter with real semantics, and the default compile path passes nothing for it ([`memory_planning.ts:112`](../../../src/compiler/passes/memory/memory_planning.ts)). There is no option on `CompilerConfig` that reaches it. A user who knows an input is dead after the call has no way to say so, and the buffer that is most often dead after a call — the one holding the previous layer's activations — is exactly a parameter. Read that as a statement about *this* mechanism and not about the compiler: a second donation exists one level up, over the executor plan's slots rather than a `PrimFunc`'s buffers, and it does run — [`plan_buffer_assignment.ts`](../../../src/compiler/passes/memory/plan_buffer_assignment.ts) reuses slots with disjoint lifetimes and donates a dying elementwise input to its output, gated by `memory.planReuse` and `memory.planDonation` and enabled by default ([`compiler.ts:383`](../../../src/compiler/pipeline/compiler.ts)), whenever a graph compiles to a multi-step plan rather than a single kernel. It infers its donations instead of being told them, so it does not close the gap this bullet describes; it does mean "the compiler cannot donate" would be the wrong sentence. [Part IX's opening](../README.md) says where that machinery is described instead.
- **The analysis is per-block, so it never crosses a fusion boundary in the direction that matters.** Candidates are drawn from one block's own read and write sets. Two adjacent blocks where the first's output is the second's only input are the commonest in-place opportunity in a chain, and they are invisible here; only fusing them into one block (Chapter 24) exposes the opportunity, at which point fusion has already removed the buffer.
- **Equal layouts is stricter than equal sizes.** `layoutsMatch` ([`inplace_analysis.ts:95`](../../../src/compiler/passes/memory/inplace_analysis.ts)) compares shape and strides dimension by dimension, so a `[4, 6]` buffer and a `[24]` buffer of the same dtype — identical byte counts, identical addresses under row-major — are not candidates. Sound, and it refuses the reshape-shaped case that in-place reuse would help most.
- **Under a dynamic shape, `layoutsMatch` compares two unknowns and calls them equal.** A dynamic dimension is the sentinel `DYNAMIC = -1` ([`types.ts:63`](../../../src/compiler/ir/graph/types.ts)), and `-1 !== -1` is false, so two buffers declared `[?, ?]` match on shape whatever their extents turn out to be. Candidates are duly produced — the lab's chain finds the same one with `dynamic_shapes` on as off. Nothing acts on them: a buffer of unknown size takes the `size < 0` path in the assignment (Chapter 50 §50.6), which gives it offset 0, size 0 and exclusion from the pool, and the reported peak then describes only the statically-sized scalars. The shape test is the weakest link in Definition 51.1 and it is currently load-bearing for nothing.

## 51.7 Read the tests

- [`tests/compiler/passes/memory/inplace.test.js`](../../../tests/compiler/passes/memory/inplace.test.js) — the candidate conditions one at a time: layout and dtype agreement, the liveness test, and the access-pattern test that refuses Counterexample 51.4's shape.
- [`tests/compiler/passes/memory/spmd-donation.test.js`](../../../tests/compiler/passes/memory/spmd-donation.test.js) — `allowedDonationParams` exercised, which is the only place in the tree where a parameter is legally overwritten.
- [`tests/compiler/passes/memory/planning.test.js`](../../../tests/compiler/passes/memory/planning.test.js) — the planner end to end: that a chain of sequential temporaries plans down to one buffer's worth of bytes, and that the in-place destination is the one temporary no `AllocateNode` is emitted for, because it has become its source.

---

**Next:** [Chapter 52 — Scheduling to lower peak memory](../ch52-scheduling-for-peak/README.md), which stops taking the program's order as given.
