# Chapter 42 — Legality

Chapters 40 and 41 kept deferring one line. `parallelize` calls `loopCarriedDependence`; `reorder` calls `reorderLegality`; `computeAt` calls `_checkRelocationDependences`. This chapter is what those three functions do, what they are allowed to assume, and what happens when the assumption is wrong.

The short version, established in §42.6, is that this compiler answers the legality question in three different places with three different mechanisms, and that only one of the three is a proof.

## 42.1 The problem: the same loop, two answers

```
for c0_8 in 0..6 {
  block matmul_1 {
    buf_5[vls0_9, vrs0_10] = (buf_5[vls0_9, vrs0_10] + (buf_1[vls0_9, vc0_11] * buf_3[vc0_11, vrs0_10]))
  }
}
```

May `c0_8` run in parallel? Every iteration reads and writes `buf_5[m,n]`, so plainly not: six threads incrementing one accumulator is the textbook race.

May `c0_8` be vectorised? Also plainly not, by the same argument — and yet `ReductionWasmRule` does exactly that on purpose ([`rules.ts:492`](../../../src/compiler/schedule/rules.ts)), and the answers come out right, because the WASM backend recognises the accumulator and emits four lanes of partial sums with a horizontal reduce at the end. Which is `rfactor` by 4, performed by the code generator.

So "may this loop run out of order?" is not one question. It is at least three: *is there a dependence*, *does the transformation reverse it*, and *does the consumer of this annotation handle it anyway*. This compiler asks all three, in three different places, and they do not have to agree.

## 42.2 Intuition: distance and direction

Two iterations of a nest conflict if they touch the same memory location and at least one writes. The interesting question is not whether that happens but **which one comes first**, because a transformation is illegal exactly when it makes the later one happen earlier.

Iterations are points of a lattice, ordered lexicographically: `(0,5)` before `(1,0)` because the first coordinate decides. For each conflicting pair, record for each loop level whether the source iteration's coordinate is less than, equal to, or greater than the sink's. That triple of symbols is a **direction vector**, and it summarises an unbounded number of conflicting pairs into one object per subscript pair.

The rule that follows is short enough to remember. A dependence is respected iff its direction vector is *lexicographically positive* — the first non-`=` entry is `<`. Reordering the loops permutes the entries of the vector. A permutation is legal iff no dependence's vector becomes lexicographically negative under it.

That is the classical statement. What this compiler adds is a shortcut around it, and the shortcut is where the interest lies: if the block *declares* an axis independent (Chapter 33's `DataPar`), the analysis is skipped and the answer is "legal". The declaration is faster and, when true, exact. Nothing checks that it is true.

## 42.3 Theory

Fix a perfectly nested chain of loops `L₁,…,L_k` with iteration points ordered lexicographically.

> **Definition 42.1 (Dependence).** **(classical)** Iterations `p` and `q` with `p <_lex q` are *dependent* on buffer `b` if they access a common element of `b` and at least one access is a write. The dependence is RAW, WAR or WAW according to which of the two writes.

> **Definition 42.2 (Direction vector).** **(classical)** For a dependence between `p` and `q`, the direction vector `d ∈ {<, =, >}^k` has `d_i = sign(q_i − p_i)`. A *direction mask* is a subset of `{<, =, >}` at each level, representing every direction the analysis could not rule out.

> **Definition 42.3 (Carried at level `ℓ`).** **(classical)** A dependence is *carried by* loop `L_ℓ` if `d_i = ` `=` for all `i < ℓ` and `d_ℓ ≠ ` `=`. Iterations differing only in levels `≥ ℓ` are then ordered by `L_ℓ`.

> **Theorem 42.4 (Permutation legality).** **(classical)** A permutation `π` of `L₁,…,L_k` preserves semantics iff, for every dependence with direction vector `d`, the permuted vector `π(d)` is lexicographically positive.

*Proof sketch.* The permuted program visits the same iteration points; only the order changes. Two iterations `p <_lex q` in the original are executed in the opposite order in the permuted program exactly when `π(q) <_lex π(p)`, i.e. when `π(d)` is lexicographically negative. Reversing the order of a dependent pair changes which value a read observes or which write survives, so it changes the result; leaving every dependent pair in order leaves every read observing the same write. ∎

> **Corollary 42.5 (Parallelisation).** **(classical)** Loop `L_ℓ` may run its iterations in any order, including concurrently, iff no dependence is carried at level `ℓ`.

*Proof.* Running `L_ℓ` out of order permutes the relative order of iterations that differ at level `ℓ` and agree above it — exactly the pairs Definition 42.3 calls carried. ∎

The compiler's dependence test is not exact and does not claim to be.

> **Proposition 42.6 (Conservative masks).** **(invariant)** Assume the two ends of every dependence are named in execution order, so that direction vectors are lexicographically positive. Then `accessDependence` returns either "independent" or a direction mask that contains the true direction. It never returns a mask excluding a direction that actually occurs.

*Proof sketch.* Four tests of decreasing precision (Chapter 36). One subscript involving one loop level with equal coefficients yields an exact distance; a GCD test rules out whole subscripts; Banerjee bounds then clear directions the loop ranges make impossible, widened back to `d | negate(d)` so the result survives lexicographic normalisation; anything else defaults to `ANY_DIRECTION`. A `null` form — a non-affine subscript — also yields `ANY_DIRECTION` ([`dependence.ts:131`](../../../src/compiler/analysis/dependence.ts)). Every fallback widens the mask, and a wider mask forbids more. ∎

> **The hypothesis is not free.** The two ends of a pair are chosen by *textual* position ([`dependence.ts:251`](../../../src/compiler/analysis/dependence.ts)), which coincides with execution order for a loop-independent dependence and not for a mixed one, so `accessDependence` normalises before returning (Chapter 36 §36.7). Without that step the analyser can report the *reverse* of a true direction — a mask excluding what occurs and including what does not — which turns the failure mode from "refuses a legal transformation" into "permits an illegal one".
>
> Keep the two failure modes apart, because only one of them is designed. **Proposition 42.8's is designed**: a false declaration deliberately overrules a correct analysis, and Counterexample 42.9 is its demonstration — a trade the compiler makes on purpose and prices. An unnormalised direction vector is the other kind: wrong before any declaration is consulted, so Counterexample 42.9's nest would be accepted *even with the axes untyped*. A conservative-by-design analysis and a conservative-in-fact one are different claims, and only the second protects you.

And now the shortcut, which is where soundness stops being derived:

> **Definition 42.7 (Kind policy).** **(stated here)** A *policy* is a set of iteration-variable kinds. Loop `L` *satisfies* policy `P` if every block under `L` has affine bindings, typed iteration variables, no direct use of `L`'s variable in the body, and every axis `L` feeds carries a kind in `P`.

> **Proposition 42.8 (The declaration overrules the analysis).** **(invariant)** If `L` satisfies the policy, `loopCarriedDependence` and `reorderLegality` return "legal" *without regard to the dependences they have already computed*. The result is sound if and only if every declaration involved is true, in the sense of Definition 33.5.

*Proof.* By inspection of [`legality.ts:40`](../../../src/compiler/schedule/legality.ts) and [`legality.ts:48`](../../../src/compiler/schedule/legality.ts): a dependence is found, then `blockAbstractionPermits` is consulted, and a `true` from it returns `null` — legal. Soundness is then exactly Proposition 33.6's hypothesis. ∎

> **Counterexample 42.9.** A block computing `A[i+1, j] = A[i, j+1] + 1` over `i, j ∈ [0,4)` has one RAW dependence with direction `(<, >)`, which reverses under the swap. With the axes untyped, `reorder(j, i)` is refused. With the axes declared `DataPar` — the way every lowering rule emits them — the same `reorder(j, i)` is accepted, and the program computes something else. §42.7 runs both.

## 42.4 In mlfw

### The three policies

[`legality.ts:17`](../../../src/compiler/schedule/legality.ts) is the whole vocabulary:

```ts
export const IterVarPolicy = Object.freeze({
  SPATIAL: new Set([IterVarKind.DATA_PAR]),
  REORDERABLE: new Set([IterVarKind.DATA_PAR, IterVarKind.COMM_REDUCE]),
  ACCUMULABLE: new Set([IterVarKind.DATA_PAR, IterVarKind.COMM_REDUCE]),
});
```

`REORDERABLE` and `ACCUMULABLE` are the same set under two names, which is not an oversight — they answer different questions that happen to have the same answer. `REORDERABLE` says "a reduction axis may be permuted with a spatial one", which is Theorem 42.4 applied to an accumulator: the direction vector is `(=,…,=,*)` and every permutation keeps it positive. `ACCUMULABLE` says "a reduction axis may be vectorised", which is *not* a consequence of Theorem 42.4 — it is the claim that the consumer of the annotation knows how to reduce across lanes. Two claims, one set, and nothing marks which is which.

### `blockAbstractionPermits`

[`legality.ts:23`](../../../src/compiler/schedule/legality.ts), 16 lines, and every early `return false` is a case where the declaration is not trusted:

```ts
function blockAbstractionPermits(state, enclosingLoop, loopVarNames, allowedKinds, byBlock): boolean {
  const sref = state.tree.getSRef(enclosingLoop);
  if (!sref) return false;
  const blocks = sref.childBlocks();
  if (blocks.length === 0) return false;
  for (const blockSRef of blocks) {
    const info = byBlock.get(blockSRef.node as BlockNode);
    if (!info) return false;
    for (const name of loopVarNames) {
      const kinds = info.iterKindsOfLoopVar(name);
      if (!kinds) return false;
      for (const kind of kinds) if (!allowedKinds.has(kind as string)) return false;
    }
  }
  return true;
}
```

Note the quantifier: **every** block under the loop must permit, so a loop enclosing two blocks needs both declarations. And `iterKindsOfLoopVar` (Chapter 33) contributes three more refusals: a non-affine binding, an untyped iteration variable, and a loop variable used directly in the block body rather than through a binding. The last is the one that keeps the abstraction honest — a body that reaches around its own bindings is not described by them.

### `loopCarriedDependence`

[`legality.ts:40`](../../../src/compiler/schedule/legality.ts), six lines, and the order of the last three is the whole design:

```ts
export function loopCarriedDependence(state: ScheduleState, loop: ForNode, allowedKinds: ReadonlySet<string>): string | null {
  const { info, deps } = state.nestAnalysis(loop);
  const dep = carriesDependence(deps, loop, (buffer) => isPrivateToLoop(info, buffer, loop));
  if (!dep) return null;
  if (blockAbstractionPermits(state, loop, [loop.loopVar.name], allowedKinds, info.byBlock)) return null;
  return `loop '${loop.loopVar.name}' carries a ${dep.kind} dependence on buffer '${dep.buffer.name}'`;
}
```

Line 3 is Corollary 42.5, with an escape hatch: `isPrivateToLoop` drops dependences on buffers allocated *inside* the loop, since each iteration would get its own copy. Line 4 is the fast exit when the analysis found nothing. Line 5 is Proposition 42.8. The analysis has already run when the declaration is consulted, so the shortcut saves no work here — it only changes the answer.

### `reorderLegality`

[`legality.ts:48`](../../../src/compiler/schedule/legality.ts), ten lines, Theorem 42.4 with the same shortcut in front of it:

```ts
export function reorderLegality(state: ScheduleState, chain: readonly ForNode[], after: readonly ForNode[]): string | null {
  const permuted = chain.filter((loop, i) => loop !== after[i]);
  if (permuted.length === 0) return null;
  const names = permuted.map((l) => l.loopVar.name);
  const { info, deps } = state.nestAnalysis(chain[0]);
  if (blockAbstractionPermits(state, chain[chain.length - 1], names, IterVarPolicy.REORDERABLE, info.byBlock)) return null;
  const dep = permutationPreservesDependences(deps, chain, after);
  if (!dep) return null;
  return `permutation violates a ${dep.kind} dependence on buffer '${dep.buffer.name}'`;
}
```

Line 2 is worth noticing: only the loops that actually *moved* are named, so reordering `(a,b,c)` to `(a,c,b)` asks about `b` and `c` and not about `a`. Line 6 is the shortcut, and here it comes *before* the expensive test rather than after it.

`permutationPreservesDependences` ([`dependence.ts:327`](../../../src/compiler/analysis/dependence.ts)) is Theorem 42.4 restricted to the permuted window, and it has two conservative exits:

```ts
    if (!outerEq || covered === 0) continue;
    if (covered !== before.length) return dep;
    if (windowViolation(masks, afterOrder)) return dep;
```

A dependence whose loops do not all lie inside the window is refused outright (`covered !== before.length`), and a dependence carried strictly above the window is ignored (`!outerEq`). Both are the safe direction.

`windowViolation` ([`dependence.ts:284`](../../../src/compiler/analysis/dependence.ts)) is the lexicographic test written for masks rather than exact directions: scanning the permuted order from innermost outward, it asks whether some level that *may* be `>` could become leading while a level that *may* be `<` has been pushed inside it.

### `ScheduleValidator`

[`schedule/validator.ts`](../../../src/compiler/schedule/validator.ts), 303 lines, is a second, independent opinion on the same IR. It walks a `PrimFunc` and reports seven kinds of error:

| Check | Line |
|---|---|
| duplicate loop variable | [`validator.ts:104`](../../../src/compiler/schedule/validator.ts) |
| thread tag missing, or bound twice on one path | [`validator.ts:108`](../../../src/compiler/schedule/validator.ts) |
| a parallel loop nested inside a parallel or thread-bound loop | [`validator.ts:172`](../../../src/compiler/schedule/validator.ts) |
| a reduction-bearing block whose init contract is violated | [`validator.ts:189`](../../../src/compiler/schedule/validator.ts) |
| a parallel or vectorized loop carrying a reduction | [`validator.ts:202`](../../../src/compiler/schedule/validator.ts) |
| a subscript provably out of bounds | [`validator.ts:252`](../../../src/compiler/schedule/validator.ts) |
| two parallel loops of different extents in one function | [`validator.ts:47`](../../../src/compiler/schedule/validator.ts) |

Two of these overlap with `legality.ts` and disagree with it. The fifth uses `reductionLoopVars` — a *syntactic* test, "this loop variable is read in the block and never appears in a write index" — and rejects a vectorized reduction axis that `IterVarPolicy.ACCUMULABLE` had allowed. The seventh objects to a function with parallel loops of two different extents, because the WASM worker pool partitions a single axis; the WASM backend defends itself against that case separately, by emitting a serial loop for any parallel loop whose extent differs from the first one it saw ([`backend/wasm/codegen.ts:455`](../../../src/backend/wasm/codegen.ts)).

And here is the fact that makes the disagreement survivable:

```ts
  verify(): string[] {
    return ScheduleValidator.validate(this.func);
  }
```

`Schedule.verify` ([`schedule.ts:1128`](../../../src/compiler/schedule/schedule.ts)) has no caller in `src/`. The validator's one production caller is the autotuner's evaluation session ([`autotune/session.ts:186`](../../../src/compiler/autotune/session.ts)), which discards any candidate that fails it. **A schedule the rule policy produced is never validated; a schedule the search produced always is.** The two paths are held to different standards, and the stricter one is applied to the path that has a fallback.

## 42.5 The refusals, collected

Every legality check in Part VII, and what asks it:

| Primitive | Check | Refuses |
|---|---|---|
| `parallelize` | `loopCarriedDependence(SPATIAL)` | a carried dependence, unless every axis the loop feeds is `DataPar` |
| `vectorize` | `loopCarriedDependence(ACCUMULABLE)` | a carried dependence, unless every axis is `DataPar` or `CommReduce` |
| `unroll` | — | nothing |
| `bindThread` | tag ∈ six strings | an unknown thread tag |
| `reorder` | `reorderLegality` | a permutation that reverses a dependence, unless the declaration permits |
| `reorder` | `_collectReorderChain` | loops not on one path; a block, a two-way `if`, or a multi-statement `SeqNode` between them |
| `split`, `fuseLoops` | constant extents | a dynamic loop bound |
| `rfactor` | Definition 41.1 + operator set + `1 < f < K`, `f ∣ K` | a body that is not a single accumulating store, a non-associative operator, a bad factor |
| `decomposeReduction` | `initBody` present | every block the lowering rules produce |
| `computeInline` | Definition 41.5, seven clauses | a reduction, a self-reference, indirect indexing, a non-invertible write index, a free variable, a co-produced operand, no consumers |
| `computeAt` | `_alignedLoopPairs` + Proposition 39.5 | mismatched iteration domains; a move across a dependent block |
| `cacheRead`/`cacheWrite` | the buffer is in the declared read/write set | a buffer the block does not name |

Two patterns are worth naming. The checks divide into questions about *order* — the three that call a dependence analysis — and questions about *shape*: whether the nest, the block body or the argument has the form the primitive needs. Shape questions are decidable and are decided; order questions are not, and are answered conservatively. And the two rows that consult the block's declaration instead of the analysis are the two whose answer can be wrong.

## 42.6 Lab — three layers, three answers

```bash
node docs/part7/ch42-legality/labs/01-what-the-primitive-refuses.mjs
```

The matmul's reduction axis, asked of every layer that has an opinion. First what the block says about itself:

```
  vls0_9     bound to ls0_6    kind DataPar
  vrs0_10    bound to rs0_7    kind DataPar
  vc0_11     bound to c0_8     kind CommReduce
```

Then the primitives:

```
  parallelize  REFUSED    Cannot parallelize: loop 'c0_8' carries a WAW dependence on buffer 'buf_5'
  vectorize    ACCEPTED   loop kind is now @vectorized
  unroll       ACCEPTED   loop kind is now @unrolled
```

Same loop, same dependence — found in both cases — and opposite answers, because `SPATIAL` excludes `CommReduce` and `ACCUMULABLE` includes it. `unroll` is accepted because it asks nothing, and correctly: unrolling preserves the order exactly.

Then the validator, on the IR `vectorize` just produced:

```
  Vectorized loop 'c0_8' carries a reduction in block 'matmul_1': the loop variable is read
  but never written, so parallel iterations race on the accumulator
```

Two mechanisms in the same repository, disagreeing about the same three lines of IR. Nothing resolves the disagreement, because nothing on this path runs the validator.

Then the backends:

```
  CPU  : SIMD in the emitted JavaScript? no — the annotation is inert here
  WASM : 0 SIMD opcodes — `_vectorizationIsLegal` ends with `!loopCarriedDependenceIn(body)` and declines.

  WASM, through compile(): 15 SIMD opcodes, 4 extract_lane
  and the answer matches the reference: true
```

Three different behaviours, all correct. CPU ignores the annotation. WASM, handed this IR directly, re-derives the dependence itself and falls back to a scalar loop — `_vectorizationIsLegal` ([`backend/wasm/codegen.ts:1606`](../../../src/backend/wasm/codegen.ts)) ends with `!loopCarriedDependenceIn(body)`, which is the same question a third time. And WASM through the shipping pipeline, where `AccumulatorDetectionPass` has turned the accumulation into an `LIRAccumulatorNode` first, emits four lanes of partial sums and four `extract_lane` opcodes to fold them — a correct vectorised reduction.

That last one is the honest summary of `ACCUMULABLE`. Vectorising a reduction axis is not legal by Theorem 42.4 and it is not a mistake: it is a *request* that the backend perform a horizontal reduce, and the one backend that can do so, does. What is missing is any statement of that contract. A backend that vectorised the loop naively would be wrong, and nothing tells it not to except the fact that all three currently re-check.

## 42.7 Lab — direction vectors, and a declaration that lies

```bash
node docs/part7/ch42-legality/labs/02-direction-vectors.mjs
```

The matmul's dependences, printed as direction vectors:

```
  WAW  on buf_5  over (di0_12, di1_14)  direction (=, =)
  WAW  on buf_5  over (ls0_6, rs0_7, c0_8)  direction (=, =, *)
  RAW  on buf_5  over (ls0_6, rs0_7, c0_8)  direction (=, =, *)
```

`(=, =, *)` — the two spatial axes agree, and the contraction axis is unconstrained. This is the accumulator: iterations `(m,n,k)` and `(m,n,k′)` touch the same element for every pair `k, k′`, and since the axis does not appear in the subscript the analysis cannot order them. That mask is what `carriesDependence` reads to refuse `parallelize(k)`. It also permits every permutation:

```
  (ls0_6, rs0_7, c0_8)  accepted     (rs0_7, c0_8, ls0_6)  accepted
  (ls0_6, c0_8, rs0_7)  accepted     (c0_8, ls0_6, rs0_7)  accepted
  (rs0_7, ls0_6, c0_8)  accepted     (c0_8, rs0_7, ls0_6)  accepted
```

All six, and the reason is worth getting right, because the short version of it is wrong. `*` does not mean "read it whichever way suits you"; it means the analysis could not rule out `<`, `=` or `>`, and a conservative test has to survive all three readings. What rescues this vector is not the `*` — it is the two `=`s in front of it.

A dependence runs from an earlier iteration to a later one, so its first non-`=` component is `<` by construction; a direction vector beginning `(>, …)` is the same pair of iterations read backwards, not a second dependence. With `=` at every level above it, the `*` here *is* the first non-`=` component, so `>` is not a realisable reading and only `<` and `=` remain — and both survive any permutation, because a dependence that constrains only iterations agreeing on `(m, n)` cannot be reordered by moving `k`. `windowViolation` ([`dependence.ts:284`](../../../src/compiler/analysis/dependence.ts)) is that argument in code: it calls a `GT` bit a violation only when some level that could be `<` sits above it and has moved inward. Here nothing above it could be `<`.

Move the `*` off the front and the argument evaporates. That is the gather case in [`dependence.test.js`](../../../tests/compiler/analysis/dependence.test.js): masks `(*, *)`, no `=` prefix to protect the leading component, interchange refused. So a matmul may be written in any of six loop orders — the folklore is true — but not because a `*` is free to be anything.

One caveat about the six lines just printed, since this chapter is about which layer answers. They came out of `sch.reorder`, and `reorderLegality` asks `blockAbstractionPermits` **first** ([`legality.ts:48`](../../../src/compiler/schedule/legality.ts)); the matmul's axes are declared `DataPar`, `DataPar`, `CommReduce`, all admitted by `IterVarPolicy.REORDERABLE`, so it returned "legal" without looking at a dependence at all. The mask does prove all six independently — strip the block header off the same nest and `permutationPreservesDependences` still accepts every one — but the run above is the declaration talking, which is the subject of the rest of this section.

Then Counterexample 42.9, built by hand because no program this compiler lowers has the shape:

```
=== a stencil, A[i+1,j] = A[i,j+1] + 1 ===

  WAW  on A      over (i, j)  direction (=, =)
  RAW  on A      over (i, j)  direction (<, >)
```

`(<, >)`: iteration `(i, j)` writes `A[i+1, j]`, and iteration `(i+1, j−1)` reads it — later in `i`, earlier in `j`. Positive under `(i, j)`; negative under `(j, i)`. And so:

```
=== every permutation of (i, j) ===        [axes untyped]

  (i, j)  accepted
  (j, i)  reorder: permutation violates a RAW dependence on buffer 'A'

=== every permutation of (i, j) ===        [the same nest, axes declared DataPar]

  (i, j)  accepted
  (j, i)  accepted
```

Nothing about the program changed between the two runs. The only difference is the block header: in the first the iteration variables have no `kind`, so `iterKindsOfLoopVar` returns `null` and the analysis is consulted; in the second they are `BlockRealizeNode`s, whose kind defaults to `DataPar`, so `blockAbstractionPermits` returns `true` and the dependence — already computed, sitting in the same function — is not looked at.

And the swap changes the answer:

```
  (i, j)   1  1  1  1  0  2  2  2  1  0  3  3  2  1  0  4
  (j, i)   1  1  1  1  0  1  1  1  1  0  1  1  1  1  0  1
```

Proposition 42.8's "if and only if", executed. A false declaration produced a transformation the analysis had already refused, and no later stage noticed: the verifier of Chapter 33 does not look at kinds, `ScheduleValidator` does not look at kinds, and the backends see a legal-looking nest.

What protects the compiler in practice is the argument Chapter 33 gave and this lab cannot improve on: the five call sites of `markCommReduce` are the five rules that build accumulations, every other rule emits a store whose only same-buffer read is at the same subscript, and so every declaration the compiler makes is in fact true. That is a property of the current rule set, not of the design, and Appendix F's "add an operation end to end" is precisely where a reader could break it.

## 42.8 Traps and limits

- **The declaration is checked by nothing and believed by two functions.** §42.7. Corollary 33.7 restated: the protection is the discipline of the lowering rules, and the failure mode is silent.
- **`ScheduleValidator` runs only for autotuned schedules.** [`autotune/session.ts:186`](../../../src/compiler/autotune/session.ts) is its one production caller, and `Schedule.verify` has none. Every finding in this chapter that the validator would catch — the vectorised reduction, the two-extent parallel partition, a nested parallel loop — is caught for searched schedules and not for rule-produced ones.
- **The rule policy routinely produces the two-extent parallel partition the validator rejects.** A `sum` over a 64×64 input gets `@parallel` on an 8-iteration split of the init nest and `@parallel` on the 64-iteration accumulation loop. `_checkPartitionConsistency` ([`validator.ts:47`](../../../src/compiler/schedule/validator.ts)) would call that an ambiguous partition. The WASM backend's own extent check makes it harmless, by demoting the second loop to serial.
- **`ACCUMULABLE` and `REORDERABLE` are the same set, and are not the same claim.** §42.4. If a future backend needed to distinguish "may be permuted" from "may be reduced across lanes", the two names are already there and the two sets are not.
- **`permutationPreservesDependences` refuses any dependence not fully inside the window.** `covered !== before.length` ([`dependence.ts:328`](../../../src/compiler/analysis/dependence.ts)) returns the dependence — i.e. "illegal" — for a dependence spanning loops the caller did not name. Conservative, and it means reordering an inner pair of a deep nest can be refused because of a dependence involving an outer loop, even when the outer loop's direction is `=`.
- **Legality is checked against the nest, and nothing rechecks after the edit.** `split` inserts a guard, `reorder` moves loops past it, and the arrangement is fixed up structurally by `_arrangeChain` rather than re-verified. A primitive that produced an ill-formed nest would be caught only by the TIR verifier between passes.
- **`_resolveLoop` on a name that does not exist yields a silent no-op or a confusing error, never "no such loop".** Chapter 40's trap, and it lands hardest here: `reorder('typo', 'i')` reports `reorder expects ForNode arguments`.

## 42.9 Read the tests

- [`tests/compiler/schedule/legality.test.js`](../../../tests/compiler/schedule/legality.test.js) — the matmul's `k` axis refused by `parallelize` and by `vectorize` on an *untyped* fixture; `A[i] = A[i−1] + 1` refused; and the `ScheduleValidator` reduction-race check. Reading it beside §42.6 is the clearest way to see how much the block's declaration changes.
- [`tests/compiler/schedule/bounds-validator.test.js`](../../../tests/compiler/schedule/bounds-validator.test.js) — the provably-out-of-bounds check, and the `condDepth` guard that suppresses it under a conditional.
- [`tests/compiler/analysis/dependence.test.js`](../../../tests/compiler/analysis/dependence.test.js) — direction vectors for the shapes Chapter 36 enumerated, and the permutation test used here.
- [`tests/compiler/schedule/compute-inline.test.js`](../../../tests/compiler/schedule/compute-inline.test.js) — `computeAt` refused across a RAW dependence, which is Proposition 39.5's only executable form.

---

**Next:** [Chapter 43 — Scheduling for GPUs](../ch43-scheduling-for-gpus/README.md), where the loop annotation stops being advisory, the legality question becomes a race question, and the compiler's fastest kernel turns out not to be a schedule at all.
