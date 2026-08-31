# Chapter 36 — Dependence analysis

This is the theoretical centre of the book.

Everything Part VII does — split, reorder, parallelise, vectorise, bind to threads, cache a tile in shared memory — is a change to the order in which iterations run. A change of order is safe exactly when no two iterations that touch the same memory swap places. Deciding that, from the subscripts of Chapter 35, is dependence analysis, and every loop compiler ever written has one.

## 36.1 The problem: reordering is either free or catastrophic

Two loops, one character apart:

```
for i in 0..n:  A[i]   = A[i]   + 1        // (1)
for i in 0..n:  A[i+1] = A[i]   + 1        // (2)
```

Run (1) on eight threads and it is eight times faster and exactly as correct. Run (2) on eight threads and it computes something that depends on thread scheduling — a different answer on every run, and no error from anywhere.

The compiler cannot tell them apart by looking at the loop. It has to look at the subscripts, and the question it has to answer is: **is there a pair of iterations `i ≠ j` such that one writes an element the other reads or writes?**

In (1) the answer is no: iteration `i` touches `A[i]` and nothing else. In (2) iteration `i` writes `A[i+1]`, which iteration `i+1` reads. The equation that separates them is `i + 1 = j`, which has solutions in range; for (1) it is `i = j`, which does not have solutions with `i ≠ j`.

That is the whole subject: **turn each pair of accesses into an equation, and try to prove it has no solution in whole numbers inside the loop bounds.** Equations solved over the integers rather than the reals are called *Diophantine*, and the distinction is the point — `2i = 2j + 1` has plenty of real solutions and no integer ones, which is exactly the kind of fact that makes a loop parallelisable.

## 36.2 Intuition: prove there is no collision, or give up safely

Notice the shape of the goal. The compiler wants to say "these never collide", and if it cannot, it must say "they might" — never the reverse. Dependence analysis is one-sided in exactly the way Chapter 37's interval arithmetic is one-sided, and for the same reason: a wrong "no dependence" produces a race, and a wrong "dependence" produces slow correct code.

So every test in this chapter has the same form. Set the two subscripts equal. Ask whether the resulting equation has an integer solution inside the loop bounds. Three outcomes:

- **Provably no solution** → the accesses are independent, and the transformation is free.
- **A solution exists, and you know where** → there is a dependence, and you know its *distance*: how many iterations apart the two ends are. Distance is what tells you whether reordering is still possible.
- **You cannot decide** → assume the worst.

The three outcomes correspond to three tests of increasing generality and decreasing precision, and mlfw implements all three.

## 36.3 Theory

Fix a loop nest of depth `d` and two accesses to the same buffer, each a list of subscripts that are affine forms (Definition 35.1) over the loop variables. Write iteration vectors as `I = (i₁,…,i_d)`, ordered lexicographically — dictionary order, outermost component first, which is the order the nest actually runs them in (Definition 32.2). `I ≺ J` therefore means "`I` executes before `J`".

> **Definition 36.1 (Dependence).** **(classical)** Let an *access instance* be a pair `(I, s)` of an iteration vector and a statement position within that iteration, ordered lexicographically: `(I, s) ≺ (J, t)` iff `I ≺ J`, or `I = J` and `s` precedes `t` in program order. Two access instances `(I, s) ≺ (J, t)` are *dependent* through a buffer if they access the same element and at least one access is a write. The dependence is **RAW** (or *flow*) if the earlier one writes and the later reads, **WAR** (*anti*) if the earlier reads and the later writes, **WAW** (*output*) if both write.

**The statement position is not decoration — without it the next definition has no cases left.** A dependence has to relate an *earlier* access to a *later* one, so the ordering must be strict. If strictness were imposed on the iteration vectors alone, `I = J` would be excluded, and Definition 36.2's *loop-independent* dependence — direction `=` at every level — would be the empty relation. But loop-independent dependences are real and common: `A[i] = B[i]; C[i] = A[i]` inside one loop is a RAW within a single iteration, and it is the reason a loop body cannot be reordered freely. Ordering instances by `(iteration, statement)` rather than by iteration alone is what admits them, and it is the convention the classical literature uses.

> **Definition 36.2 (Distance and direction).** **(classical)** For a dependence from `(I, s)` to `(J, t)`, the *distance vector* is `J − I`. The *direction vector* is its sign, componentwise: `<`, `=` or `>`. A dependence is *carried by level `ℓ`* if its direction is `=` at every level above `ℓ` and `<` at level `ℓ`; it is *loop-independent* if the direction is `=` everywhere, in which case the two instances lie in the same iteration and their order is fixed by `s ≺ t` rather than by any loop.

> **Direction vectors are lexicographically positive, by construction.** Because Definition 36.1 orders the two instances, the leftmost non-`=` component of a direction vector is always `<`. A pair whose difference has a leading `>` is not a different dependence; it is *the same* dependence with its endpoints named the wrong way round, and the fix is to swap them and negate every component. This is a normalization step, it is not optional, and §36.7 is where its absence from this implementation becomes a wrong answer.

Definition 36.2 is where the payoff lives:

> **Theorem 36.3 (Loop parallelism).** **(classical)** The iterations of the loop at level `ℓ` may be run in any order — including concurrently — if and only if no dependence in the nest is carried by level `ℓ`.

*Proof sketch.* (⇐) Take any dependence and look at its direction at level `ℓ`. Either it is `=`, in which case both ends lie in the same iteration of `ℓ` and reordering `ℓ` does not separate them — this covers loop-independent dependences and those carried at a level *below* `ℓ`. Or it is not `=`, in which case some level above `ℓ` must already be non-`=`, since otherwise the dependence would be carried at `ℓ`; the first such level is carried, so the two ends lie in different iterations of an *outer* loop whose order is untouched. Either way no dependent pair is reordered. (⇒) If a dependence is carried at `ℓ`, its two ends lie in different iterations of `ℓ` with identical outer indices; running those iterations in reverse or concurrently reorders them, and by Definition 36.1 one of the two accesses is a write. ∎

### The three tests

> **Theorem 36.4 (GCD test; Banerjee, Towle, 1976).** **(classical)** Let one subscript be `a·i + c_s` and the other `b·j + c_d` over the same loop level. If `gcd(a, b)` does not divide `c_d − c_s`, no integer solution to `a·i = b·j + (c_d − c_s)` exists, and the accesses are independent.

*Proof.* Every integer of the form `a·i − b·j` is a multiple of `gcd(a,b)`, so if the constant difference is not, there is nothing to solve. ∎

The GCD test is cheap, always applicable, and weak: it ignores the loop bounds entirely, so it proves independence only when the *unbounded* equation has no solution.

> **Theorem 36.5 (Strong SIV test).** **(classical)** If a single loop level appears, with the *same* coefficient `a ≠ 0` in both subscripts, then a dependence exists if and only if `a` divides `δ = c_d − c_s` and `|δ/a| < e`, where `e` is the loop's extent. When it exists, the distance at that level is exactly `−δ/a`.

*Proof.* The equation is `a·i + c_s = a·j + c_d`, so `a(j − i) = −δ`. An integer solution requires `a | δ`, and then `j − i = −δ/a` is forced — a single value, not a family. Both `i` and `j` must lie in `[0, e)`, which is possible exactly when `|j − i| < e`. ∎

"SIV" is *single index variable*. Theorem 36.5 is exact — it gives the distance, not merely existence — and it covers the overwhelming majority of subscripts in real code, because a subscript like `A[i]` versus `A[i+1]` is precisely this case.

> **Definition 36.6 (MIV).** **(classical)** A subscript pair is *multiple index variable* if more than one loop level appears in it. The MIV path here is Theorem 36.4 applied to the gcd of all involved coefficients, followed by Theorem 36.6b, which is the part that reads the loop bounds.

> **Theorem 36.6b (Banerjee bounds, one direction at a time).** **(classical)** Write the dependence equation as `Σ_k (a_k i_k − b_k j_k) = δ`. Fix a level `k` and a direction `d ∈ {<, =, >}` for it, leaving every other level unconstrained. The left-hand side then ranges over a product of one triangular region (level `k`) and rectangles (the rest); each factor is an integer polytope with a linear objective, so its extremes are attained at vertices. If `δ` falls outside `[min, max]`, direction `d` is impossible at level `k`. If no direction survives at some level, the accesses are independent.

*Proof.* The regions are, in `(p, q) = (i_k − lo, j_k − lo)` coordinates with `span = e − 1`: `{(0,0), (span,span)}` for `=`, `{(0,1), (0,span), (span−1,span)}` for `<`, `{(1,0), (span,0), (span,span−1)}` for `>`, and the four corners for an unconstrained level ([`dependence.ts:72`](../../../src/compiler/analysis/dependence.ts)). Each is the vertex set of the corresponding polytope, and `a_k p − b_k q` is linear, so its extremes over the polytope are attained there. Summing the per-level extremes gives the extremes of the sum, because the levels are independent. Leaving the other levels free only widens the interval, so a `δ` outside it is outside every narrower one too — the test is sound, and weaker than a full direction-vector hierarchy. ∎

Two caveats on how the result is reported. Every involved level must have a literal `min` and `extent`, or the refinement is skipped entirely — an unbounded level makes the sum unbounded. And because `accessDependence` normalises a direction vector to lexicographic order by negating the whole vector, a per-level mask is widened to `d | negate(d)` before it is returned ([`dependence.ts:93`](../../../src/compiler/analysis/dependence.ts)): the analysis knows the direction in source-to-destination terms, but what it reports must survive that flip.

There is one MIV case the compiler does decide exactly, and it comes straight out of Chapter 35:

> **Theorem 36.7 (Coincidence of a mixed-radix subscript).** **(stated here)** If both subscripts are the *same* affine form `f`, `f` is in mixed-radix form (Definition 35.7), and the constant difference is zero, then the two accesses coincide exactly when every involved index agrees — that is, the direction is `=` at every involved level.

*Proof.* By Theorem 35.8, `f` takes each value in its range exactly once as the involved variables range over their domain. So `f(I) = f(J)` forces `I = J` on those components. ∎

Without Theorem 36.7 a subscript like `A[3i + j]` in both accesses would fall to the gcd test, which would return "maybe" and forbid parallelising either loop. With it, the compiler concludes the only collisions are the trivial ones and both loops are free.

### Permutation

> **Theorem 36.8 (Legality of loop interchange).** **(classical)** A permutation `σ` of the loops in a perfect nest is legal if and only if, for every dependence, the permuted direction vector is not lexicographically negative — that is, its first non-`=` component is not `>`.

The proof is the same argument as Theorem 36.3 one level up: a lexicographically negative direction vector means the dependence now runs backwards, which is a source executing after its sink. This chapter builds the machinery; Chapter 42 is where it is used and where the counterexample is worked.

## 36.4 In mlfw: collect, then test

Two files. [`analysis/buffer_access.ts`](../../../src/compiler/analysis/buffer_access.ts), 274 lines, walks a nest and records every access. [`analysis/dependence.ts`](../../../src/compiler/analysis/dependence.ts), 280 lines, is Theorems 36.4 to 36.7.

### Collection

`collectBufferAccesses` ([`buffer_access.ts:110`](../../../src/compiler/analysis/buffer_access.ts)) is a single walk maintaining four stacks: the loop ranges in scope, the affine form bound to each variable, the enclosing loop levels, and the enclosing blocks. Every `BufferStoreNode` and `BufferLoadNode` produces a record ([`buffer_access.ts:152`](../../../src/compiler/analysis/buffer_access.ts)):

```ts
    for (const idx of indices) {
      const raw = idx ? toLinearForm(idx) : null;
      regions.push(coverRangeOfForm(raw, loopRanges));
      forms.push(composeForm(raw, varForms));
    }
```

Two versions of each subscript, for two different questions. The *raw* form, over whatever variables are written in the subscript, feeds `coverRangeOfForm` — Theorem 35.8 — to get the region of the buffer this access covers. The *composed* form, substituted down to loop variables, is what the dependence test needs, because two accesses can only be compared in a common vocabulary.

Note what a loop whose `min` or `extent` is not an integer literal contributes ([`buffer_access.ts:180`](../../../src/compiler/analysis/buffer_access.ts)): a level that still records the loop node and the loop variable, carrying `null` for both bounds. The variable is still bound to its own linear form, so subscripts written over it still compose, and the level still occupies its position in every direction vector. What is missing is the *range*, and the range is needed by three of the refinements below — every one of which can only ever prove *more* independence than the test would otherwise report. A symbolic extent therefore costs precision, not visibility, and the distinction is worth the paragraph: getting it the other way round is a soundness bug rather than a slow program, and §36.8's fourth trap is where that history is.

### The common nest

Two accesses can only be compared at the loop levels they share ([`dependence.ts:45`](../../../src/compiler/analysis/dependence.ts)):

```ts
function commonNest(srcSpace: readonly IterLevel[], dstSpace: readonly IterLevel[]): IterLevel[] {
  const n = Math.min(srcSpace.length, dstSpace.length);
  const loops: IterLevel[] = [];
  for (let i = 0; i < n; i++) {
    if (srcSpace[i].node !== dstSpace[i].node) break;
    loops.push(srcSpace[i]);
  }
  return loops;
}
```

Identity comparison on the loop node, and nothing else: whether the level's bounds are known is not this function's business. It stops at the first node mismatch, which is what separates two accesses in sibling nests — they share nothing, so their dependence has an empty direction vector, and `carriesDependence` never attributes it to any loop. That is correct: parallelising a loop inside one nest does not reorder that nest against the other.

It is also precisely why a level with unknown bounds must not stop this loop. Dropping such a level would reach the same "attributable to no loop" conclusion for a loop that genuinely encloses *both* accesses — and `carriesDependence`, asked about that loop, would answer that nothing is carried by it.

### The tests

`subscriptDirections` ([`dependence.ts:144`](../../../src/compiler/analysis/dependence.ts)) is Theorems 36.4 to 36.7, in the order of decreasing precision. Zero involved levels:

```ts
  if (involved.length === 0) return delta === 0 ? new Array<DirectionMask>(n).fill(ANY_DIRECTION) : INDEPENDENT;
```

Two constants: they collide iff they are equal. Then the SIV cases ([`dependence.ts:148`](../../../src/compiler/analysis/dependence.ts)):

```ts
  if (involved.length === 1) {
    const k = involved[0];
    const a = src.coeffs[k];
    const b = dst.coeffs[k];
    const { min, extent } = loops[k];
    const ranged = min !== null && extent !== null;

    if (a === b) {
      if (delta % a !== 0) return INDEPENDENT;
      const distance = -delta / a;
      if (ranged && Math.abs(distance) >= extent) return INDEPENDENT;
      masks[k] = distance > 0 ? Direction.LT : (distance === 0 ? Direction.EQ : Direction.GT);
      return masks;
    }
```

Theorem 36.5, line for line: divisibility, then the bound, then the exact direction. `ranged` is where a symbolic extent is paid for, and it is worth seeing how little it buys: divisibility and the direction survive without a bound, because neither mentions one, and only the refinement that would have proved *independence* is dropped. The two weak-zero cases follow — one subscript not mentioning the level at all, so the other is pinned to a single value that must be in range, which is where `ranged` appears again — and then the general two-coefficient case falls back to the gcd:

```ts
    if (delta % gcd(a, b) !== 0) return INDEPENDENT;
    return masks;
```

The MIV branch is Theorem 36.7, then Theorem 36.4, then Theorem 36.6b ([`dependence.ts:181`](../../../src/compiler/analysis/dependence.ts)):

```ts
  const uniform = involved.every((k) => src.coeffs[k] === dst.coeffs[k]);
  if (uniform && delta === 0 && mixedRadixDecomposition(srcForm, varRanges) !== null) {
    for (const k of involved) masks[k] = Direction.EQ;
    return masks;
  }

  let g = 0;
  for (const k of involved) g = gcd(g, gcd(src.coeffs[k], dst.coeffs[k]));
  if (g !== 0 && delta % g !== 0) return INDEPENDENT;
  if (!banerjeeRefine(src, dst, delta, loops, involved, masks)) return INDEPENDENT;
  return masks;
```

The order matters for cost, not for strength: gcd is a handful of divisions and rejects the arithmetically impossible cases before the bounds test allocates anything.

Every path that cannot decide returns a mask of `ANY_DIRECTION`, which is `LT | EQ | GT` — the "assume the worst" of §36.2, encoded as a bitmask so that intersecting across subscripts is an `&`.

That intersection is `accessDependence` ([`dependence.ts:230`](../../../src/compiler/analysis/dependence.ts)):

```ts
  for (let d = 0; d < rank; d++) {
    const dims = subscriptDirections(src.forms[d], dst.forms[d], loops, levelIndex, varRanges);
    if (dims === INDEPENDENT) return null;
    for (let k = 0; k < loops.length; k++) {
      masks[k] &= dims[k];
      if (masks[k] === 0) return null;
    }
  }
```

One subscript proving independence is enough — `A[i][j]` and `A[i][j+1]` collide only if both dimensions collide — and an empty mask after intersection is the same conclusion arrived at from two directions at once.

### Using the answer

`carriesDependence` ([`dependence.ts:272`](../../../src/compiler/analysis/dependence.ts)) is Definition 36.2's "carried by level `ℓ`" and Theorem 36.3's test:

```ts
    const level = dep.loops.findIndex((l) => l.node === loopNode);
    if (level < 0) continue;
    let outerEq = true;
    for (let m = 0; m < level; m++) {
      if (!(dep.masks[m] & Direction.EQ)) { outerEq = false; break; }
    }
    if (!outerEq) continue;
    if (dep.masks[level] & (Direction.LT | Direction.GT)) return dep;
```

and `permutationPreservesDependences` ([`dependence.ts:307`](../../../src/compiler/analysis/dependence.ts)) is Theorem 36.8, with `windowViolation` ([`dependence.ts:284`](../../../src/compiler/analysis/dependence.ts)) checking for a lexicographically negative permuted vector over the window of loops being permuted.

Both are called from [`schedule/legality.ts`](../../../src/compiler/schedule/legality.ts) — and Chapter 33 has already shown the twist: when the block's iteration-variable kinds are available and permit the movement, the computed dependence is **overruled**. Dependence analysis is the fallback, not the primary mechanism. It runs when the declaration does not apply: a non-affine binding, an untyped iteration variable, or a loop variable used directly in a block body.

The analysis is recomputed per nest and cached on the schedule state ([`schedule_state.ts:122`](../../../src/compiler/schedule/schedule_state.ts)), invalidated whenever a primitive edits the IR.

## 36.5 Lab — which loops may run in parallel

```bash
node docs/part6/ch36-dependence-analysis/labs/01-which-loops-may-run-in-parallel.mjs
```

Scheduling is off by default; the lab turns it on and reads the loop kinds off the scheduled IR.

Two things in the output need a word first. A loop annotated `@parallel` or `@vectorized` is one the scheduler decided may run out of order — that decision is what this chapter is about, and the primitives that carry it out are Chapter 40's. And a name like `i1_6_o_0` next to `i1_6_i_1` is one original loop `i1_6` that has been **split** into an outer and an inner half, `_o_` over the tiles and `_i_` within one: 64 iterations becoming 8 × 8. Splitting is Chapter 40 as well; here it matters only because it is what a legal loop gets and an illegal one does not.

```
=== elementwise: x * x + x ===
  i0_5           extent 64     @parallel
    i1_6_o_0       extent 8      serial
      i1_6_i_1       extent 8      @vectorized
        [mul_block_0]

=== reduction: x.sum(1) ===
  si0_5_o_4      extent 8      @parallel
    si0_5_i_5      extent 8      @vectorized
      [reduce_init_0]
  sa0_7          extent 64     @parallel
    r0_9           extent 64     serial
      [reduce_acc_1]

=== contraction: x @ y ===
  ls0_6_o_8      extent 1      @parallel
    rs0_7_o_10     extent 1      serial
      ls0_6_i_9      extent 64     serial
        rs0_7_i_11     extent 64     serial
          c0_8           extent 64     serial
            [matmul_1]

=== recurrence: scan(c -> 0.9*c + x_t) ===
  t_9            extent 8      @recurrence
    i0_11          extent 4      @parallel
      [scan_in_1]
    ...
```

Four programs, three answers.

**The elementwise nest is fully independent.** Both subscripts are the bare loop variables in both accesses, so the strong SIV test gives distance 0 at every level and the direction is `=` everywhere. No level carries anything, and Theorem 36.3 permits every order.

**The reduction splits.** `reduce_acc_1` writes `buf_3[sa0_7]` and reads `buf_3[sa0_7]`. The subscript is a single affine form mentioning one level, so the strong SIV test applies there: same coefficient, `δ = 0`, distance 0, direction `=`. The reduction level `r0_9` appears in **neither** subscript, so it is not among the involved levels at all and its mask is left at its initial `ANY`. The analysis says nothing about it — and by Theorem 36.3 saying nothing is the same as carrying a dependence, because `ANY` contains `<`. Spatial loop parallel; reduction loop serial.

**The contraction is the same shape one dimension up.** `ls0_6` and `rs0_7` appear in the write subscript; `c0_8` does not. Only `c0_8` is left alone.

**The recurrence is never asked.** `scan` lowers its time loop with `ForKind.RECURRENCE` ([`rules/control_flow.ts:202`](../../../src/compiler/passes/lowering/rules/control_flow.ts)), and the scheduler's rule set excludes such a nest from parallelisation by inspection ([`schedule/rules.ts:183`](../../../src/compiler/schedule/rules.ts)) rather than by analysing it. That is the same trade as Chapter 33's iteration-variable kinds: a declaration that costs nothing to make and saves an analysis that would have had to look through a carried buffer copy to reach the same conclusion.

## 36.6 Lab — reading the dependence off the IR

```bash
node docs/part6/ch36-dependence-analysis/labs/02-reading-the-dependence.mjs
```

The second lab does by hand, on the printed IR, the presence test that `subscriptDirections` does properly:

```
=== x.sum(1) ===
  block            loop       in write idx  self read  carried   scheduled
  reduce_init_0    si0_5      true          false      -         parallel
  reduce_acc_1     sa0_7      true          true       -         parallel
  reduce_acc_1     r0_9       false         true       RAW       serial

=== x @ y ===
  block            loop       in write idx  self read  carried   scheduled
  matmul_init_0    di0_12     true          false      -         parallel
  matmul_init_0    di1_14     true          false      -         serial + vectorized
  matmul_1         ls0_6      true          true       -         parallel
  matmul_1         rs0_7      true          true       -         serial
  matmul_1         c0_8       false         true       RAW       serial
```

Every row marked RAW came back serial, and no row marked RAW was split, vectorised or parallelised. The rows that are serial without a dependence are serial for a scheduling reason, not a legality one — the policy ran out of parallelism worth extracting, which is Part VII's subject and not this chapter's.

The hand version is weaker than the real one in exactly one way, and it is the way that matters. It asks *"does the loop variable appear in the write subscript?"* The real test asks *"does the equation have a solution?"* — which lets it answer "independent" for subscripts that **do** contain the variable. `A[2i]` against `A[2i+1]` is the canonical case: the variable is present in both, and Theorem 36.5's divisibility condition fails, because `2` does not divide `1`. The accesses are independent and the loop is free. A presence test would have said "carried".

**Try this.** The interesting programs for this chapter are the ones a tensor language cannot express, because every tensor operation writes each output element once. Reach for the schedule primitives of Part VII instead: `cache_write` introduces a second buffer with a shifted subscript, and `rfactor` splits a reduction into two, and both are legality questions this machinery answers.

## 36.7 Normalising the direction vectors

Definition 36.2 closed with a claim that needs establishing rather than assuming: a direction vector is lexicographically positive, because the two access instances were named in execution order. Nothing in the computation so far guarantees it. Look at how the two ends of a pair are chosen ([`dependence.ts:252`](../../../src/compiler/analysis/dependence.ts)):

```ts
      const src = write.position <= other.position ? write : other;
      const dst = write.position <= other.position ? other : write;
```

`position` is the access's index in the **textual** walk of the body — which statement was written first — not a statement about which *instance* runs first. For a loop-independent dependence the two coincide. For a dependence carried across iterations with a **mixed** direction they do not:

> **Counterexample 36.9.** In the nest
>
> ```
> for i in 0..4:
>   for j in 0..4:
>     A[i, j] = A[i+1, j-1]
> ```
>
> iteration `(i, j)` reads the element that iteration `(i+1, j-1)` writes. Since `(i, j) ≺ (i+1, j-1)` lexicographically, the read happens **first**: this is a **WAR** dependence with direction `(<, >)`. Named by textual position, with the write as source, it comes out as RAW with direction `(>, <)` — the wrong kind and the reverse direction.

Drawn on the iteration space, with the pair at `(1,2)` and `(2,1)`:

```
   j=0   j=1   j=2   j=3
 i=0  .     .     .     .        execution order is row by row, left to right,
 i=1  .     .    (R)    .        so (1,2) runs BEFORE (2,1)
 i=2  .    (W)    .     .
 i=3  .     .     .     .        (R) at (1,2) reads  A[2,1]
                                 (W) at (2,1) writes A[2,1]
   difference (2,1) - (1,2) = (+1, -1)  ->  direction (<, >)
   earlier instance READS, later instance WRITES        ->  WAR

   named by textual position instead, with the store as source:
   difference (1,2) - (2,1) = (-1, +1)  ->  direction (>, <)   and kind RAW
                                              both wrong
```

That matters because `permutationPreservesDependences` is Theorem 36.8's test: *does a permutation reverse a direction vector?* Interchanging `i` and `j` maps the true `(<, >)` to `(>, <)`, which is lexicographically negative and therefore **illegal**. It maps the reversed reading back to `(<, >)`, which looks fine — so the interchange is accepted, and after it iteration `(j, i)` reads an element that has already been overwritten. **That is the failure mode dependence analysis exists to make impossible**: not "I could not tell", but a confident answer wrong in the permissive direction (Chapter 42 §42.4).

So `accessDependence` normalises before returning. A vector is *definitely* lexicographically negative when its leftmost non-`=` level admits only `>`; in that case the ends were named in the wrong order, so they are swapped, every component negated, and the kind recomputed:

```ts
function isLexNegative(masks: readonly DirectionMask[]): boolean {
  for (const mask of masks) {
    if (mask === Direction.EQ) continue;
    return mask === Direction.GT;
  }
  return false;
}
```

Two details make this sound rather than merely plausible. It flips **only when the leftmost non-`=` mask is exactly `GT`** — a mask that also admits `<` can already be read as positive, and flipping it would throw that reading away; leaving it costs precision and never soundness. And the *kind* is recomputed rather than carried, because RAW and WAR are defined by which end is earlier, so reversing the pair turns one into the other. The nest above reports `kind=WAR dirs=[<,>]`, and the interchange is refused.

**The test that checks this had to be written carefully, and that is the transferable part.** A brute-force oracle over the iteration space is the obvious way to validate a dependence analyser — enumerate every pair of instances, find the colliding ones, take the difference. But if the oracle pairs them the same way the implementation does, with the write always as source, it computes the same reversed answer and confirms the analyser against itself. §36.9's oracle orders each colliding pair lexicographically *before* taking the difference, so it derives the direction from execution order independently. **An oracle that shares an assumption with the code under test does not test that assumption**, and orientation is exactly the kind of assumption that is invisible until something forces the two apart.

## 36.8 Traps and limits

- **The declaration usually wins before the analysis is consulted.** [`legality.ts:40`](../../../src/compiler/schedule/legality.ts) computes the dependence, finds one, and returns "legal anyway" if the block's iteration-variable kinds permit. On well-formed lowered IR the two always agree; nothing checks that they do, and Chapter 33's Corollary 33.7 is the exposure.
- **MIV bounds are tested one level at a time, not as a vector.** `A[i + 64j]` against `A[i + 64j + 32]` in a nest with `i` of extent 32 is decided: gcd is 1 and divides 32, so Theorem 36.4 passes it through, and Theorem 36.6b then finds no surviving direction at the `j` level and returns independence. What is *not* implemented is the direction-vector hierarchy — Theorem 36.6b constrains one level and leaves the rest free, so a dependence that is impossible only for a *combination* of directions across two levels is still reported. An exact integer-programming test (Omega) would decide those; that is the remaining gap, and it is a smaller one than it was.
- **A dynamic extent costs the bound-dependent refinements, and only those.** A loop with a non-literal extent contributes a level with `null` bounds ([`buffer_access.ts:180`](../../../src/compiler/analysis/buffer_access.ts)), and `subscriptDirections` skips the three refinements that need a bound: `|δ/a| < e`, the in-range check on the single colliding iteration of a weak-zero SIV pair, and — because `mixedRadixDecomposition` needs a range per variable — the exact-coincidence case of Theorem 36.7. Divisibility still runs and the sign of the distance is still exact. The residual cost is precision: `A[i] = A[i+8]` in a loop of extent 8 is provably independent with a literal extent and reads as a dependence with a symbolic one, so a model compiled with symbolic shapes gets less parallelism than the same model with static ones, silently.
- **That trap used to say something stronger, and the stronger version was a bug.** The level was dropped from the common nest outright, so `carriesDependence` found no level to attribute the dependence to and returned `null` — and `parallelize` accepted a reduction loop it refuses when the same extent is a literal, which is a race, not a lost optimisation. The lesson generalises past this chapter: an analysis that expresses "I could not look" by returning nothing is, at the call site, indistinguishable from one returning "I looked and there is nothing there". The fix was to keep the level and lose only the bound.
- **Conditional accesses are recorded and not used.** `BufferAccess.conditional` marks an access under an `if` or a `while` ([`buffer_access.ts:162`](../../../src/compiler/analysis/buffer_access.ts)). `accessDependence` ignores it, which is the conservative direction — a dependence that only exists on one branch is reported unconditionally. `buffer_dataflow.ts` uses the flag; the dependence tester does not.
- **`selfReferential` is likewise informational here.** It marks an access whose own subscript or stored value loads from the same buffer — an indirect access such as `A[B[i]]` where `A` is `B`. The dependence tester does not consult it; the scheduler does, separately ([`schedule.ts:858`](../../../src/compiler/schedule/schedule.ts)).
- **Direction is computed, distance is discarded.** Theorem 36.5 produces an exact distance and `subscriptDirections` immediately reduces it to a three-valued sign ([`dependence.ts:159`](../../../src/compiler/analysis/dependence.ts)). Distance is what a legality test for *skewing* and for software pipelining needs, and neither exists here, so nothing has yet wanted it. A `Dependence` carries `masks` and no distances.
- **The pairing is quadratic in accesses per buffer.** `bufferDependences` ([`dependence.ts:246`](../../../src/compiler/analysis/dependence.ts)) is a double loop over the accesses to one buffer. For the nests this compiler produces — a handful of accesses each — that is nothing; for a heavily unrolled nest it is the dominant cost of a legality query, and the caching in `schedule_state` is what keeps it off the critical path.
- **An oracle can share an assumption with the code it checks.** §36.7 is the worked example: the analyser and its brute-force oracle must derive direction orientation by independent routes, or the test confirms the implementation against itself. When you write an oracle for an analysis, the question to ask is which of the analysis's assumptions the oracle also makes.
- **Non-affine subscripts are `null`, and `null` means "any".** `subscriptDirections` opens with `if (!srcForm || !dstForm) return ... ANY_DIRECTION`. A gather's data-dependent subscript therefore blocks every reordering of every loop around it — correct, and the reason a gather-heavy kernel schedules badly.

## 36.9 Read the tests

- [`tests/compiler/analysis/dependence.test.js`](../../../tests/compiler/analysis/dependence.test.js) — the three kinds, the direction vectors, the SIV distance cases including the out-of-range one, the gcd rejection, and the symbolic-extent levels of §36.7 in both directions: still in the direction vector, minus the range refinements. The Banerjee cases are the interesting ones to read: each first asserts that the gcd test *would* have let the pair through, then that the analysis returns independence anyway, and the refinement cases assert only that the reported mask is a **superset** of the brute-force one — the soundness direction, which is the only direction a widened mask can be checked in.
- [`tests/compiler/schedule/legality.test.js`](../../../tests/compiler/schedule/legality.test.js) — the same question asked through `parallelize` and `reorder`, which is where a level going missing turns into a race.
- [`tests/compiler/analysis/buffer-dataflow.test.js`](../../../tests/compiler/analysis/buffer-dataflow.test.js) — the collection side, and what the `conditional` and `selfReferential` flags are for.
- [`tests/compiler/analysis/gpu-race.test.js`](../../../tests/compiler/analysis/gpu-race.test.js) — the same question asked about threads rather than iterations, which is Chapter 43.
- [`tests/compiler/analysis/iter-map.test.js`](../../../tests/compiler/analysis/iter-map.test.js) — the mixed-radix recogniser that Theorem 36.7 rests on.

---

**Next:** [Chapter 37 — Proving things about indices](../ch37-proving-things-about-indices/README.md). Every test in this chapter needed a range — `|distance| < extent`, `0 ≤ f_R ≤ c−1`, "is this index in bounds". The last chapter of Part VI is the analysis that supplies them, and the precise sense in which it is allowed to be wrong.
