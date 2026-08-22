# Chapter 40 — Loop primitives

Four primitives change the shape of a nest: `split` cuts one loop into two, `fuseLoops` welds two into one, `reorder` permutes a chain, and `tile` is the composition that gives the family its usual name. Everything Part VIII searches over is built from them, and everything they can go wrong about is visible in one arithmetic fact — that an extent need not be divisible by a factor.

## 40.1 The problem: the loop bound is not negotiable, and the tile size is

A cache line is 64 bytes. A SIMD register is four or eight lanes. A CUDA block is 256 threads. A GPU shared-memory tile is 16×16. Every number that matters for performance is fixed by hardware, and none of them divides 4096, 1000, 768 or 50257 in any useful combination.

So a scheduling language whose primitives require divisibility can only schedule the shapes it happens to like. `split(i, 8)` on a loop of 100 has to mean something, and there are exactly three things it can mean:

1. **Refuse.** Correct, and useless: no vocabulary size is a multiple of eight.
2. **Round down.** Run 96 iterations and lose four elements. Not a scheduling primitive; a different program.
3. **Round up and guard.** Run 104 iterations, and skip the last four.

Option 3 is what the compiler does, and it is not free: a predicate evaluated 104 times, and a nest that can no longer be reasoned about as a rectangle. This chapter is about paying for that exactly once, at the primitive, rather than at every consumer.

## 40.2 Intuition: renaming the iteration space

Every one of the four primitives is the same move in different clothes: **give the iteration space a different set of names, and rewrite the body's references in terms of the new names.**

`split(i, c)` renames one coordinate `i` as a pair `(i_o, i_i)` with `i = i_o·c + i_i`. That is division with remainder, read forwards. `fuseLoops(o, n)` renames a pair as one coordinate `f` with `o = f div n` and `n_var = f mod n` — the same statement read backwards. `reorder` renames nothing at all; it just visits the same points in a different sequence. `tile` is two splits and a reorder.

The picture worth holding is a rectangle of lattice points. `split` draws horizontal lines across it; `fuse` erases the lines and numbers the points serially; `reorder` changes the direction you sweep. None of them adds or removes a point — *except* that `split` cannot draw lines at an arbitrary spacing without extending the rectangle to the next multiple. The guard is the part of the extended rectangle that is not the original.

## 40.3 Theory

Fix a loop `for i in [0, n)` with body `B(i)`, and let `c > 0`. **The lower bound being zero is a hypothesis, not a convenience.** A `ForNode` carries a `min` as well as an `extent` ([`nodes.ts:106`](../../../src/compiler/ir/tensor/nodes.ts)), so a loop over `[m, m + n)` is representable and `split` has to carry the offset through; §40.7 is where that hypothesis is discharged.

> **Definition 40.1 (Split).** **(stated here)** `split(i, c)` replaces the loop by
> `for i_o in [0, ⌈n/c⌉) { for i_i in [0, c) { if (i_o·c + i_i < n) B(i_o·c + i_i) } }`,
> and omits the predicate when `c ∣ n`.

> **Theorem 40.2 (Split is sound, for every extent).** **(stated here)** For every `n ≥ 0` and `c > 0`, the split loop executes `B(v)` exactly once for each `v ∈ [0, n)` and for no other `v`. When `c ∣ n` the predicate is universally true and may be omitted.

*Proof.* The map `(i_o, i_i) ↦ i_o·c + i_i` from `[0, ⌈n/c⌉) × [0, c)` to `[0, ⌈n/c⌉·c)` is a bijection: it is the base-`c` numeral with two digits, so surjective by construction and injective because `i_i < c` makes the low digit unique. Its image is `[0, ⌈n/c⌉·c) ⊇ [0, n)`, and the predicate `i_o·c + i_i < n` selects exactly the preimage of `[0, n)`. Each `v ∈ [0, n)` therefore has exactly one preimage and it passes the predicate; each `v ∈ [n, ⌈n/c⌉·c)` has exactly one preimage and it fails. If `c ∣ n` then `⌈n/c⌉·c = n`, the image is exactly `[0, n)`, and no `(i_o, i_i)` fails the predicate. ∎

> **Counterexample 40.3 (Without the guard).** `n = 12`, `c = 5`. The split space is `3 × 5 = 15` points, so `B(12)`, `B(13)` and `B(14)` run. If `B(v)` writes `A[v]` into a buffer of 12 elements, three elements past the end are written. §40.6 does this to a real program and prints the result.

The other three primitives are the same theorem in different arrangements.

> **Proposition 40.4 (Fuse is sound, and needs no guard).** **(stated here)** For adjacent loops `for o in [0, p) { for q in [0, r) { B(o, q) } }`, `fuseLoops` produces `for f in [0, p·r) { B(f div r, f mod r) }`, which executes `B(o, q)` exactly once for each pair.

*Proof.* Division with remainder again, now read as a decoding: `f ↦ (f div r, f mod r)` is the inverse of `(o, q) ↦ o·r + q`, which is a bijection `[0, p) × [0, r) → [0, p·r)` by the argument of Theorem 40.2 with `n = p·r`, where the divisibility case applies. No point of the fused space is outside the product, so no predicate is needed. ∎

> **Corollary 40.5 (Fuse does not undo split).** **(stated here)** If `c ∤ n`, then `fuseLoops(split(i, c))` yields a loop of extent `⌈n/c⌉·c > n` carrying the predicate, with body references of the form `(f div c)·c + (f mod c)`.

*Proof.* Immediate: `split` produced a `⌈n/c⌉ × c` space and a predicate, and `fuse` is a bijection onto `[0, ⌈n/c⌉·c)` that rewrites `i_o` and `i_i` but deletes nothing. ∎

Corollary 40.5 matters because `(f div c)·c + (f mod c) = f` for every `f ≥ 0`, so the round trip leaves an expression that is provably the identity. §40.6 shows the compiler emitting exactly that expression, per element, at the end of the real pipeline.

For `reorder` this chapter states only the easy half and defers the rest:

> **Definition 40.6 (Reorder).** **(stated here)** Given a chain of perfectly nested loops `L₁,…,L_k` and a permutation `π`, `reorder` produces the chain `L_{π(1)},…,L_{π(k)}` with the same innermost body.

Reorder visits exactly the same set of iteration points, in a different order. Whether that is *sound* is Definition 38.2's question and depends on the body: Chapter 42 answers it with direction vectors. Everything up to that point — that the loops must form a chain, that no block may separate them, that a two-way conditional between them is fatal — is structural, and this chapter's `_collectReorderChain` handles it.

## 40.4 In mlfw

### `split`

[`schedule.ts:257`](../../../src/compiler/schedule/schedule.ts), 46 lines, and Theorem 40.2 is legible in it:

```ts
    const outerExtent = Math.ceil(extent / factor);
    const outerVar = freshVar(`${loop.loopVar.name}_o`);
    const innerVar = freshVar(`${loop.loopVar.name}_i`);
    const oldVarName = loop.loopVar.name;

    const clonedBody = cloneExprTree(loop.body);
    const innerLoop = new ForNode(innerVar, new IntImmNode(0), new IntImmNode(factor),
      loop.kind, clonedBody, loop.threadTag);

    const needsGuard = extent % factor !== 0;
    if (needsGuard) {
      const flatIdx = new MathOpNode('+',
        new MathOpNode('*', outerVar, new IntImmNode(factor)), innerVar);
      const guard = new MathOpNode('<', flatIdx, new IntImmNode(extent));
      const guarded = new IfThenElseNode(guard, innerLoop.body);
      innerLoop.body = guarded;
      innerLoop._setChild('body', guarded);
    }
```

Then the outer loop is built and the substitution runs:

```ts
    substituteVar(innerLoop.body, oldVarName, () =>
      new MathOpNode('+',
        new MathOpNode('*', outerVar, new IntImmNode(factor)),
        innerVar
      )
    );
```

`substituteVar` walks the cloned body and replaces every `VariableNode` named `i` with a fresh copy of `i_o·c + i_i` — a fresh copy per occurrence, because `exprFactory` is a function and TIR nodes carry parent pointers ([`schedule.ts:47`](../../../src/compiler/schedule/schedule.ts)). Note *where* it runs: on `innerLoop.body`, after the guard has been installed, so the guard's own `i_o·c + i_i` is built directly and the body's is substituted. Two syntactically identical expressions, separately constructed, which is why CSE at the LIR level rather than here is what removes the duplicate.

Two preconditions, both checked before anything is built: a non-constant extent is refused outright, and so is a factor that is not a positive integer. The first is why `SchedulePolicy.applyToBlock` refuses to schedule *any* block with a dynamic extent on a GPU ([`rules.ts:560`](../../../src/compiler/schedule/rules.ts)) — with no constant extent there is no `⌈n/c⌉` and hence no grid.

`split` preserves `loop.kind` on **both** halves, and refuses outright to split a thread-bound loop.

The `kind` case is a wart: splitting a `@vectorized` loop gives two vectorised loops, which is very likely not what the caller meant. No rule does it, and nothing prevents it.

The thread tag is a sharper problem and gets a sharper answer. A binding is a statement about *how many* parallel instances there are and which index each gets, so copying `threadIdx.x` onto both halves of a split would claim that the outer and inner loops are each the full thread axis — and the launch geometry Chapter 43 derives from those tags would see one axis bound twice, at two different extents, describing neither the original contract nor a coherent new one. Moving the tag to the inner loop, or to the outer, are both defensible readings of what the caller might have meant, which is precisely the reason not to guess:

```ts
    if (loop.threadTag) {
      throw new Error(`Cannot split loop '${loop.loopVar.name}': it is bound to '${loop.threadTag}', ...`);
    }
```

The refusal costs nothing, because every binding site in this compiler splits *first* and binds afterwards — `sketch_generators.ts` and the GPU rules in `rules.ts` all follow `split(...)` with `bindThread` on the halves. Splitting a bound loop is a sequence nobody writes, and Definition 38.1's "each primitive refuses rather than repairs" is exactly the licence to leave it that way.

### `fuseLoops`

[`schedule.ts:488`](../../../src/compiler/schedule/schedule.ts), 46 lines, Proposition 40.4:

```ts
    substituteVar(fusedLoop.body, outerName, () =>
      new MathOpNode('//', fusedVar, new IntImmNode(innerExtent))
    );
    substituteVar(fusedLoop.body, innerName, () =>
      new MathOpNode('%', fusedVar, new IntImmNode(innerExtent))
    );
```

`//` and `%` are floor division and floor modulo — the decision of Chapter 35, and the reason this is correct without a non-negativity side condition: a loop variable is non-negative, so floor and truncation agree, and the simplifier may substitute `tdiv`/`tmod` later under exactly that proof.

The structural precondition is strict: `if (outer.body !== inner)` throws. The inner loop must be the *direct* child of the outer one — not separated by a `SeqNode`, an `IfThenElseNode` or anything else. That is why `ElementwiseGPURule` re-fetches the loop list and tests `findDirectChild` before each fuse ([`rules.ts:211`](../../../src/compiler/schedule/rules.ts)): after fusing `i0` and `i1`, the new fused loop is the direct parent of `i2`, but only if nothing was in between.

`fuseLoops` keeps the *outer* loop's kind and drops both thread tags: the `ForNode` it builds is constructed without the sixth argument, so a fused loop is never thread-bound however its halves were annotated.

### `reorder`

[`schedule.ts:320`](../../../src/compiler/schedule/schedule.ts) is the longest of the four at 47 lines plus four helpers, and almost all of it is structure rather than legality. `_collectReorderChain` ([`schedule.ts:448`](../../../src/compiler/schedule/schedule.ts)) walks down from the topmost requested loop collecting *links* — the loops being permuted plus everything transparent between them — and refuses four situations by name:

| Situation | Message |
|---|---|
| a two-way `if` separates the loops | `a two-way conditional separates the reordered loops, so they do not form a single chain` |
| a block separates the loops | `block '…' separates the reordered loops, which therefore belong to different block scopes` |
| a `SeqNode` with more than one statement | `multiple statements separate the reordered loops, so they do not form a single chain` |
| the loops are not all on one path | `loops do not form a single chain` |

Then the permutation is applied to the loop positions only, and `_arrangeChain` ([`schedule.ts:401`](../../../src/compiler/schedule/schedule.ts)) fixes up the transparent links. That helper exists for one reason: a one-armed `if` in the chain is a guard from an earlier `split`, and it mentions loop variables. Moving a loop outward past its own guard would put the guard above the loop that binds it. `_arrangeChain` therefore defers any link whose condition names a not-yet-bound loop variable, and releases it as soon as its variables are bound.

The legality check is one line ([`schedule.ts:346`](../../../src/compiler/schedule/schedule.ts)), and it is Chapter 42:

```ts
    const reason = reorderLegality(this.state, links.filter(isLoop), after.filter(isLoop));
    if (reason) throw new Error(`reorder: ${reason}`);
```

### `tile`

[`schedule.ts:534`](../../../src/compiler/schedule/schedule.ts) is a composition and nothing else: split each named loop by its tile size, then reorder so that all the outer halves precede all the inner halves. It is worth reading for one detail, which is how it survives its own edits:

```ts
    for (let i = 0; i < targetLoops.length; i++) {
      const currentLoops = this.getLoops(blockName);
      const loop = currentLoops.find(l =>
        l.loopVar.name === targetLoops[i].loopVar.name ||
        l === targetLoops[i]
      );
      if (!loop) throw new Error(`tile: lost track of loop at index ${i}`);
      const [outer, inner] = this.split(loop, tileSizes[i]);
```

The loop list is re-fetched on every iteration and the wanted loop is found *by name*, because the previous `split` replaced a subtree and every `ForNode` under it is a different object now (Chapter 39). `tile` records no trace step of its own; the two splits and the reorder record themselves, so a replayed trace reconstructs the tiling from its parts.

### The annotations

`vectorize`, `unroll`, `parallelize` and `bindThread` ([`schedule.ts:575`–`617`](../../../src/compiler/schedule/schedule.ts)) are four near-identical short methods that set `loop.kind` and, for the last, `loop.threadTag`. They mutate a node in place rather than replacing a subtree, which is why they call `this.state.invalidate()` directly instead of going through `replaceNode` — as do the other three field-setting primitives, `annotate`, `setScope` and `storageAlign`.

Their asymmetry is the whole of Chapter 42 in one table:

| Primitive | Legality check | Policy |
|---|---|---|
| `parallelize` | `loopCarriedDependence` | `IterVarPolicy.SPATIAL` |
| `vectorize` | `loopCarriedDependence` | `IterVarPolicy.ACCUMULABLE` |
| `unroll` | none | — |
| `bindThread` | the tag is one of six strings | — |

`unroll` is unchecked because unrolling never changes the execution order — it is the one annotation that is sound unconditionally. `bindThread` is unchecked because Chapter 43 checks it at the backend instead.

The fifth member of the family is `annotate` ([`schedule.ts:1081`](../../../src/compiler/schedule/schedule.ts)), which writes an arbitrary key and value into `loop.annotations` rather than setting `kind`. It is the extension point for a backend-specific hint that does not deserve a `ForKind` — a software-pipeline depth, a prefetch distance. Nothing in `src/` calls it, and the only code that touches `annotations` is `simplify_tir.ts:53`, which copies the field forward when it rebuilds a loop — so the mechanism is preserved end to end and is at present a well-formed hole.

## 40.5 Lab — the four primitives on one nest

```bash
node docs/part7/ch40-loop-primitives/labs/01-split-fuse-reorder-tile.mjs
```

A 12×8 by 8×6 matmul, lowered, then each primitive applied to a fresh copy. The nest as lowered:

```
  for ls0_6 in 0..12 {
    for rs0_7 in 0..6 {
      for c0_8 in 0..8 {
        block matmul_1 {
          bind vls0_9 = ls0_6
```

`split(m, 4)`, where `4 ∣ 12`:

```
  for ls0_6_o_0 in 0..3 {
    for ls0_6_i_1 in 0..4 {
      for rs0_7 in 0..6 {
          bind vls0_9 = ((ls0_6_o_0 * 4) + ls0_6_i_1)
```

`split(m, 5)`, where it does not:

```
  for ls0_6_o_0 in 0..3 {
    for ls0_6_i_1 in 0..5 {
      if ((((ls0_6_o_0 * 5) + ls0_6_i_1) < 12)) {
        for rs0_7 in 0..6 {
```

Three things to notice. The guard sits *inside* the inner loop and outside everything else, so the two loops below it run 0 or 48 times rather than being predicated element by element. The binding is unchanged in form — `outer·factor + inner` in both cases — so a consumer that only reads bindings cannot tell a guarded split from an unguarded one. And the outer extent is `⌈12/5⌉ = 3` in the second case and `12/4 = 3` in the first: the same number, reached two different ways, which is a reminder that the extent alone does not tell you whether a guard exists.

`fuseLoops(m, n)`:

```
  for ls0_6_rs0_7_fused_0 in 0..72 {
    for c0_8 in 0..8 {
        bind vls0_9 = (ls0_6_rs0_7_fused_0 // 6)
        bind vrs0_10 = (ls0_6_rs0_7_fused_0 % 6)
```

72 = 12 × 6, no guard, and the two coordinates recovered by a division and a modulo. This is Chapter 35's mixed-radix decoding, now written by the scheduler rather than by a `reshape` rule.

`reorder(k, m, n)` moves the reduction axis outermost and is accepted — for a matmul all six permutations are legal, and Chapter 42 says why. `tile('matmul_1', [0,1], [4,3])` produces `ls_o, rs_o, ls_i, rs_i, c` and a trace of three steps:

```
  trace: [{"primitive":"split","args":["ls0_6",4]},
          {"primitive":"split","args":["rs0_7",3]},
          {"primitive":"reorder","args":[["ls0_6_o_0","rs0_7_o_2","ls0_6_i_1","rs0_7_i_3"]]}]
```

**Try this.** Ask for `tile('matmul_1', [0, 2], [4, 3])` — the M axis and the reduction axis. It is accepted, and the reduction loop ends up split with its outer half hoisted above the N loop. Nothing in `tile` knows which axes are spatial.

## 40.6 Lab — the guard, removed

```bash
node docs/part7/ch40-loop-primitives/labs/02-the-guard.mjs
```

A 12-element `mul`, run into an output buffer with four elements of slack past the end, pre-filled with `-1` so that a stray write is visible. Unscheduled and split-with-guard both give:

```
  output: 2 4 6 8 10 12 14 16 18 20 22 24 -1 -1 -1 -1
```

The lab then walks the IR and splices out the one `IfThenElseNode`, leaving the schedule otherwise identical:

```
  for i0_5_o_0 in 0..3 {
    for i0_5_i_1 in 0..5 {
      block mul_block_0 {
        bind v0_6 = ((i0_5_o_0 * 5) + i0_5_i_1)
        buf_3[v0_6] = (buf_1[v0_6] * buf_4[])

  output: 2 4 6 8 10 12 14 16 18 20 22 24 NaN NaN NaN -1
```

Counterexample 40.3, executed. Three iterations of the padded space read past the input — `undefined` from a `Float32Array`, hence `NaN` — and wrote past the output. In this lab the damage is confined to the slack because the output happens to be its own array. In the compiler's arena (Chapter 50) the three floats land in whatever buffer was placed next.

Then the round trip of Corollary 40.5:

```
  for i0_5_o_0_i0_5_i_1_fused_2 in 0..15 {
    if (((((i0_5_o_0_i0_5_i_1_fused_2 // 5) * 5) + (i0_5_o_0_i0_5_i_1_fused_2 % 5)) < 12)) {
        bind v0_6 = (((i0_5_o_0_i0_5_i_1_fused_2 // 5) * 5) + (i0_5_o_0_i0_5_i_1_fused_2 % 5))
```

Fifteen iterations for twelve elements, a guard testing an expression that is the identity, and a subscript that is the same identity. The answer is right; the arithmetic is three operations per element more than the program started with.

And it is not an artefact of driving the primitives by hand. The last section compiles a 12×5 `mul` for CUDA through the ordinary `compile()` entry point, where `ElementwiseGPURule` fuses the two loops itself:

```
__global__ void Object(float* buf_1, float* buf_3) {
  const int i0_5_i1_6_fused_3 = threadIdx.x;
  const int v0_7 = (i0_5_i1_6_fused_3 / 5);
  const int v1_8 = (i0_5_i1_6_fused_3 % 5);
  buf_3[((v0_7 * 5) + v1_8)] = (buf_1[((v0_7 * 5) + v1_8)] * buf_4[0]);
}
```

`(f / 5) * 5 + f % 5`, emitted twice per element, in shipping output, having passed through `SimplifyPass` and every TIR and LIR pass after it. The analyzer did its half of the job — the operators are C's `/` and `%`, the truncating pair, which the simplifier substitutes only where it has proved the dividend non-negative (Theorem 37.6). It holds the proof that the subscript is `f` and emits four operations anyway, which is exactly the identity Chapter 35 watched the simplifier decline to fold, now reachable from the scheduler as well as from `reshape`.

## 40.7 Traps and limits

> **Counterexample 40.7 (The other hypothesis).** A loop over `[m, m + n)` is representable, and a `split` that ignores `m` is wrong on it: computing the extent, building two loops both starting at `0`, and substituting `i ↦ i_o·c + i_i` is the mapping for `m = 0`, so `for i in [2, 6)` split by 2 runs the body at `i ∈ {0, 1, 2, 3}` where it should run at `{2, 3, 4, 5}` — every access off by `m`. The substitution therefore carries the offset, `i ↦ m + (i_o·c + i_i)`, while the guard stays `i_o·c + i_i < n` because it is a test in the *shifted* space. The offset is emitted only when `min` is not the constant zero, so the ordinary nest is unchanged and no simplifier has to clean up a `+ 0`.

**Note how little of the compiler would notice if that were wrong.** Every loop the lowering rules in Chapter 34 emit has `min = 0`, so no compilation reaches the offset path — and the TIR printer does not render `min` at all, so a discrepancy would not appear in the one tool you would use to look for it. Neither is a reason to skip it: `split` is documented as a primitive over `ForNode`, and `ForNode` has a `min`. **A primitive's contract is its signature, not the subset of inputs the rest of the compiler happens to produce.**

- **`split` propagates the loop kind to both halves.** Splitting a `@parallel` loop produces two `@parallel` loops, which `ScheduleValidator._checkNoNestedParallel` ([`validator.ts:167`](../../../src/compiler/schedule/validator.ts)) would report as nested parallelism — on the paths that run the validator. No rule does this, because every rule splits before it annotates.
- **The guard's index expression is built twice.** §40.4. The predicate and the body each hold their own `i_o·c + i_i`, structurally equal and not shared. Nothing at the TIR level commons them; the CPU backend's expression peephole and the LIR CSE do.
- **`reorder` with fewer than two loops silently returns.** `if (newOrder.length < 2) return;` ([`schedule.ts:323`](../../../src/compiler/schedule/schedule.ts)) — before the arguments are checked to be loops at all, so `reorder('not_a_loop')` is a no-op rather than an error.
- **`_resolveLoop` returns its argument when the name is not found.** `return (found ?? ref) as unknown as ForNode;` ([`schedule.ts:254`](../../../src/compiler/schedule/schedule.ts)). A misspelled loop name therefore reaches the primitive as a *string*, and what the caller sees depends on which primitive it was: `parallelize('typo')` reports `parallelize expects ForNode`, `reorder('typo', 'also')` reports `reorder expects ForNode arguments`, `reorder('typo')` is silently a no-op, and `split('typo', 4)` — which has no type check at all — throws `TypeError: Cannot read properties of undefined (reading 'name')` from inside its own error message. In none of the four is the name reported.
- **`tile`'s reorder is unconditional and its result is unchecked.** After the splits it calls `this.reorder(...outers, ...inners)` if both lists are non-empty. If the permutation is illegal, `reorder` throws — from inside `tile`, with the loops already split. The schedule is left half-tiled, and `applyToBlock`'s `catch` reports "rule rejected" for a nest that has in fact been modified. This is the one place in the file where a failed primitive does not leave the IR untouched.
- **Tiling by the extent produces a degenerate outer loop, and the rule then parallelises it.** `MatmulTiledCPURule` computes `tileDim = max(8, min(64, ⌊√(L1/4)⌋))`, which is 64 for every target with an L1 cache of 16 KiB or more. On a 64×64 matmul both tiled axes have extent 64, so both outer loops have extent 1 — and `apply` finishes with `schedule.parallelize(outerLoops[0])`, marking a one-iteration loop parallel while the 64-iteration inner loop stays serial. The nest is two levels deeper and no more parallel than before.
- **The trace records loop *names*, and names are minted from a process-global counter.** `freshVar` ([`schedule.ts:194`](../../../src/compiler/schedule/schedule.ts)) increments `_varId`, reset only by the exported `resetVarCounter`. Replaying a trace onto a fresh `PrimFunc` in the same process therefore produces a program with different loop names than the original, and any trace step naming a loop that a *previous* step created — every step after a `split` — depends on the counter being at the same value. `tests/compiler/schedule/trace.test.js` calls `resetVarCounter()` in a `beforeEach` for this reason.

## 40.8 Read the tests

- [`tests/compiler/schedule/primitives.test.js`](../../../tests/compiler/schedule/primitives.test.js) — split with and without a divisor, fuse, reorder, tile, and the four annotations, each against the nest it should produce.
- [`tests/compiler/schedule/incremental-sref.test.js`](../../../tests/compiler/schedule/incremental-sref.test.js) — every primitive here, checked for the tree-patching property of Chapter 39.
- [`tests/compiler/schedule/trace.test.js`](../../../tests/compiler/schedule/trace.test.js) — record, serialize, replay, and the counter reset the last trap describes.

---

**Next:** [Chapter 41 — Memory and reduction primitives](../ch41-memory-and-reduction-primitives/README.md), where the primitives stop being bijections on the iteration space and start allocating buffers.
