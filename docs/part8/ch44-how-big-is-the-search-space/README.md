# Chapter 44 — How big is the search space

Part VII ended with twenty-two primitives and a promise: any sequence of them that does not throw produces a program that computes what the original computed. That is a licence to try things. This chapter is about how many things there are to try.

The question is not rhetorical and it is not a warm-up. Every decision an autotuner makes downstream — which skeletons to generate, whether to predict or to measure, how to spend thirty seconds — follows from one number, and the number is not intuitive. A matrix multiply is three loops. Three loops sound like a handful of arrangements. The handful turns out to be five figures for one block, and that is before the compiler has been asked about a second block, a second operator, or a second machine.

## 44.1 The problem: how many schedules does one matmul have?

Here is the nest, lowered from `a.matmul(b)` on two 64×64 matrices as Chapter 32 produced it, with the block's bindings and declared access sets elided:

```
for ls0_6 in 0..64 {
  for rs0_7 in 0..64 {
    for c0_8 in 0..64 {
      block matmul_1 {
        buf_5[vls0_9, vrs0_10] = (buf_5[vls0_9, vrs0_10] + (buf_1[vls0_9, vc0_11] * buf_3[vc0_11, vrs0_10]))
```

Three loops, one block, one statement. Chapter 40's `split` can cut any of them into two; the halves can be cut again; `reorder` can permute the result; `parallelize` and `vectorize` can mark two of them. If you ask "how many programs is that", the honest answer is that the question is not well posed, because a sequence of primitives is an unbounded object — you can split a loop of extent 64 into 64 loops of extent 1 and keep going with guards.

What makes the question answerable is that a real autotuner does not search over sequences of primitives. It searches over *parameters of a fixed skeleton*, and the skeleton bounds the sequence. Chapter 45 is about the skeletons; this chapter is about the arithmetic of the parameters, because the arithmetic is what decides that the search is a search and not an enumeration.

For the nest above the compiler offers one skeleton with two parameters — a factorisation of `ls0_6` into four levels, and a factorisation of `rs0_7` into four levels — and the answer is **2,304**, of a mathematically available 7,056. Two more skeletons add six points and a fourth adds 16,128 it cannot deliver. §44.5 counts all of them.

## 44.2 Intuition: a product of independent choices

A schedule is a filled-in form. Each field has a small menu — this axis is cut into four pieces whose sizes multiply to 64; that one likewise; the vector width is one of five numbers — and the number of forms is the product of the menu lengths, because the fields do not constrain each other.

Products grow fast, and they grow fast in the direction that is easy to miss. Doubling the number of candidates for one axis doubles the space. Adding one more *level* to the tiling structure does not add a constant: by Theorem 44.2 it multiplies the per-axis menu by `∏_p (e_p + L)/L`, which for a 64-long axis moving from four levels to five is 2.5, taking 84 to 210. And adding one more axis multiplies by the whole per-axis menu: two axes of extent 64 at four levels is 84 × 84, three of them is 84³ = 592,704.

The second intuition is the one that motivates the rest of the part. A menu of 2,304 is far too many to compile and run — even at a millisecond apiece, once, that is two seconds per block, and a real network has hundreds of blocks and the tuner wants to try each candidate more than once. But 2,304 is far too *few* to give up and enumerate the good ones by hand, because nothing about the block tells you which of the 2,304 is fast. That gap — too many to measure, too many to reason about — is exactly the gap a cost model and a search fill.

## 44.3 Theory

Fix a block and a skeleton. The skeleton exposes a finite list of named choices.

> **Definition 44.1 (Schedule space, stated here).** Let `S` be a *sketch*: a name, a finite list of *search variables* `V₁,…,V_k` with finite candidate sets `C₁,…,C_k`, and a function that applies a parameter assignment to a schedule. The *schedule space* of `S` is the product `C₁ × ⋯ × C_k`. The schedule space of a block is the disjoint union of the spaces of every sketch derived for it.

Nothing in Definition 44.1 says the map from parameters to programs is injective; it usually is not, and Chapter 45 measures how far from injective. What the definition gives is a finite object to search, which a sequence of primitives is not.

The dominant factor in that product, for every skeleton in this compiler that does real work, is the number of ways to cut an axis.

> **Theorem 44.2 (Ordered factorisations).** *(Classical; stars and bars applied to each prime exponent.)* Let `n = ∏_p p^{e_p}`. The number of ordered `L`-tuples `(f₁,…,f_L)` of positive integers with `∏ f_i = n` is
> ```
>   F(n, L) = ∏_p C(e_p + L − 1, L − 1).
> ```

*Proof.* An ordered `L`-tuple with product `n` is the same thing as, for each prime `p` dividing `n`, an ordered `L`-tuple of non-negative exponents summing to `e_p`; the choices for distinct primes are independent. The number of ways to write `e` as an ordered sum of `L` non-negative integers is the number of ways to place `L−1` bars among `e` stars, which is `C(e + L − 1, L − 1)`. Multiplying over primes gives the claim. ∎

> **Corollary 44.3 (Size of a multi-level tiling space, stated here).** For a block with spatial axes of extents `n₁,…,n_s` and reduction axes of extents `m₁,…,m_r`, tiled by a structure with `L_S` spatial and `L_R` reduction levels, the schedule space has
> ```
>   ∏_{i≤s} F(n_i, L_S) · ∏_{j≤r} F(m_j, L_R)
> ```
> points. For a 64×64×64 matrix multiply under this compiler's default CPU structure (`L_S = 4`, `L_R = 1`) that is `F(64,4)² · F(64,1) = 84² · 1 = 7,056`.

`F(n, 1) = 1` for every `n`, and that single fact decides more than it looks. A tiling structure with one reduction level does not split the reduction axis at all — Chapter 45 returns to it.

The number the search actually gets is smaller, and not by a uniform thinning.

> **Proposition 44.4 (The offered space is a biased subset, stated here).** `enumerateFactorizations(n, L, m)` returns at most `m` tuples. It halts the enumeration once `m·8` tuples have been generated ([`factorization.ts:48`](../../../src/compiler/autotune/factorization.ts)) and only then subsamples. The recursion fixes the outermost factor first and walks divisors in ascending order ([`factorization.ts:55`](../../../src/compiler/autotune/factorization.ts)), so when the halt is reached the retained tuples are exactly those whose leading factor is smallest.

*Proof sketch.* `rec` iterates `divisorsOf(remaining)`, which is sorted ascending, and recurses depth-first; so the generation order is lexicographic in the factor tuple. The `all.length >= limit` test terminates generation, leaving a prefix of that order — a prefix which, being lexicographic, contains every tuple with a small first factor and none with a large one. `selectDiverse` then samples the *sorted unique* prefix at even index intervals, which redistributes within the prefix and cannot recover what was never generated. ∎

For `F(n,4) ≤ 384` the halt never fires and the subsample is a genuine even thinning of the whole space. `F(4096, 4) = 455`, so a 4096-extent axis is the first place it bites; §44.5 shows which tuples are lost.

Now the other half of the chapter's question: why not just pick one?

> **Definition 44.5 (Regret, stated here).** Let `T(p)` be the running time of the program at point `p` of a schedule space `P`, and let `p*` minimise it. The *regret* of a selection procedure that returns `p̂` is `T(p̂) − T(p*)`, and its *speedup gap* is `T(p̂)/T(p*)`.

> **Proposition 44.6 (A heuristic is constant on its blind spot, stated here).** A schedule heuristic is a function `h(P, T)` of the program and the target's declared attributes. If two workloads agree on everything `h` reads, `h` returns the same schedule for both; if their optima differ, its regret on at least one of them is the full gap between the two optima.

*Proof.* Immediate from `h` being a function. ∎

Proposition 44.6 is a tautology, and it is worth writing down because it says precisely where a heuristic can be improved and where it cannot: not by making it cleverer about what it already reads, but only by making it read something else — which for a schedule means reading a measurement.

> **Corollary 44.7 (The shipped CPU matmul tile is shape-independent, stated here).** `MatmulTiledCPURule` computes `tileDim = max(8, min(64, ⌊√(L1/4)⌋))` ([`rules.ts:317`](../../../src/compiler/schedule/rules.ts)), a function of `target.l1CacheBytes` alone. Every CPU target with `l1CacheBytes ≥ 16384` gets `tileDim = 64`, so every matrix multiply this rule fires on receives the same tile dimension whatever its extents. Chapter 40's finding 32 is the case where that costs the whole of the parallelism: on a 64×64 matmul both tiled axes have extent 64, so both outer loops have extent 1, and the rule parallelises one of them.

Chapter 4's cost model is what makes Corollary 44.7 a criticism rather than an observation. The time of a tiled nest depends on whether the tile's working set fits in cache, and the working set is `O(tile² · element size)` while the number of tiles is `O((n/tile)²)`: both depend on `n`. A tile chosen from the cache size alone is optimal for the one `n` at which the two happen to balance.

## 44.4 In mlfw

### `enumerateFactorizations`

[`factorization.ts:45`](../../../src/compiler/autotune/factorization.ts), 20 lines, and the whole of Theorem 44.2 in the repository:

```ts
export function enumerateFactorizations(extent: number, levels: number, maxCandidates = 48): Factorization[] {
  if (!Number.isFinite(extent) || extent < 1 || levels <= 1) return [[extent]];
  const all: Factorization[] = [];
  const limit = maxCandidates * 8;
  const rec = (remaining: number, depth: number, acc: number[]): void => {
    if (all.length >= limit) return;
    if (depth === levels - 1) {
      all.push([...acc, remaining]);
      return;
    }
    for (const d of divisorsOf(remaining)) {
      acc.push(d);
      rec(remaining / d, depth + 1, acc);
      acc.pop();
      if (all.length >= limit) return;
    }
  };
  rec(extent, 0, []);
  return selectDiverse(all, maxCandidates);
}
```

Line 2 is `F(n, 1) = 1`, written as an early return. The recursion fills the first `L−1` slots from the divisors of what is left and puts the remainder in the last, which enumerates each tuple exactly once. Two guards mention `limit`, and Proposition 44.4 is about them.

`selectDiverse` ([`factorization.ts:23`](../../../src/compiler/autotune/factorization.ts)) deduplicates, sorts lexicographically, and then — if there are still too many — picks indices at even intervals:

```ts
  for (let i = 0; i < maxCandidates; i++) {
    const idx = Math.floor((i * (uniq.length - 1)) / (maxCandidates - 1));
```

The intent is a spread rather than a prefix, and within what was generated it delivers one. It is worth noticing that the sort key is the tuple read left to right, so "even intervals in the sorted order" means even intervals in the *outermost* factor first — a reasonable proxy for coverage of the coarse-to-fine axis.

### The tile structures

[`tile_structure.ts:11`](../../../src/compiler/autotune/tile_structure.ts) is three literals and nothing else:

```ts
const CPU_TILING: TileStructure = {
  name: 'mlt_cpu',
  order: [['S', 0], ['S', 1], ['S', 2], ['S', 3], ['R', 0]],
  roles: { S0: 'parallelize', S3: 'vectorize' }
};
```

`order` is the loop nesting after tiling, outermost first, written as (kind, level) pairs; `roles` says which level gets which annotation. `levelCounts` ([`tile_structure.ts:33`](../../../src/compiler/autotune/tile_structure.ts)) reads the maximum level of each kind back out, and that is what feeds `L_S` and `L_R` into Corollary 44.3. The GPU structure has three spatial levels and one reduction level; `CPU_TILING_SSRSRS` has four and two, and is the only structure in the compiler that splits a contraction axis.

### Where the count is assembled

`tilingVariables` ([`tiling.ts:94`](../../../src/compiler/autotune/tiling.ts)) is Corollary 44.3 as three lines:

```ts
  spatialLoops.forEach((l, i) => variables.push(new SearchVariable(`s${i}`, enumerateFactorizations(staticExtent(l) as number, spatialLevels))));
  reductionLoops.forEach((l, j) => variables.push(new SearchVariable(`r${j}`, enumerateFactorizations(staticExtent(l) as number, reductionLevels))));
```

One search variable per axis, named `s0`, `s1`, … and `r0`, `r1`, …, each holding its own menu. The product of the menu lengths is the space, and nothing anywhere computes that product: the search samples the variables independently and never asks how many combinations there are.

`staticExtent` returns `null` for a non-constant extent, and `createMultiLevelTilingSketch` refuses the whole sketch if any axis has one ([`tiling.ts:106`](../../../src/compiler/autotune/tiling.ts)). That is the point at which Chapter 62's symbolic shapes leave the search space entirely: a dynamic extent has no divisors to enumerate.

## 44.5 Lab — counting the space

```bash
node docs/part8/ch44-how-big-is-the-search-space/labs/01-counting-the-space.mjs
```

Theorem 44.2 against the enumerator, with and without the cap:

```
  extent  levels   Theorem 44.2   enumerateFactorizations   uncapped
       8       3             10                        10         10
      12       3             18                        18         18
      96       2             12                        12         12
       7       4              4                         4          4
      64       4             84                        48         84
      64       5            210                        48        210
     256       4            165                        48        165
    1024       4            286                        48        286
    2048       4            364                        48        364
    4096       4            455                        48        455
     720       4           1400                        48       1400
```

The closed form and the uncapped enumerator agree on every row, including the prime case `F(7,4) = C(1+3,3) = 4` — the only freedom is which of the four slots holds the 7. From extent 64 upwards the shipped `maxCandidates = 48` binds. And the last two rows are the reason size is the wrong intuition: 2048 has 364 factorisations at four levels and 720 has 1,400, because `720 = 2⁴·3²·5` has three primes to distribute and `2048 = 2¹¹` has one.

Then the block:

```
  block                 axes              sketch           search variables                   points
  matmul_1              S=2 R=1 reads=2   mlt_cpu          s0[48] s1[48] r0[1]                  2304
  matmul_1              S=2 R=1 reads=2   ssrsrs_cpu       s0[48] s1[48] r0[7]                 16128
  matmul_1              S=2 R=1 reads=2   rfactor          rf_factor[5]                            5
  matmul_1              S=2 R=1 reads=2   reduction_cpu    (none)                                  1
                                          —                block total                         18438
  matmul_init_0         S=2 R=0 reads=0   elementwise_cpu  vector_width[5]                         5
                                          —                block total                             5
```

Four sketches for the accumulation block, one for the zeroing block, and 92,190 combinations for the pair. Three numbers in that table are worth pausing on.

`r0[1]` is `F(64, 1) = 1`: under `mlt_cpu` the contraction axis is a search variable with exactly one candidate, `[64]`, and `multiLevelSplit` on a one-element factorisation performs zero splits. The compiler's default CPU tiling does not tile `k`.

`ssrsrs_cpu` offers `r0[7]` — the seven ordered pairs with product 64 — and 16,128 points in total, which is 87% of the block's advertised space. Chapter 45 shows that all 16,128 throw.

`reads=0` on `matmul_init_0` is Chapter 33's finding 9 arriving here: the init block's declared read set is empty, because `bufRefs` builds it from the operation's operands and a zeroing block has none. `analyzeBlockStructure` reads that declaration ([`block_analysis.ts:47`](../../../src/compiler/autotune/block_analysis.ts)), and the derivation rule at priority 10 tests `reads >= 2`. Here the declaration is right and nothing turns on it, but it is another of the small handful of places where the unverified read set is load-bearing.

Finally the truncation:

```
  extent 4096, 4 levels: 455 tuples exist, 48 are offered.
  leading factors present in all 455: 1 2 4 8 16 32 64 128 256 512 1024 2048 4096
  leading factors present in the 48 offered: 1 2 4 8 16 32 64
  largest tuple offered: [64,2,32,1]
```

Proposition 44.4, executed. The outermost tile of a 4096-long axis can be any of thirteen sizes and the search is shown seven of them, all at the fine end. For a CPU tiling whose level 0 is the parallelised loop, that means the search never sees a schedule with fewer than 64 parallel chunks — the coarse-grained options are not rejected, they are never generated.

## 44.6 Lab — what the space is worth

```bash
node docs/part8/ch44-how-big-is-the-search-space/labs/02-what-the-space-is-worth.mjs
```

Counting a space says nothing about whether searching it pays. Eight points of `mlt_cpu`, compiled and run on a 256×256 matmul:

```
  schedule                        loops   max |err| vs scalar   model score   median ms (MEASURED)
  (no tiling — the lowered nest)       5               1.61e-6      2.497682                33.910
  s0=[1,1,1,256] s1=[1,1,1,256]      11               1.61e-6      4.270409                33.679
  s0=[256,1,1,1] s1=[256,1,1,1]      11               1.61e-6      4.270409                32.615
  s0=[8,1,4,8]   s1=[4,2,8,4]        11               1.61e-6      4.270409                31.313
  s0=[8,2,2,8]   s1=[8,2,2,8]        11               1.61e-6      4.270409                31.288
  s0=[1,1,256,1] s1=[1,1,1,256]      11               1.61e-6      4.270409                33.121
  s0=[32,1,8,1]  s1=[1,2,8,16]       11               1.61e-6      4.270409                33.824
  s0=[2,2,2,32]  s1=[2,2,2,32]       11               1.61e-6      4.270409                31.573
```

The last column is measured and will not reproduce; the run above showed a spread of 1.08× between the fastest and the slowest point. Everything to its left is deterministic and does reproduce.

Three readings, in increasing order of importance.

**The spread is small, and that is a fact about the backend, not about tiling.** The CPU backend emits single-threaded JavaScript. Chapter 42 measured what it does with a `@vectorized` annotation — nothing — and it treats `@parallel` the same way. So of everything a schedule can express, the only thing that reaches this backend is the *order* in which the loop nest walks memory, and on a 256×256 problem whose three matrices total 768 KiB the order is worth about eight percent. On WASM, where `@parallel` becomes a worker-pool partition and `@vectorized` becomes SIMD, the same eight points would not be within eight percent of each other. The lesson generalises past this compiler: the payoff of a search is bounded above by what the code generator can be told, and measuring the payoff before checking that bound is how tuning efforts get abandoned for the wrong reason.

**Every point is correct.** One error bound, identical across the eight. That is Proposition 38.4 delivering what Part VII promised, for this sketch: `mlt_cpu`'s `apply` calls only `split`, `reorder`, `parallelize` and `vectorize`, all sound, so no parameter can make it wrong however badly the search behaves. It is the single most valuable structural property an autotuner can have, because over such a sketch the search needs no correctness oracle — only a clock. Chapter 45 states the hypothesis properly and names the two sketches in this compiler that do not satisfy it.

**Every tiled point has the same score.** One value, `4.270409`, for seven different programs whose measured times spanned two and a half milliseconds in the run above. That is not a rounding artefact and it is not specific to these eight points: §46.5 evaluates all 2,304 and gets one distinct score. The objective the search is climbing is flat over the space it is searching, which is the subject of Chapter 46 and the reason this part is ordered the way it is.

## 44.7 Traps and limits

- **`enumerateFactorizations` truncates before it subsamples.** [`factorization.ts:48`](../../../src/compiler/autotune/factorization.ts) stops the recursion at `maxCandidates * 8` tuples, and because the recursion is lexicographic by outermost factor the loss is entirely at the coarse end. The subsample to `maxCandidates` happens either way; what the halt changes is *what it samples from*. Below 384 tuples the recursion completes, so the 48 survivors are spread over the whole space; above it they are spread over a lexicographic prefix. Whether an axis is below the threshold depends on its factorisation rather than its size: a 2048-long axis at four levels has 364 tuples and is sampled from all of them, a 720-long one has 1,400 and is sampled from the 384 with the smallest leading factors. At 4096 the search is offered leading factors 1 through 64 out of 1 through 4096.
- **`maxCandidates` is a per-axis cap, so the joint truncation is its square.** Two axes at 48 candidates each is 2,304 of 7,056 for a 64×64 matmul — 33% of the space — and the fraction falls as the extents grow.
- **The reduction axis is never tiled by the default CPU structure.** `CPU_TILING` has one `R` level ([`tile_structure.ts:13`](../../../src/compiler/autotune/tile_structure.ts)), and `F(n,1) = 1`. The structure that does have two, `CPU_TILING_SSRSRS`, cannot be applied to any lowered block (Chapter 45, finding 23). So the standard cache-blocking of a matrix multiply — block `M`, block `N`, *block `K`* — is not in this compiler's reachable search space at all.
- **A non-constant extent removes the block from the space.** `createMultiLevelTilingSketch` returns `null` if any axis has a symbolic extent ([`tiling.ts:106`](../../../src/compiler/autotune/tiling.ts)), and the derivation then falls through to the parameterless `reduction_cpu` sketch. A dynamically-shaped contraction is therefore untunable, silently; a dynamically-shaped elementwise block still gets its sketch, because `createElementwiseCPUSketch` tests the extent's node type rather than requiring one.
- **Nothing computes the size of the space.** The product in Corollary 44.3 appears nowhere in `src/`; the search draws from each variable independently and the trace emits `blockCount`, not a space size ([`schedule_pass.ts:55`](../../../src/compiler/passes/schedule/schedule_pass.ts)). A user cannot tell from any diagnostic whether a budget of 320 evaluations covered 14% of a space or 0.0002% of one.
- **The heuristic the search competes against is shape-independent.** Corollary 44.7. When a comparison between "tuned" and "default" is reported, the default is a single point chosen from `l1CacheBytes`, which is the same point for every matrix multiply on the machine.

## 44.8 Read the tests

- [`tests/compiler/autotune/ansor.test.js`](../../../tests/compiler/autotune/ansor.test.js) — the first three tests are Theorem 44.2's executable form: every tuple multiplies back to the extent, has the requested length, includes the degenerate `[1,…,1,n]` tuple so that a prime extent still tiles, and the result is bounded by 48 and deterministic. They pin the properties the enumerator must have; none of them pins what the cap discards.
- [`tests/compiler/autotune/autotuner.test.js`](../../../tests/compiler/autotune/autotuner.test.js) — `derives multi-level-tiling + reduction sketches …` asserts the exact sketch list per block and per target, which is the row of §44.5's table that this chapter counts.
- [`tests/compiler/schedule/primitives.test.js`](../../../tests/compiler/schedule/primitives.test.js) — the primitives the space is built from, one test each, and the reason Proposition 38.4 can be relied on when the search composes them blindly.

---

**Next:** [Chapter 45 — Sketches](../ch45-sketches/README.md), which is about the object that turns an unbounded sequence of primitives into the finite product this chapter counted — and about the two skeletons in this compiler that generate a space nothing can reach.
