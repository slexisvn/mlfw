# Chapter 46 — Cost models

Chapter 45 produced a generator. Turn its crank 2,304 times and you have 2,304 legal programs, all computing the same matrix product, differing only in how they walk memory. Now pick one.

Running all of them is not an option: compiling and timing each takes a few milliseconds, a network has hundreds of blocks, and the budget is thirty seconds. So the search needs a *proxy* — something that takes a scheduled `PrimFunc` and returns a number, fast, without executing it. That object is the cost model, and this chapter is about what it can be asked to do, what it is graded on, and the gap between those two things.

The gap is the point. A cost model is fitted with squared error and used as a comparator. Those are different objectives, they are usually aligned, and the places where they come apart are exactly the places where a tuner's behaviour stops making sense to the person watching it.

## 46.1 The problem: what does "this schedule is better" mean to a function?

Here are two schedules of the same 64×64 matmul, differing in one parameter:

```
  s0 = [1,1,1,64]   the outermost, parallelised loop has extent 1
  s0 = [64,1,1,1]   the outermost, parallelised loop has extent 64
```

Any engineer would rank these instantly: the second has 64 independent chunks of work and the first has one. A function that must decide it from the IR has to notice that `ForKind.PARALLEL` is on a loop whose `extent` is an `IntImmNode` with value 64 rather than 1, and has to have a term in which that extent appears.

The shipped model has neither. Its parallelism term is `numParallelLoops / numLoops` ([`cost_model.ts:113`](../../../src/compiler/autotune/cost_model.ts)) — a ratio of counts. Both schedules have one parallel loop out of nine. They score identically, and so does every other point of the space.

That is not a bug in a single line; it is what happens when a feature vector is designed before the space it has to discriminate. The chapter's job is to make the failure mode legible: which features exist, which of them vary over a sketch's space, and what the score is therefore a function of.

## 46.2 Intuition: two ways to be wrong

A cost model can fail in two independent ways, and conflating them is the most common mistake in reading one.

**It can be wrong about the number.** Predicting 5 ms for a kernel that takes 40. This is what a regression metric measures, and it is what you notice first because it is the thing you can print.

**It can be wrong about the order.** Predicting 5 ms for a kernel that takes 40 and 6 ms for one that takes 4. This is what the search actually consumes: nothing downstream of the model reads its value, only its comparisons.

The two are related — a model with zero error has perfect order — and they are not the same, in both directions. A model that adds a thousand to every prediction has enormous error and identical behaviour. A model that is within one percent everywhere except on the two fastest candidates, which it swaps, has excellent error and does the one thing the search asked it not to.

The second intuition is about what a feature vector is *for*. A feature is a hypothesis: "programs that differ in this number differ in speed". A feature that is constant over the space you are searching is not a weak hypothesis, it is not a hypothesis at all — it contributes exactly nothing, and if all of them are constant the model is a constant function and the search is a random draw wearing a suit.

## 46.3 Theory

> **Definition 46.1 (Feature map, stated here).** A *feature map* is a function `φ : PrimFunc → ℝ^d`. It is *discriminating on a space `P`* if `φ` is non-constant on `{ program(p) : p ∈ P }`.

> **Definition 46.2 (Cost model, stated here).** A *cost model* is a function `ĉ : PrimFunc → ℝ`, used by the search only through comparisons: the search computes `argmax`, sorts by `ĉ`, and keeps a running maximum. Higher is better by convention, so a model fitted to measured time predicts its negation.

> **Theorem 46.3 (Only the induced order matters, stated here).** Let `g : ℝ → ℝ` be strictly increasing and let `ĉ' = g ∘ ĉ`. Then a search that consumes `ĉ` only through comparisons visits the same candidates, in the same order, and returns the same result under `ĉ'` as under `ĉ`.

*Proof.* By inspection of the three places a score is consumed. `scored.sort((a,b) => b.score − a.score)` ([`search.ts:134`](../../../src/compiler/autotune/search.ts)) and the final sort at [`search.ts:164`](../../../src/compiler/autotune/search.ts) depend only on the sign of a difference, which `g` preserves; `_consider` ([`session.ts:235`](../../../src/compiler/autotune/session.ts)) is `rec.measuredScore > this._best.measuredScore`, likewise. The remaining reads of the score — the memo key ([`search.ts:117`](../../../src/compiler/autotune/search.ts)), the elite count, the mutation draws — do not involve it. The sorts are stable, so ties break identically. ∎

> **Corollary 46.4 (Absolute error is not identifiable from search behaviour, stated here).** For any `ĉ` and any `M > 0` there is a `ĉ'` inducing the same search whose mean squared error exceeds `M`. Hence no statement of the form "the model's error is `E`" constrains what the search will do, and no observation of the search bounds the model's error.

*Proof.* Let `r_i = ĉ(x_i) − y_i` be the residuals and take `g(x) = x + c`, strictly increasing for every `c`, so Theorem 46.3 applies. Then `MSE(c) = mean((r_i + c)²) = mean(r²) + 2c·mean(r) + c²`, a quadratic in `c` with positive leading coefficient; it tends to infinity with `c`, so some `c` puts it above `M`. ∎

The shift has to be chosen against the residuals rather than fixed in advance, and it is worth seeing why, because the obvious `g(x) = x + √M` does not work. For `c = −mean(r)` the expression above *falls*: a model predicting `−√M` everywhere against a truth of `0` has `MSE = M`, and adding `√M` takes it to `0`. Monotone shifts can move the error anywhere, in either direction, which is the whole content of the corollary.

> **Counterexample 46.5 (Lower error, higher regret).** *(Stated in predicted cost rather than score: Definition 46.2's convention is higher-is-better, and a model fitted to time predicts its negation, so here the selection is an arg-min.)* Three candidates with true costs `(1, 2, 100)` ms. Model B predicts `(2, 1, 100)`: mean squared error `2/3`, and it selects candidate 1, whose regret is 1 ms. Model C predicts `(50, 51, 60)`: mean squared error `2134`, and it selects candidate 0, whose regret is 0. C's error is 3,201 times B's and its regret is smaller. §46.6 runs the arithmetic.

Corollary 46.4 says a regression metric cannot *prove* anything about search quality; Counterexample 46.5 says it cannot even be relied on as a heuristic in the small. What survives is the practical claim, which §46.6 also measures: on real data a model that fits better usually ranks better, and fitting is worth doing — just not worth *reporting* as if it were the objective.

Now the chapter's sharpest statement, which is about this compiler rather than about cost models in general.

> **Proposition 46.6 (The analytic model is constant on the multi-level tiling space, stated here).** Let `B` be a block with `s` spatial and `r` reduction axes of constant extent, tiled by `CPU_TILING`, and let `p, q` be any two points of the `mlt_cpu` schedule space. Then `AnalyticalCostModel.score(program(p)) = AnalyticalCostModel.score(program(q))` on a CPU target.

*Proof.* The score is a fixed linear combination of seven terms ([`cost_model.ts:87`](../../../src/compiler/autotune/cost_model.ts)) that on a CPU target read exactly ten features: `numLoops`, `numParallelLoops`, `numVectorizedLoops`, `numSerialLoops`, `innermostExtent`, `strideOneAccesses`, `nonStrideOneAccesses`, `arithmeticIntensity`, `numMathOps`, `numExternCalls`. Take each in turn.

`multiLevelSplit` performs exactly `L−1` splits per axis regardless of the factors (Proposition 45.5), so every point produces the same loop count; `applyRoles` performs exactly one `parallelize` and one `vectorize`; hence `numLoops`, `numParallelLoops`, `numVectorizedLoops` and `numSerialLoops` are the same for all points.

Splitting rewrites a block's *bindings* and never its body, so the set of `BufferLoad`/`BufferStore` nodes and their subscripts is unchanged; `_checkStride` ([`features.ts:352`](../../../src/compiler/autotune/features.ts)) classifies an access by the last entry of its subscript list, which is an iteration variable in every case, so `strideOneAccesses` and `nonStrideOneAccesses` are unchanged, as are `numMathOps` and `numExternCalls`. The buffer set is unchanged, so `totalBufferBytes` and therefore `arithmeticIntensity = (math + extern)/bytes` ([`features.ts:103`](../../../src/compiler/autotune/features.ts)) are unchanged.

Finally `innermostExtent` is assigned on every `ForNode` visit ([`features.ts:267`](../../../src/compiler/autotune/features.ts)) and therefore holds the extent of the last loop the depth-first walk enters, which under `CPU_TILING`'s order `S0 S1 S2 S3 R0` is the reduction loop — and `L_R = 1`, so that loop is never split. All ten features agree, so the scores agree. ∎

Two hypotheses in that proof deserve to be named, because they are what a fix would have to change. The first is `L_R = 1`: with two reduction levels `innermostExtent` would vary. The second is that the walk's last `ForNode` is the innermost one, which is an accident of the traversal rather than a stated intent — the feature is named `innermostExtent` and is used by `_scoreVectorization` as if it were the extent of the *vectorised* loop.

> **Proposition 46.7 (The statement aggregation is max-dominated, stated here).** `aggregateStatements` ([`cost_model.ts:20`](../../../src/compiler/autotune/cost_model.ts)) reduces a function's statement vectors component-wise: by maximum for the seven features in `MAX_FEATURE_NAMES`, by mean for `arithmeticIntensity`, by sum otherwise. Consequently, if a schedule change alters a max-aggregated feature only in a statement that is not the arg-max, the aggregated row is unchanged and the learned model cannot see the change.

*Proof.* Immediate from `max` being insensitive to changes below the maximum. ∎

Proposition 46.7 bites whenever the model is applied to a whole function containing more than one block. §46.6 exhibits five schedules of one block of a two-block function collapsing to a single row — and also shows why the shipped path escapes it, which is that `BlockTuningSession` evaluates a single-block *mini* function instead ([`session.ts:104`](../../../src/compiler/autotune/session.ts)).

## 46.4 In mlfw

### Two extractors, two feature sets

[`features.ts`](../../../src/compiler/autotune/features.ts) contains two unrelated things that share a class name.

`FeatureExtractor.extract` ([`features.ts:83`](../../../src/compiler/autotune/features.ts)) walks the whole function once and returns a `ScheduleFeatures` object with 23 scalar fields: loop counts by kind, buffer bytes, read and write counts, arithmetic op counts, `innermostExtent` and `outermostExtent`, reduction depth, thread-block size, grid size, and stride-one versus strided access counts. This is the analytic model's input.

`FeatureExtractor.extractStatements` ([`features.ts:108`](../../../src/compiler/autotune/features.ts)) walks the same function and returns one 22-element vector per `BufferStore`, whose schema is a named constant:

```ts
export const STATEMENT_FEATURE_SCHEMA = [
  'iterCount', 'depth',
  'parallelLoops', 'vectorizedLoops', 'unrolledLoops', 'threadBoundLoops', 'serialLoops',
  'threadBlockSize', 'gridSize', 'underReduction',
  'numMathOps', 'numExternCalls', 'numReads', 'numWrites',
  'stride1Accesses', 'stridedAccesses', 'reuseCount', 'touchedBytes',
  'arithmeticIntensity', 'vectorized', 'parallelized', 'innermostExtent'
];
```

This is the learned model's input. The two sets overlap in intent and share no code: `iterCount` (the product of enclosing extents) exists only in the statement schema, and `totalBufferBytes` only in the whole-function one. Neither model can consume the other's features, and there is no path by which the learned model's `iterCount` — the one feature in either set that measures how much work the nest does — can inform the analytic one.

### The seven terms

[`cost_model.ts:45`](../../../src/compiler/autotune/cost_model.ts) is the whole weighting:

```ts
const DEFAULT_COST_WEIGHTS: CostWeightMap = {
  parallelism: 2.0,
  vectorization: 1.5,
  memoryCoalescing: 2.0,
  occupancy: 1.0,
  arithmeticIntensity: 1.0,
  loopOverhead: -0.5,
  codeSize: -0.3
};
```

Every term is computed to lie in `[0, 1]`, so the score lies in `[−0.8, 7.5]` and is dimensionless. A target may override the weights through `costModelWeights` ([`cost_model.ts:63`](../../../src/compiler/autotune/cost_model.ts)); no shipped target does.

Each term is one expression. Reading the branch each takes on a CPU target, with the line it lives on:

| Term | CPU-path value | Line |
|---|---|---|
| `parallelism` | `f.numParallelLoops / Math.max(f.numLoops, 1)` | [113](../../../src/compiler/autotune/cost_model.ts) |
| `vectorization` | `f.numVectorizedLoops > 0 ? Math.min(1.0, f.innermostExtent / this.target.vectorWidth) : 0` | [123](../../../src/compiler/autotune/cost_model.ts) |
| `memoryCoalescing` | `f.strideOneAccesses / total` | [129](../../../src/compiler/autotune/cost_model.ts) |
| `occupancy` | `1.0`, returned before anything is read | [133](../../../src/compiler/autotune/cost_model.ts) |
| `arithmeticIntensity` | `Math.min(1.0, f.arithmeticIntensity * 10 * boost)` | [143](../../../src/compiler/autotune/cost_model.ts) |
| `loopOverhead` | `f.numSerialLoops / Math.max(f.numLoops, 1)` | [147](../../../src/compiler/autotune/cost_model.ts) |
| `codeSize` | `Math.min(1.0, (f.numMathOps + f.numExternCalls) / 256)` | [151](../../../src/compiler/autotune/cost_model.ts) |

Three of them — parallelism, memory, overhead — are ratios of counts. `occupancy` is the constant 1 on CPU. `codeSize` and `intensity` depend on the block body, which a schedule does not change. That leaves `vectorization` as the only term on a CPU target that a schedule can move, and it moves it through `innermostExtent` alone.

`arithmeticIntensity` deserves one sentence because its name promises Chapter 4's quantity and does not deliver it: it is `(numMathOps + numExternCalls) / totalBufferBytes` ([`features.ts:103`](../../../src/compiler/autotune/features.ts)), where the numerator counts *syntactic* operations in the source text and the denominator sums the declared sizes of every buffer the walk saw. Chapter 4's arithmetic intensity is dynamic FLOPs per byte *moved*. The trip count appears in neither, so the formula's value *falls* as the matrices grow while Chapter 4's quantity rises: a 64×64 matmul is given 4,096 times the arithmetic intensity of a 4096×4096 one, when the true ratio is `n/6` and runs the other way.

### The learned model

`LearnedCostModel` ([`cost_model.ts:159`](../../../src/compiler/autotune/cost_model.ts)) is a thin wrapper: it accumulates `(aggregate(statement vectors), measuredScore)` pairs and refits gradient-boosted trees from scratch on every `train()`. `GradientBoostedTrees` ([`gbt.ts`](../../../src/compiler/autotune/gbt.ts), 136 lines) is a textbook implementation — exact split search on every feature by sorted order, variance-reduction criterion, depth-3 trees, learning rate 0.1, sixty of them:

```ts
      const cost = (leftSumSq - (leftSum * leftSum) / leftCount) + (rightSumSq - (rightSum * rightSum) / rightCount);
```

which is the sum of squared errors of the two children, so the fit is least-squares and the model is a regressor. `buildFeatureOrder` ([`gbt.ts:59`](../../../src/compiler/autotune/gbt.ts)) pre-sorts row indices once per fit, which is what keeps a full refit affordable at these sample counts.

`addSample` drops non-finite labels ([`cost_model.ts:179`](../../../src/compiler/autotune/cost_model.ts)) — a measurement that returned `Infinity` cannot poison the fit — and drops empty statement lists. It does not deduplicate, so the same schedule measured twice contributes two rows, which is the right behaviour for averaging noise and the wrong one when the two rows come from aliased parameters (Chapter 45).

### The handover

`GuidedCostModel` ([`cost_model.ts:213`](../../../src/compiler/autotune/cost_model.ts)) is twenty-three lines and one decision:

```ts
  score(primFunc: PrimFunc): number {
    if (this._learnedConfident()) {
      return (this.learned as LearnedCostModel).predict(FeatureExtractor.extractStatements(primFunc));
    }
    return this.analytical.score(primFunc);
  }
```

`_learnedConfident` is `trained && sampleCount >= confidenceSamples` ([`cost_model.ts:224`](../../../src/compiler/autotune/cost_model.ts)); `confidenceSamples` defaults to 8 and `topKForBenchmark` to 5, so the learned model takes over during the second measured round and never hands back. The two models live on unrelated scales — the analytic one in `[−0.8, 7.5]`, the learned one in negative milliseconds — and Theorem 46.3 is why that costs nothing: within any single sort, only one of them is talking.

## 46.5 Lab — what the analytic model sees

```bash
node docs/part8/ch46-cost-models/labs/01-what-the-analytic-model-sees.mjs
```

First, how much of the feature vector is live:

```
  FeatureExtractor.extract        -> 23 whole-function scalars, for the analytic model
  FeatureExtractor.extractStatements -> 1 vector(s) of 22, for the learned model

  of the 23 scalars, the CPU path of estimateFromFeatures reads 10:
    numLoops, numParallelLoops, numVectorizedLoops, numSerialLoops, innermostExtent, strideOneAccesses, nonStrideOneAccesses, arithmeticIntensity, numMathOps, numExternCalls
  two more are GPU-only (threadBlockSize, gridSize), and 11 are extracted and never read:
    numBlocks, totalIterations, maxLoopDepth, numUnrolledLoops, numThreadBound, totalBufferBytes, numBufferReads, numBufferWrites, outermostExtent, hasReduction, reductionDepth
```

Eleven of twenty-three are computed and discarded, and the list is worth reading rather than skimming. `totalIterations` is the product of every loop extent — the closest thing in either feature set to "how much work is this". `numUnrolledLoops` means `unroll` is invisible to the model, so the `R1: 'unroll'` role in `CPU_TILING_SSRSRS` would be unscored even if that structure could run. `outermostExtent` is the extent of the loop `applyRoles` parallelises, and it is extracted, and it is not read.

Then the breakdown, on the lowered 64×64 matmul before any scheduling:

```
  term                  raw        weight    contribution
  parallelism            0.000000        2        0.000000
  vectorization          0.000000      1.5        0.000000
  memoryCoalescing       1.000000        2        2.000000
  occupancy              1.000000        1        1.000000
  arithmeticIntensity    0.000407        1        0.000407
  loopOverhead           1.000000     -0.5       -0.500000
  codeSize               0.007813     -0.3       -0.002344
  total                                           2.498063
```

Two terms are pinned at their maximum for structural reasons — `occupancy` is `1.0` on any non-GPU target, and `memoryCoalescing` is 1 because every subscript in this nest ends in a plain iteration variable. `arithmeticIntensity` contributes 0.0004 of a possible 1. And a schedule changes neither the block body nor its subscripts, so on a CPU target five of the seven terms are constant *under scheduling*: for this block the score is `2·parallelism + 1.5·vectorization − 0.5·overhead` plus a fixed 2.998063.

Next a space where that is enough:

```
  vector_width   loops  par  vec  innermostExtent   vectorization term      score
             1       3    1    1                1               0.1250     3.6875
             2       3    1    1                2               0.2500     3.8750
             4       3    1    1                4               0.5000     4.2500
             8       3    1    1                8               1.0000     5.0000
            16       3    1    1               16               1.0000     5.0000
```

A genuine gradient, and every bit of it is the vectorization term: `min(1, innermostExtent/8)` with the CPU target's `vectorWidth = 8`. The model's entire preference over the elementwise space is "make the innermost extent at least the vector width", which is a defensible thing to prefer — and Chapter 42 measured what the CPU backend does with the resulting `@vectorized` annotation, which is nothing. On WASM, where the annotation becomes SIMD, the term earns its weight; on CPU the model is rewarding a marker the code generator ignores. The two widths that saturate the term, 8 and 16, tie, so the model has no opinion between a schedule that matches the vector width and one that doubles it.

And the space where it is not enough:

```
  points instantiated and validated: 2304
  distinct analytic scores:          1
    4.331396484375   first reached at s0=[1,1,1,64] s1=[1,1,1,64]

  why: the features the model reads are the same for every point.
  feature              [1,1,1,64]x[1,1,1,64]   [64,1,1,1]x[8,2,2,2]
  numLoops                           9.00000                9.00000
  numParallelLoops                   1.00000                1.00000
  numVectorizedLoops                 1.00000                1.00000
  numSerialLoops                     7.00000                7.00000
  innermostExtent                    64.0000                64.0000
  strideOneAccesses                  4.00000                4.00000
  nonStrideOneAccesses               0.00000                0.00000
  arithmeticIntensity           0.0000406901           0.0000406901
  numMathOps                         2.00000                2.00000
  numExternCalls                     0.00000                0.00000
```

Proposition 46.6, exhaustively. All 2,304 points instantiate, all pass `ScheduleValidator`, and all score `4.331396484375`. The two columns are the proof made concrete: the left-hand schedule parallelises a loop of extent 1 and the right-hand one a loop of extent 64, and every number the model reads is identical, because `numParallelLoops / numLoops` counts loops and `innermostExtent` reports the reduction axis.

The consequence for Chapter 47 is worth stating plainly. A search maximising a constant function is a search that returns whichever candidate its sort happened to put first. Everything the evolutionary search does — elitism, crossover, mutation — is defined in terms of a fitness that does not vary, so on this space the search degenerates to its initialisation.

On a GPU target the picture changes:

```
  mlt_gpu: 784 points, 31 distinct scores
```

because `_scoreParallelism` and `_scoreOccupancy` both read `threadBlockSize` and `gridSize`, which `_statementVector` and `_visitIterative` compute from the *extents* of the thread-bound loops ([`features.ts:274`](../../../src/compiler/autotune/features.ts)). Binding a loop to `threadIdx.x` makes its extent visible to the model; parallelising it does not. Thirty-one values over 784 points is still coarse — most of the space ties — but it is a function of the parameters, which is the minimum a search needs.

## 46.6 Lab — ranking, not regression

```bash
node docs/part8/ch46-cost-models/labs/02-ranking-not-regression.mjs
```

What a training sample is, on a four-statement function:

```
  feature                    stmt0      stmt1      stmt2      stmt3   aggregate   how
  iterCount                  1.000      256.0   1.049e+6   2.684e+8   2.6948e+8   sum
  numMathOps                 0.000      0.000      2.000      0.000      2.0000   sum
  numReads                   0.000      0.000      3.000      2.000      5.0000   sum
  numWrites                  1.000      1.000      1.000      1.000      4.0000   sum
  depth                      0.000      2.000      5.000      7.000      7.0000   max
  innermostExtent            0.000      16.00      16.00      16.00      16.000   max
  vectorized                 0.000      0.000      0.000      0.000      0.0000   max
  arithmeticIntensity        0.000      0.000  0.0004883  0.0004873   0.00024390   mean
```

Then Theorem 46.3, executed by running the same evolutionary search four times under four monotone transforms of one scoring function:

```
  score transform s                best = {"s0":[1,4,1,4],"s1":[4,4,1,1],"r0":[16]}  identical ranking: true
  score transform 3s - 1000        best = {"s0":[1,4,1,4],"s1":[4,4,1,1],"r0":[16]}  identical ranking: true
  score transform exp(s / 10)      best = {"s0":[1,4,1,4],"s1":[4,4,1,1],"r0":[16]}  identical ranking: true
  score transform -1 / (s + 100)   best = {"s0":[1,4,1,4],"s1":[4,4,1,1],"r0":[16]}  identical ranking: true
```

Four models whose predictions differ by many orders of magnitude, one trajectory, one answer, and the returned candidate lists identical position by position.

Counterexample 46.5:

```
  Definition 46.2 scores higher-is-better; this table is in predicted
  *cost*, lower-is-better, so the model picks an argmin.

  model                prediction            MSE   picks   true cost   regret
  A  perfect          [1,2,100]                0.00       0         1.0      0.0
  B  small error      [2,1,100]                0.67       1         2.0      1.0
  C  enormous error   [50,51,60]            2134.00       0         1.0      0.0
```

Then the honest other half, on the representation the session actually builds — twenty samples of one block's mini function across four problem sizes and five vector widths, labelled by a fixed function of the features so the table is deterministic:

```
  trees        MSE     model argmax   regret (ms)   discordant pairs
      1    3.301e-4       w=1 n=16        0.0010              15.5%
      2    2.685e-4       w=1 n=16        0.0010              13.9%
      4    1.779e-4       w=2 n=16        0.0004               8.6%
      8    7.824e-5       w=4 n=16        0.0002               5.3%
     16    1.516e-5       w=4 n=16        0.0002               3.7%
     32    6.555e-7      w=16 n=16        0.0000               0.0%
     60    7.660e-9      w=16 n=16        0.0000               0.0%
```

Error falls five orders of magnitude and the ranking converges with it. This is the usual case, and it is why fitting is worth doing at all. The default of sixty trees is comfortably past the point where the ranking has stabilised on twenty samples; the interesting question for a real tuner is what the curve looks like at eight samples, which is where `GuidedCostModel` hands over.

Then Proposition 46.7, and the reason it does not bite:

```
  vector_width   innermostExtent per statement (whole func)   max   mini func
             1                                    [0,64,1]    64   [1]
             2                                    [0,64,2]    64   [2]
             4                                    [0,64,4]    64   [4]
             8                                    [0,64,8]    64   [8]
            16                                   [0,64,16]    64   [16]

  distinct aggregated rows over the five widths — whole function: 1, mini function: 5
```

On the whole function, all five schedules give the identical aggregated row: the block being tuned has innermost extent 1 through 16, the `mul` block beside it has 64, and the max erases the difference. Every sample the learned model would receive has the same features and a different label, so it would fit their mean and rank them all equal.

The session escapes that by evaluating a *mini* function — `extractBlockMini` ([`tune_ir.ts:68`](../../../src/compiler/autotune/tune_ir.ts)) rebuilds the block and its enclosing loops as a standalone `PrimFunc` — which has one statement, so the max is over one element and is the identity. It is a genuine fix and it was made for a different reason: the mini function is what lets the search score a block without the rest of the program's noise.

The label is not given the same treatment. `_measure` benchmarks the *whole* scheduled function and pairs its median with the *mini* function's features ([`session.ts:229`](../../../src/compiler/autotune/session.ts), [`session.ts:231`](../../../src/compiler/autotune/session.ts)):

```ts
    const result = (this.benchmarkRunner as BenchmarkRunnerLike).run(scheduled);
    if (!result) return null;
    return { result, features: FeatureExtractor.extractStatements(miniScheduled) };
```

In a program with `k` blocks, every sample for every block carries the time of all `k`. The differences between candidates for one block survive — the other blocks contribute the same constant to each — so the *ranking within a block* is unaffected, which is what Theorem 46.3 says is all that matters within a round. What is affected is the model itself, since samples from different blocks share one `LearnedCostModel` ([`autotuner.ts:158`](../../../src/compiler/autotune/autotuner.ts)) and are fitted against labels that all include one another's time.

Last, the handover:

```
  samples   trained   guided.score(mini)   = analytic?   = learned?
        0   false               2.504167   true          false
        7   true                2.504167   true          false
        8   true               -0.001795   false         true
        9   true               -0.001255   false         true
```

Discontinuous in value, continuous in behaviour.

## 46.7 Traps and limits

- **The analytic model is a constant on the `mlt_cpu` space.** Proposition 46.6, and §46.5 evaluates all 2,304 points to confirm it. Since `mlt_cpu` is the largest reachable sketch for the only block class the compiler considers worth tiling, the cost model contributes nothing to the selection of a matmul schedule on CPU.
- **`_scoreParallelism` on CPU counts loops, not iterations.** [`cost_model.ts:113`](../../../src/compiler/autotune/cost_model.ts). A `@parallel` loop of extent 1 and one of extent 64 score identically, and `outermostExtent` — the number that would distinguish them — is extracted and never read.
- **`innermostExtent` is the last loop the walk visits, not the vectorised one.** [`features.ts:267`](../../../src/compiler/autotune/features.ts) assigns it on every `ForNode`. Under `CPU_TILING` the innermost loop is the reduction axis, so `_scoreVectorization` scores a loop that has nothing to do with the `@vectorized` annotation it is gating on.
- **Eleven of twenty-three whole-function features are dead.** §46.5 lists them. The ones that matter are `totalIterations` — the model has no term for the amount of work — and `numUnrolledLoops`, which makes `unroll` unscoreable.
- **`arithmeticIntensity` is not Chapter 4's arithmetic intensity.** [`features.ts:103`](../../../src/compiler/autotune/features.ts) divides a *syntactic* operation count by the sum of declared buffer sizes, with no trip count in either. The value is a function of the block body and the buffer set alone, so no schedule can change it, and a larger tensor with the same body gets a *lower* one.
- **The two models share no features.** `ScheduleFeatures` (23 scalars) and `STATEMENT_FEATURE_SCHEMA` (22 per statement) are built by separate walks with separate definitions of the same names — `innermostExtent` means "last loop visited in the function" in one and "extent of this statement's innermost enclosing loop" in the other.
- **The max aggregation hides a per-block change in a multi-block function.** Proposition 46.7 and §46.6. It is inert today only because `BlockTuningSession` evaluates a single-block mini function; the `needsWholeFunc` branch that would reintroduce it ([`session.ts:96`](../../../src/compiler/autotune/session.ts)) is the one guarded by the `fused` sketch, which Chapter 45 showed is never derived.
- **One learned model is shared across every block of every function in a compile.** `Autotuner` constructs one ([`autotuner.ts:158`](../../../src/compiler/autotune/autotuner.ts)) and hands it to every session. Samples from a matmul and from an elementwise block are fitted together, with labels that both include the other's time.

## 46.8 Read the tests

- [`tests/compiler/autotune/cost-model.test.js`](../../../tests/compiler/autotune/cost-model.test.js) — the learned model against a threshold target, a non-finite label, XOR, and a serialize/deserialize round trip. The XOR test is the one that earns its place: it pins that the model is non-linear, which is the property that makes it worth having over the analytic one. The `AnalyticalCostModel` test asserts only that the score is finite and has a breakdown; nothing there would fail if the model were constant.
- [`tests/compiler/autotune/ansor.test.js`](../../../tests/compiler/autotune/ansor.test.js) — `GuidedCostModel closes the loop` walks the handover sample by sample, asserting the score equals the analytic one below the threshold and the learned one above it. It is the executable form of §46.6's last table.
- Several of these tests pass options the current model does not have — `new LearnedCostModel(null, { seed: 2, epochs: 1500 })` at [`ansor.test.js:126`](../../../tests/compiler/autotune/ansor.test.js), and a `seed` in eight other places — left over from a neural implementation. They are inert: `LearnedCostModel` reads only `numTrees`, `maxDepth`, `lr` and `minSamples` ([`cost_model.ts:166`](../../../src/compiler/autotune/cost_model.ts)). Worth knowing before you go looking for where the seed is used.

---

**Next:** [Chapter 47 — Search and measurement](../ch47-search-and-measurement/README.md), which takes a space too large to enumerate and an objective that may or may not vary over it, and spends thirty seconds — and finds that at the shipped default seed it spends them all on one sketch.
