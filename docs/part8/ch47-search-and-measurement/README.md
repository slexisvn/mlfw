# Chapter 47 — Search and measurement

Everything so far has been static. A sketch generates a space; a cost model orders it; neither has executed a kernel. This chapter is where the compiler starts spending real time: it draws candidates, scores them, compiles a few, runs them on the machine, learns from what the clock said, and stops when a budget runs out. Then it writes the answer down so the next compilation does not have to repeat it.

Four mechanisms, and each has a precise obligation. The search has to produce diverse candidates from a seeded generator. The benchmark has to turn a noisy clock into one number the search can trust. The budget has to stop the whole thing at a stated time. The database has to recognise, tomorrow, that tomorrow's problem is today's problem — and to refuse when it is not.

## 47.1 The problem: thirty seconds, two thousand candidates, a noisy clock

`timeBudgetMs` defaults to 30,000 ([`autotuner.ts:121`](../../../src/compiler/autotune/autotuner.ts)). A 64×64 matmul has one block worth tuning, with 2,304 points in the largest sketch that can actually run. Compiling and timing one candidate on this machine takes a few milliseconds; ten repeats and three warm-ups is more. Thirty seconds is a few thousand measurements if nothing else happens, and a real model has more than one block.

So the loop cannot be "measure everything". It has to be: draw a population, score all of it with the model, measure only the top few, use those measurements to improve the model, repeat. That is the standard autotuning loop and this compiler implements all of it. The interesting questions are the ones that only appear when you run it:

- Where do the candidates come from, and are they diverse?
- What does a "measurement" summarise away?
- When the budget expires mid-generation, how far past it does the search run?
- Two blocks, one budget: who gets the rounds?
- And the next compilation: what makes tomorrow's problem the same problem?

## 47.2 Intuition: a population, a clock, and a ledger

**The search** is a genetic algorithm with elitism. Hold a population of parameter vectors; score them; keep the best fifth unchanged; fill the rest by mutating and crossing the keepers; repeat. Elitism is what makes it monotone — the best individual is never lost — and mutation is what stops it collapsing onto one point.

**The measurement** is a median. Run the kernel a few times, sort the timings, take the middle one. The median throws away the tail, which is what you want when the operating system schedules something else on your core halfway through, and it throws away the minimum too, which is a choice with two sides. The minimum is the sample least contaminated by whatever else the machine was doing, which is the argument for it; it is also a downward-biased estimator whose expectation keeps falling as you add repeats, so it answers "what is the best this kernel has ever done" rather than "what will it cost".

**The ledger** is a hash table from a description of the problem to the winning parameters — coarse enough that two 512×512 matmuls in different models share an entry, fine enough that a 512×512 and a 513×512 do not, and carrying a version that can invalidate everything at once when the compiler changes underneath it.

## 47.3 Theory

> **Definition 47.1 (Elitist evolutionary search, stated here).** Given a space `P`, a fitness `f : P → ℝ`, a population size `N`, an elite ratio `ρ` and a mutation operator `μ`, the search maintains a **multiset** `P₀` of `N` elements of `P` and iterates: score `P_g`; let `E_g` be the `⌈ρN⌉` highest-scoring members; set `P_{g+1} = E_g ⊎ { μ(crossover(a,b)) : a,b ∈ E_g }` filled to size `N`. It returns `argmax_{p ∈ P_G} f(p)`.

**A multiset, not a subset — and the difference is the subject of the next two results.** It is natural to write `P₀ ⊂ P` with `|P₀| = N` and it would misdescribe every generation this search produces. Nothing forbids two individuals from being the same parameter vector: mutation may return its argument, crossover of two identical parents is that parent, and elitism copies the keepers forward verbatim. So `N` is a count of *individuals*, and the number of *distinct points* it covers is at most `N` and routinely far less — Corollary 47.4 exhibits an initial population of `N` individuals covering, from the third onward, a single sketch.

This matters in two places downstream. The memoisation in `search.ts` keys on the parameter vector, so duplicates cost nothing to score — which is why Proposition 47.9's `2N` bound is an over-estimate. And the diversity the algorithm relies on to avoid collapsing is a property of the distinct set, not of `N`, so a population that looks healthy by count can be exploring one point.

> **Proposition 47.2 (Elitism makes the best monotone, stated here).** If `f` is a function of the candidate — the same candidate always scores the same — then `max_{p ∈ P_{g+1}} f(p) ≥ max_{p ∈ P_g} f(p)` for every `g`.

*Proof.* `E_g` contains an arg-max of `P_g` and `E_g ⊆ P_{g+1}` by construction ([`search.ts:140`](../../../src/compiler/autotune/search.ts)). ∎

The hypothesis is not free. `f` here is `evaluator`, which clones the function, applies the sketch, runs `ScheduleValidator` and calls the cost model — deterministic, but a *learned* cost model changes between rounds, so `f` is a function only within one call to `search`. The code makes the hypothesis hold where it needs to by memoising on the parameter vector ([`search.ts:116`](../../../src/compiler/autotune/search.ts)), which also means an elite is never re-evaluated and so cannot be re-scored differently.

Now the generator. `_rng(max)` is `state % max` over an LCG with a power-of-two modulus, and that combination has a classical failure mode.

> **Proposition 47.3 (The initial population's sketch index reaches a fixed point, stated here).** Let `x_{n+1} = (a·x_n + c) mod 2^31` with `a = 1664525`, `c = 1013904223`, and let `_rng(m) = x mod m`. Since `a ≡ 1 (mod 4)` and `c ≡ 3 (mod 4)`, we have `x_{n+1} ≡ x_n + 3 (mod 4)`. Suppose the sketch list has length 4 and `_initPopulation` draws one variate for the index and one per search variable of the sketch it drew. Then the index of individual `i+1` is `idx_i + 3·(1 + |V_{idx_i}|) (mod 4)`. An index whose sketch has three search variables is a fixed point; from an index whose sketch has one, the next index is `idx + 2`; from one whose sketch has none, `idx + 3`.

*Proof.* `1664525 = 4·416131 + 1` and `1013904223 = 4·253476055 + 3`, so reducing the recurrence mod 4 gives `x ↦ x + 3`. Advancing `k` draws advances the residue by `3k ≡ −k (mod 4)`. With `k = 1 + |V|`: `|V| = 3` gives `k = 4` and no advance; `|V| = 1` gives `k = 2` and an advance of `−2 ≡ 2`; `|V| = 0` gives `k = 1` and an advance of `3`. ∎

> **Corollary 47.4 (stated here).** On a CPU target the four derived sketches for a contraction block have `3, 3, 1, 0` search variables at indices `0, 1, 2, 3`. The index map is therefore `0 ↦ 0`, `1 ↦ 1`, `2 ↦ 0`, `3 ↦ 2`; every orbit reaches `0` or `1` in at most two steps and stays. The initial population contains at most two distinct sketches and, from the third individual on, exactly one — which one being decided entirely by the seed.

`RandomSearch` is immune, because it iterates over the sketch list rather than sampling it ([`search.ts:62`](../../../src/compiler/autotune/search.ts)).

Next, the cache key.

> **Definition 47.5 (Workload key, stated here).** The *workload key* of a block is the FNV-1a hash of: the shapes and dtypes of its declared read and write buffers; a pre-order serialisation of its body's expression tree in which every load carries its buffer's shape and dtype; the sorted serialisations of the bodies of every other block that declares a read of its output buffer; and the target's `name` and `kind`.

> **Proposition 47.6 (What the key does and does not determine, stated here).** Two blocks with equal *descriptions* have equal declared buffer shapes and dtypes, equal expression trees up to buffer naming, and equal target name and kind. They need not have equal iteration domains, equal loop counts, or equal target attributes other than name and kind. The key is the 32-bit FNV-1a hash of the description, so equal *keys* imply equal descriptions only under a no-collision assumption, which nothing checks.

*Proof.* The first three follow from the construction, since the description is the concatenation of exactly those parts ([`workload_key.ts:30`](../../../src/compiler/autotune/workload_key.ts) to [`workload_key.ts:58`](../../../src/compiler/autotune/workload_key.ts)). The negative half follows from the same construction: `collectBlockOps` descends into a `ForNode` by recursing on its body ([`workload_key.ts:133`](../../../src/compiler/autotune/workload_key.ts)) without emitting anything for the loop itself, and the enclosing loops are outside the block. `target` contributes only `name` and `kind`. The last sentence is `fnv1a` ([`workload_key.ts:152`](../../../src/compiler/autotune/workload_key.ts)) mapping into a 32-bit space; §47.6 exhibits two descriptions that collide. ∎

> **Counterexample 47.7 (Two ways to share a key).** Three nests over the same two 64-element buffers with the same body and loop extents 64, 32 and 3 have the same description, hence the same key. Their `elementwise_cpu` sketches admit different vector widths: the third can only use width 1, because the sketch's split is guarded by `extent >= vector_width * 2`. Separately, the same elementwise block over buffers of 10,039 and of 11,827 elements has two *different* descriptions and one key, because 32 bits collide. §47.6 runs both.

Then the measurement, where the theory is classical and slightly discouraging.

> **Theorem 47.8 (Selection bias in a noisy minimum).** *(Classical.)* Let `T̂₁,…,T̂_n` be measurements with `E[T̂_i] = T_i`. Then `E[min_i T̂_i] ≤ min_i T_i`, with equality only if the minimising index is almost surely the one with the smallest mean.

*Proof.* `min_i T̂_i ≤ T̂_j` pointwise for every `j`, so `E[min_i T̂_i] ≤ E[T̂_j] = T_j` for every `j`, hence `≤ min_j T_j`. ∎

The tuner's reported best is a minimum over candidates of a median over repeats, so Theorem 47.8 applies at the candidate level, not the sample level: it is the `min` *over candidates* that is optimistic. The size of the bias grows with the number of candidates measured and with the residual noise in each median — which is why `maxCv` exists, and why its default of 0 (which disables it) means nothing bounds the residual.

**Applying the theorem to this estimator needs one step the theorem does not supply, and it is worth not skipping.** Theorem 47.8 assumes `E[T̂_i] = T_i` — that each per-candidate estimate is *unbiased*. The tuner's per-candidate estimate is the upper median of a small sample ([`benchmark.ts:71`](../../../src/compiler/autotune/benchmark.ts)), and **a median is not an unbiased estimator of a mean**. It is unbiased for the *population median*, and the two coincide only for a symmetric distribution. Benchmark timings are not symmetric: they have a hard floor at the true cost and an unbounded right tail of interference, so the median sits below the mean, systematically.

That does not rescue the tuner — it makes the situation slightly worse and slightly different:

- Taking `T_i` to mean *the population median of candidate i's timing*, Theorem 47.8 applies exactly as stated, because the sample median is (asymptotically) unbiased for it. This is the reading the chapter uses, and under it "optimism over candidates" is the whole of the bias.
- Taking `T_i` to mean *the candidate's true cost*, there is a second, per-candidate downward bias on top of the selection bias, and the two compound.

The right-skew is also the argument for preferring the median here in the first place: it is what makes a single interfering sample unable to move the estimate, which a mean cannot claim. So the estimator is a good choice and the phrase to avoid is "unbiased" — what it is, is *robust*, which is a different property and the one that matters when the tail is interference rather than signal. (Chapter 15 §15.7 works through the same trade in the opposite direction, for a quantity where the minimum is defensible.)

> **Proposition 47.9 (Budget overshoot, stated here).** Let one cost-model evaluation cost at most `e` and one measurement at most `b`. Then `RandomSearch` starts no trial after its deadline has passed and overruns only by the evaluation already in flight; `EvolutionarySearch` performs at most `2N` further evaluations, `N` for the generation it had already entered and `N` for the final scoring pass, which tests no deadline ([`search.ts:157`](../../../src/compiler/autotune/search.ts)); `BlockTuningSession._measureAndLearn` performs at most one further measurement ([`session.ts:203`](../../../src/compiler/autotune/session.ts)); and `TaskScheduler` starts at most one further round ([`task_scheduler.ts:54`](../../../src/compiler/autotune/task_scheduler.ts)). The total overshoot is bounded by `2Ne + kb` where `k = topKForBenchmark`.

*Proof.* By inspection of the four loops and the position of their deadline tests. The memoisation makes `2N` an over-estimate whenever the population contains duplicates. ∎

## 47.4 In mlfw

### The loop, top to bottom

`Autotuner.tune` ([`autotuner.ts:186`](../../../src/compiler/autotune/autotuner.ts)) is sixty-five lines and the whole architecture:

```ts
    for (const name of blockNames) {
      const key = computeWorkloadKey(primFunc, name, this.target, blockMap);
      keyByBlock.set(name, key);
      const existing = tasksByKey.get(key);
      if (existing) { existing.weight++; continue; }

      if (this.config.useTuningDB && this.db.has(key)) {
        tasksByKey.set(key, { key, kind: 'cache', cached: this.db.lookup(key), weight: 1 });
        continue;
      }

      const sketches = getSketchesForBlock(primFunc, name, this.target, blockMap, { … });
      if (sketches.length === 0) {
        tasksByKey.set(key, { key, kind: 'empty', weight: 1 });
        continue;
      }
      …
      tasksByKey.set(key, { key, kind: 'session', session, weight: 1 });
    }
```

Blocks are grouped by key before anything is searched, so two identical convolutions in a network are one tuning task with `weight: 2`. Three task kinds: served from cache, nothing to do, or a live session. The live ones go to the scheduler, which runs rounds until the deadline; then the results are collected and, for a session that produced a best, written back to the database.

`tuneAndApply` ([`autotuner.ts:252`](../../../src/compiler/autotune/autotuner.ts)) then builds *two* schedules — the tuned one and the rule-based default — and adopts the tuned one only if the baseline is not a "strong" backend schedule, or the tuned one is strong and a hardware measurer was used ([`autotuner.ts:270`](../../../src/compiler/autotune/autotuner.ts)). That gate is Chapter 43's register-blocked GPU kernel defending itself against a cost-model-only search.

Whatever the tuning did or did not produce, `_scheduleResidualBlocks` ([`autotuner.ts:343`](../../../src/compiler/autotune/autotuner.ts)) walks every block that is still unscheduled and applies the rule policy. **The autotuner cannot leave a block unscheduled**, and that is why a search that finds nothing is invisible in the output.

### One round

`BlockTuningSession.runRound` ([`session.ts:122`](../../../src/compiler/autotune/session.ts)) is seventeen lines:

```ts
  runRound(): number {
    const prev = this._best ? this._best.measuredScore : -Infinity;
    const candidates = this._produceCandidates();
    if (candidates.length === 0) {
      this.plateaued = true;
      return 0;
    }
    if (this.benchmarkRunner) {
      this._measureAndLearn(candidates);
    } else {
      const top = candidates[0];
      this._consider({ sketchName: top.sketchName, params: top.params, score: top.score, measuredScore: top.score });
      this.plateaued = true;
    }
    const now = this._best ? this._best.measuredScore : -Infinity;
    return Math.max(0, now - prev);
  }
```

Two paths. With a benchmark runner: score the population, measure the top `k`, learn from the timings. Without one: take `candidates[0]` and declare the task finished. The second path is the shipped CPU behaviour, because `enableBenchmark` defaults to `hardwareMeasure || !!measurer` ([`autotuner.ts:127`](../../../src/compiler/autotune/autotuner.ts)) and both are off.

The return value is the round's improvement, and its first value is `Math.max(0, s − (−∞)) = ∞` — an arithmetic detail with consequences for the budget allocator below, which §47.7 follows up.

### The candidate filter

`_evaluate` ([`session.ts:181`](../../../src/compiler/autotune/session.ts)) is where a candidate becomes a score or a `null`:

```ts
      const cloned = clonePrimFunc(this.evalFunc);
      const sch = new Schedule(cloned);
      sketch.instantiate(params)(sch, this.evalBlockName, this.target);
      const errors = ScheduleValidator.validate(cloned);
      if (errors.length > 0) return null;
      const blockLimit = this.target.maxThreadsPerBlock;
      if (this.target.isGPU && this.target.isGPU() && blockLimit && gpuThreadBlockSize(cloned) > blockLimit) return null;
      return { score: this.costModel.score(cloned) };
```

This is the one production caller of `ScheduleValidator` in the compiler: a searched schedule is validated, a rule-produced one is not. Chapter 42 reaches the same asymmetry from the other end.

The `catch` around it warns once per sketch name ([`session.ts:192`](../../../src/compiler/autotune/session.ts)) and returns `null`, which is how 6,125 `ssrsrs_cpu` refusals become one warning line.

### Measurement

`BenchmarkRunner.run` ([`benchmark.ts:151`](../../../src/compiler/autotune/benchmark.ts)) compiles the `PrimFunc` through the real backend pipeline, builds a callable with `new Function`, warms it up, and collects samples. Two details are worth having:

```ts
    for (const buf of buffers) {
      for (let i = 0; i < buf.length; i++) buf[i] = random() * 2 - 1;
    }
```

Buffers are cached by shape signature and refilled with fresh uniform random data on every call ([`benchmark.ts:145`](../../../src/compiler/autotune/benchmark.ts)), so no candidate is timed on whatever the previous candidate left in memory — which matters more than it sounds, since a buffer of zeros and a buffer of denormals do not cost the same to multiply. They are always `Float32Array`, whatever the buffer declares; §47.7 says what that costs.

```ts
    for (let round = 0; round <= this.maxReMeasures; round++) {
      this._collect(fn, buffers, samples);
      stats = robustStats(samples);
      if (this.maxCv <= 0 || stats.cv <= this.maxCv) break;
    }
```

Measure, summarize, and measure again if the samples were too noisy to trust. That is the shape every benchmark harness converges on, and §1.8's reporting rule is why the summary is a median rather than a minimum: `robustStats` ([`benchmark.ts:71`](../../../src/compiler/autotune/benchmark.ts)) returns the upper median, the minimum, a 10%-trimmed mean and a coefficient of variation, and the search learns from the median ([`session.ts:206`](../../../src/compiler/autotune/session.ts)). The loop's default settings turn most of that machinery off, which §47.7 takes up.

### The task scheduler

`GradientSchedulerPolicy.pick` ([`task_scheduler.ts:17`](../../../src/compiler/autotune/task_scheduler.ts)) is the budget allocator:

```ts
    const cold = live.filter(t => t.rounds === 0);
    if (cold.length > 0) return cold[0];
    let best: TuningTask | null = null;
    let bestPriority = -Infinity;
    for (const t of live) {
      const gain = t.gainEwma !== undefined ? t.gainEwma : t.lastGain;
      const priority = t.weight * gain;
      if (priority > bestPriority) {
        bestPriority = priority;
        best = t;
      }
    }
```

Every task gets one round first, then rounds go to the task with the highest `weight × recent gain` — the standard "spend where the marginal return is highest" rule, weighted by how many blocks in the program share this workload. `gainEwma` is an exponential moving average with `α = 0.5` ([`task_scheduler.ts:60`](../../../src/compiler/autotune/task_scheduler.ts)), and a task retires when it plateaus, goes stale twice, or hits `maxRoundsPerTask`.

The arithmetic of that averaging is where §47.5 finds something.

### The database

[`tuning_db.ts`](../../../src/compiler/autotune/tuning_db.ts), 158 lines: `store` appends, re-sorts by `rankRecords` ([`tuning_db.ts:53`](../../../src/compiler/autotune/tuning_db.ts)) and truncates to ten records per key, and invalidation is the single line

```ts
    if (data.codegenVersion !== undefined && data.codegenVersion !== CODEGEN_VERSION) {
```

against `CODEGEN_VERSION = 'mlfw-codegen-1'` ([`tuning_db.ts:27`](../../../src/compiler/autotune/tuning_db.ts)). Bumping that string discards every stored record, which is the right lever: a change to a lowering rule or a backend can make yesterday's winner today's loser. §47.6 runs both the ranking and the guard.

## 47.5 Lab — the search loop

```bash
node docs/part8/ch47-search-and-measurement/labs/01-the-search-loop.mjs
```

The generator first:

```
  multiplier 1664525 mod 4 = 1    increment 1013904223 mod 4 = 3
  so x mod 4 obeys  x <- (x + 3) mod 4,  and x mod 2 obeys  x <- (x + 1) mod 2

  _rng( 2) from seed 42:  1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0
  _rng( 4) from seed 42:  1 0 3 2 1 0 3 2 1 0 3 2 1 0 3 2
  _rng( 8) from seed 42:  1 4 3 6 5 0 7 2 1 4 3 6 5 0 7 2
  _rng( 5) from seed 42:  3 3 4 4 1 2 0 2 1 1 2 1 1 4 0 4
  _rng(48) from seed 42:  1 28 27 30 21 16 47 34 9 36 35 38 29 8 23 26
```

The last two rows are why this is easy to miss: `_rng(48)` and `_rng(5)` look perfectly random, because a non-power-of-two modulus mixes in the high bits. Every *parameter* draw in this compiler uses a candidate-list length of 5, 6, 20, 35 or 48, and those are fine. The one draw that uses a power of two is the choice of sketch.

```
  sketches: 0=mlt_cpu(3v)  1=ssrsrs_cpu(3v)  2=rfactor(1v)  3=reduction_cpu(0v)

  seed   1:  indices 000000000000   population drawn from {mlt_cpu}
  seed   7:  indices 200000000000   population drawn from {rfactor, mlt_cpu}
  seed  42:  indices 111111111111   population drawn from {ssrsrs_cpu}
  seed 123:  indices 200000000000   population drawn from {rfactor, mlt_cpu}
```

Corollary 47.4, executed. Every orbit reaches a fixed point immediately or after one step, and the fixed points are the two three-variable sketches. Which one the seed lands on decides the entire initial population, and since elitism preserves the sketch of every elite and `_mutate` never changes it ([`search.ts:187`](../../../src/compiler/autotune/search.ts)), it decides every subsequent generation too.

And the shipped default seed lands on the sketch that always throws:

```
  evolutionary, seed 42 (the default)  matmul_init_0 -> elementwise_cpu
  evolutionary, seed 1                 matmul_1 -> mlt_cpu   matmul_init_0 -> elementwise_cpu
  evolutionary, seed 7                 matmul_1 -> mlt_cpu   matmul_init_0 -> elementwise_cpu
  random, seed 42                      matmul_1 -> mlt_cpu   matmul_init_0 -> elementwise_cpu
```

At seed 42 every member of the initial population is `ssrsrs_cpu`; every one of them raises `decomposeReduction: block 'matmul_1' has no initBody`; `scored` is empty so generation 0 breaks immediately; `finalScored` is empty too; `candidates.length === 0`, so `runRound` returns 0 and marks the task plateaued; `best()` is `null` and `tune` skips the block. **The matmul — the only block in the function whose schedule costs anything — receives no tuning result at the compiler's default configuration.**

The output is still correct, and that is the reason this has never been noticed: `_scheduleResidualBlocks` gives the block the rule policy's schedule, and the end-to-end tests assert that the tuned kernel matches the untuned one, which it does.

Then elitism:

```
  populationSize 10 x numGenerations 6 = 60 slots
  evaluator calls: 29   distinct parameter vectors: 29
  best found: 9.9014 at {"s0":[64,1,1,1],"s1":[16,2,1,2],"r0":[64]}
  best-so-far after each block of 10 evaluations: 7.9056 -> 9.9014 -> 9.9014
```

Proposition 47.2 in action, and a second fact worth noticing: 60 population slots produced 29 evaluator calls. The memo absorbed 31 of them, which is elites being carried forward unchanged and children colliding with their parents. That is the cheap half of elitism — the expensive half is that a population which has converged spends its whole budget re-drawing points it has already seen.

The budget:

```
  RandomSearch(numTrials 1000)         budget 50 "ms", each evaluation costs 10: 5 evaluations, clock at 50
  EvolutionarySearch(pop 8, gens 50)   budget 50 "ms", each evaluation costs 10: 10 evaluations, clock at 100
```

Proposition 47.9 with `N = 8`: the random search stops exactly on time, because its deadline test comes before the trial, and the evolutionary one runs to twice the budget. The 10 rather than 16 is the memo: the dummy sketch has one search variable and the mutation rate is 0.3, so seven children in ten are copies of the parent they came from, and the elite is copied unmutated.

Finally the task scheduler, run twice with different gain reports:

```
  every round reports a finite gain of 1:   weight-1  weight-10  weight-10  weight-10  weight-10  weight-1  weight-1  weight-1
  every round reports Infinity:             weight-1  weight-10  weight-1  weight-1  weight-1  weight-10  weight-10  weight-10
```

The first row is what the policy is for: after the cold round each, the weight-10 task takes its four rounds and the weight-1 task takes what is left. The second row is list order.

And the second row is the real one. `runRound` computes its improvement against a starting best of `-Infinity` ([`session.ts:123`](../../../src/compiler/autotune/session.ts)), so the first round that produces any candidate at all returns `Infinity`. Then `gainEwma = 0.5·Infinity + 0.5·gainEwma` stays `Infinity` for ever, `weight × Infinity` is `Infinity` for every weight, and `priority > bestPriority` is false between two infinities. **The weight the tuner computes by grouping blocks by workload key never affects the allocation.** The scheduler is a round-robin over the task list, one round each until each retires.

## 47.6 Lab — measurement and the cache

```bash
node docs/part8/ch47-search-and-measurement/labs/02-measurement-and-the-cache.mjs
```

What a benchmark summarises away:

```
  samples                              median     min   trimmedMean       cv
  ten clean runs                         1.00    1.00        1.0000   0.0000
  nine clean, one interrupted            1.00    1.00        1.0000   0.0000
  a slow first run (no warmup)           1.00    1.00        1.0000   0.0000
  a machine under load                   3.00    1.00        3.5000   0.7636
```

Rows two and three are the design working: one interrupted run and one cold run are both invisible in every statistic, because the median ignores them and the 10% trim removes exactly one sample from each end. Row four is the case the statistics cannot rescue — a machine that is loaded throughout gives a median three times the minimum and a cv of 0.76, and `maxCv`'s default of 0 means nothing acts on it. The number that reaches the search is the 3.00.

The workload key:

```
  program                            block           key
  matmul 8x8 @ 8x8                   matmul_init_0   4915ab70
  matmul 8x8 @ 8x8                   matmul_1        1529aa07
  matmul 16x8 @ 8x8                  matmul_init_0   29040c58
  matmul 16x8 @ 8x8                  matmul_1        94ea1a2c
  matmul 8x8 @ 8x16                  matmul_init_0   eff4068c
  matmul 8x8 @ 8x16                  matmul_1        996c9738

  the 8x8 block on WasmTarget()      matmul_1        ea1596bc
  ... on CPUTarget({vectorWidth:4})  matmul_1        1529aa07
```

Shapes separate, targets of different kinds separate, and two CPU targets with different vector widths do not — the key ends with `target.name` and `target.kind`, and both are `cpu_generic`/`cpu`. `_scoreVectorization` divides by `target.vectorWidth`, so those two targets have different cost models and one cache entry.

Counterexample 47.7:

```
  loop extent   key         elementwise_cpu behaviour
           64   cd60d238    widths that split: {1, 2, 4, 8, 16}
           32   cd60d238    widths that split: {1, 2, 4, 8, 16}
            3   cd60d238    widths that split: {1}
```

One key, three iteration domains. The key describes the *data* — buffer shapes, dtypes, the expression tree — and never the nest, because `collectBlockOps` walks through a `ForNode` to its body and emits nothing for the loop. For a matmul the buffer shapes determine the extents and the key is adequate. For a block whose domain is not its buffer shape — a padded region, a partial reduction, a nest over a slice — it is not, and the failure is silent: a cached `vector_width: 8` applied to the extent-3 nest takes the else branch and produces a bare `parallelize`.

And the second half of Counterexample 47.7, which is about the hash rather than the description:

```
=== two different workloads, one key ===

  the same block over buffers of shape [10039]   key = 9a89ea08
  the same block over buffers of shape [10040]   key = 0b811cfb
  the same block over buffers of shape [11827]   key = 9a89ea08
```

Two genuinely different problems, one cache entry. `fnv1a` ([`workload_key.ts:152`](../../../src/compiler/autotune/workload_key.ts)) maps the description into 32 bits, so a birthday search over the descriptions this compiler builds — vary one buffer extent, hash, repeat — finds a pair inside the range of ordinary tensor sizes. This is not a defect in FNV-1a, which is doing what a 32-bit hash does; it is that nothing downstream can notice. `TuningRecord` keeps the key and not the string it hashed ([`tuning_db.ts:29`](../../../src/compiler/autotune/tuning_db.ts)), so `lookup` cannot compare the description it was asked for against the one that was stored, and the 11,827-element block silently inherits the 10,039-element block's parameters. A canonical description beside the key would turn a wrong answer into a miss.

The database:

```
  three records under one key, after `rankRecords` (tuning_db.ts:53):
    {"vector_width":4}     score  -0.9   medianMs 0.9
    {"vector_width":8}     score     5   medianMs null
    {"vector_width":2}     score   3.9   medianMs null

  codegenVersion written:       mlfw-codegen-1
  reload with the same version: 3 records
  reload with a different one:  0 records
  reload with the field absent: 3 records
  a record claiming version 9, stored into a version-1 database: kept, version 9
```

The ranking is right and the invalidation has a hole in it. `data.codegenVersion !== undefined && …` means a serialised database with no `codegenVersion` field loads unconditionally — and a file with no `codegenVersion` field is precisely a file written before the field existed, which is the one file the mechanism is there to reject. The per-record `version`, which `TuningRecord` stores and `Autotuner` fills from `this.db.version`, is written, serialised, restored, and compared to nothing.

Finally the whole loop, through the public compiler:

```
  after the first compile the database holds 3 record(s)
  the second compile reports 3/3 cache hits
  the two kernels agree up to loop-variable numbering: true
  byte-for-byte identical: false
  and against the untuned baseline: max |difference| = 0.00e+0
```

Three blocks, three keys, three records; the second compilation hits all three and regenerates the same kernel — up to the fresh-variable counter, which has advanced. The last line is the property that matters most and is easiest to take for granted: the tuned kernel and the untuned one agree exactly, because every point of the space is a sound schedule.

And the shipped default:

```
  new Autotuner(CPUTarget(), {}):  hardwareMeasure=false  measurer=null  enableBenchmark=false
  benchmarkRunner: null
```

So on a CPU compile with `scheduling: { autotune: true }`, none of §47.5's measurement machinery runs. `runRound` takes the second branch, adopts `candidates[0]`, and marks the task plateaued after one round. Combined with Chapter 46's Proposition 46.6 — the cost model is constant on the tiling space — the shipped CPU tuner selects the first candidate its sort happened to return, from a population drawn from one sketch. Turning `enableBenchmark: true` on changes all of that, and it is one option away.

## 47.7 Traps and limits

- **The evolutionary search draws its whole initial population from one sketch.** Proposition 47.3 and Corollary 47.4. The sketch is decided by the seed, and the shipped default of 42 ([`autotuner.ts:120`](../../../src/compiler/autotune/autotuner.ts)) selects `ssrsrs_cpu`, which throws on every lowered block — so a CPU matmul gets no tuning result at all. `RandomSearch` and seeds 1 or 7 do not have the problem. The output is still correct, because `_scheduleResidualBlocks` applies the rule policy to whatever the search left alone.
- **`gainEwma` becomes `Infinity` on the first round and never decays**, because `runRound` measures its first improvement against `-Infinity` ([`session.ts:123`](../../../src/compiler/autotune/session.ts)). `weight × Infinity` is the same for every weight, so `GradientSchedulerPolicy` degenerates to list order and the block-grouping weight computed at [`autotuner.ts:200`](../../../src/compiler/autotune/autotuner.ts) is inert.
- **A round in which every measurement fails returns `NaN`**, since `-Infinity − (−Infinity)` is `NaN` and `Math.max(0, NaN)` is `NaN`. `NaN` is neither `<= 0` (so the task is not marked stale) nor `> bestPriority` (so it is never picked again, and its round counter never reaches `maxRoundsPerTask` either). A task in that state starves; if every live task is in it, `pick` returns `null` and the loop exits as though everything had plateaued.
- **Benchmarking is off by default.** `enableBenchmark` is `hardwareMeasure || !!measurer` ([`autotuner.ts:127`](../../../src/compiler/autotune/autotuner.ts)), and a CPU compile sets neither, so `BenchmarkRunner` — which works perfectly well on a CPU target — is not constructed. The tuner is a cost-model search with no feedback loop unless the caller asks for one.
- **`maxCv` defaults to 0, which disables the re-measurement it gates.** [`autotuner.ts:130`](../../../src/compiler/autotune/autotuner.ts) and [`benchmark.ts:187`](../../../src/compiler/autotune/benchmark.ts). The cv is computed on every measurement and read only by that test, so with the default it is computed and discarded. And when it *is* enabled, `_collect` appends to the existing sample array rather than replacing it, so round two's statistics are computed over both rounds.
- **Benchmark buffers are `Float32Array` whatever the buffer declares.** [`benchmark.ts:142`](../../../src/compiler/autotune/benchmark.ts). Neutral for the `f32` kernels the tuner meets in practice, and not in general: an `f64` buffer is handed half the bytes it declares, and an `i64` one the wrong array class entirely, so the timing would be of a different code path or of nothing at all.
- **The workload key is 32 bits and nothing re-checks a hit.** `computeWorkloadKey` returns `fnv1a(...)` ([`workload_key.ts:152`](../../../src/compiler/autotune/workload_key.ts)); §47.6 exhibits a collision between two descriptions the compiler really builds, for buffers of 10,039 and 11,827 elements. `TuningRecord` stores the key and never the description ([`tuning_db.ts:29`](../../../src/compiler/autotune/tuning_db.ts)), so `lookup` has nothing to compare against and returns whichever workload was tuned first.
- **The workload key does not include the iteration domain.** Counterexample 47.7. It also identifies a target by `name` and `kind` only, so two `CPUTarget`s differing in `vectorWidth`, `numCores` or `l1CacheBytes` — all of which the cost model or the rule policy reads — share every cache entry.
- **A database written before `codegenVersion` existed loads unconditionally.** [`tuning_db.ts:129`](../../../src/compiler/autotune/tuning_db.ts). The `!== undefined` guard exempts exactly the files the check exists for.
- **`TuningRecord.version` is stored and never compared.** `TuningDatabase.deserialize` reads it into the reconstructed record ([`tuning_db.ts:133`](../../../src/compiler/autotune/tuning_db.ts)) and no code path tests it against `db.version`.
- **`EvolutionarySearch` overshoots its deadline by up to two populations.** Proposition 47.9; the final scoring pass at [`search.ts:157`](../../../src/compiler/autotune/search.ts) has no deadline test. With the default `populationSize: 32` that is up to 64 evaluations past the stated budget.
- **`_crossover` has a branch its caller cannot reach.** [`search.ts:179`](../../../src/compiler/autotune/search.ts) returns `{...a.params}` when the two parents have different sketches, but the only call site is the `else` of `if (parentA.sketch !== parentB.sketch)` ([`search.ts:149`](../../../src/compiler/autotune/search.ts)), so the parents always match. Harmless, and a hint that the two-sketch case was once handled differently.

## 47.8 Read the tests

- [`tests/compiler/autotune/autotuner.test.js`](../../../tests/compiler/autotune/autotuner.test.js) — the end-to-end block: five shapes, none a power of two, compiled with both search strategies on CPU and WASM and compared against the untuned baseline. It is the test that makes this chapter's findings survivable, and also the reason they survived: what it asserts is that the *answer* is right, and the answer is right whether or not tuning did anything. The same file has the budget tests (a zero budget still compiles correctly) and the cross-session cache test, which normalises `_\d+` out of the generated source before comparing — Chapter 48's subject, worked around.
- [`tests/compiler/autotune/ansor.test.js`](../../../tests/compiler/autotune/ansor.test.js) — `TaskScheduler allocates budget by weight` asserts exactly the first row of §47.5's last table, using a fake session whose `runRound` returns a finite `1`. The policy is correct and the test is correct; what neither covers is the value the real session returns.
- [`tests/compiler/autotune/benchmark.test.js`](../../../tests/compiler/autotune/benchmark.test.js) — `robustStats` against a spike and against uniform samples, and the buffer-refill property. Its two "iteration bound" tests re-implement `_collect`'s loop condition inside the test body rather than calling `_collect`, so they pin a copy of the logic.
- [`tests/compiler/autotune/workload-key.test.js`](../../../tests/compiler/autotune/workload-key.test.js) — that the key ignores buffer *names* and distinguishes buffer *shapes*. Both are properties Proposition 47.6 asserts; the negative half of that proposition has no test.

---

**Next:** [Chapter 48 — Reproducibility](../ch48-reproducibility/README.md), which asks what has to be written down for a tuning result to survive the process that produced it — and finds that the object designed for the job is written, stored, serialised, and read by nothing.
