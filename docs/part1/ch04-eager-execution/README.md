# Chapter 4 — Eager execution, and where it hurts

Chapter 2 measured a twelve-operation chain running 2.57× faster compiled than eager at 16 × 16, and only 1.08× faster at 512 × 512 — and then the same chain, with two function calls swapped out, running 14.54× and 4.57× faster at those same sizes. This chapter explains all four numbers, and by the end you will be able to predict them.

That is a strong claim, so here is the test we will hold ourselves to. Near the end of the chapter we will predict the runtime of a twelve-operation chain from two measured constants, before running it — and we will keep the model only to the accuracy it actually earns, which turns out to be the right *ranking* and the right *order of magnitude*, not the third digit.

## 4.1 The problem: what does one operation cost?

Start with the simplest possible question. You write:

```js
const c = a.add(b);
```

What did that cost? The obvious answer — "one addition per element" — turns out to be wrong in both directions. For small tensors it is far too optimistic. For large ones it is not the arithmetic you are paying for.

## 4.2 What eager execution actually does

"Eager" means each operation runs when you write it: no waiting, no plan, a tensor comes back immediately. Concretely, `a.add(b)` goes through four steps.

1. **Dispatch.** The framework decides which implementation of `add` applies, based on the device the tensors live on and their dtypes. This is a table lookup in [`src/dispatcher/`](../../../src/dispatcher/), modelled on PyTorch's dispatcher; Chapter 60 covers it.
2. **Shape and dtype resolution.** Broadcasting is worked out, the result dtype is computed, argument shapes are validated.
3. **Allocation.** An output tensor is allocated — a fresh typed array of the right size.
4. **The kernel.** A loop runs over the elements.

Here is the part that surprises people, and it matters for everything that follows. For most operations, step 4 is not a hand-written loop. It is **compiled, per operation, on first use** — [`src/dispatcher/jit_cache.ts:158`](../../../src/dispatcher/jit_cache.ts):

```ts
export function jitCompile(opName: string, tensorArgs: readonly TensorLike[], scalarArgs: ScalarArgs = null, target: TargetLike): CacheEntry {
  const key = _cacheKey(opName, tensorArgs, scalarArgs, target);
  let entry = _cache.get(key);
  if (entry) return entry;

  const func = _buildGraphFunc(opName, tensorArgs, scalarArgs);

  const mod = new GraphModule(opName + '_jit_mod');
  mod.addFunction(func as unknown as IRGraphFunction);
  const pm = new PassManager();
  pm.addPass(new DecompositionPass());
  pm.addPass(new CanonicalizePass());
  pm.addPass(new DCEPass());
  pm.run(mod);
```

Read what that does. It builds a one-operation graph, runs three compiler passes over it, lowers it, generates code, and caches the result under a key made of the operation name, the argument shapes and dtypes, and the target.

Not every operation takes this path. Registration skips any operation that already has a kernel ([`jit_dispatch.ts:382`](../../../src/dispatcher/jit_dispatch.ts): `if (handle.entry.hasKernel(key)) continue`), and two families do. **View operations** — `reshape`, `transpose`, `permute`, `broadcast_in_dim`, `slice`, `squeeze` and friends — are implemented by rewriting shape and stride metadata without touching data at all, which is why a transpose is free at the eager level. **Composite operations** — `sort`, `topk`, `cumsum`, `roll` and similar — have hand-written implementations because they are awkward to express as a single graph operation. Everything else, including all of the arithmetic this chapter measures, is compiled on demand.

So for the operations that matter here, the eager path and the compiled path **share the same code generator**. This is unusually convenient for a book, because it isolates the variable we care about. When we compare eager against compiled, we are not comparing a good code generator against a bad one. We are comparing *the same code generator given one operation at a time* against *the same code generator given the whole program at once*.

Everything Part I claims about compilation reduces to that difference in scope.

## 4.3 Lab 1 — The anatomy of one operation

```bash
node docs/part1/ch04-eager-execution/labs/01-anatomy-of-an-op.mjs
```

The lab times a single `add` across six sizes. Each row is the median of 21 rounds, with the relative half-IQR beside it so you can see which rows are trustworthy (Node 24.9, 2026-08-21):

```
one eager add, by size   (median of 21 rounds)
      n      elements       us/call      ns/element        rel. IQR
      1             1          1.44         1436.97           5.8%
      4            16          1.43           89.58           5.3%
     16           256          2.40            9.36           2.6%
     64          4096         12.43            3.03          11.8%
    256         65536        160.39            2.45           9.7%
   1024       1048576       1648.33            1.57           4.7%
```

The middle column rises by a factor of a thousand while the right column falls by a factor of nine hundred. That shape is the signature of a fixed cost being amortized, and it justifies a very simple model.

> **Definition 4.1 (Per-operation cost model).** **(stated here)** The wall-clock cost of one eager operation on `n` elements is
>
> **T(n) = α + βn**
>
> where **α** is the per-call cost that does not depend on the data — dispatch, shape resolution, allocation, cache lookup — and **β** is the marginal cost per element.

Now actually fit it, because there is a shortcut here that is easy to take and worth not taking. It is tempting to read α straight off the top row (α = T(1)) and β straight off the bottom (β = T(N)/N) and call that a fit. It is not one: those two numbers pass through those two points by construction, they cannot disagree with the data, and there is no residual to report. Worse, T(N)/N is *systematically* wrong — it equals β + α/N, so it always overstates the marginal cost, by a lot at small N and imperceptibly at large N.

So the lab solves for both parameters at once, over all six points, by weighted least squares. The weights are 1/T² — that is, the fit minimizes *relative* error. Without them the residual sum is dominated entirely by the 1,048,576-element row, whose absolute time is a thousand times everything else, and the "fit" quietly degenerates back into reading β off the bottom row. It prints both, side by side:

```
weighted least-squares fit of T(m) = alpha + beta*m over all 6 points
fixed cost per call   alpha = 1.40 us
marginal cost         beta  = 1.84 ns/element
worst relative residual     = 29.1%
break-even size             = 760 elements (a 28x28 tensor)

for comparison, the two-point shortcut alpha=T(1), beta=T(N)/N:
  alpha = 1.47 us, beta = 1.62 ns/element
```

**The residual is the interesting output, and it is 29%.** A two-parameter affine model is not a good description of this data across six orders of magnitude, and the fit is honest enough to say so. Look back at the `ns/element` column: it is *still falling* at a million elements — 3.03, 2.45, 1.57 — so there is no flat asymptote for β to be. What is really happening is at least two regimes with a cache transition between them, summarized by one straight line. Both parameters are therefore ballpark figures: α is good to about 10%, β to about 30%, and T(N)/N is an upper bound on the asymptotic marginal cost rather than an estimate of it.

That is enough for what the model is used for. It is a *ranking* device, not a predictor, and §4.7 shows exactly where the difference bites.

> **Corollary 4.2 (Break-even size).** **(stated here)** Below n = α/β elements, an operation spends more time being *arranged* than being *performed*.

Here α/β ≈ 760 elements — a 28 × 28 tensor. Given the 29% residual, read that as "a few hundred elements", not as 760. Every operation on anything much smaller is dominated by framework overhead rather than by your model's arithmetic. If you have ever wondered why a small network on a fast machine feels sluggish, this is usually why: at batch size 1 with modest hidden sizes, a great deal of deep learning happens below the break-even point.

**Try this.** Add rows for `n = 2` and `n = 2048`. They probe opposite ends of the model: at n = 2 the four elements are lost inside α, so `ns/element` climbs to several hundred; at n = 2048 it edges closer to β without ever reaching it, because α is divided by a larger number but never becomes zero.

## 4.4 Lab 3 — The first call is not like the others

*(Out of order on purpose: this lab belongs to the argument of §4.2, and Lab 2 belongs to the argument of §4.7. Run them in whichever order you read them.)*

```bash
node docs/part1/ch04-eager-execution/labs/03-eager-is-compiled-too.mjs
```

Since §4.2 told us that arithmetic operations are compiled on first use at each new shape, we should be able to see that cost:

```
   shape        first call      steady state       ratio
  8x8             13.719 ms           4.10 us        3345x
  9x9              1.029 ms           4.62 us         223x
  10x10           1.691 ms           3.93 us         431x
  11x11           1.075 ms           5.80 us         185x
```

The very first operation in the process costs 13.7 ms, because it warms up the compiler itself. Each subsequent *new shape* costs about 1 ms — one trip through graph construction, three passes, lowering and code generation. Every repeat is a few microseconds.

Two things follow, and both come back later in the book.

**Compilation is not free, so it must be cached.** The cache key includes the shapes. That is a deliberate trade: specializing on exact shapes produces better code (loop bounds are constants, as you saw in Chapter 2's kernels) at the cost of recompiling when shapes change.

**A program that keeps producing new shapes keeps paying.** A thousand distinct sequence lengths means a thousand compilations. This is precisely the problem that dynamic shapes and guards exist to solve, and Chapter 62 is devoted to it.

> **Note on comparing these numbers to Lab 1.** Lab 1 amortizes the clock over thousands of calls in a tight loop; Lab 3 times individual calls, which includes the cost of reading the clock. The steady-state microsecond figures here are therefore a little higher than α. Within a lab the comparisons are sound; across labs, trust the ratios rather than the absolute values.

## 4.5 Where does β go? Memory, mostly

β is about 2 ns per element. On the hardware, a floating-point addition is one instruction and costs a fraction of a nanosecond. So most of β is not the addition.

Count the memory instead. An elementwise `add` on n elements reads two arrays and writes one: **12n bytes** for `float32`. A unary operation such as `tanh` reads one array and writes one: **8n bytes**. Keep those apart — §4.6 needs the difference. At 1024 × 1024 the lab reports:

```
at 1024x1024 (4 MB per tensor)
  add   1.622 ms  (n=21, IQR 1.578-1.692, min 1.402, max 2.039)
         12 MB moved -> 7.76 GB/s
  tanh  24.899 ms  (n=21, IQR 23.920-25.511, min 20.394, max 52.015)
         8 MB moved -> 0.34 GB/s
```

12.6 MB moved in 1.62 ms. That is the number to hold on to: the `add` is moving data, and the arithmetic is a rounding error on top. The `tanh` is doing something else entirely — it moves *two thirds* as many bytes and takes fifteen times as long, which is §4.6's whole point arriving early.

(The table in §4.3 reported 1.65 ms for the same `add`, measured in a different batch a moment earlier. Run-to-run variation of a few percent is normal and worth getting used to — which is why every timing in this book comes with its IQR, and why a benchmark that reproduces to three digits is usually measuring the wrong thing.)

> **Definition 4.3 (Arithmetic intensity; classical).** The arithmetic intensity of a computation is the ratio of arithmetic operations performed to bytes of memory traffic required:
>
> **I = FLOP / bytes**

For elementwise `add`: one FLOP per 12 bytes, so I = 0.083. For `tanh`, one transcendental evaluation per 8 bytes.

For a matrix multiply of two n × n matrices the count needs a stated assumption, and it is a strong one. The FLOP count is 2n³ and is not negotiable. The byte count is: **12n² bytes is the *minimum* traffic**, achieved only if each of the three matrices crosses the memory boundary exactly once — which requires enough on-chip storage to hold the working set, and a loop order that exploits it. That gives I = n/6, about 171 at n = 1024, and it is a *lower bound on traffic*, hence an *upper bound on intensity*. A naive triple loop that re-reads a row of A and a column of B for every output element moves Θ(n³) bytes and has intensity Θ(1) — memory-bound, not compute-bound, on the same mathematics. The gap between those two numbers is what tiling exists to close, and Part VII is where the compiler goes after it. Whenever this book quotes 12n² it means the ideal, and the phrase to attach is "if the reuse is achieved".

> **Theorem 4.4 (Roofline bound; Williams, Waterman and Patterson, 2009).** A computation with arithmetic intensity I, running on a machine with peak compute rate P (FLOP/s) and peak bandwidth B (bytes/s), cannot exceed
>
> **min(P, I · B)** FLOP/s.
>
> *Proof sketch.* Executing F floating-point operations takes at least F/P seconds. Moving the required F/I bytes takes at least F/(I·B) seconds. Both must elapse, so total time ≥ max(F/P, F/(I·B)), and rate = F/time ≤ min(P, I·B). ∎

The consequence is the vocabulary you need for the rest of the book. A computation is **memory-bound** when I·B < P — it will finish faster only if you move less data. It is **compute-bound** when I·B > P — it will finish faster only if you do less arithmetic, or do it with better instructions.

Elementwise operations, with I ≈ 0.08, are firmly memory-bound. And that is exactly why fusion is the most valuable optimization in this domain: **fusion reduces bytes moved without changing the arithmetic at all.**

## 4.6 Not all arithmetic is equal

One more measurement from Lab 1:

```
  tanh  24.899 ms
         8 MB moved -> 0.34 GB/s
  tanh costs 15.4x an add while moving 0.67x the bytes
```

Same tensor, **two thirds** the traffic, fifteen times the time. Say that carefully, because the tempting version — "same tensor, same traffic" — is false and would break the argument rather than support it. The binary `add` reads two arrays and writes one; the unary `tanh` reads one and writes one. If traffic were what mattered, `tanh` should be the *cheaper* of the two. It is fifteen times dearer, so whatever is costing the time is not memory, and that is a far stronger conclusion than the equal-traffic version would have licensed.

What is costing it is the arithmetic. `tanh` is a *transcendental* function: it has no single-instruction implementation on mainstream CPUs and is evaluated by an approximation — typically a polynomial or rational one over a reduced range, sometimes with a table. The ECMAScript specification does not say which: it requires only that `Math.tanh` return an implementation-approximated result, explicitly permitting implementations to differ. So the honest claim is architectural rather than specific — many machine-level operations per element instead of one, which no amount of memory optimization makes cheaper — and if you want the actual sequence for your machine you must read your engine's math library, not the standard.

This breaks the tidy story from §4.5. An elementwise chain is memory-bound only if its elements are *cheap*. Put a `tanh` in it and the chain becomes compute-bound, at which point removing memory traffic optimizes something that was not the bottleneck.

## 4.7 Lab 2 — Where fusion wins, and where it does not

```bash
node docs/part1/ch04-eager-execution/labs/02-where-fusion-wins.mjs
```

Two chains, twelve operations each, same tensor, same compiler. The only difference: one chain contains two `tanh` calls, the other replaces them with two multiplications.

```
twelve elementwise operations on a 1024x1024 tensor (4 MB)

  chain              eager ms   compiled ms     speedup
  10 cheap + 2 tanh       80.2          73.1        1.10x
  12 cheap ops            32.2           7.2        4.47x
```

**The same compiler, on the same shape, delivers 1.10× or 4.47× depending on two function calls.** (The 4.47× reproduces within a few percent anywhere; the 1.10× is a coin toss around 1.0 and §4.10 says why.)

Now use the model from §4.3 and §4.6 to predict those eager numbers before looking at them. The constants have to come from the *same session* as the chain measurements, or the comparison is meaningless: in that session an `add` at this size cost 2.62 ms and a `tanh` cost 26.95 ms. (The re-measured figures quoted in §4.5 — 1.62 ms and 24.90 ms — are from a different, faster machine two weeks later, and are not interchangeable with these. Constants and predictions travel together or not at all.)

- Twelve cheap operations: 12 × 2.62 = **31.4 ms**. Measured: 32.2 ms.
- Ten cheap operations plus two `tanh`: 10 × 2.62 + 2 × 26.95 = 26.2 + 53.9 = **80.1 ms**. Measured: 80.2 ms.

Both predictions land within three percent on this run, and the second within 0.2%. Do not read too much into that third digit — the next box shows it does not survive re-measurement — but do read the shape of it. A model that says nothing except *add up the operations* gets the total roughly right, and the reason tells you something real about eager execution: **there is very little else going on.** Eager execution is close to the sum of its operations, because each one runs to completion in isolation before the next begins. No overlap, no reuse, almost no interaction. That is the property compilation removes.

> **Does this reproduce? Partly, and the part that fails is instructive.** The tables above come from one session. Re-running both labs on a different machine — Node 24.9, three paired runs of Lab 1 and Lab 2 — reproduced every structural claim and none of the precision. The constants drifted as expected (α = 1.34 μs, β = 1.62 ns, `add` 1.65–1.99 ms, `tanh` 11–15× `add`). But the predictions came out 10% to 60% away from measurement, in both directions: 12 × 1.99 = 23.9 ms predicted against 21.8 measured, and 10 × 1.99 + 2 × 24.38 = 68.7 ms predicted against 42.9 measured.
>
> So the sum-of-operations model is not the 0.2%-accurate instrument the first run makes it look like. What it reliably gets right is the ordering and the rough magnitude: the `tanh` chain always costs about twice the cheap chain, `tanh` always dominates the total when it is present, and the cheap chain always fuses several times better than the `tanh` chain. What it gets wrong is any specific total, by up to a factor of 1.6 — because §4.10's caveat is real and load-bearing: `add` and `tanh` were timed *alone*, and an operation inside a twelve-deep chain competes for cache with eleven others. The individual numbers in this chapter are disposable. The model is worth carrying for what it ranks, not for what it predicts, and §4.10 returns to why that distinction is the whole subject of Part VIII.

For the compiled numbers, apply Amdahl's law.

> **Theorem 4.5 (Amdahl's law, 1967; in the form we need).** **(classical)** If a fraction *p* of a program's runtime is spent in work that an optimization cannot touch, the speedup from that optimization is at most **1/p**.

Applying it needs a number the measurements above do not contain, and it is worth being explicit about the gap rather than papering over it. The two `tanh` calls account for 53.9 ms of the 80.2 ms — but that 53.9 ms is *the whole eager cost of two `tanh` operations*, and fusion does touch part of it. A fused `tanh` still pays for its dispatch (α), still allocates no intermediate, and reads its input from a register rather than from RAM. What fusion genuinely cannot remove is only the **transcendental arithmetic itself**, and the labs never measured that in isolation; every `tanh` timing here is arithmetic plus 8n bytes of traffic plus α.

So there are two bounds, and only one of them is derived from the data:

- **A lower bound on the ceiling.** Take the *whole* 53.9 ms as untouchable: p = 0.672 and speedup ≤ 1/0.672 ≈ **1.49×**. This is not the true ceiling — it is a ceiling computed by assuming fusion helps the `tanh` operations not at all, so the real ceiling is higher.
- **The genuine ceiling** would need the transcendental arithmetic timed apart from its memory traffic. Subtracting an `add`-sized traffic cost as a rough proxy for the memory half of each `tanh` moves p to roughly 0.63 and the ceiling to about 1.6×. That number is an estimate built on a proxy, not a measurement, and it is quoted only to show which direction the correction goes.

Measured: 1.10×, comfortably under both. The lesson survives the imprecision intact — when two thirds of the time is arithmetic that fusion cannot touch, no traffic-removing optimization can do much — but "1.49×" is a bound derived under a stated pessimistic assumption, not a prediction, and it should not be quoted as one.

In the cheap chain none of this applies: essentially all the time is traffic, and fusion collapses twelve passes over memory into one for 4.47×, or the entire chain for the price of 2.7 `add`s.

### A model for the compiled side too

The fused kernel has its own two-term structure: one pass over memory, plus the arithmetic, which grows with the number of operations. Halving the chain to six cheap operations and measuring both versions gives (a separate run, hence 32.8 ms where Lab 2 reported 32.2 ms):

```
  ops    eager ms   compiled ms   speedup
    6       16.0           6.1      2.62x
   12       32.8           7.3      4.50x
```

Eager halved almost exactly, as its model demands. Compiled fell by only 16%, from 7.3 ms to 6.1 ms — and those two points determine the second model. Fitting **T_fused = M + k·c** gives c ≈ 0.2 ms per operation and M ≈ 4.9 ms for the single pass over memory; check it against the other point: 4.9 + 12 × 0.2 = 7.3 ✓.

Now the speedup has a closed form:

> **speedup(k) = k · T_op / (M + k · c)**

which rises with the chain length k and saturates at **T_op / c** ≈ 2.62 / 0.2 ≈ 13×. So longer chains fuse better, but not without limit: past a certain length the fused kernel is doing arithmetic, not moving memory, and you are back in the compute-bound regime by a different route.

Note what this predicts and what it does not. It says a six-operation chain will land near 2.6×, not near the 2.2× a naive "half the operations, half the benefit" guess would give — because M does not halve when k does. Getting that wrong is the most common error people make when reasoning about fusion, and it is why the model is worth carrying.

**Try this.** Predict the speedup for a chain of 24 cheap operations before running it. The model says 24 × 2.62 / (4.9 + 24 × 0.2) ≈ 6.4×. Then run it, and take the disagreement seriously: at 24 operations the fused kernel's working set and register pressure start to matter, and neither appears anywhere in this model.

## 4.8 The three regimes

Everything above collapses into one table, which is worth remembering because it tells you which chapters of this book will help with a given slow program.

| Regime | Symptom | Where the time goes | What helps |
|---|---|---|---|
| **Overhead-bound** | Small tensors, many operations; time barely changes with tensor size | α — dispatch, allocation, bookkeeping | Fusion, graph capture, bigger batches (Part IV) |
| **Bandwidth-bound** | Large tensors, cheap arithmetic; time scales with bytes | Memory traffic | Fusion, layout, tiling, in-place reuse (Parts IV, VII, IX) |
| **Compute-bound** | Expensive arithmetic, or a matmul at scale | The FLOPs themselves | Better instructions, tensor cores, lower precision, better algorithms (Parts VII, X) |

Chapter 2's tables walk down this column. The `tanh` chain at 16 is overhead-bound and fusion collects 2.57×; at 128 it is in transition at 1.36×; at 512 the fixed costs are gone and the two `tanh` calls have made it compute-bound, leaving fusion almost nothing it is allowed to remove: 1.08×. The cheap chain never reaches the third row at all — it goes from overhead-bound (14.54×) to bandwidth-bound (4.57×), which is the regime where fusion is at its most valuable.

## 4.9 What eager execution cannot do, at any speed

Everything so far has been about constants. There is a second category of loss that no amount of tuning recovers, because it is structural. Executing one operation at a time means:

- **No fusion.** To fuse `add` into `maximum` you must know the `maximum` is coming. When `add` executes, that fact does not exist yet.
- **No memory planning.** Every intermediate is allocated because it might be needed; nothing knows it will be dead in two operations. Part IX plans an entire program's buffers at once.
- **No layout choice.** A single operation cannot know that storing its output transposed would let the next three operations run faster.
- **No cross-operation algebra.** `x.transpose().dot(w)` becomes a `dot` with swapped indices only if someone sees both operations together — which you watched happen in Chapter 3.
- **No whole-program gradient.** Reverse-mode differentiation is a transformation *of a program*. Without a program, autograd must record a tape at runtime and pay for it. Part V.

Each of those is a chapter or a part of this book, and all of them require the same precondition: **the whole computation must be visible at once**.

That precondition is not free either. Getting it costs you dynamism, and the bill comes due in the next chapter.

## 4.10 Traps and limits

- **These constants are this machine's, and so is the sign of the `tanh` result.** α, β and the cost of `tanh` will differ on your hardware and your Node version. The *structure* — a fixed cost, a marginal cost, a break-even point, expensive transcendentals — is what transfers. The 1.10× in §4.7 does not: on a second machine the same lab reported 0.95×, 0.99× and 1.07× across three runs, meaning fusion sometimes made the `tanh` chain *slower*. That is not a contradiction of the argument, it is the argument's conclusion taken one step further — when almost all the time is in work an optimization cannot touch, the optimization's own overhead is free to dominate what little is left.
- **`add` and `tanh` were measured in isolation.** In a real model, operations compete for cache with everything around them, and measurements taken alone can flatter or slander an operation. Chapter 46 returns to this when it builds a cost model the autotuner has to trust.
- **The fused model was fitted on cheap operations, and does not extrapolate.** Applying `T_fused = M + k·c` to the `tanh` chain — substituting the measured cost of a `tanh` for `c` on two of the twelve terms — predicts about 55 ms against a measured 73 ms. Something the model does not represent is costing 18 ms. Candidate explanations: the generated code calls `Math.tanh` in a context the engine optimizes less well than the one-operation kernel does, or the long fused expression puts pressure on registers. The honest position is that we do not know which, and that a cost model is a hypothesis you keep testing, not a fact you have established. Part VIII is about compilers that face this problem at scale, and their answer is to stop predicting and start measuring.
- **The roofline model ignores latency.** It assumes you can saturate either compute or bandwidth. A dependent chain of scalar operations saturates neither, and no roofline predicts it.
- **JavaScript is not the peak.** 4.81 GB/s is what this eager path achieves, not what the machine can do. Chapter 55 gets closer with WebAssembly SIMD; Chapter 56 changes the machine entirely.

## 4.11 Read the tests

- [`tests/dispatcher/`](../../../tests/dispatcher/) — the dispatch path and the JIT cache: what gets cached, under what key, and when the cache is invalidated.
- [`tests/perf/`](../../../tests/perf/) — the performance tests, which assert the *shape* of scaling behaviour rather than absolute timings. Chapter 66 explains why that is the only kind of performance test worth writing.

---

**Next:** [Chapter 5 — From a sequence of calls to a program](../ch05-calls-to-program/README.md), which asks what it takes to see the whole computation at once, and what you give up to get it.
