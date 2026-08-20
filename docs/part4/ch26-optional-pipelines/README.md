# Chapter 26 — Three optional pipelines

Chapters 19 through 24 described transformations that run on every compilation. Chapter 25 described one that does not. This chapter describes the other three that do not, and it is deliberately an overview: each of them deserves a chapter of its own and gets an appendix instead.

They share a shape worth naming before the details. Each one **trades something you have for something you want** — memory for time, accuracy for speed, a single kernel for two devices — and each therefore needs a number from the user that the compiler cannot supply: a budget, a calibration, a device list. That is why they are opt-in. An optimization with a free lunch can be on by default; an optimization that spends one resource to buy another cannot, because only the user knows which resource is scarce.

## 26.1 Rematerialization: memory for recomputation

### The problem

Training a network requires keeping every intermediate activation from the forward pass alive until the backward pass consumes it (Part V). For a deep model that is the dominant memory cost, and it scales with depth: a 48-layer transformer holds 48 layers' worth of activations at the moment the backward pass begins.

When that exceeds the device, you have three options: shrink the batch, shrink the model, or **do not keep the activation** — recompute it from its inputs when the backward pass asks for it. The third is rematerialization, and it is the only one that does not change the model.

### The theory

> **Definition 26.1 (Rematerialization).** Let `v` be a value with more than one use, produced by a pure operation `f`. *Rematerializing* `v` at a use `u` means inserting a fresh evaluation of `f` immediately before `u` and rewiring `u` to the fresh result, shortening `v`'s live interval.

Chapter 49 defines live intervals properly; here it is enough that a value is *live* from its definition to its last use, and that peak memory is the maximum over program points of the total bytes live there.

> **Theorem 26.2 (√n checkpointing).** *(Chen et al., 2016.)* For a chain of `n` layers, storing every `√n`-th activation and recomputing the rest gives `O(√n)` memory at the cost of one extra forward pass.

That is the result the technique is famous for, and it applies to a chain. The general graph case has no such clean bound, and the implementation here does not attempt one: it is a greedy loop that repeatedly rematerializes the highest-scoring candidate until the peak fits the budget or it runs out of candidates.

The score is a ratio ([`rematerialization.ts:104`](../../../src/compiler/passes/memory/rematerialization.ts)):

```ts
      candidates.push({
        value,
        definingOp: defOp,
        memorySaved,
        recomputeCost,
        score: memorySaved / recomputeCost
      });
```

Bytes freed per unit of work added — the natural greedy criterion for a knapsack-shaped problem, and like most greedy criteria it is not optimal.

### In mlfw

[`passes/memory/rematerialization.ts`](../../../src/compiler/passes/memory/rematerialization.ts), 228 lines, a `FunctionPass` requiring `LivenessAnalysis`. The entry condition is the budget ([`rematerialization.ts:60`](../../../src/compiler/passes/memory/rematerialization.ts)):

```ts
    if (this.config.memoryBudget === Infinity) return PassResult.UNCHANGED;
```

No budget, no work — which is the opt-in. Candidates must be pure, region-free, non-constant, and, crucially, used more than once ([`rematerialization.ts:125`](../../../src/compiler/passes/memory/rematerialization.ts)):

```ts
    if (value.useCount <= 1) return false;
```

A value used once has nothing to shorten: its live interval already ends at its only use. Rematerialization is about values that are produced early and consumed late — exactly the shape of a forward activation waiting for its gradient.

### Lab

```bash
node docs/part4/ch26-optional-pipelines/labs/01-memory-for-recomputation.mjs
```

Three values, each used twice, all live simultaneously:

```
=== rematerialization off (the default) ===
  graph: 7 operations -- exp, log, add, sub, add, add, sub
  planned peak memory: 786432 bytes across 6 temporaries
  1.423 ms

=== memory budget 512 KiB ===
  graph: 9 operations -- exp, log, add, sub, exp, add, log, add, sub
  pass: 2 rematerialization(s), live pressure 1048576 bytes against a budget of 524288
  planned peak memory: 524288 bytes across 8 temporaries
  2.181 ms
```

The `exp` and the `log` each appear twice in the second graph — the pass inserted a second evaluation of each next to its later use. Peak memory falls from 768 KiB to **exactly 512 KiB, the budget**; runtime rises from 1.42 ms to 2.18 ms. Memory down 33%, time up 53%: the trade, priced.

And then the third run, which is the part to remember:

```
=== memory budget 128 KiB ===
  graph: 9 operations -- exp, log, add, sub, exp, add, log, add, sub
  pass: 2 rematerialization(s), live pressure 1048576 bytes against a budget of 131072
  planned peak memory: 524288 bytes across 8 temporaries
  2.167 ms
```

Asked for 128 KiB, delivered 512 KiB, **reported success**. The loop exits when it runs out of candidates ([`rematerialization.ts:71`](../../../src/compiler/passes/memory/rematerialization.ts): `if (candidates.length === 0) break;`) and nothing distinguishes that exit from the one where the budget was met. A user who set a budget because the device has that much memory will find out at run time.

The budget is a *goal*, not a guarantee, and the pass does not say which one it achieved. The information is in the trace — the `pass_detail` event carries both `peakPressure` and `budget` — so a caller can compare them, and no caller does.

## 26.2 Quantization: accuracy for speed

### The problem

An f32 weight is four bytes. An int8 weight is one. For a memory-bound network — which by Chapter 22 is most of them — a 4× reduction in bytes moved is close to a 4× reduction in time, and on hardware with integer tensor units it is more.

The cost is precision. Mapping a range of floats onto 256 integer levels loses information, and how much depends entirely on choosing the range well.

### The theory

> **Definition 26.3 (Affine quantization).** A quantization of a float range onto `b` bits is a pair `(s, z)` — scale and zero-point — with `q = round(x/s) + z` and `x̂ = s(q − z)`. It is *symmetric* if `z = 0`.

> **Definition 26.4 (Calibration).** *Calibration* is the process of choosing `s` per tensor by observing the values that tensor actually takes on representative inputs.

Calibration is the whole game. Weights can be inspected statically — they are constants. *Activations* cannot: their range depends on the input distribution, and a compiler that guesses gets the accuracy shown below.

### In mlfw

[`passes/quantization/quantization_pass.ts`](../../../src/compiler/passes/quantization/quantization_pass.ts), 450 lines, the largest single graph pass. It rewrites each quantizable operation into `quantize → quantized_op → dequantize`, and relies on Chapters 19 and 24 to cancel adjacent `dequantize`/`quantize` pairs and fuse what is left.

Calibration is a *phase*, not a pass — it runs before the graph passes ([`compiler.ts:317`](../../../src/compiler/pipeline/compiler.ts)), compiles an instrumented copy of the graph that captures activation ranges, runs it on user-supplied batches, and feeds the result back into the quantization config.

### Lab

```bash
node docs/part4/ch26-optional-pipelines/labs/02-quantization-as-a-rewrite.mjs
```

```
=== float32 ===
  7 operations: dot, constant, fusion, add, maximum, dot, add
=== int8, default activation range ===
  14 operations: quantize, quantize, quantized_dot, constant, fusion, dequantize, add, maximum, quantize, quantize, quantized_dot, fusion, dequantize, add

relative error against the float32 result: 18.26%
first output element: 0.04237501695752144 -> 0.048035264015197754
```

Two `dot`s became two `quantized_dot`s with four `quantize`s and two `dequantize`s around them, and the answer is **18% wrong**.

That number is the chapter's point. Nothing failed: the rewrite is correct, the arithmetic is correct, the int8 kernel computes exactly what an int8 kernel should. The default activation range is `[-6, 6]` and these activations live in roughly `[-0.1, 0.1]`, so 250 of the 256 available levels are never used and the effective precision is about four bits. Quantization without calibration is not a fast version of your model; it is a different model.

The repository's own test suite states this as an assertion rather than a footnote — [`calibration-exec.test.js`](../../../tests/compiler/passes/quantization/calibration-exec.test.js) has a case named *"default [-6,6] activation range is lossy on small activations (the bug)"* that requires the error to exceed 5%.

## 26.3 Partitioning and BYOC: one graph, several executors

### The problem

Two versions of the same question. **Partitioning:** a graph that does not fit on one device, or whose parts run better on different devices, must be cut into subgraphs with explicit transfers between them. **BYOC — bring your own codegen:** a vendor library implements some operations better than any generated kernel, so those operations should be handed to it and the rest compiled normally.

Both are the same graph problem: choose a set of subgraphs, assign each to an executor, and pay for every edge that crosses a boundary.

### The theory

> **Definition 26.5 (Partition).** A *partition* assigns every operation to an executor. Its cost is the sum of per-operation costs on the assigned executor plus a transfer cost for every value crossing a boundary.

> **Note.** The subgraph handed to an executor must be *convex*: if `a` and `c` are in it and `b` lies on a path from `a` to `c`, then `b` must be in it too. Otherwise the subgraph cannot be executed as one unit — the same acyclicity requirement as Theorem 23.2, in a different costume.

Which is worth stating explicitly, because it means the whole of Chapter 23 applies here unchanged. Fusion, partitioning and BYOC are three names for grouping operations under a convexity constraint, differing only in the cost function.

### In mlfw

Three implementations sit side by side in [`passes/partition/`](../../../src/compiler/passes/partition/):

| File | What it splits on |
|---|---|
| [`partition_pass.ts`](../../../src/compiler/passes/partition/partition_pass.ts) | device assignment across a target list |
| [`cublas_split.ts`](../../../src/compiler/passes/partition/cublas_split.ts) | operations cuBLAS should own |
| [`scan_split.ts`](../../../src/compiler/passes/partition/scan_split.ts) | a `scan` too large for one kernel |

and the generic mechanism they feed is [`pipeline/external_codegen.ts`](../../../src/compiler/pipeline/external_codegen.ts), a registry of providers that can claim operations, contribute their own graph passes, and annotate the TIR module (Chapter 58).

Partitioning is enabled only when at least two targets are configured ([`compiler.ts:196`](../../../src/compiler/pipeline/compiler.ts)):

```ts
  get usePartition(): boolean {
    return this.partition.enabled && this.partition.targets.length >= 2;
  }
```

which is the same opt-in shape as the other two: the pass needs a fact only the user has.

There is no lab for this section, because the interesting cases need two devices and this book's labs run on one. [`tests/compiler/analysis/partitioner.test.js`](../../../tests/compiler/analysis/partitioner.test.js) and [`tests/compiler/pipeline/graph-split.test.js`](../../../tests/compiler/pipeline/graph-split.test.js) are where the behaviour is pinned, and Chapter 58 returns to the external-codegen interface with a working cuBLAS example.

## 26.4 What the three have in common

Three transformations, three resources, one structure.

| | Spends | Buys | The number the user must supply |
|---|---|---|---|
| Rematerialization | time | memory | `memoryBudget` |
| Quantization | accuracy | time and memory | calibration data |
| Partitioning | transfers | device capacity, library kernels | a target list |

And each fails in the same characteristic way when the number is missing or wrong: not with an error, but with a *plausible* result. A remat pass that cannot reach its budget returns success. A quantizer without calibration returns a model that runs and is 18% wrong. A partitioner with one target does nothing at all.

That is the general lesson of the chapter, and it generalizes past these three: **an optimization that trades resources needs to report what it actually achieved, not just that it ran.** Chapter 18's trace stream is where that reporting would live, and all three of these emit enough into it to tell you — the remat pass emits its peak against its budget, the quantizer's calibration phase emits its ranges — and nothing in the pipeline compares the two and complains.

## 26.5 Traps and limits

- **The remat budget is not a guarantee and success is not reported.** §26.1's third run. The loop exits identically whether the budget was met or the candidates ran out.
- **Remat only considers values used more than once.** A long chain of single-use activations — the ordinary shape of a forward pass before automatic differentiation adds the second uses — has no candidates at all. The pass is designed for the joint forward-backward graph of Chapter 29 and does nothing useful before it.
- **`recomputeCost` is a FLOP estimate with the same flaw as Chapter 22's.** Without a registry `getFlops`, an `exp` costs the same as an `add`, so the score ranks a transcendental recomputation as cheaply as an arithmetic one.
- **Quantization's `calibrationData` is not usable through the public `compile()` path.** `collectCalibration` runs the instrumented graph with the batch as its complete argument list ([`calibrate_exec.ts:101`](../../../src/compiler/analysis/calibrate_exec.ts)), but a traced model's graph also takes its captured parameters as arguments — which the user does not have. Supplying a batch of user inputs therefore misaligns the arguments and fails with `Cannot set properties of undefined`. Calibration works through the lower-level `Compiler` API, which is how the tests use it.
- **The default activation range is `[-6, 6]` and nothing warns.** §26.2 measures 18% error on a model whose activations are two orders of magnitude smaller. The compiler has the calibration machinery and, absent data, silently uses a constant.
- **Partitioning needs two targets and reports nothing when it has one.** `usePartition` returns false and the phase is skipped; a user who configured `partition.enabled` and one target gets no error and no partitioning.

## 26.6 Read the tests

- [`tests/compiler/passes/memory/remat.test.js`](../../../tests/compiler/passes/memory/remat.test.js) — rematerialization candidate selection and the greedy loop.
- [`tests/compiler/passes/quantization/calibration-exec.test.js`](../../../tests/compiler/passes/quantization/calibration-exec.test.js) — end-to-end accuracy with and without calibration, including the named `(the bug)` case.
- [`tests/compiler/analysis/partitioner.test.js`](../../../tests/compiler/analysis/partitioner.test.js) and [`tests/compiler/pipeline/graph-split.test.js`](../../../tests/compiler/pipeline/graph-split.test.js) — assignment and the convexity of the resulting subgraphs.

---

**Part IV ends here.** The graph that entered was a trace of user calls; the graph that leaves has been folded, deduplicated, swept, algebraically rewritten under a licence it declares, expanded into primitives and re-collapsed into kernels by a cost model, with a layout it declined to change and three optional trades it did not take.

What it has *not* been is lowered. Every operation in it is still a tensor operation over whole tensors, with no loops, no indices and no buffers. The next two parts introduce those: Part V differentiates the graph while it is still at this level, because that is the only level where the chain rule is expressible as a graph rewrite — and Part VI turns the result into loops.

**Next:** Part V — Automatic differentiation, which takes the forward graph you now know how to optimize and constructs the backward one.
