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
  2.333 ms

=== memory budget 512 KiB ===
  graph: 9 operations -- exp, log, add, sub, exp, add, log, add, sub
  pass: 2 rematerialization(s), live pressure 1048576 bytes against a budget of 524288
  planned peak memory: 524288 bytes across 8 temporaries
  warning: budget not met: peak live pressure is 1048576 bytes against a budget of 524288 bytes
  3.813 ms
```

The `exp` and the `log` each appear twice in the second graph — the pass inserted a second evaluation of each next to its later use. Planned peak memory falls from 768 KiB to **exactly 512 KiB, the budget**; runtime rises. Memory down 33%, time up: the trade, priced.

And then the third run, which is the part to remember:

```
=== memory budget 128 KiB ===
  graph: 9 operations -- exp, log, add, sub, exp, add, log, add, sub
  pass: 2 rematerialization(s), live pressure 1048576 bytes against a budget of 131072
  planned peak memory: 524288 bytes across 8 temporaries
  warning: budget not met: peak live pressure is 1048576 bytes against a budget of 131072 bytes
  3.754 ms
```

Asked for 128 KiB, delivered 512 KiB. Until recently it *reported success*: the loop exits when it runs out of candidates ([`rematerialization.ts:71`](../../../src/compiler/passes/memory/rematerialization.ts): `if (candidates.length === 0) break;`) and nothing distinguished that exit from the one where the budget was met. A user who set a budget because the device has that much memory found out at run time. The pass now compares its final peak against the budget and emits a `trace.warn` naming both numbers when it fell short — and it still returns `CHANGED` rather than failing, because a budget is a goal and missing it is not a compilation error.

Read the two warnings together, though, because they say something the third run alone does not. The 512 KiB run *also* warns, and its planned peak is exactly 512 KiB. Those are two different peaks. `peakPressure` is what `LivenessAnalysis` measures — the total bytes of values simultaneously live at the busiest program point, which is what the greedy loop is steering. `peakMemory` is what the memory planner reports afterwards, and the planner reuses buffers, so it lands lower. The pass warns against the quantity it actually controls, which is the honest thing for it to do and is not the quantity the user is thinking of. Reconciling the two is Chapter 49's problem, not this pass's.

## 26.2 Quantization: accuracy for speed

### The problem

An f32 weight is four bytes. An int8 weight is one. For a memory-bound network — which by Chapter 22 is most of them — a 4× reduction in bytes moved is close to a 4× reduction in time, and on hardware with integer tensor units it is more.

The cost is precision. Mapping a range of floats onto 256 integer levels loses information, and how much depends entirely on choosing the range well.

### The theory

> **Definition 26.3 (Affine quantization).** A quantization of a float range onto `b` bits is a pair `(s, z)` — scale and zero-point — with `q = round(x/s) + z` and `x̂ = s(q − z)`. It is *symmetric* if `z = 0`.

> **Definition 26.4 (Calibration).** *Calibration* is the process of choosing `s` per tensor by observing the values that tensor actually takes on representative inputs.

Calibration is the whole game. Weights can be inspected statically — they are constants. *Activations* cannot: their range depends on the input distribution, and a compiler that guesses gets the accuracy shown below.

### In mlfw

[`passes/quantization/quantization_pass.ts`](../../../src/compiler/passes/quantization/quantization_pass.ts), 450 lines, the largest single graph pass. It rewrites each quantizable operation into `quantize → quantized_op → dequantize`, and leaves the resulting elementwise litter to Chapter 24's fusion engine — the `dequantize`s and `add`s in §26.2's output end up inside regions, not as separate kernels.

Two adjacent quantized operations should also not dequantize and requantize between them, and that cleanup is delegated twice over. The pass avoids creating most such pairs: it carries a `quantizedValues` set, and an operand already in it is consumed as-is with the producing `quantize`'s scale copied onto the consumer as an attribute ([`quantization_pass.ts:269`](../../../src/compiler/passes/quantization/quantization_pass.ts)). Whatever survives that is deleted by `QuantizeDequantizeIdentity`, a rewrite rule declared on `quantize`'s registry entry ([`ops/quantization.ts:56`](../../../src/compiler/ir/graph/ops/quantization.ts)) which matches a `quantize` whose operand is a `dequantize` with equal parameters and replaces it with the original ([`quantization_patterns.ts:28`](../../../src/compiler/ir/graph/quantization_patterns.ts)). It carries benefit 20, the highest in the compiler, and it is the reason `buildGraphPipeline` places a fresh `CanonicalizePass` immediately after the quantization pass ([`graph_pipeline.ts:67`](../../../src/compiler/pipeline/graph_pipeline.ts)) rather than relying on the fixed-point group that already ran.

That ordering is the interesting part. The quantization rewrite is written as if it could litter freely, because a canonicalizer runs directly behind it — which is Chapter 19's rule that a pass should make things unnecessary rather than remove them, applied to a pass big enough that doing its own cleanup would have doubled it.

Calibration is a *pass*, and where it sits in the list is the whole of its correctness. [`CalibrationPass`](../../../src/compiler/passes/quantization/calibration_pass.ts) compiles an instrumented copy of the graph that returns the operands of every quantizable operation as extra outputs, runs it on user-supplied batches, and writes the observed ranges into the quantization config. It observes *values* — `CalibrationResult` is a map keyed on `Value` identity ([`calibration.ts:232`](../../../src/compiler/analysis/calibration.ts)) — so anything that replaces a value between the observation and the lookup throws the observation away silently, and `_getQuantParams` falls back to `[-6, 6]`.

The fix for that is not a mechanism, it is an ordering: `buildGraphPipeline` emits the calibration pass immediately before the quantization pass that consumes it ([`graph_pipeline.ts:58`](../../../src/compiler/pipeline/graph_pipeline.ts)), with nothing in between. Calibration used to be a compile *phase* running ahead of every graph pass, which meant it observed a graph the quantizer never saw: for the traced `Sequential` below, the traced form is `transpose(%1) → dot(%0, transposed)`, canonicalization folds the transpose into the `dot`'s contracting-dimension attributes, and the right operand the quantizer then asks about is `%1` — never observed. Supplying `calibrationData` bought 18.26% to 17.46%. Run adjacent to its consumer, the same data buys 18.26% to 0.60%.

The instrumented graph is run with the batch as its **complete** argument list ([`calibrate_exec.ts:107`](../../../src/compiler/analysis/calibrate_exec.ts)), which is fine for a hand-built `GraphFunction` whose only argument is the input, and wrong for a traced model, whose graph also takes every captured parameter as an argument. Supplying `calibrationData` through `compile()` therefore used to misalign the arguments and fail with `Cannot set properties of undefined (setting '0')` — the output buffers landed past the end of the list. Calibration was reachable only through the lower-level `Compiler` API, which is how the tests reach it. The `compile()` wrapper now assembles the graph-level argument list the same way it assembles one for an ordinary call — user inputs, then captured parameters ([`compile.ts:311`](../../../src/tracing/compile.ts)) — and converts tensors in a batch to contiguous arrays, so a batch of the same tensors you would pass to the compiled function is now what the pass wants.

### Lab

```bash
node docs/part4/ch26-optional-pipelines/labs/02-quantization-as-a-rewrite.mjs
```

```
=== float32 ===
  7 operations: dot, constant, fusion, add, maximum, dot, add

=== int8, default activation range ===
  14 operations: quantize, quantize, quantized_dot, constant, fusion, dequantize, add, maximum, quantize, quantize, quantized_dot, fusion, dequantize, add
  relative error against float32: 18.26%
  first output element: 0.04237501695752144 -> 0.048035264015197754

=== int8, calibration ===
  14 operations: quantize, quantize, quantized_dot, constant, fusion, dequantize, add, maximum, quantize, quantize, quantized_dot, fusion, dequantize, add
  relative error against float32: 0.60%
  first output element: 0.04237501695752144 -> 0.042650580406188965

=== int8, folded weights ===
  16 operations: constant, constant, quantize, quantize, quantized_dot, constant, fusion, dequantize, add, maximum, quantize, quantize, quantized_dot, fusion, dequantize, add
  relative error against float32: 5.49%
  first output element: 0.04237501695752144 -> 0.03840856999158859

=== int8, folded weights + calibration ===
  16 operations: constant, constant, quantize, quantize, quantized_dot, constant, fusion, dequantize, add, maximum, quantize, quantize, quantized_dot, fusion, dequantize, add
  relative error against float32: 0.60%
  first output element: 0.04237501695752144 -> 0.042650580406188965
```

Two `dot`s became two `quantized_dot`s with four `quantize`s and two `dequantize`s around them, and the first configuration's answer is **18% wrong**.

That number is the chapter's point. Nothing failed: the rewrite is correct, the arithmetic is correct, the int8 kernel computes exactly what an int8 kernel should. The default activation range is `[-6, 6]`, and symmetric per-tensor int8 spreads that over ±127 levels — `scale = 6/127 ≈ 0.0472`, which is the number every `quantize` in the compiled graph carries as an attribute — print `ir` in the lab instead of the operation names to see it. The tensors actually reaching these two `dot`s span about `[-0.5, 0.5]` at the input and `[0, 0.57]` after the ReLU, so they land on roughly 20 and 12 distinct levels out of 255. The other 230-odd levels are spent representing values the model never produces. Eight bits were paid for and about four arrived. Quantization without calibration is not a fast version of your model; it is a different model.

The next three runs take the 18% apart. Calibration alone takes it to **0.60%**, which is int8-grade and is what this whole apparatus is for. `foldWeights` alone — which turns the captured parameters into graph constants, so `_getQuantParams` reads their exact range off the constant array instead of guessing ([`quantization_pass.ts:152`](../../../src/compiler/passes/quantization/quantization_pass.ts)) — takes it only to **5.49%**, and adding calibration on top of it lands on the same 0.60%.

Read the three together and the two switches are one mechanism seen from either side. Every operand of a quantized `dot` needs a real range; a *constant* operand can be measured statically, and one that is still a graph argument has to be observed on data. `foldWeights` converts operands of the second kind into the first — which is why it helps on its own, two thirds of the original error being the *weights* quantized against an activation default. Calibration covers the second kind directly, and the first kind never needed it. With neither switch on, all four operands take `[-6, 6]` and the answer is 18% wrong.

The two runs at 0.60% are not the same graph — the folded one has 16 operations to the unfolded one's 14 — and `foldWeights` still decides whether the weights are baked into the kernel or passed in, which matters for const buffers and for BYOC. It is no longer an accuracy switch.

The repository's own test suite states the underlying problem as an assertion rather than a footnote — [`calibration-exec.test.js`](../../../tests/compiler/passes/quantization/calibration-exec.test.js) has a case named *"default [-6,6] activation range is lossy on small activations (the bug)"* that requires the error to exceed 5%. Alongside it, *"calibration survives the rewrites that run before quantization"* builds `dot(x, transpose(wT))` directly, asserts from the post-pass IR dump that the transpose really was folded away, and then requires the calibrated error on the operand that replaced it to come in under 2% — the ordering above, pinned as a test rather than as a comment.

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

That is the general lesson of the chapter, and it generalizes past these three: **an optimization that trades resources needs to report what it actually achieved, not just that it ran.** Chapter 18's trace stream is where that reporting would live, and all three of these emit enough into it to tell you — the remat pass emits its peak against its budget, the calibration pass emits how many values it managed to observe — and nothing in the pipeline compares the two and complains.

## 26.5 Traps and limits

- **The remat budget is a goal, not a guarantee.** §26.1's third run: the greedy loop can run out of candidates well short of the number it was given, and all it can do about that is say so. The warning is a `trace.warn` at `INFO`, so a caller that supplies no trace sink still learns nothing; the pass has no way to fail, because failing a compilation over a soft budget would be worse.
- **The pass warns about live pressure, the planner reports peak memory, and they are different numbers.** §26.1. On the 512 KiB run the warning fires and the planner reports exactly 512 KiB, because buffer reuse happens after the pass has finished steering.
- **Remat only considers values used more than once.** A long chain of single-use activations — the ordinary shape of a forward pass before automatic differentiation adds the second uses — has no candidates at all. The pass is designed for the joint forward-backward graph of Chapter 29 and does nothing useful before it.
- **`recomputeCost` is a FLOP estimate with the same flaw as Chapter 22's.** Without a registry `getFlops`, an `exp` costs the same as an `add`, so the score ranks a transcendental recomputation as cheaply as an arithmetic one.
- **The default activation range is `[-6, 6]` and nothing warns.** §26.2 measures 18% error on a model whose activations are two orders of magnitude smaller. The compiler has the calibration machinery and, absent data, silently uses a constant. This is the finding the `compile()`-path fix makes *addressable* rather than the finding it removes: the default is unchanged, and a user who does not know to supply batches still gets it.
- **`foldWeights` alone is a partial fix that reads like a complete one.** §26.2. It takes the model to 5.49%, small enough to look like success next to 18% and eight times the 0.60% the same model reaches with calibration. Nothing says the remaining error is a missing switch rather than the price of int8.
- **A calibrated range is still keyed on `Value` identity, and only the ordering protects it.** `CalibrationResult` is a map from `Value` to observer, so an observation survives exactly as long as the value does. Nothing runs between the calibration pass and the quantization pass today, which is what makes it work; inserting a pass there would break calibration silently, and no test outside `calibration-exec.test.js` would notice. A `quantization.calibration` result supplied directly by the caller — the documented route for a GPU target, which cannot run the synchronous instrumented graph ([`calibrate_exec.ts:81`](../../../src/compiler/analysis/calibrate_exec.ts)) — is keyed on the caller's pre-pass values and does still lose most of its data.
- **Only the entry function's entry block is observed.** `CalibrationPass` calibrates `functionNames()[0]`, and `activationTargets` skips any operand not defined in that block ([`calibrate_exec.ts:34`](../../../src/compiler/analysis/calibrate_exec.ts)), because the instrumented graph can only return values it can name as outputs. A quantizable operation inside an `if` or `scan` region gets the default range, with no warning that it was skipped.
- **Partitioning needs two targets and reports nothing when it has one.** `usePartition` returns false and the phase is skipped; a user who configured `partition.enabled` and one target gets no error and no partitioning.

## 26.6 Read the tests

- [`tests/compiler/passes/memory/remat.test.js`](../../../tests/compiler/passes/memory/remat.test.js) — rematerialization candidate selection and the greedy loop.
- [`tests/compiler/passes/quantization/calibration-exec.test.js`](../../../tests/compiler/passes/quantization/calibration-exec.test.js) — end-to-end accuracy with and without calibration, including the named `(the bug)` case.
- [`tests/compiler/analysis/partitioner.test.js`](../../../tests/compiler/analysis/partitioner.test.js) and [`tests/compiler/pipeline/graph-split.test.js`](../../../tests/compiler/pipeline/graph-split.test.js) — assignment and the convexity of the resulting subgraphs.

---

**Part IV ends here.** The graph that entered was a trace of user calls; the graph that leaves has been folded, deduplicated, swept, algebraically rewritten under a licence it declares, expanded into primitives and re-collapsed into kernels by a cost model, with a layout it declined to change and three optional trades it did not take.

What it has *not* been is lowered. Every operation in it is still a tensor operation over whole tensors, with no loops, no indices and no buffers. The next two parts introduce those: Part V differentiates the graph while it is still at this level, because that is the only level where the chain rule is expressible as a graph rewrite — and Part VI turns the result into loops.

**Next:** [Part V — Automatic differentiation](../../part5/README.md), which takes the forward graph you now know how to optimize and constructs the backward one.
