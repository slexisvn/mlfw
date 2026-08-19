# Building a Machine Learning Compiler — Book Outline (overview level)

Status: overview plan. Chapter-level detail (section headings, figures, lab scripts) comes next, one Part at a time.

| Part | State |
|---|---|
| [Part 0 — Orientation](part0/README.md) | **written** — 3 chapters, 5 runnable labs |
| [Part I — Why ML needs a compiler](part1/README.md) | **written** — 4 chapters, 7 runnable labs |
| Parts II–XII | outlined below, not yet written |

## Locked decisions

| Decision | Choice |
|---|---|
| Language | English throughout |
| Scope | `src/compiler` + `src/backend` + `src/runtime` + `src/tracing` + `src/dispatcher`. Eager tensor internals, `nn`, `optim`, `lightning` appear only as callers, never as subjects |
| Theory depth | Formal definitions and short proofs, always introduced by intuition first |
| Reader | Can program, knows what a neural network is; assumed to know **nothing** about compilers |

## The teaching contract

Every chapter follows the same six-beat structure. The structure is the promise the book makes to a reader who has no compiler background: they never meet a term before it has been motivated by a problem they can see.

1. **Problem** — a concrete, measurable situation ("five elementwise ops means five passes over RAM").
2. **Intuition** — a picture or analogy. No formalism yet.
3. **Theory** — definition, algorithm, and the argument for why it is correct.
4. **In mlfw** — 15–40 lines of real source with `file:line`, walked block by block.
5. **Lab** — a runnable command; dump the IR, change one knob, observe the difference.
6. **Traps and limits** — where the implementation is deliberately conservative, and what it does not do.

### Formal apparatus

Formal material is set in labelled boxes so a reader can either absorb or skip it on first pass:

- **Definition** — precise statement of a term (SSA, dependence, legality, live interval).
- **Theorem / Lemma** — the property being claimed.
- **Proof sketch** — three to fifteen lines. Full proofs only where they are genuinely short.
- **Counterexample** — the input that breaks the naive version. Every legality rule gets one.

The named results the book will actually state and argue are listed per Part below. This is the material that separates the book from a code walkthrough: a reader should finish able to reason about *any* tensor compiler, not just this one.

**Provenance is marked on every formal statement.** Classical results carry an attribution — *(Williams et al., 2009)* for the roofline bound, *(Amdahl, 1967)*, *(classical)* for terms like arithmetic intensity and the phase-ordering problem. Statements the book formulates itself, because the property is one practitioners rely on but usually state only in prose, carry **(stated here)**. Theorem 5.3 (trace validity) and Definition 6.1 (lowering as irreversible) are the current examples. The distinction is explained to the reader in §1.6, and it matters: a classical result fails only if misapplied, a stated-here result can fail because the formulation missed a case.

### Running example

A two-layer MLP, `relu(x @ W1 + b1) @ W2 + b2` with MSE loss, threads through the entire book. The same program is shown as: user code, traced graph IR, post-fusion graph, TIR loop nest, scheduled TIR, LIR, and finally emitted CPU / WASM / CUDA source. The reader always watches one familiar program change shape.

Parts VII and VIII promote the example to a large matmul plus softmax when tiling and search need realistic sizes to be meaningful.

### Citation conventions

- Every source excerpt carries `path:line` and is quoted verbatim, never paraphrased into pseudocode.
- Each chapter ends with **Read the tests** — the test files that pin the behaviour just described. Tests are the executable specification; the book points at them rather than restating them.
- Repo statistics quoted in the text are dated, because they drift.

### Repo baseline (2026-08-19)

36,152 LOC in `src/compiler`; 64 graph ops; 33 pass classes; 63 VJP rules plus `stop_gradient` and 8 declared gradient barriers; 32 lowering rules; 23 schedule primitives; 5,131 tests passing across 302 files; `tsc --noEmit` clean under `strict`.

---

## Part 0 — Orientation (3 chapters, ~25 pages) — **written**, see [part0/](part0/README.md)

1. [**What this book is, and how to read it**](part0/ch01-what-this-book-is/README.md) — the three pictures of one program, the six-beat structure, the formal apparatus, the three reading paths.
2. [**Setting up**](part0/ch02-setting-up/README.md) — build, health check, and three labs: print the IR, measure the gap eager vs compiled, read the generated kernel.
3. [**A map of the codebase**](part0/ch03-map-of-the-codebase/README.md) — the five representations, the directory tour with sizes, the phase list, and a lab where the compiler narrates its own run.

## Part I — Why machine learning needs a compiler (4 chapters) — **written**, see [part1/](part1/README.md)

4. [**Eager execution, and where it hurts**](part1/ch04-eager-execution/README.md) — the cost model T(n) = α + βn measured; break-even size; arithmetic intensity and the roofline; why the same fusion delivers 1.10× or 4.47× depending on two `tanh` calls.
   *Formal:* Definition (per-operation cost model), Corollary (break-even size), Definition (arithmetic intensity), Theorem (roofline bound), Theorem (Amdahl, in the form needed).
5. [**From a sequence of calls to a program**](part1/ch05-calls-to-program/README.md) — four ways to extract a program (DSL, source parsing, tracing, bytecode capture with graph breaks), tracing via symbolic tensors, what the whole program buys, and the two conditions under which a trace is valid.
   *Formal:* Definition (trace), Definition (control-flow path), Theorem (trace validity, stated here), Counterexample (a branch on host state, which traces silently and is silently wrong), Definition (guard, stated here).
6. [**The pipeline in one picture**](part1/ch06-the-pipeline/README.md) — why three IRs; which optimization is expressible at which level; the phase-ordering problem and the three answers to it.
   *Formal:* Definition (lowering, irreversible), Definition (phase-ordering problem), Theorem (no fixed order is optimal for all programs).
7. [**Vocabulary**](part1/ch07-vocabulary/README.md) — every term used in the book, grouped by data / programs / transformations / execution / performance, each with the code that implements it.

## Part II — Representing programs (6 chapters)

8. **SSA and dataflow** — why "each value defined exactly once" makes analysis tractable.
   *Formal:* Definition of SSA form; Definition of a dataflow region; Theorem: an acyclic use-def graph admits a topological execution order, so textual order carries no semantics.
9. **Value, Operation, Block, Region, Function, Module** — the object model, [ir/graph/](../src/compiler/ir/graph/).
10. **The type system** — TensorType, dtypes, static / dynamic / symbolic dimensions, layout, [ir/graph/types.ts](../src/compiler/ir/graph/types.ts).
    *Formal:* shape compatibility as a partial order; broadcasting as a join.
11. **Ops as a dialect** — traits, `verify`, `inferResultTypes`, `fold`, canonicalization patterns, [op_registry.ts](../src/compiler/ir/graph/op_registry.ts).
12. **What "valid IR" means** — the verifier as executable specification, [verifier.ts](../src/compiler/ir/graph/verifier.ts).
    *Formal:* the invariant set the verifier enforces; why scope-plus-acyclicity is the right check for a dataflow IR and dominance is not required.
13. **IR as text** — print it, edit it by hand, parse it back, [printer.ts](../src/compiler/ir/graph/printer.ts) / [parser.ts](../src/compiler/ir/graph/parser.ts).

## Part III — The transformation infrastructure (5 chapters)

14. **What a pass is** — Module vs Function passes, the CHANGED / UNCHANGED / FAILED contract, [passes/pass.ts](../src/compiler/passes/pass.ts).
15. **The pass manager** — ordering, fixed-point groups, per-pass verification, [pass_manager.ts](../src/compiler/passes/pass_manager.ts).
    *Formal:* Theorem: a fixed-point group terminates; the monotone measure that makes it terminate, and why an iteration cap is still needed.
16. **Analyses and the invalidation problem** — the hardest correctness problem in pass infrastructure, [analysis_manager.ts](../src/compiler/analysis/analysis_manager.ts).
    *Formal:* Definition of a preserved analysis; Theorem: staleness propagates along the dependency DAG, hence transitive invalidation is required.
17. **Pattern rewriting** — match, rewrite, canonical form, [ir/rewrite/](../src/compiler/ir/rewrite/).
    *Formal:* confluence and termination of a rewrite set; why canonicalization must be terminating.
18. **Watching the compiler work** — trace levels, IR snapshots, `explain`, resilient mode, [pipeline/trace.ts](../src/compiler/pipeline/trace.ts).

## Part IV — Graph-level optimization (8 chapters)

19. **Constant folding, CSE, DCE** — and why memory effects make DCE non-trivial, [passes/simplify/](../src/compiler/passes/simplify/), [analysis/memory_effect.ts](../src/compiler/analysis/memory_effect.ts).
20. **Algebraic simplification meets IEEE 754** — why `x - x` is not always zero, and what a fast-math flag really licenses, [simplify/algebraic.ts](../src/compiler/passes/simplify/algebraic.ts).
    *Formal:* which identities hold over floats, which only over the reals; NaN and signed-zero counterexamples.
21. **Decomposition** — big ops into primitives, and when that loses information a library call would have kept.
22. **Fusion I: why it is the single most valuable optimization** — arithmetic intensity, the roofline argument.
    *Formal:* the memory-traffic model; Theorem: fusing a producer-consumer pair removes one full tensor round-trip.
23. **Fusion II: legality** — the cycle problem and incremental topological ordering, [graph_cycles.ts](../src/compiler/passes/fusion/graph_cycles.ts).
    *Formal:* Theorem: contracting a producer-consumer pair is legal iff it introduces no cycle; the Pearce–Kelly invariant and its amortized bound.
24. **Fusion III: the three strategies in mlfw** — dominator, priority, multi-output and epilogue fusion, plus the cost model that picks, [passes/fusion/](../src/compiler/passes/fusion/).
25. **Layout** — NCHW vs NHWC vs blocked, propagation and insertion of transforms, [passes/layout/](../src/compiler/passes/layout/).
26. **Three optional pipelines in outline** — rematerialization, quantization, partitioning and BYOC. One chapter of overview; each gets a deep appendix.

## Part V — Automatic differentiation (5 chapters)

27. **Differentiating programs** — forward vs reverse mode, and why training uses reverse.
    *Formal:* the chain rule as a linear map; Theorem: reverse mode computes a full gradient at cost proportional to the forward pass, forward mode at cost proportional to the number of inputs.
28. **Writing a VJP rule** — from the math to the registry entry, [ad/vjp_rules/](../src/compiler/ad/vjp_rules/).
29. **Building the backward graph** — from a forward graph to a joint graph, [backward_builder.ts](../src/compiler/ad/backward_builder.ts), [joint_builder.ts](../src/compiler/ad/joint_builder.ts).
30. **Trading memory for recomputation** — checkpointing policies, [checkpoint_policy.ts](../src/compiler/ad/checkpoint_policy.ts).
    *Formal:* Theorem: √n checkpointing gives O(√n) memory at one extra forward pass.
31. **Differentiating control flow** — scan and if, and the honest alternative to silent zeros: declared gradient barriers, [scan_backward.ts](../src/compiler/ad/scan_backward.ts).

## Part VI — Lowering to loops: TIR (6 chapters)

32. **From tensor algebra to loop nests** — the TensorIR model and why a second IR exists at all.
33. **Buffers, blocks, iteration variables** — spatial vs reduction axes, [ir/tensor/nodes.ts](../src/compiler/ir/tensor/nodes.ts).
    *Formal:* Definition of a block as an isolated iteration domain with declared read/write regions.
34. **Lowering rules** — one op at a time, from `add` to `conv`, [lowering/rules/](../src/compiler/passes/lowering/rules/).
35. **Index arithmetic** — linear forms, iteration maps, mixed-radix splitting, [analysis/iter_map.ts](../src/compiler/analysis/iter_map.ts).
    *Formal:* Theorem: row-major flattening is a bijection; the inverse is exactly the div/mod decomposition.
36. **Dependence analysis** — the theoretical heart of the book, [dependence.ts](../src/compiler/analysis/dependence.ts).
    *Formal:* Definitions of RAW/WAR/WAW, distance and direction vectors; Theorem (GCD test): if gcd of the coefficients does not divide the constant difference, no dependence exists; the exact SIV tests and why MIV falls back to a conservative answer.
37. **Proving things about indices** — interval bounds, provable non-negativity, provable divisibility, [analyzer.ts](../src/compiler/analysis/analyzer.ts).
    *Formal:* soundness direction of interval arithmetic — an analyzer may say "unknown", never "in bounds" wrongly.

## Part VII — Scheduling (6 chapters)

38. **Separating what from how** — the Halide/TVM idea, stated precisely.
39. **The sref tree and block scopes** — how a schedule edits IR without destroying structure, [schedule/sref.ts](../src/compiler/schedule/sref.ts), [block_scope.ts](../src/compiler/schedule/block_scope.ts).
40. **Loop primitives** — split, fuse, reorder, tile, and the predicate that appears when the extent does not divide, [schedule.ts:248](../src/compiler/schedule/schedule.ts:248).
    *Formal:* Theorem: split with a guard preserves semantics for any extent; the counterexample without the guard.
41. **Memory and reduction primitives** — cache_read/cache_write, rfactor, decompose_reduction, storage_align.
    *Formal:* Theorem: rfactor is valid only if the reduction operator is associative and commutative; the floating-point caveat.
42. **Legality** — which primitive is allowed when, decided by dependence, [schedule/legality.ts](../src/compiler/schedule/legality.ts).
    *Formal:* Theorem: a loop permutation is legal iff no dependence direction vector is reversed under it; worked counterexample.
43. **Scheduling for GPUs** — thread and block binding, shared memory, tensorization, and detecting cross-thread races, [analysis/gpu_race.ts](../src/compiler/analysis/gpu_race.ts).

## Part VIII — Autotuning (5 chapters)

44. **How big is the search space** — counting it for one matmul; why hand heuristics run out.
45. **Sketches** — generating a schedule skeleton per problem class, [autotune/sketch_generators.ts](../src/compiler/autotune/sketch_generators.ts).
46. **Cost models** — analytic features vs a learned model, and why ranking beats absolute prediction, [cost_model.ts](../src/compiler/autotune/cost_model.ts), [gbt.ts](../src/compiler/autotune/gbt.ts).
    *Formal:* the ranking objective; why regression error is the wrong metric here.
47. **Search and measurement** — evolutionary search, real benchmarking, workload keys, the tuning database and its versioning, [search.ts](../src/compiler/autotune/search.ts), [tuning_db.ts](../src/compiler/autotune/tuning_db.ts).
48. **Reproducibility** — schedule traces as serializable, replayable objects, [schedule/trace.ts](../src/compiler/schedule/trace.ts).

## Part IX — Memory (4 chapters)

49. **Buffer lifetimes** — live intervals over a linearized program, [buffer_liveness.ts](../src/compiler/passes/memory/buffer_liveness.ts).
50. **Arena allocation** — best-fit and first-fit packing, alignment, [buffer_assignment.ts](../src/compiler/passes/memory/buffer_assignment.ts).
    *Formal:* the problem as 2D strip packing; NP-hardness in general and what greedy-by-size buys in practice.
51. **In-place reuse and donation** — when overwriting an input is provably safe, [inplace_analysis.ts](../src/compiler/passes/memory/inplace_analysis.ts).
52. **Scheduling to lower peak memory** — reordering independent work to shrink the high-water mark, [memory_scheduler.ts](../src/compiler/passes/memory/memory_scheduler.ts).

## Part X — Code generation (6 chapters)

53. **LIR: why a third IR** — flattened indices, explicit accumulators, [ir/lir/](../src/compiler/ir/lir/).
54. **Generating JavaScript for the CPU** — [backend/cpu/codegen.ts](../src/backend/cpu/codegen.ts).
55. **WebAssembly** — emitting WAT, encoding the binary, SIMD, [backend/wasm/](../src/backend/wasm/).
56. **CUDA** — kernels, launch geometry, shared memory, tensor cores, [backend/cuda/](../src/backend/cuda/).
57. **WebGPU and WGSL** — the constraints a browser target imposes, [backend/webgpu/](../src/backend/webgpu/).
58. **Calling someone else's kernel** — cuBLAS/cuDNN and the external codegen interface, [pipeline/external_codegen.ts](../src/compiler/pipeline/external_codegen.ts).
    *Formal (cross-cutting):* integer division and modulo semantics — floor vs truncation — stated once and held to by every backend. **Decided:** TIR `//` and `%` are floor division and floor modulo, matching the symbolic layer and the analyzer's bounds; `tdiv`/`tmod` are the truncating primitives, introduced by the simplifier only where the dividend is provably non-negative and the two agree.

## Part XI — From compiler to framework (5 chapters)

59. **The runtime module** — loading, calling, device memory, [runtime/runtime.ts](../src/runtime/runtime.ts).
60. **A PyTorch-style dispatcher** — dispatch keys, boxing, fallbacks, [src/dispatcher/](../src/dispatcher/).
61. **Tracing** — turning eager user code into a GraphModule, [tracing/compile.ts](../src/tracing/compile.ts).
62. **Dynamic shapes** — symbolic dimensions, guards, the JIT cache, recompilation, [tracing/shape_env.ts](../src/tracing/shape_env.ts), [dispatcher/jit_cache.ts](../src/dispatcher/jit_cache.ts).
    *Formal:* Definition of a guard; Theorem: a specialized kernel is safe to reuse exactly when its guard set holds.
63. **Training end to end** — the running example, compiled, trained, measured against eager.

## Part XII — Being sure it is right (4 chapters)

64. **Verification** — boundary checks and per-pass checks, and what each catches, [pipeline/invariant_check.ts](../src/compiler/pipeline/invariant_check.ts).
65. **Differential testing** — compiled vs eager vs finite differences, [tests/e2e/](../../tests/e2e/).
    *Formal:* finite-difference error as a function of step size; how to pick a tolerance that is neither vacuous nor flaky.
66. **Fuzzing, scaling tests, and numerical conformance** — including tests that assert an algorithm is not accidentally quadratic.
67. **Debugging a wrong answer** — a repeatable bisection procedure from wrong output down to the offending pass.

## Appendices

- **A. Op reference** — all 64 ops: operands, results, attributes, traits, VJP status. *Generated from the registry.*
- **B. Pass catalog** — every pass, its pipeline position, required and preserved analyses. *Generated.*
- **C. Glossary** — every defined term with the chapter that defines it.
- **D. File-to-chapter map** — for readers who arrive from the source rather than the table of contents.
- **E. Known limits and open work** — carried from the compiler review; each entry names the chapter it constrains. Current entries: interval arithmetic answering "unknown" where a stronger analysis would prove safety (Chapter 37). *Closed:* `//` and `%` disagreeing between the symbolic layer, constant folding and the four backends — now floor everywhere, from one definition, with `tdiv`/`tmod` carrying truncation where it is provably equivalent (Chapters 35, 36, 54–58); reading a value from a symbolic tensor failing with an uninformative `Cannot read properties of null` — `SymbolicTensor` now overrides the value-reading accessors and names data-dependent control flow (Chapter 5).
- **F. Walkthrough: adding an operation end to end** — one new op taken through definition, verification, canonicalization, VJP rule, lowering, scheduling and four backends, in a single continuous exercise. Chapter 1 promises the reader they will be able to do this; the chapters teach the pieces separately, and this appendix is where they are assembled.

---

## Execution notes

### Writing order

Not 1 → 67. The technical spine first, because it fixes the vocabulary everything else borrows:

1. Parts II, III, VI, VII — object model, pass infrastructure, TIR, scheduling.
2. Parts I, IV, V — motivation and graph-level work, now expressible in settled terms.
3. Parts VIII–XII.
4. Part 0 last — an introduction is only honest once the book exists.

### Dependencies to resolve before writing

- ~~**Integer division and modulo semantics.**~~ *Resolved.* `//` and `%` are floor division and floor modulo in every layer — [util/divmod.ts](../src/util/divmod.ts) holds the single scalar definition, used by `SymInt`, TIR constant folding and all four backends. Truncation survives as the separate `tdiv`/`tmod` pair, emitted only where the simplifier has proved the dividend non-negative, so index math costs no more than before. Chapters 35, 36 and 54–58 can now state the rule and hold every backend to it.
- **Coverage reporting.** Part XII will quote test-suite numbers; coverage currently produces no report on a multi-project run.
- **Hardware-gated tests.** Chapters 56 and 57 must state where CUDA and WebGPU were actually verified, since 30 test blocks skip without hardware.

### Generated content

Appendices A and B must be generated from `registry` and the pipeline builders, not written by hand — 64 ops and 33 passes drift within weeks. Plan a small script under `docs/tools/` and regenerate on every release.

### Size estimate

67 chapters at 8–12 pages ≈ 550–800 pages, which is long for a first edition. Three levers, in the order they should be pulled:

1. Compress Part VIII (autotuning) from five chapters to three — the search machinery is easier to compress than the theory it searches over.
2. Merge Part IX (memory) into Part VI as two chapters rather than four.
3. Hold Parts IV, VI and VII at full length regardless. They carry the material that transfers to other compilers, and cutting them is what turns the book into a code tour.

Applying the first two lands near 60 chapters and 480–700 pages.
