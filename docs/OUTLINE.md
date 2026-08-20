# Building a Machine Learning Compiler — Book Outline (overview level)

Status: overview plan. Chapter-level detail (section headings, figures, lab scripts) comes next, one Part at a time.

| Part | State |
|---|---|
| [Part 0 — Orientation](part0/README.md) | **written** — 3 chapters, 5 runnable labs |
| [Part I — Why ML needs a compiler](part1/README.md) | **written** — 4 chapters, 7 runnable labs |
| [Part II — Representing programs](part2/README.md) | **written** — 6 chapters, 11 runnable labs |
| [Part III — The transformation infrastructure](part3/README.md) | **written** — 5 chapters, 12 runnable labs |
| [Part IV — Graph-level optimization](part4/README.md) | **written** — 8 chapters, 15 runnable labs |
| Parts V–XII | outlined below, not yet written |

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

36,166 LOC in `src/compiler`; 96 graph ops; 31 concrete pass classes (21 graph, 9 TIR, 1 LIR); 63 VJP rules plus `stop_gradient` and 8 declared gradient barriers; 32 lowering rules; 23 schedule primitives; 5,131 tests passing across 302 files; `tsc --noEmit` clean under `strict`.

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

## Part II — Representing programs (6 chapters) — **written**, see [part2/](part2/README.md)

8. [**SSA and dataflow**](part2/ch08-ssa-and-dataflow/README.md) — why "each value defined exactly once" makes analysis tractable; the intrusive use list; and the lab that prints a function backwards and re-parses it.
   *Formal:* Definition (SSA form), Definition (use-def graph), Lemma (O(1) producer, O(k) consumers), Theorem 8.4 (textual order carries no semantics, stated here), Counterexample (a shared mutable buffer).
9. [**Value, Operation, Block, Region, Function, Module**](part2/ch09-object-model/README.md) — the object model, [ir/graph/](../src/compiler/ir/graph/); the intrusive operation list, the version counter every edit bumps, and what a region can see.
   *Formal:* Definition 9.1 (region scope isolation, stated here).
10. [**The type system**](part2/ch10-type-system/README.md) — TensorType, dtypes, static / dynamic / symbolic dimensions, layout, [ir/graph/types.ts](../src/compiler/ir/graph/types.ts).
    *Formal:* Definition (specificity order — a genuine partial order on dimensions), Definition (compatibility as unifiability), Theorem 10.3 (compatibility is **not** transitive, hence not a partial order — the outline's earlier phrasing was wrong and the chapter says so), Definition (broadcast order), Counterexample (`broadcastDim(?, 4) = 4` is an assumption delegated to a guard).
11. [**Ops as a dialect**](part2/ch11-ops-as-a-dialect/README.md) — traits, `verify`, `inferResultTypes`, `fold`, canonicalization patterns, [op_registry.ts](../src/compiler/ir/graph/op_registry.ts); `add` and `sub` differing by one word, and CSE behaving differently because of it.
    *Formal:* Definition 11.1 (op registry).
12. [**What "valid IR" means**](part2/ch12-valid-ir/README.md) — the verifier as executable specification, [verifier.ts](../src/compiler/ir/graph/verifier.ts); three checkers at three moments, and the seven-way break lab that shows which catches what.
    *Formal:* the four load-bearing invariants; Theorem 12.1 (scope plus acyclicity suffices, stated here) — why dominance is not required here.
13. [**IR as text**](part2/ch13-ir-as-text/README.md) — print it, edit it by hand, parse it back, [printer.ts](../src/compiler/ir/graph/printer.ts) / [parser.ts](../src/compiler/ir/graph/parser.ts); the parser is two-phase because the IR is a DAG.
    *Formal:* Definition 13.1 (lossless round-trip, stated here).

## Part III — The transformation infrastructure (5 chapters) — **written**, see [part3/](part3/README.md)

14. [**What a pass is**](part3/ch14-what-a-pass-is/README.md) — module vs function passes, the CHANGED / UNCHANGED / FAILED contract, [passes/pass.ts](../src/compiler/passes/pass.ts); the ledger that shows 11 of 15 pass runs changing nothing, and `PassContext` as the switch that turns one off from outside.
    *Formal:* Definition 14.1 (pass, with the one-way verdict implication, stated here), Definition 14.2 (pass granularity, stated here).
15. [**The pass manager**](part3/ch15-the-pass-manager/README.md) — ordering, fixed-point groups, per-pass verification, [pass_manager.ts](../src/compiler/passes/pass_manager.ts); the three qualities of answer you get from the three verification levels.
    *Formal:* Definition 15.1 (pass pipeline), Definition 15.2 (fixed-point group), Lemma 15.3 (convergence costs one extra round, stated here), Theorem 15.4 (termination — and the honest finding that the iteration cap **is** the termination argument, because no monotone measure exists), Counterexample (`canonicalize: CHANGED 10 -> 10`).
16. [**Analyses and the invalidation problem**](part3/ch16-analyses-and-invalidation/README.md) — the hardest correctness problem in pass infrastructure, [analysis_manager.ts](../src/compiler/analysis/analysis_manager.ts).
    *Formal:* Definition 16.1 (analysis), Definition 16.2 (preserved analysis), Theorem 16.3 (transitive invalidation is required, stated here), Definition 16.4 (mutation version, stated here), Counterexample (a preservation that cannot be honoured serves 10 of 14 passes a stale answer, and nothing catches it).
17. [**Pattern rewriting**](part3/ch17-pattern-rewriting/README.md) — match, rewrite, canonical form, [ir/rewrite/](../src/compiler/ir/rewrite/); the worklist that turns independent rules into a cascade, and traits generating canonicalization patterns from the registry.
    *Formal:* Definition 17.1 (rewrite rule), Definition 17.2 (normal form), Theorem 17.3 (Newman's Lemma — *(Newman, 1942)*), and the stated limit that neither termination nor confluence is proved here; both are bounded.
18. [**Watching the compiler work**](part3/ch18-watching-the-compiler/README.md) — trace levels, IR snapshots, `explain`, resilient mode, [pipeline/trace.ts](../src/compiler/pipeline/trace.ts).
    *Formal:* Definition 18.1 (trace level, with the monotonicity constraint), Definition 18.2 (explanation — a decision plus the terms the decision procedure used, stated here), Definition 18.3 (transactional compilation, stated here).

## Part IV — Graph-level optimization (8 chapters) — **written**, see [part4/](part4/README.md)

19. [**Constant folding, CSE, DCE**](part4/ch19-fold-cse-dce/README.md) — and why memory effects make DCE non-trivial, [passes/simplify/](../src/compiler/passes/simplify/), [analysis/memory_effect.ts](../src/compiler/analysis/memory_effect.ts); a dead `scatter` and its whole index subgraph surviving a pass whose job is removing them.
    *Formal:* Definition 19.1 (constant-foldable), Definition 19.2 (redundant), Definition 19.3 (dead), Theorem 19.4 (soundness of DCE, stated here) and the asymmetry it exposes, Corollary 19.5 (DCE is a fixed point).
20. [**Algebraic simplification meets IEEE 754**](part4/ch20-algebra-and-ieee754/README.md) — why `x - x` is not always zero, and what a fast-math flag really licenses, [simplify/algebraic.ts](../src/compiler/passes/simplify/algebraic.ts).
    *Formal:* Theorem 20.1 (identities valid over floats), Theorem 20.2 (identities not valid, with the NaN, ±∞ and signed-zero counterexamples), Definition 20.3 (fast-math licence, stated here). Measured: two rewrites that fire with the licence withheld, one in a graph pattern and one in a backend's string peephole.
21. [**Decomposition**](part4/ch21-decomposition/README.md) — big ops into primitives, and when that loses information a library call would have kept.
    *Formal:* Definition 21.1 (decomposition rule), Definition 21.2 (primitive set, stated here), Definition 21.3 (lossy vs neutral decomposition, stated here); the numerical-form note, since `softmax`'s rule is where max-subtraction is decided for every backend.
22. [**Fusion I: why it is the single most valuable optimization**](part4/ch22-fusion-why/README.md) — arithmetic intensity, the roofline argument, measured 2.55× flat across four orders of magnitude.
    *Formal:* Definition 22.1 (memory traffic), Definition 22.2 (fusion of a group), Theorem 22.3 (one round trip removed per internalized value, stated here), Corollary 22.4 (a chain of k gains a factor k).
23. [**Fusion II: legality**](part4/ch23-fusion-legality/README.md) — the cycle problem and incremental topological ordering, [graph_cycles.ts](../src/compiler/passes/fusion/graph_cycles.ts).
    *Formal:* Definition 23.1 (contraction), Theorem 23.2 (legal iff the contraction is acyclic), Definition 23.3 (incremental topological order), Theorem 23.4 (Pearce–Kelly, *(Pearce and Kelly, 2006)*), Corollary 23.5 (windowed cycle detection, stated here). Measured exponent 1.59 — and the finding that the cost is candidate re-evaluation, not the cycle check.
24. [**Fusion III: the three strategies in mlfw**](part4/ch24-fusion-strategies/README.md) — dominator, priority, multi-output and epilogue fusion, plus the cost model that picks, [passes/fusion/](../src/compiler/passes/fusion/).
    *Formal:* Definition 24.1 (fusion partition), Theorem 24.2 (NP-hard, classical), Definition 24.3 (monotone candidate set, stated here), Definition 24.4 (stale candidate, stated here). Measured: 2.4× between the default strategy and the same strategy with one cost-model constant corrected.
25. [**Layout**](part4/ch25-layout/README.md) — NCHW vs NHWC vs blocked, propagation and insertion of transforms, [passes/layout/](../src/compiler/passes/layout/); off by default, inert when enabled, and slower when properly enabled.
    *Formal:* Definition 25.1 (layout as a permutation), Definition 25.2 (layout assignment problem, stated here), Theorem 25.3 (NP-hard as multiway cut, classical), Definition 25.4 (greedy propagation with local accept, stated here).
26. [**Three optional pipelines in outline**](part4/ch26-optional-pipelines/README.md) — rematerialization, quantization, partitioning and BYOC. One chapter of overview; each gets a deep appendix.
    *Formal:* Definition 26.1 (rematerialization), Theorem 26.2 (√n checkpointing, *(Chen et al., 2016)*), Definition 26.3 (affine quantization), Definition 26.4 (calibration), Definition 26.5 (partition) plus the convexity note that makes Theorem 23.2 reappear unchanged.

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
65. **Differential testing** — compiled vs eager vs finite differences, [tests/e2e/](../tests/e2e/).
    *Formal:* finite-difference error as a function of step size; how to pick a tolerance that is neither vacuous nor flaky.
66. **Fuzzing, scaling tests, and numerical conformance** — including tests that assert an algorithm is not accidentally quadratic.
67. **Debugging a wrong answer** — a repeatable bisection procedure from wrong output down to the offending pass.

## Appendices

- **A. Op reference** — all 96 ops: operands, results, attributes, traits, VJP status. *Generated from the registry.*
- **B. Pass catalog** — every pass, its pipeline position, required and preserved analyses. *Generated.*
- **C. Glossary** — every defined term with the chapter that defines it.
- **D. File-to-chapter map** — for readers who arrive from the source rather than the table of contents.
- **E. Known limits and open work** — carried from the compiler review; each entry names the chapter it constrains. Current entries: interval arithmetic answering "unknown" where a stronger analysis would prove safety (Chapter 37).

  *Found while writing Part IV, each reproducible by a lab in that part, none fixed in the code the book describes:*
  1. `AddZero` (`patterns.ts:167`) applies `x + 0 → x` with no fast-math gate; unsound for `−0`, where IEEE gives `+0`. Its three neighbouring rules take the gate. (Chapter 20)
  2. The CPU backend's expression peephole (`backend/cpu/codegen.ts:472`) folds `x * 0 → 0` on rendered strings, with no dtype and no flag in scope; `∞ × 0` and `NaN × 0` become `0`. No other backend does this, so CPU and CUDA disagree on non-finite inputs. (Chapters 20, 65)
  3. `scatter` declares `sideEffects: WRITE` (`ops/shape.ts:368`) although its lowering writes only its own output buffer; a dead `scatter` and its whole index-computation subgraph survive DCE, CSE and folding. (Chapter 19)
  4. `target.sharedMemoryBytes || …` (`fusion_cost.ts:62`, `priority_fusion.ts:62`) reads a CPU's declared `0` as "unspecified" and applies a 48 KiB GPU shared-memory budget; measured 2.4× on a three-operation diamond. (Chapter 24)
  5. `layoutAwareOps` is empty for every shipped target (`target.ts:129`), so `LayoutTransformPass` proposes conversions and discards all of them, reporting UNCHANGED — indistinguishable from having nothing to do. (Chapter 25)
  6. `RematerializationPass` exits identically whether it met its `memoryBudget` or ran out of candidates; asked for 128 KiB it delivered 512 KiB and reported success. (Chapter 26)
  7. Quantization's `calibrationData` is unusable through the public `compile()` path: `collectCalibration` (`calibrate_exec.ts:101`) passes the batch as the graph's complete argument list, but a traced model's captured parameters are arguments too. Without calibration the default `[-6, 6]` activation range gives 18% relative error, silently. (Chapter 26) *Closed:* `//` and `%` disagreeing between the symbolic layer, constant folding and the four backends — now floor everywhere, from one definition, with `tdiv`/`tmod` carrying truncation where it is provably equivalent (Chapters 35, 36, 54–58); reading a value from a symbolic tensor failing with an uninformative `Cannot read properties of null` — `SymbolicTensor` now overrides the value-reading accessors and names data-dependent control flow (Chapter 5).
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

Appendices A and B must be generated from `registry` and the pipeline builders, not written by hand — 96 ops and 33 passes drift within weeks, and a hand count silently misses every op registered inside a loop, which is how an earlier draft of this outline came to claim 64. Plan a small script under `docs/tools/` and regenerate on every release.

### Size estimate

67 chapters at 8–12 pages ≈ 550–800 pages, which is long for a first edition. Three levers, in the order they should be pulled:

1. Compress Part VIII (autotuning) from five chapters to three — the search machinery is easier to compress than the theory it searches over.
2. Merge Part IX (memory) into Part VI as two chapters rather than four.
3. Hold Parts IV, VI and VII at full length regardless. They carry the material that transfers to other compilers, and cutting them is what turns the book into a code tour.

Applying the first two lands near 60 chapters and 480–700 pages.
