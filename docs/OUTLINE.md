# Building a Machine Learning Compiler — Book Outline (overview level)

Status: overview plan. Chapter-level detail (section headings, figures, lab scripts) comes next, one Part at a time.

| Part | State |
|---|---|
| [Part 0 — Orientation](part0/README.md) | **written** — 3 chapters, 5 runnable labs |
| [Part I — Why ML needs a compiler](part1/README.md) | **written** — 4 chapters, 7 runnable labs |
| [Part II — Representing programs](part2/README.md) | **written** — 6 chapters, 11 runnable labs |
| [Part III — The transformation infrastructure](part3/README.md) | **written** — 5 chapters, 12 runnable labs |
| [Part IV — Graph-level optimization](part4/README.md) | **written** — 8 chapters, 15 runnable labs |
| [Part V — Automatic differentiation](part5/README.md) | **written** — 5 chapters, 9 runnable labs |
| [Part VI — Lowering to loops: TIR](part6/README.md) | **written** — 6 chapters, 12 runnable labs |
| [Part VII — Scheduling](part7/README.md) | **written** — 6 chapters, 12 runnable labs |
| [Part VIII — Autotuning](part8/README.md) | **written** — 5 chapters, 10 runnable labs |
| [Part IX — Memory](part9/README.md) | **written** — 4 chapters, 4 runnable labs |
| Parts X–XII | outlined below, not yet written |

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

**Provenance is marked on every formal statement, and there are four kinds.** The distinction is explained to the reader in §1.6, and it matters because the four fail in four different ways.

- **(classical)** — a mathematical result that can be looked up, carrying an attribution: *(Williams et al., 2009)* for the roofline bound, *(Amdahl, 1967)*, *(classical)* for terms like arithmetic intensity and the phase-ordering problem. Fails only if misapplied.
- **(stated here)** — a property practitioners rely on but usually state only in prose, which the book formulates. Theorem 5.3 (trace validity) and Definition 6.1 (lowering as irreversible) are the canonical examples. Can fail because the formulation missed a case.
- **(invariant)** — a contract this codebase intends to maintain, stated together with the mechanism that enforces it, and with a note where nothing enforces it but convention. Can be violated by a bug; §1.11 lists the contracts currently broken.
- **(measured)** — an observation on one machine, one date, one revision, reported as a median with its spread. Never used as a premise in a proof.

A labelled block that makes a claim about *this implementation* is **(invariant)** or **(measured)**, never **(stated here)**. Getting that boundary wrong is how a book ends up asserting that a compiler is sound because its documentation says so.

**The numerical-equivalence ladder is book-wide.** Definition 1.4 fixes four levels — N0 bit-identical, N1 same operations in the same order, N2 reassociated, N3 algebraically rewritten — and every legality claim in every Part names the level it means. Reduction order is part of the contract: any transformation that changes the number of partial accumulators is N2 and must be labelled one.

### Running example

A two-layer MLP, `relu(x @ W1 + b1) @ W2 + b2` with MSE loss, threads through the entire book. The same program is shown as: user code, traced graph IR, post-fusion graph, TIR loop nest, scheduled TIR, LIR, and finally emitted CPU / WASM / CUDA source. The reader always watches one familiar program change shape.

Parts VII and VIII promote the example to a large matmul plus softmax when tiling and search need realistic sizes to be meaningful.

### Citation conventions

- Every source excerpt carries `path:line` and is quoted verbatim, never paraphrased into pseudocode.
- Each chapter ends with **Read the tests** — the test files that pin the behaviour just described. Tests are the executable specification; the book points at them rather than restating them.
- Repo statistics quoted in the text are dated, because they drift.

### Repo baseline (2026-08-19; AD and lowering figures re-measured 2026-08-20)

36,166 LOC in `src/compiler`; 96 graph ops; 31 concrete pass classes (21 graph, 9 TIR, 1 LIR); 63 VJP rules (including `stop_gradient`) plus 8 further declared gradient barriers, and 43 JVP rules; **68 of the 96 ops have a lowering rule**, the other 28 being 21 removed by decomposition and 7 handled structurally (Definition 34.6); 21 TIR node kinds; **22 schedule primitives** (28 public members of `Schedule` less six queries; the earlier count of 23 was made by hand); 5,131 tests passing across 302 files; `tsc --noEmit` clean under `strict`.

The lowering figure was "32" in earlier drafts. That number counted literal `registerLoweringRule('name', …)` sites and missed the 33 elementwise rules registered in a loop, `argmax`/`argmin` registered by a helper, and the one target-specific rule — the same failure mode that once made this outline claim 64 ops. It is now measured from the strategy registry, as Appendix A will be.

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
22. [**Fusion I: why it is the single most valuable optimization**](part4/ch22-fusion-why/README.md) — arithmetic intensity, the roofline argument, measured 1.5–2.4×, flat across four orders of magnitude, with the interquartile ranges reported so the flatness is checkable.
    *Formal:* Definition 22.1 (memory traffic), Definition 22.2 (fusion of a group), Theorem 22.3 (one round trip removed per internalized value, stated here), Corollary 22.4 (a chain of k gains a factor k).
23. [**Fusion II: legality**](part4/ch23-fusion-legality/README.md) — the cycle problem and incremental topological ordering, [graph_cycles.ts](../src/compiler/passes/fusion/graph_cycles.ts).
    *Formal:* Definition 23.1 (contraction), Theorem 23.2 (legal iff the contraction is acyclic), Definition 23.3 (incremental topological order), Theorem 23.4 (Pearce–Kelly, *(Pearce and Kelly, 2006)*), Corollary 23.5 (windowed cycle detection, stated here). Measured exponent 1.59 — and the finding that the cost is candidate re-evaluation, not the cycle check.
24. [**Fusion III: the three strategies in mlfw**](part4/ch24-fusion-strategies/README.md) — dominator, priority, multi-output and epilogue fusion, plus the cost model that picks, [passes/fusion/](../src/compiler/passes/fusion/).
    *Formal:* Definition 24.1 (fusion partition), Theorem 24.2 (NP-hard, classical), Definition 24.3 (monotone candidate set, stated here), Counterexample 24.4 (legality is not monotone), Definition 24.5 (stale candidate, stated here). Measured 2026-08-21: 2.3× between the default strategy and the same strategy with one cost-model constant corrected; the ratio reproduces, the absolute times do not.
25. [**Layout**](part4/ch25-layout/README.md) — NCHW vs NHWC vs blocked, propagation and insertion of transforms, [passes/layout/](../src/compiler/passes/layout/); off by default, inert when enabled, and slower when properly enabled.
    *Formal:* Definition 25.1 (layout as a permutation), Definition 25.2 (layout assignment problem, stated here), Theorem 25.3 (NP-hard as multiway cut, classical), Definition 25.4 (greedy propagation with local accept, stated here).
26. [**Three optional pipelines in outline**](part4/ch26-optional-pipelines/README.md) — rematerialization, quantization, partitioning and BYOC. One chapter of overview; each gets a deep appendix.
    *Formal:* Definition 26.1 (rematerialization), Theorem 26.2 (√n checkpointing, *(Chen et al., 2016)*), Definition 26.3 (affine quantization), Definition 26.4 (calibration), Definition 26.5 (partition) plus the convexity note that makes Theorem 23.2 reappear unchanged.

## Part V — Automatic differentiation (5 chapters) — **written**, see [part5/](part5/README.md)

27. **Differentiating programs** — forward vs reverse mode, and why training uses reverse.
    *Formal:* the chain rule as a linear map; Theorem: reverse mode computes a full gradient at cost proportional to the forward pass, forward mode at cost proportional to the number of inputs.
28. **Writing a VJP rule** — from the math to the registry entry, [ad/vjp_rules/](../src/compiler/ad/vjp_rules/).
29. **Building the backward graph** — from a forward graph to a joint graph, [backward_builder.ts](../src/compiler/ad/backward_builder.ts), [joint_builder.ts](../src/compiler/ad/joint_builder.ts).
30. **Trading memory for recomputation** — checkpointing policies, [checkpoint_policy.ts](../src/compiler/ad/checkpoint_policy.ts).
    *Formal:* Theorem: √n checkpointing gives O(√n) memory at one extra forward pass.
31. **Differentiating control flow** — scan and if, and the honest alternative to silent zeros: declared gradient barriers, [scan_backward.ts](../src/compiler/ad/scan_backward.ts).

## Part VI — Lowering to loops: TIR (6 chapters) — **written**, see [part6/](part6/README.md)

32. [**From tensor algebra to loop nests**](part6/ch32-tensor-algebra-to-loops/README.md) — why a second IR exists; the five-walk driver, the two decisions no rule can make, and the 21-node language that comes out of a 96-operation one.
    *Formal:* Definition 32.1 (iteration domain), Definition 32.2 (loop nest), Definition 32.3 (lowering), Theorem 32.4 (lowering is not injective, stated here) with a character-identical counterexample, Corollary 32.5 (irreversibility — Definition 6.1 where it bites), Definition 32.6 (lowering rule, and what it may not see). Measured: 6 graph ops become 16 loops.
33. [**Buffers, blocks, iteration variables**](part6/ch33-buffers-blocks-itervars/README.md) — the loops are the plan and the block is the computation; spatial vs reduction axes, `local` scratch, and the declaration the scheduler trusts, [ir/tensor/nodes.ts](../src/compiler/ir/tensor/nodes.ts).
    *Formal:* Definition 33.1 (buffer, and layout as strides), Definition 33.2 (block), Definition 33.3 (iteration variable kind), Definition 33.4 (block abstraction, stated here), Definition 33.5 (what `DataPar` claims, stated here), Proposition 33.6 (kind-based legality, stated here), Corollary 33.7 (the declaration is load-bearing — and is checked by nothing).
34. [**Lowering rules**](part6/ch34-lowering-rules/README.md) — 68 rules, a two-level priority scheme, two override points, and five shared skeletons, [lowering/rules/](../src/compiler/passes/lowering/rules/).
    *Formal:* Definition 34.1 (op strategy), Definition 34.2 (implementation selection — by priority alone, never by shape), Definition 34.3 (coverage), Proposition 34.4 (coverage is a joint property of registry and pipeline, stated here), Definition 34.5 (rule skeleton, stated here). Measured 2026-08-21: 68 ruled + 21 decomposed + 7 structural = 96, with nothing left over; Definition 34.6 fixes the three classes so the counts stay comparable across Parts.
35. [**Index arithmetic**](part6/ch35-index-arithmetic/README.md) — linear forms, iteration maps, mixed-radix splitting, [analysis/iter_map.ts](../src/compiler/analysis/iter_map.ts).
    *Formal:* Definition 35.1 (affine form, and why `split` stays inside it while a loop fuse does not), Definition 35.2 and Theorem 35.3 (row-major flattening is a bijection; the inverse is exactly the div/mod recurrence), Definition 35.4 and Theorem 35.5 (exact divisor split, stated here), Corollary 35.6, Definition 35.7 and Theorem 35.8 (mixed-radix forms are exact covers, stated here). Measured: 1.10× between two reshapes of the same 98,304 elements, one of which satisfies Theorem 35.5.
36. [**Dependence analysis**](part6/ch36-dependence-analysis/README.md) — the theoretical heart of the book, [dependence.ts](../src/compiler/analysis/dependence.ts).
    *Formal:* Definition 36.1 (RAW/WAR/WAW), Definition 36.2 (distance and direction vectors, carried level), Theorem 36.3 (loop parallelism iff nothing carried at that level), Theorem 36.4 (GCD test, *(Banerjee, Towle, 1976)*), Theorem 36.5 (strong SIV, exact), Definition 36.6 (MIV and the conservative fallback), Theorem 36.7 (coincidence of a mixed-radix subscript, stated here — Theorem 35.8 reused), Theorem 36.8 (interchange legality, classical, used in Chapter 42).
37. [**Proving things about indices**](part6/ch37-proving-things-about-indices/README.md) — interval bounds, provable non-negativity, and the licence for `tdiv`, [analyzer.ts](../src/compiler/analysis/analyzer.ts).
    *Formal:* Definition 37.1 (interval abstraction), Definition 37.2 (sound abstraction), Theorem 37.3 (soundness, *(Moore, 1966; Cousot and Cousot, 1977)*), Theorem 37.4 (one-sided decidability — an analyser may say "unknown", never "in bounds" wrongly), Definition 37.5 (non-relational domain and what it costs), Theorem 37.6 (floor and truncation agree on non-negative dividends — the licence for the `tdiv`/`tmod` substitution that Chapters 54–58 depend on). Measured: `pad` emits four comparisons and two reach the kernel.

## Part VII — Scheduling (6 chapters) — **written**, see [part7/](part7/README.md)

38. [**Separating what from how**](part7/ch38-separating-what-from-how/README.md) — the Halide/TVM idea, stated precisely; the 22 primitives and the 9 rules; one program scheduled for four machines; and what an annotation is worth on a backend that ignores it.
    *Formal:* Definition 38.1 (schedule as a sequence of partial functions), Definition 38.2 (semantic equivalence for a `PrimFunc`, stated here), Definition 38.3 (sound primitive), Proposition 38.4 (soundness composes — why a search cannot produce a wrong program, and the one primitive whose unsoundness it therefore does not cover), Definition 38.5 (advisory annotation, stated here).
39. [**The sref tree and block scopes**](part7/ch39-sref-tree-and-block-scopes/README.md) — how a schedule edits IR without destroying structure, [schedule/sref.ts](../src/compiler/schedule/sref.ts), [block_scope.ts](../src/compiler/schedule/block_scope.ts).
    *Formal:* Definition 39.1 (sref tree), Definition 39.2 (block scope), Definition 39.3 (scope dependence), Lemma 39.4 (scope edges over-approximate, stated here), Proposition 39.5 (relocation legality, stated here).
40. [**Loop primitives**](part7/ch40-loop-primitives/README.md) — split, fuse, reorder, tile, and the predicate that appears when the extent does not divide, [schedule.ts](../src/compiler/schedule/schedule.ts).
    *Formal:* Definition 40.1 (split), Theorem 40.2 (split with a guard preserves semantics for any extent), Counterexample 40.3 (the same split without the guard, executed), Proposition 40.4 (fuse needs no guard), Corollary 40.5 (fuse does not undo split, stated here), Definition 40.6 (reorder).
41. [**Memory and reduction primitives**](part7/ch41-memory-and-reduction-primitives/README.md) — cache_read/cache_write, rfactor, decompose_reduction, storage_align, compute_inline, compute_at.
    *Formal:* Definition 41.1 (reduction block), Theorem 41.2 (rfactor is sound iff the operator is associative and commutative), Counterexample 41.3 (f32 addition is not — measured: serial 3, rfactor(4) 6, on the same eight values), Proposition 41.4 (interposing a buffer, stated here), Definition 41.5 (inlinable producer, stated here), Proposition 41.6 (inlining is sound for one).
42. [**Legality**](part7/ch42-legality/README.md) — which primitive is allowed when, decided by dependence, [schedule/legality.ts](../src/compiler/schedule/legality.ts); and the three layers that answer the same question differently.
    *Formal:* Definitions 42.1–42.3 (dependence, direction vector, carried level), Theorem 42.4 (a permutation is legal iff no direction vector is reversed — *classical*), Corollary 42.5 (parallelisation), Proposition 42.6 (conservative masks, stated here), Definition 42.7 (kind policy), Proposition 42.8 (the declaration overrules the analysis, stated here), Counterexample 42.9 (a false `DataPar` buying an illegal permutation, executed).
43. [**Scheduling for GPUs**](part7/ch43-scheduling-for-gpus/README.md) — thread and block binding, shared memory, tensorization, and detecting cross-thread races, [analysis/gpu_race.ts](../src/compiler/analysis/gpu_race.ts); and the finding that the fastest GPU kernel is not a schedule.
    *Formal:* Definition 43.1 (thread binding), Definition 43.2 (launch geometry), Definition 43.3 (binding signature), Theorem 43.4 (cross-block RAW is unrepairable, stated here), Proposition 43.5 (cross-thread sharing is repairable, stated here), Definition 43.6 (tensorisation).

## Part VIII — Autotuning (5 chapters) — **written**, see [part8/](part8/README.md)

44. [**How big is the search space**](part8/ch44-how-big-is-the-search-space/README.md) — counting it in closed form for one matmul; the cap that turns the count into a biased subset; and why a heuristic that reads only the target cannot be optimal for more than one shape.
    *Formal:* Definition 44.1 (schedule space, stated here), Theorem 44.2 (ordered factorisations — *classical*, stars and bars per prime exponent), Corollary 44.3 (size of a multi-level tiling space, stated here), Proposition 44.4 (the offered space is a biased subset, stated here), Definition 44.5 (regret, stated here), Proposition 44.6 (a heuristic is constant on its blind spot, stated here), Corollary 44.7 (the shipped CPU matmul tile is shape-independent, stated here). Measured: 84 factorisations per 64-long axis at four levels, of which 48 are offered; 2,304 points for one matmul block and 92,190 for the two blocks of the function; eight points of that space 1.08x apart on a 256x256 matmul, and one cost-model score between them.
45. [**Sketches**](part8/ch45-sketches/README.md) — a skeleton of primitive calls with typed holes, derived per block by a three-rule decision list, [autotune/sketch_generators.ts](../src/compiler/autotune/sketch_generators.ts), [tiling.ts](../src/compiler/autotune/tiling.ts), [derivation.ts](../src/compiler/autotune/derivation.ts); and the two skeletons that generate a space nothing can reach.
    *Formal:* Definition 45.1 (sketch, stated here), Definition 45.2 (derivation as a decision list, stated here), Proposition 45.3 (derivation is total on a supported target, stated here), Definition 45.4 (tile structure, stated here), Proposition 45.5 (a multi-level split realises a factorisation exactly and emits no guard, stated here), Corollary 45.6 (the bindings are a mixed-radix reconstruction — Theorem 35.3 reused), Theorem 45.7 (a sketch space is sound if its primitives are, stated here), Counterexample 45.8 (the two ways that hypothesis fails: `rfactor`, and a sketch that calls no primitive). Measured: 1,225 points of `mlt_cpu` all distinct and validator-clean; 6,125 points of `ssrsrs_cpu` all refused; six GPU block sizes collapsing to four programs; and `rfactor` accepting a spatial axis, which the validator then rejects.
46. [**Cost models**](part8/ch46-cost-models/README.md) — two disjoint feature sets, seven bounded analytic terms, gradient-boosted trees, and why the model is graded on error and used as a comparator, [cost_model.ts](../src/compiler/autotune/cost_model.ts), [features.ts](../src/compiler/autotune/features.ts), [gbt.ts](../src/compiler/autotune/gbt.ts).
    *Formal:* Definition 46.1 (feature map, and what "discriminating on a space" means, stated here), Definition 46.2 (cost model as a comparator, stated here), Theorem 46.3 (only the induced order matters, stated here — executed under four monotone transforms), Corollary 46.4 (absolute error is not identifiable from search behaviour, stated here), Counterexample 46.5 (a model with 3,201x the squared error and lower regret), Proposition 46.6 (the analytic model is constant on the multi-level tiling space, stated here), Proposition 46.7 (the statement aggregation is max-dominated, stated here). Measured: 2,304 tiling points, one distinct score; 784 GPU points, 31; 11 of 23 whole-function features never read.
47. [**Search and measurement**](part8/ch47-search-and-measurement/README.md) — an elitist genetic algorithm over a seeded LCG, a median over repeats, a deadline, a per-workload task scheduler, and the tuning database with its codegen versioning, [search.ts](../src/compiler/autotune/search.ts), [benchmark.ts](../src/compiler/autotune/benchmark.ts), [session.ts](../src/compiler/autotune/session.ts), [task_scheduler.ts](../src/compiler/autotune/task_scheduler.ts), [workload_key.ts](../src/compiler/autotune/workload_key.ts), [tuning_db.ts](../src/compiler/autotune/tuning_db.ts), [autotuner.ts](../src/compiler/autotune/autotuner.ts).
    *Formal:* Definition 47.1 (elitist evolutionary search, stated here), Proposition 47.2 (elitism makes the best monotone — and the memo is what makes the hypothesis hold, stated here), Proposition 47.3 (the initial population's sketch index reaches a fixed point, stated here), Corollary 47.4 (at most two sketches, then one), Definition 47.5 (workload key, stated here), Proposition 47.6 (what the key does and does not determine, stated here), Counterexample 47.7 (three iteration domains, one key, executed), Theorem 47.8 (selection bias in a noisy minimum — *classical*), Proposition 47.9 (budget overshoot, stated here). Measured: at the shipped default seed a CPU matmul receives no tuning result; an evolutionary search runs to twice its stated budget; and two elementwise blocks of 10,039 and 11,827 elements share a workload key.
48. [**Reproducibility**](part8/ch48-reproducibility/README.md) — schedule traces as serialisable, replayable objects, [schedule/trace.ts](../src/compiler/schedule/trace.ts), [autotune/tune_ir.ts](../src/compiler/autotune/tune_ir.ts); and the difference between replaying a derivation and re-deriving from parameters.
    *Formal:* Definition 48.1 (schedule trace, stated here), Definition 48.2 (replayable versus faithful, stated here), Theorem 48.3 (conditions for faithful replay, stated here — the fresh-variable counter is condition (ii)), Proposition 48.4 (faithful replay requires complete recording, stated here), Counterexample 48.5 (`tensorize` and the register-block sketch record nothing), Corollary 48.6 (a trace is a recipe, not a certificate, stated here), Definition 48.7 (provenance of a tuning result, stated here). Measured: 22 primitives, 18 recording a step of their own; every stored trace replaced with garbage leaves a cached compilation's output unchanged; three of six `annotate` values survive a JSON round trip.

## Part IX — Memory (4 chapters) — **written**, see [part9/](part9/README.md)

49. [**Buffer lifetimes**](part9/ch49-buffer-lifetimes/README.md) — live intervals over a linearized program, [buffer_liveness.ts](../src/compiler/passes/memory/buffer_liveness.ts).
    *Formal:* Definition 49.1 (live interval), Definition 49.2 (interference), Theorem 49.3 (disjoint intervals may share storage), Lemma 49.4 (region extension — why a loop widens an interval), Definition 49.5 (peak).
50. [**Arena allocation**](part9/ch50-arena-allocation/README.md) — best-fit and first-fit packing, alignment, [buffer_assignment.ts](../src/compiler/passes/memory/buffer_assignment.ts).
    *Formal:* Definitions 50.1–50.2 (assignment, validity), Theorem 50.3 (width is a lower bound), Theorem 50.4 (dynamic storage allocation is NP-complete, *(Garey and Johnson, 1979)*), Proposition 50.5 (greedy placement is valid), Definition 50.6 (first-fit and best-fit).
51. [**In-place reuse and donation**](part9/ch51-inplace-and-donation/README.md) — when overwriting an input is provably safe, [inplace_analysis.ts](../src/compiler/passes/memory/inplace_analysis.ts).
    *Formal:* Definitions 51.1–51.2 (candidate, aliasing), Theorem 51.3 (index equality suffices), Counterexample 51.4 (`D[i] = S[i] * S[0]`). **And the finding that the shipped default plans a 1.9x lower peak while allocating 2.5x more, because in-place candidates are recorded but never materialized and are excluded from the aliasing that would have been.**
52. [**Scheduling to lower peak memory**](part9/ch52-scheduling-for-peak/README.md) — reordering independent work to shrink the high-water mark, [memory_scheduler.ts](../src/compiler/passes/memory/memory_scheduler.ts).
    *Formal:* Definitions 52.1–52.2 (schedule, peak of a schedule), Theorem 52.3 (register sufficiency is NP-complete, *(Sethi, 1975)*), Definitions 52.4–52.5 (list scheduling, depth-first by subgraph weight), Proposition 52.6 (best-of-k is never worse). Measured 2.99x.

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
61. **Tracing** — turning eager user code into a GraphModule, [tracing/compile.ts](../src/tracing/compile.ts). Also the two things `compile()` does around the compiler rather than inside it: folding parameters into constants (`foldWeights`), and the optimization gate that compiles candidate configurations, measures them and keeps the winner, [pipeline/opt_gate.ts](../src/compiler/pipeline/opt_gate.ts). Chapters 19, 20 and 25 defer to both.
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

  *Found while writing Part IV, each reproducible by a lab in that part, still open:*
  1. `scatter` declares `sideEffects: WRITE` (`ops/shape.ts:368`) although its lowering writes only its own output buffer; a dead `scatter` and its whole index-computation subgraph survive DCE, CSE and folding. (Chapter 19)
  2. `layoutAwareOps` is empty for every shipped target (`target.ts:129`), so `LayoutTransformPass` proposes conversions and discards all of them, reporting UNCHANGED — indistinguishable from having nothing to do. (Chapter 25)
  *Found while writing Part V, each reproducible by a lab in that part, still open:*
  3. `CheckpointPolicy` and its four segmenters (`checkpoint_policy.ts`) are complete and tested, and no path through `compileWithBackward` can select one — `compile_backward.ts:95` builds a `RematPolicy` and nothing looks for a checkpoint policy. √n checkpointing is implemented and unreachable. (Chapter 30)
  4. `scanCheckpoint` (`backward_builder.ts:43`) is the same story one level down: the option that bounds the memory of an unrolled `scan` backward has no caller. (Chapters 30, 31)
  5. Differentiating a `scan` unrolls it at compile time (`scan_backward.ts:285`), so the backward graph is Θ(T) where the forward graph is Θ(1) — measured 76 operations per timestep. `T` must also be a static dimension, so a `scan` over a symbolic sequence length cannot be differentiated. (Chapter 31)
  6. `findUnsupportedGradOps` (`vjp_registry.ts:39`) exists to report un-differentiable operations before the sweep starts and has no callers; it would also mis-report `scan` and `if`, which are handled by the separate region-VJP registry. (Chapter 31)
  7. `maxRematDepth` (`remat_policy.ts:5`) is declared, assigned in the constructor and read by nothing, so rematerialization has no depth bound. (Chapter 30)

  *Found while writing Part VI, each reproducible by a lab in that part, still open. Unlike Part V's, none of these is unreachable code: every one runs on every compilation, and is either less precise than its own proof allows or a declaration nothing checks:*
  8. `(f tdiv c) * c + (f tmod c)` is emitted where the compiler already holds the proof that it equals `f`. Truncating division is substituted for floor division exactly when the dividend is proved non-negative (`ir_arith.ts:165`), and for a non-negative dividend that expression is the identity — but the rewrite is not in `RewriteSimplify` (`ir_arith.ts:210`). Every reshape that fails the exact-split test pays a division, a modulo, a multiply and an add per element; measured 1.10× against an equivalent reshape that does satisfy it, on the same 98,304 elements. (Chapter 35)
  9. A block's declared read set omits every buffer it accumulates into, because `bufRefs` (`lowering_registry.ts:421`) builds the set from the operation's operands and the accumulator is a result. Nothing verifies the declaration. Everything that must be right walks the body instead; what consumes the declaration alone is the autotuner's workload key and block DAG. (Chapters 33, 45, 47)
  10. A block's declared read/write *regions* do not exist. `BufferRegion` (`buffer.ts:69`) is exported and constructed nowhere in `src/` or `tests/`, and the optional `min`/`extent` on `BufferRegionLike` are never set, so every declared region prints and behaves as "this buffer, unspecified" and is recomputed from the body on every query. (Chapter 33)
  11. Iteration-variable kinds are trusted in preference to a computed dependence and are checked by nothing. `loopCarriedDependence` (`legality.ts:40`) finds a dependence and returns "legal anyway" when the block's kinds permit; the TIR verifier does not look at kinds at all. The protection is that the five `markCommReduce` call sites are the five rules that build accumulations. (Chapters 33, 36)
  12. `BlockNode.initBody` is implemented on every side — constructor, verifier, printer, cloner — and set by no lowering rule; reductions are emitted as two sibling blocks instead. (Chapter 33)
  13. `TensorIRPrinter` implements 17 visitors for 21 node kinds and falls back to `[UnknownNode: …]` (`printer.ts:36`), so every lowered `scan` prints with a hole where its barrier is; and `visitMathOpNode` (`printer.ts:181`) emits the operator only for binary nodes, so `neg` and `logical_not` print as a bare parenthesis. Both are reporting bugs only — the backends read the node — and TIR has no parser, so there is no round-trip property to violate. (Chapter 32)
  14. A rule that cannot fire: `registerLoweringRule('broadcast', …)` (`rules/shape.ts:15`) registers a strategy for a name the op registry does not contain, as does the inline fusion builder for the same name (`rules/fusion.ts:56`). (Chapter 34)
  15. A missing lowering rule throws on the first offender (`graph_to_tensor.ts:155`), so a module with three unlowerable operations reports one. `hasLoweringRule` (`lowering_registry.ts:68`) exists and is not used to check a module up front — the same shape as finding 6. (Chapter 32)
  16. `_modBound` (`analyzer.ts:115`) requires a constant positive divisor, so `x % n` with a symbolic `n` gets no bound although `0 ≤ x % n < n` holds for every positive `n` under the floor semantics the compiler guarantees. `IntBound` holds `number`s, so the symbolic upper bound is not expressible; the cost is that dynamic shapes lose guard elision on every subscript containing a modulo. (Chapter 37)
  17. A `fusion` region containing one operation without an inline builder is lowered operation by operation, materialising every intermediate (`rules/fusion.ts:219`). The fusion pass that formed the group does not consult the inline-builder list, and the degradation emits no trace event, so a silently un-fused group is indistinguishable from a fused one in the trace. (Chapters 24, 34)

  *Found while writing Part VII, each reproducible by a lab in that part, still open. They fall into two groups — primitives nothing calls, and a legality question with three answers:*

  **Unreachable, and therefore never tested against a real backend.**
  18. Nine of the twenty-two schedule primitives have no caller in `src/`: `cacheRead`, `cacheWrite`, `setScope`, `storageAlign`, `computeInline`, `computeAt`, `reverseComputeAt`, `annotate` and `blockize`. They are implemented and unit-tested; no compilation runs them. (Chapters 38, 41, 43)
  19. `cacheWrite` (`schedule.ts:752`) redirects a block's accumulator to a fresh cache buffer and does not initialise it, because the zeroing lives in a sibling init block that still targets the original. Correct on CPU only because a fresh `Float32Array` is zero-filled; on CUDA the emitted `float ..._cachew[20];` is uninitialised and the kernel returns garbage. (Chapter 41)
  20. `cacheRead` and `cacheWrite` bind each staging loop variable to itself, `new BlockRealizeNode(v, v)` (`schedule.ts:767`, `schedule.ts:1016`), which the CUDA backend emits as `const int x = x;`. The TIR verifier's scoping check passes it because the name is in scope — it binds itself. (Chapter 41)
  21. `buildBlockScopes` — the four-pass construction in `block_scope.ts` that builds a producer-consumer graph over sibling blocks, plus `regionCover` and `stagePipeline` — is reached only through `ScheduleState.scopes`, whose only callers are `_checkRelocationDependences` (from `computeAt`/`reverseComputeAt`) and `ScheduleState.blockInfo`, which has none. It does not run during compilation, and `producersOf`, `consumersOf`, `writersOf`, `stagePipeline` and `regionCover` have no reader anywhere in `src/`. The exception is `linkAccessUnits` (`block_scope.ts:158`), thirteen of the file's 257 lines, which `MemorySchedulePass` (`memory_scheduler.ts:6`) imports directly and runs on every compilation — the one reusable piece was factored out and the machinery around it was not. (Chapters 39, 52)
  22. `SRefTree.root` is `null` for every function whose body is a `SeqNode` — that is, every multi-statement function — because only `ForNode` and `BlockNode` are registered (`sref.ts:150`). `SRef.isRoot` and the `wasRoot` branches of `replaceNode`/`removeNode` are dead for the common case; nothing notices, because every query goes through the name and node maps. `ScheduleState.dependences` (`schedule_state.ts:182`) is dead for the same reason: every legality question uses `nestAnalysis` instead. (Chapter 39)
  23. `decomposeReduction` (`schedule.ts:718`) opens by requiring `initBody`, which no lowering rule sets (finding 12), so it throws on every block the compiler produces and succeeds only on a block `rfactor` built. `createSSRSRSTilingSketch` (`autotune/tiling.ts:131`) opens with a call to it, so the SSRSRS tiling structure cannot be applied to any lowered block; the sketch throws and the search proceeds without it. (Chapters 41, 45)
  24. `rfactor` defaults its identity element to integer `0` (`schedule.ts:656`) for all four operators in `RFACTOR_ASSOCIATIVE_OPS` — right for `+`, wrong for `*`, `min` and `max`. Latent rather than live: the reduce rule emits its identity as a separate init block, so the only blocks matching the accumulating-store pattern are `+` accumulations. It also reassociates with no `fastMath` gate and no dtype check, unlike Chapter 20's algebraic patterns. (Chapter 41)
  25. `tensorize` (`schedule.ts:1093`) is the one mutating primitive that records no trace step, so a tuned schedule that tensorises cannot be replayed faithfully. (Chapters 43, 48)

  **Legality answered three times, with only the third a proof.**
  26. `IterVarPolicy.SPATIAL` and `ACCUMULABLE` differ by `CommReduce` (`legality.ts:17`), so `parallelize` refuses a matmul's reduction axis and `vectorize` accepts it — the same dependence, found in both cases, overruled in one by the block's declaration. `ScheduleValidator` then calls the result a race. Its one production caller is the autotuner's session (`autotune/session.ts:186`); `Schedule.verify()` has none, so a rule-produced schedule is never validated and a searched one always is. The WASM backend re-derives the dependence a third time (`backend/wasm/codegen.ts:1606`) and handles it correctly. (Chapter 42)
  27. Counterexample 42.9, executed: a hand-built nest with a genuine `(<, >)` dependence has `reorder(j, i)` refused when its axes are untyped and *accepted* when they are declared `DataPar` — the way every lowering rule emits them — and the two programs compute different values. Corollary 33.7 is live; the protection is the discipline of the five `markCommReduce` call sites, not the design. (Chapters 33, 42)
  28. The rule policy routinely produces the two-extent parallel partition `ScheduleValidator._checkPartitionConsistency` (`validator.ts:47`) rejects: a `sum` over 64x64 gets `@parallel` at extent 8 on the init nest and extent 64 on the accumulation. Harmless only because the WASM backend independently demotes any parallel loop whose extent differs from the first it saw. (Chapter 42)
  29. A CUDA compilation with the shipped defaults schedules only what `applyDeterministicGpuSchedule` recognises. `CUDATarget` declares `{ gpuTiling: true }` (`target.ts:225`) and not `enabled`, and `SchedulePass` reads `if (!handled && sCfg.enabled)` (`schedule_pass.ts:61`), so every kernel that is not a matmul or a convolution is emitted as a serial loop inside a `__global__` function — one thread for the whole tensor. `WebGPUTarget` declares `{ enabled: true }` (`target.ts:261`) and does not have the problem; the difference between the two GPU targets is one key in one attribute table. (Chapters 38, 43)
  30. When the CUDA backend detects an unrepairable cross-block race it sets the launch to 1x1x1 and emits the thread-bound loops as ordinary `for` loops (`cuda/codegen.ts:562`), recording the reason on `_launchDiagnosis` — which no trace event reads. A silently serialised kernel is indistinguishable from a parallel one in the trace. Same shape as finding 17. (Chapter 43)
  31. `bindFusedSpatialGPU` caps the thread block at `Math.min(target.maxThreadsPerBlock, 256)` (`rules.ts:180`), so a device advertising 1024 is given 256; and `maxBlockDimY`, `maxGridDimY` and `maxGridDimZ` are declared on every GPU target and read by nothing in `src/compiler/`, so a binding that exceeds them compiles and fails at launch. (Chapter 43)
  32. `MatmulTiledCPURule` computes `tileDim = max(8, min(64, floor(sqrt(L1/4))))`, which is 64 for any L1 of 16 KiB or more. On a 64x64 matmul both tiled axes have extent 64, so both outer loops have extent 1 — and the rule finishes by parallelising `outerLoops[0]`, marking a one-iteration loop parallel while the 64-iteration inner loop stays serial. Two extra loop levels, no extra parallelism. (Chapter 40)
  33. `tile` is the one primitive that can leave the IR modified after failing: it splits every named axis and then calls `reorder`, which may throw, at which point the splits have already been applied and `applyToBlock` reports "rule rejected" for a nest that was in fact changed (`schedule.ts:569`). (Chapter 40)
  34. `_resolveLoop` returns its argument unchanged when the name is not found (`schedule.ts:254`), so a misspelled loop name reaches the primitive as a string and the error names the type rather than the name — or, for `reorder` with fewer than two arguments, is silently a no-op (`schedule.ts:323`). (Chapters 40, 42)
  35. Chapter 35's finding 8 is reachable from the scheduler as well as from `reshape`. `fuseLoops(split(i, c))` leaves `(f // c) * c + (f % c)`, and so does `ElementwiseGPURule` on any multi-dimensional elementwise nest: a 12x5 `mul` compiled for CUDA through the public `compile()` emits `buf_3[((v0_7 * 5) + v1_8)]` with `v0_7 = f / 5` and `v1_8 = f % 5` — the identity, computed with four operations per element, after `SimplifyPass` and every pass below it. (Chapters 35, 40)

  *Found while writing Part VIII, each reproducible by a lab in that part, still open. Every component of the autotuner is individually tested against a fixture that has the property it needs; these are the seams between them, where one component's assumption about its neighbour does not hold. None of them produces a wrong kernel, because a searched schedule that fails is replaced by a rule-produced one:*

  **A component written against a block shape the lowering rules do not emit.** Finding 23 is the first of these; two more follow.
  36. `findFusibleConsumer` (`block_dag.ts:119`, and again at `:124` and `:130`) compares a producer's store-subscript variable names against its enclosing loop variable names. Those are two disjoint namespaces for every block a lowering rule emits — the loop is `ls0_6`, the iteration variable is `vls0_9` — so the comparison always fails, the `fused` sketch is never derived, `createFusedTilingSketch` has no reachable caller, and `BlockTuningSession`'s `needsWholeFunc` path (`session.ts:96`) is dead with it. It fires correctly on a hand-built pair whose blocks bind each iteration variable to itself. (Chapter 45)
  37. The autotuner derives no sketches at all for a WASM target: `deriveSketches` returns the empty list unless the target is `TargetKind.CPU` or `isGPU()` (`derivation.ts:78`), and `WasmTarget` is neither. Every block becomes an `empty` task, `tune` returns an empty map, and the rule policy runs — the correct output, silently obtained without tuning. WASM is the one shipped backend that acts on both `@parallel` and `@vectorized`. (Chapter 45)
  38. `gpuThreadCap` (`sketch_generators.ts:10`) clamps every candidate thread-block size to `min(maxThreadsPerBlock, 256)`, so two of the six `BLOCK_SIZE_CANDIDATES` are unreachable on every device and three name the same kernel on a 256-thread one. `EvolutionarySearch` memoises on `sketch.name + JSON.stringify(params)` (`search.ts:117`), which is parameter identity rather than program identity, so aliased points are scored and — with a benchmark runner attached — measured once each. Same constant and same consequence as finding 31, in a different file. (Chapter 45)

  **An objective that is constant over the space it ranks.**
  39. `AnalyticalCostModel.score` is constant on the entire `mlt_cpu` schedule space (Proposition 46.6; measured: 2,304 points, one distinct value). `_scoreParallelism` on CPU is `numParallelLoops / numLoops` (`cost_model.ts:113`), a count ratio blind to extents, so parallelising a loop of extent 1 scores exactly as much as one of extent 64; and `innermostExtent` (`features.ts:267`) is assigned on every `ForNode` visit and therefore holds the extent of the last loop the walk enters, which under `CPU_TILING` is the reduction axis — the one axis that structure never splits. The only term a schedule can move on a CPU target is `vectorization`, which is gated on an annotation the CPU backend ignores. (Chapter 46)
  40. Eleven of the 23 whole-function features are extracted on every scoring call and never read: `numBlocks`, `totalIterations`, `maxLoopDepth`, `numUnrolledLoops`, `numThreadBound`, `totalBufferBytes`, `numBufferReads`, `numBufferWrites`, `outermostExtent`, `hasReduction`, `reductionDepth`. `totalIterations` is the model's only possible term for how much work the nest does; `outermostExtent` is the extent of the loop `applyRoles` parallelises; `numUnrolledLoops` makes `unroll` unscoreable. (Chapter 46)
  41. `arithmeticIntensity` (`features.ts:103`) is a *syntactic* operation count divided by the sum of declared buffer sizes, with no trip count in either. Its value therefore falls as the tensors grow while Chapter 4's quantity rises: a 64x64 matmul is credited with 4,096 times the arithmetic intensity of a 4096x4096 one. (Chapters 4, 46)
  42. The learned model is trained on mismatched pairs: `_measure` benchmarks the whole scheduled function and labels the *mini* function's features with the result (`session.ts:229`, `session.ts:231`), so in a k-block program every sample carries the time of all k. One `LearnedCostModel` (`autotuner.ts:158`) is shared across every block of every function in a compilation. The within-block ranking survives — the other blocks contribute the same constant to every candidate — but the fit does not. (Chapter 46)

  **A generator whose low bits collapse a choice, and three budgets that do not bind.**
  43. `EvolutionarySearch._initPopulation` draws its whole population from one sketch. `_rng(m)` is `state % m` over an LCG with modulus 2^31 (`search.ts:21`) whose multiplier is 1 and increment 3 modulo 4, so a four-element sketch list advances the index by minus one plus the drawn sketch's variable count, modulo 4, and reaches a fixed point within two draws; the fixed points are the sketches with three search variables. At the shipped default seed 42 (`autotuner.ts:120`) that sketch is `ssrsrs_cpu`, which throws on every lowered block (finding 23), so the population scores nothing, `candidates` comes back empty, and **a CPU matmul receives no tuning result at all**. Seeds 1 and 7 and `strategy: random` do not have the problem; `RandomSearch` iterates the sketch list rather than sampling it (`search.ts:62`). (Chapters 45, 47)
  44. `GradientSchedulerPolicy` degenerates to list order. `runRound` measures its first improvement against a starting best of minus infinity (`session.ts:123`), so every task's first productive round returns `Infinity`, and `gainEwma = 0.5 * Infinity + 0.5 * gainEwma` never decays; `weight * Infinity` is equal for every weight, so the block-grouping weight computed at `autotuner.ts:200` never affects the allocation. A round in which candidates exist but every measurement fails returns `Math.max(0, NaN)`, which is `NaN` — neither at most 0, so the task is not marked stale, nor greater than `bestPriority`, so it is never picked again — and the scheduler then exits as if everything had plateaued. (Chapter 47)
  45. Benchmarking is off by default: `enableBenchmark` is `hardwareMeasure || !!measurer` (`autotuner.ts:127`) and a CPU compile sets neither, so `BenchmarkRunner` — which works on a CPU target — is not constructed, and `runRound` takes `candidates[0]` and plateaus after one round. `benchmarkMaxCv` defaults to 0 (`autotuner.ts:130`), which disables the re-measurement loop it gates (`benchmark.ts:187`); when it is enabled, `_collect` appends to the existing sample array, so a second round's statistics are computed over both rounds. (Chapter 47)
  46. `EvolutionarySearch` overshoots its deadline by up to two populations: it tests the deadline once per generation (`search.ts:125`) and then evaluates a whole population, and the final scoring pass (`search.ts:157`) tests it not at all. With the default `populationSize: 32` that is up to 64 evaluations past the stated budget. `_crossover` also has an unreachable branch (`search.ts:179`), since its only call site is the else arm of a test that the two parents' sketches differ. (Chapter 47)

  **A cache key and a version that do not carry what they promise.**
  47. The workload key does not include the iteration domain. `collectBlockOps` walks through a `ForNode` to its body (`workload_key.ts:133`) and emits nothing for the loop, and the enclosing loops are outside the block, so three nests over the same buffers with the same body and extents 64, 32 and 3 share one key while admitting different vector widths. The key also identifies a target by `name` and `kind` alone (`workload_key.ts:57`), so two `CPUTarget`s differing in `vectorWidth`, `numCores` or `l1CacheBytes` — all of which the cost model or the rule policy reads — share every cache entry. (Chapter 47)
  48. `TuningDatabase.deserialize` accepts any serialised database with no `codegenVersion` field, because the guard is `data.codegenVersion !== undefined && data.codegenVersion !== CODEGEN_VERSION` (`tuning_db.ts:129`) — and a file with no such field is exactly a file written before the field existed, which is the one file the mechanism exists to reject. `TuningRecord.version` is written, serialised and restored (`tuning_db.ts:133`) and compared to nothing. (Chapter 47)

  **A persistence format that is complete, fragile, and unread.**
  49. `TuningRecord.traceData` is computed by `bestTrace()` (`session.ts:151`), stored, ranked alongside, serialised to JSON and written to disk — and read by no code in `src/`; `ScheduleTrace.replay` has no caller there either. Replacing every stored trace with a step naming a nonexistent primitive leaves a cached compilation's output unchanged, because a cache hit is served from `sketchName` and `params` (`autotuner.ts:230`) and re-derives the sketch. (Chapter 48)
  50. Faithful replay of a trace requires the fresh-variable counter to be at its record-time value, and `resetVarCounter` (`schedule.ts:198`) has no caller in `src/`, so the counter never returns to a previous value within a process. A trace whose later steps name loops an earlier step created — which is every trace a tiling sketch produces — fails on its second step with `reorder expects ForNode arguments`, because `_resolveLoop` returns its string argument unchanged (finding 34). It is also why two cached compilations of the same graph agree only up to renumbering of the trailing digits, which `autotuner.test.js` normalises away before comparing. (Chapters 40, 48)
  51. `tensorize` (`schedule.ts:1093`) records no trace step — finding 25, now with its consequence measured — and `createMatmulRegisterBlockGPUSketch` records none either, because it assigns `schedule.func.body` directly (`gpu_matmul_sketch.ts:393`) instead of calling primitives. Its `TuningRecord.traceData` is the empty list, which replays to the unscheduled function and is indistinguishable from a schedule that did nothing. The compiler's fastest GPU matmul is the kernel the trace mechanism describes least. (Chapters 43, 45, 48)

  **Two places where the search's coverage is silently bounded.**
  52. `enumerateFactorizations` truncates before it subsamples: the recursion halts at `maxCandidates * 8` tuples (`factorization.ts:48`) and walks divisors in ascending order, so the tuples never generated are exactly those with a large outermost factor. For a 4096-extent axis at four levels the search is offered leading factors 1 through 64 out of 1 through 4096 — the coarse-grained tilings are not rejected, they are never built. Below 384 tuples nothing is lost. (Chapter 44)
  53. Nothing computes or reports the size of a search space. The product of the search variables' candidate counts appears nowhere in `src/`, and `trace.autotuneStats` emits `blockCount` and `cacheHits` (`schedule_pass.ts:55`) and no coverage figure, so a user cannot tell from any diagnostic whether a budget covered 14% of a space or 0.0002% of one. Same shape as findings 17 and 30. (Chapter 44)

  **A precondition a primitive relies on its callers to hold.**
  54. `rfactor` (`schedule.ts:629`) never checks that the axis it is asked to factor is a reduction axis. It tests the loop, the extent, the factor and the block body's operator, and the body of a matmul block is an accumulating store whichever loop you name — so `rfactor('matmul_1', 'ls0_6', 2)`, reassociating over the row axis, is accepted. Definition 41.1's requirement that the accumulator's subscript not involve the factored axis is the missing hypothesis: the combine nest is built over the surviving axes only, and the emitted kernel references the partial nest's loop variables out of scope. Two things keep it latent: `createRfactorSketch` names only axes from `blockInfo.reductionLoopVars` (`sketch_generators.ts:34`), and `ScheduleValidator` rejects the result on the searched path (`session.ts:186`) — the one path that runs it. (Chapters 41, 45)

  **Two formats narrower than the thing they carry.**
  55. The workload key is a 32-bit FNV-1a hash (`workload_key.ts:152`) and nothing re-checks a hit. A birthday search over the descriptions the compiler actually builds finds a collision inside the range of ordinary tensor sizes: the same 1-D elementwise block over buffers of 10,039 and of 11,827 elements both key to `9a89ea08`. `TuningRecord` stores the key and never the description it hashed (`tuning_db.ts:29`), so `lookup` has nothing to compare against and returns whichever workload was tuned first. Storing the canonical description beside the key would turn a wrong answer into a miss. (Chapter 47)
  56. A trace step's arguments are typed `readonly unknown[]` (`trace.ts:1`), and `annotate` records its `value: unknown` verbatim (`schedule.ts:1089`). A BigInt makes `JSON.stringify` throw, which inside `saveToFile` (`tuning_db.ts:146`) loses the entire database rather than the step; a function or `undefined` serialises silently to `null`, so replay sets an annotation the recorded schedule never had. Definition 48.1 describes the arguments as JSON values and nothing enforces it. Unreachable today only because `annotate` is one of the nine primitives with no caller in `src/` (finding 18). (Chapter 48)

  *Closed:* `//` and `%` disagreeing between the symbolic layer, constant folding and the four backends — now floor everywhere, from one definition, with `tdiv`/`tmod` carrying truncation where it is provably equivalent (Chapters 35, 36, 54–58); reading a value from a symbolic tensor failing with an uninformative `Cannot read properties of null` — `SymbolicTensor` now overrides the value-reading accessors and names data-dependent control flow (Chapter 5); `AddZero` (`patterns.ts:167`) applying `x + 0 → x` with no fast-math gate, unsound for `−0` — the pattern now takes the same `fastMath` argument and dtype-or-licence check its three neighbours already had, and defaults to integers-only where canonicalize constructs it (Chapter 20); the CPU backend's expression peephole (`backend/cpu/codegen.ts:470`) folding `x + 0` and `x * 0` on rendered strings, so `∞ × 0` and `NaN × 0` became `0` on CPU and stayed NaN on CUDA — the zero identities are now gated on `inferDtype`, which keeps them for the integer index arithmetic they were serving and declines on floats, and `_zeroLit` supplies a zero of the right width (Chapters 20, 65); `target.sharedMemoryBytes || …` (`fusion_cost.ts:62`, `priority_fusion.ts:60`) reading a CPU's declared `0` as "unspecified" and applying a 48 KiB GPU budget — a stated `0` now means the device exposes no such budget and the constraint does not apply, which took the default fusion strategy from 1.47 ms to 0.62 ms on the three-operation diamond, and `registersPerThread` is resolved the same way (Chapter 24); `RematerializationPass` exiting identically whether it met its `memoryBudget` or ran out of candidates — it now compares its final peak against the budget and emits a `trace.warn` naming both, while still returning CHANGED because a missed soft budget is not a compilation failure (Chapter 26); quantization's `calibrationData` being unreachable through the public `compile()` path, where a traced model's captured parameters are graph arguments the user does not have — the wrapper now assembles the graph-level argument list the way it does for an ordinary call and converts tensors in a batch (Chapter 26); calibration then observing a graph the quantizer never sees — it ran as a compile phase ahead of every graph pass and keys its ranges on `Value` identity, so a `transpose` folded into a `dot`'s contracting dimensions took its observation with it and the operand that replaced it fell back to `[-6, 6]`, leaving `calibrationData` alone worth 18.26% to 17.46% — it is now a `CalibrationPass` emitted immediately before the quantization pass that consumes it, which takes the same model to 0.60% on `calibrationData` alone rather than only in combination with `foldWeights`, and removes `calibrate` from the compile phase list (Chapters 15, 18, 26); `joint` mode recomputing the forward pass on every `backward()` call (`compile_backward.ts:455`) — `cf(x)` now defers the single kernel run and either an output read or `backward()` forces it with the real cotangent, making joint 0.044 ms against `separate`'s 0.048 rather than 15% slower (Chapter 29).

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
