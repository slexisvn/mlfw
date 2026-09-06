# Part IV — Graph-level optimization

Part III built the machinery for running transformations. This part is the transformations.

Everything here works on the graph IR of Part II: whole tensors, no loops, no indices, no buffers. That level of abstraction is a constraint and an advantage. A pass cannot reason about cache lines or register pressure because it cannot see them — but it can see the entire dataflow of the program at once, which is where the largest optimization in this book lives.

| Chapter | Title | The question it answers |
|---|---|---|
| [19](ch19-fold-cse-dce/README.md) | Constant folding, CSE, and dead code elimination | Which work in this graph is provably unnecessary — and why is "nobody reads this" harder to establish than it sounds? |
| [20](ch20-algebra-and-ieee754/README.md) | Algebraic simplification meets IEEE 754 | Which schoolbook identities are true of the numbers a machine actually has, and what does a fast-math flag really license? |
| [21](ch21-decomposition/README.md) | Decomposition | Why would a compiler deliberately make the graph five times bigger? |
| [22](ch22-fusion-why/README.md) | Fusion I: why it is the single most valuable optimization | Where does the ~1.9× come from, and why does it not depend on tensor size? |
| [23](ch23-fusion-legality/README.md) | Fusion II: legality | When is merging two adjacent operations not merely unprofitable but impossible? |
| [24](ch24-fusion-strategies/README.md) | Fusion III: the three strategies | Given many legal, profitable merges, in what order should you make them — and what does the wrong order cost? |
| [25](ch25-layout/README.md) | Layout | The numbers are the same; only the addresses change. When is that worth a pass over memory? |
| [26](ch26-optional-pipelines/README.md) | Three optional pipelines | Rematerialization, quantization, partitioning: what do you have to give up, and who decides? |

## The argument in one paragraph

A traced graph arrives full of work nobody asked for — duplicated subexpressions, constants waiting to be computed, branches nobody reads — and three cheap passes remove it, of which only dead-code elimination is subtle, because "nobody observes this" is a claim about state outside the dataflow graph and is only as sound as the side-effect declarations behind it (Chapter 19). A fourth pass rewrites what remains using algebraic identities, and here the compiler has to decide which algebra it is working in: four of the identities are false over IEEE 754 floats, the compiler gates two of them behind a user licence, and measurement finds two more that slip through ungated (Chapter 20). Then the graph is deliberately *expanded*: composite operations like `softmax` and `layer_norm` are rewritten into primitives, because a pass can only exploit structure it can see, and the name of a composite is a wall (Chapter 21). What makes that affordable is fusion, which merges chains of elementwise work into single kernels and removes both a full tensor round trip per internalized value and a redundant read per repeated input — worth a measured ~1.9× that holds steady across a 256-fold range of tensor size, because the saving and the total both scale with `n` (Chapter 22). Fusion has exactly one hard legality condition — contracting a group must leave the kernel graph acyclic, which a union-find over an incrementally maintained topological order decides in bounded time (Chapter 23) — and a large space of legal choices, over which three different greedy engines disagreed by more than a factor of two on a three-operation program because a CPU inherits a GPU's shared-memory budget through a falsy zero (Chapter 24). Layout changes addresses rather than values, and pays only where a kernel is keyed to the layout it chose: blocking a convolution's channels by eight is worth a measured 1.26× on this CPU, flat across depth because the layout propagates through the elementwise operations between layers, and 0.94× wherever propagation stops and a convolution is left with one blocked operand and one plain one (Chapter 25). And three further pipelines trade memory, accuracy or locality for speed, each needing a number only the user has, and each failing quietly rather than loudly when that number is missing (Chapter 26).

## What Part IV establishes for later parts

- **The side-effect discipline** (Chapter 19) that every later reordering and deletion relies on, and the asymmetry that makes over-declaration cost dead work while under-declaration costs correctness.
- **The fast-math licence** (Definition 20.3) and the rule that an algebraic rewrite belongs at the highest level that can see the types — which Chapters 35 and 54–58 have to hold to.
- **Decomposition to primitives** (Chapter 21), which is why Part VI needs 68 lowering rules rather than 96 — 21 of the remaining 28 operations are rewritten away before lowering sees them, and 7 are handled structurally (Definition 34.6 fixes the three classes) — and why Part V needs gradient rules only for primitives.
- **The memory-traffic model** (Theorem 22.3), reused by Chapter 41's scheduling decisions and Chapter 50's allocator.
- **Contraction acyclicity** (Theorem 23.2), which reappears unchanged as the convexity requirement for partitioning and BYOC in Chapters 26 and 58.
- **`fusion` regions** as the unit that reaches lowering: after this part, a "kernel" is a region, and Part VI's job is to turn one region into one loop nest.

## The order these passes run in

Part III said the pipeline is *built* rather than written down, which means no chapter of this part can show you the list. Here it is, as [`buildGraphPipeline`](../../src/compiler/pipeline/graph_pipeline.ts) assembles it, with the conditions that decide whether each entry exists at all. Everything unmarked runs on every compile.

```
  registry passes for phase 'pre'
  CallInlinerPass                                        ch14 (the one ModulePass)
  DecompositionPass                                      ch21
  FixedPointGroup 'canonicalize', bound 8:               ch15
        CanonicalizePass                                 ch17, ch20
        AlgebraicSimplificationPass                      ch20
        ConstantFoldPass                                 ch19
        CSEPass                                          ch19
        DCEPass                                          ch19
  --- if optimization.layout and a target ---            ch25  (off by default)
  LayoutTransformPass ; DCEPass
  --- if quantization.enabled ---                        ch26  (off by default)
  CalibrationPass  (only when the caller supplied batches, and immediately before:)
  QuantizationPass ; CanonicalizePass ; DCEPass
  --- if fusion.enabled and target.enableEpilogueFusion ---   ch24 (CUDA only)
  EpilogueFusionPass ; DCEPass
  --- if fusion.enabled, one of three branches ---        ch22, ch23, ch24
  strategy 'priority' (default): PriorityFusionPass ; MultiOutputFusionPass
  strategy 'dominator':          DominatorFusionPass
  anything else:                 FusionPass ; FusionMergerPass ; MultiOutputFusionPass
  DCEPass
  external codegen providers' own passes                 ch58
  registry passes for phase 'post'
```

Three things are worth reading off it before any chapter does. `DCEPass` appears five times, because Chapter 19's rule — *a rewrite pass should make things unnecessary, not remove them* — means somebody has to sweep after every pass that leaves orphans. Rematerialization and partitioning are not in this list at all: they are TIR-level and phase-level respectively, which is why Chapter 26 covers them as pipelines rather than as entries. And on a default CPU compile only nine of the twenty-one graph passes are ever constructed.

## Labs

```bash
npm run build   # once, if you have not already

node docs/part4/ch19-fold-cse-dce/labs/01-three-passes.mjs
node docs/part4/ch19-fold-cse-dce/labs/02-what-dce-may-not-remove.mjs
node docs/part4/ch20-algebra-and-ieee754/labs/01-identities-under-ieee754.mjs
node docs/part4/ch20-algebra-and-ieee754/labs/02-where-the-rewrite-happened.mjs
node docs/part4/ch21-decomposition/labs/01-one-op-becomes-ten.mjs
node docs/part4/ch21-decomposition/labs/02-the-catalogue.mjs
node docs/part4/ch21-decomposition/labs/03-keeping-an-op-whole.mjs
node docs/part4/ch22-fusion-why/labs/01-the-traffic-model.mjs
node docs/part4/ch23-fusion-legality/labs/01-the-cycle-that-blocks-fusion.mjs
node docs/part4/ch23-fusion-legality/labs/02-how-the-check-scales.mjs
node docs/part4/ch24-fusion-strategies/labs/01-three-strategies.mjs
node docs/part4/ch24-fusion-strategies/labs/02-epilogue-fusion.mjs
node docs/part4/ch25-layout/labs/01-the-layout-a-target-asks-for.mjs
node docs/part4/ch25-layout/labs/02-what-a-blocked-tensor-becomes.mjs
node docs/part4/ch26-optional-pipelines/labs/01-memory-for-recomputation.mjs
node docs/part4/ch26-optional-pipelines/labs/02-quantization-as-a-rewrite.mjs
```

Unlike Parts II and III, half of these measure time. The IR they print is deterministic; the milliseconds are not, and each chapter says which of its numbers should reproduce and which are machine-specific. Where a lab reports a speedup, what should hold anywhere is the *ordering* and the rough magnitude — "between two and three times" for fusing a four-operation chain, not any particular third digit. Timings are reported as medians with their interquartile range for that reason (Chapter 1 §1.8); when two ranges overlap, the lab says so and the ratio between them means nothing.

Several labs reach past the public surface in the way Part III established: `compile` accepts a `passContext`, and a target object can be handed extra attributes before being passed in. Chapters 21 and 24 use the second of those to enable machinery no shipped target enables, and say so where they do.

## A note on what this part found

Five of the eight chapters end with a measurement that contradicts something the code implies, and they yield six findings between them. `x + 0` and `x * 0` changed results on non-finite inputs without any flag being set; a dead `scatter` and its whole index subgraph survived a pass whose job is removing them; the default fusion strategy left more than a factor of two on a three-operation program; layout transformation was inert unless you set a field no target set, and slower when you did; a memory budget reported success without meeting it; quantization's only escape from a lossy default range could not be reached through the public API at all.

All six are now fixed, and four of those chapters show the measurement on both sides of the fix rather than only after it. Chapter 25 is the exception, because what was fixed there was not a line but the absence of a mechanism; what its lab contrasts instead is the case the layout pass now gets right against the one it still gets wrong — which is where that chapter's remaining open item lives (§25.8). That is the point of keeping a finding in the text: the interesting content of §20.6 is not that `AddZero` needs a dtype check, it is the four-layer chase that located it, and the interesting content of §24.5 is that a one-character `||` cost more than a factor of two while failing no test and printing nothing. Each finding names the file and line, each is reproducible by a lab in this part, and each is carried in the outline's Appendix E.
