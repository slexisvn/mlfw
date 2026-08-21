# Part VIII — Autotuning

Part VII gave the compiler a vocabulary for saying how a loop nest should run, and a guarantee: any sequence of the twenty-two primitives that does not throw produces a program computing what the original computed. What it did not give is a way to choose. Nine scheduling rules pick one arrangement per block from the target's declared attributes, and Chapter 40 measured what that costs on a matmul whose extents happen not to suit the rule.

This part replaces the choice with a search. That is a different kind of compiler activity from everything before it. A pass either fires or does not, and its correctness argument is about the program; a search has a *budget*, a *sample*, an *estimate* and a *result that could have been better*. Nothing here is provably optimal, and for most of the space nothing needs to be: where a sketch is a composition of Part VII's primitives, soundness composes and the search can only be slow, never wrong. That is what turns an intractable correctness problem into a tractable engineering one — and Chapter 45 names the two sketches to which it does not apply, one because it reassociates floating point and one because it does not call primitives at all.

The part follows one loop nest through the whole machine. How many arrangements does it have; what generates them; how are they ordered without being run; how are the few worth running actually run; and what has to be written down so that tomorrow's compilation does not repeat today's thirty seconds.

| Chapter | Title | The question it answers |
|---|---|---|
| [44](ch44-how-big-is-the-search-space/README.md) | How big is the search space | How many schedules does one matmul have, and what does the count decide about the method? |
| [45](ch45-sketches/README.md) | Sketches | What turns an unbounded sequence of primitives into a finite space in which almost every point is legal? |
| [46](ch46-cost-models/README.md) | Cost models | How do you order candidates you cannot afford to run, and what is such a model graded on? |
| [47](ch47-search-and-measurement/README.md) | Search and measurement | Where do candidates come from, what does one measurement mean, and what makes tomorrow's problem the same problem? |
| [48](ch48-reproducibility/README.md) | Reproducibility | What has to be written down for a tuned kernel to survive the process that produced it? |

## The argument in one paragraph

A schedule space is a product of independent finite choices, and the dominant factor is the number of ordered factorisations of a loop extent, which is a closed form — 84 for a 64-long axis at four levels, 7,056 for a two-axis tiling, 92,190 for the two blocks of one matrix multiply — a count too large to enumerate and too structureless to reason about, which is precisely the range in which searching is the right method (Chapter 44). Turning that count into something a program can draw from is the job of a *sketch*: a skeleton of primitive calls with typed holes, generated per block by a three-rule decision list on the block's structure, so that Part VII's soundness composes over the whole space wherever the skeleton is a composition of primitives, which is everywhere but two named places (Chapter 45). Ordering the space without running it needs a cost model, and the thing to understand about one is that it is fitted with squared error and consumed only through comparisons, so any strictly increasing transform of it is the same model — which makes ranking the property to reason about and makes a feature that is constant over the space no feature at all (Chapter 46). Then the search itself: an elitist genetic algorithm over a seeded generator, a benchmark that turns a noisy clock into a median, a deadline that bounds the whole thing to within one population, and a database keyed on a structural description of the block so that the second compilation of a model pays nothing (Chapter 47). And finally the object that makes the last claim meaningful — a schedule trace, a list of primitive names and JSON arguments, replayable into a fresh program, and faithful exactly when the interpreter is in the state it was recorded in (Chapter 48).

## What Part VIII establishes for later parts

- **The schedule space as a product** (Definition 44.1) and its size in closed form (Theorem 44.2), which Chapter 62 needs in the negative: a symbolic extent has no divisors, so a dynamically-shaped block leaves the search space entirely.
- **The sketch as the unit of tuning** (Definition 45.1) and `getSketchesForBlock` as the interface a new operation must be classifiable by — Appendix F's end-to-end walkthrough acquires a tuning step here.
- **Theorem 45.7**: a sketch space contains no wrong programs *provided* every primitive its `apply` calls is sound, so over that part of the space a tuner needs a clock and not an oracle. Counterexample 45.8 gives the two sketches whose hypothesis fails — `rfactor`, which reassociates floating point, and the register-blocked GPU matmul, which writes a body rather than scheduling one — and those two are what Part XII's differential testing still has to cover.
- **Theorem 46.3** (only the induced order matters) as the statement that separates a cost model's error from its usefulness, and Definition 44.5 (regret) as the quantity a tuner should actually be reported against.
- **The workload key** (Definition 47.5) as the identity a compiled artefact is cached under, which Part XI's JIT cache sits directly beside — one keyed on shapes and guards, the other on a block's structure.
- **`ScheduleTrace`** (Definition 48.1) as the serialisable form of a schedule, and **Theorem 48.3** as the precise statement of what a replay needs — the fresh-variable counter of [`schedule.ts:193`](../../src/compiler/schedule/schedule.ts) being the piece of interpreter state that neither the trace nor the database carries.
- **`ScheduleValidator` running only on searched schedules** ([`session.ts:186`](../../src/compiler/autotune/session.ts)), which Chapter 42 named and this part is the other half of: the autotuner is the validator's only production caller, so everything it checks is checked exactly when a search produced the schedule and never when a rule did.

## Labs

Part VIII's labs drive the autotuner by hand, and nothing under `src/compiler/autotune/` is part of the package's public surface — a user asks for `scheduling: { autotune: true }` and never sees a sketch, a cost model or a workload key. Each lab therefore imports [`docs/part8/_internals.mjs`](_internals.mjs), which bundles the internal surface listed in [`docs/tools/internals-entry.ts`](../tools/internals-entry.ts) with esbuild, a devDependency the repository already has. The bundle lands in the OS temp directory and takes about a tenth of a second to build; nothing is written inside the repository, and there is still no build step to run.

```bash
npm install   # once, if you have not already — the labs need esbuild

node docs/part8/ch44-how-big-is-the-search-space/labs/01-counting-the-space.mjs
node docs/part8/ch44-how-big-is-the-search-space/labs/02-what-the-space-is-worth.mjs
node docs/part8/ch45-sketches/labs/01-deriving-sketches.mjs
node docs/part8/ch45-sketches/labs/02-how-many-distinct-programs.mjs
node docs/part8/ch46-cost-models/labs/01-what-the-analytic-model-sees.mjs
node docs/part8/ch46-cost-models/labs/02-ranking-not-regression.mjs
node docs/part8/ch47-search-and-measurement/labs/01-the-search-loop.mjs
node docs/part8/ch47-search-and-measurement/labs/02-measurement-and-the-cache.mjs
node docs/part8/ch48-reproducibility/labs/01-record-and-replay.mjs
node docs/part8/ch48-reproducibility/labs/02-what-the-trace-omits.mjs
```

Three helpers carry over from Part VII. `lowerToTir(fn, inputs, target)` traces a function and lowers it, giving a real `PrimFunc` — the same object the autotuner receives; it calls `resetVarCounter()` first, which is what makes loop names reproducible across sections and across runs. `toKernel(primFunc, target)` compiles one to callable source. `printTensorIR` is Chapter 32's printer. Two labs also use `compileGraph` with `scheduling: { autotune: true }`, where the point is to show what the shipping pipeline does rather than what the pieces can do.

Every sketch name, parameter tuple, loop extent, feature value, cost-model score, workload key and trace step in this part is deterministic and should reproduce exactly. The exception is §44.6, which compiles eight schedules of a 256×256 matmul and *times* them: those numbers are labelled MEASURED in both the lab and the chapter, they will differ on your machine and between runs, and nothing else in the part depends on them. The search itself is seeded, so even the evolutionary runs reproduce. The slowest lab takes about four and a half seconds.

## A note on what this part found

Part VI's findings were about *precision* and Part VII's about *reach*. Part VIII's are about **seams**. Every component is individually well built and individually tested: the factorisation enumerator is checked for product, length, degeneracy and determinism, the tiling sketch produces a validator-clean schedule at all 1,225 of its points, the database round-trips and invalidates on a version mismatch. What is not tested is any *pair* of them, and the assumptions each makes about its neighbour are where the findings live.

Three shapes recur. A component is written against a block shape the lowering rules do not emit, so it is derived, counted, sampled and always refused — the SSRSRS tiling sketch and the producer–consumer fusion sketch are both this, and between them they account for most of a matmul's advertised space. An objective is constant over the space it is meant to rank: the analytic cost model returns one value for all 2,304 points of the multi-level tiling space, provably and by measurement. And a generator's low bits collapse a choice: the LCG's period-4 residue makes the initial population a single sketch, which at the shipped default seed is the sketch that always throws.

None of it produces a wrong kernel, because a searched schedule that fails is replaced by a rule-produced one and every end-to-end test asserts the answer rather than the schedule. Each finding is named with its file and line in the chapter that meets it, each is reproducible by a lab here, and all twenty are collected in the outline's [Appendix E](../OUTLINE.md).
