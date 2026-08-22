# Part I — Why machine learning needs a compiler

Part 0 showed you a compiler working. This part asks whether it should exist at all, and answers with measurements rather than assertions.

| Chapter | Title | The question it answers |
|---|---|---|
| [4](ch04-eager-execution/README.md) | Eager execution, and where it hurts | What does one operation cost, and why does the answer change by 30× with tensor size? |
| [5](ch05-calls-to-program/README.md) | From a sequence of calls to a program | How do you turn a method into a data structure, and what do you lose doing it? |
| [6](ch06-the-pipeline/README.md) | The pipeline in one picture | Why three intermediate representations instead of one? |
| [7](ch07-vocabulary/README.md) | Vocabulary | Every term, defined once, with the code that implements it |

## The argument in one paragraph

An eager operation costs roughly `α + βn` — a fixed framework cost plus a per-element cost — and below a few hundred elements the fixed part dominates (Chapter 4, which also fits the model properly and finds it good to about 30%). Above it, the per-element part is mostly memory traffic, which fusion can remove, unless the arithmetic is expensive, in which case fusion removes something that was not the bottleneck. Getting fusion at all requires seeing the whole computation, which requires turning the model into a graph, which requires either a new language, a parser, or tracing — and tracing works only for inputs resembling the one traced, in two precise senses (Chapter 5). Once you have the graph, the journey to machine code passes through several representations, because each optimization is expressible at exactly one level of abstraction (Chapter 6).

## What the cost models assume

Part I states two quantitative models and later parts lean on both. Neither is a law, and both are stated under assumptions that are easy to lose track of once the formula is memorized. They are collected here so that a later chapter can be checked against them.

**Definition 4.1, `T(n) = α + βn`, assumes:**

- **One regime.** A single β across all sizes, which the fit in §4.3 shows is false: `ns/element` is still falling at a million elements, and the affine model carries a ~30% worst-case residual across the measured range. Read α and β as ballpark constants for *ranking* two programs, not as calibrated parameters for predicting either.
- **Isolation.** α and β are measured for one operation running alone, with the whole cache to itself. An operation inside a chain competes with its neighbours, and §4.7's box shows predictions drifting by up to 1.6× because of it.
- **Steady state.** The first call at each new shape pays for compilation (§4.4) and is excluded. A program that keeps producing fresh shapes never reaches the regime this model describes.
- **One backend, one dtype.** Everything here is the CPU backend on `float32` in Node. α is dominated by dispatch and allocation in JavaScript; on a device backend α is dominated by launch latency instead, and the two numbers have nothing to do with each other.
- **Allocation is inside α.** Every eager operation allocates its output. That cost is real, it is charged to α, and it is one of the things fusion removes — which is why the model cannot be used to predict fused performance.

**Theorem 4.4, the roofline bound, assumes:**

- **Achieved reuse.** Byte counts such as `12n²` for a matmul are *minimum* traffic, reached only if each matrix crosses the memory boundary once. A naive loop nest over the same mathematics moves Θ(n³) bytes and sits on the other roof. Quoting an intensity always means quoting the reuse assumption with it.
- **Saturation is possible.** The bound assumes the machine can be driven to either peak compute or peak bandwidth. A latency-bound dependent chain saturates neither, and no roofline predicts it (§4.10).
- **Peaks are the hardware's, not the runtime's.** The 7.76 GB/s measured in §4.5 is what this eager JavaScript path achieves on one machine, not what that machine can do.

## What Part I establishes for later parts

- **A cost model** (Definition 4.1) that later chapters use to explain why a transformation helps — under the assumptions above.
- **Roofline reasoning** (Theorem 4.4) — the memory-bound / compute-bound distinction that decides which optimization can possibly matter.
- **Trace validity** (Theorem 5.3) — the precondition every compiled artifact in this framework relies on, and the reason guards exist.
- **Lowering as irreversible translation** (Definition 6.1) and the phase-ordering problem (Theorem 6.3), which together justify the pipeline's shape.

## Labs

```bash
npm run build   # once, if you have not already

node docs/part1/ch04-eager-execution/labs/01-anatomy-of-an-op.mjs
node docs/part1/ch04-eager-execution/labs/02-where-fusion-wins.mjs
node docs/part1/ch04-eager-execution/labs/03-eager-is-compiled-too.mjs
node docs/part1/ch05-calls-to-program/labs/01-what-the-whole-program-buys.mjs
node docs/part1/ch05-calls-to-program/labs/02-control-flow.mjs
node docs/part1/ch05-calls-to-program/labs/03-guards-and-recompilation.mjs
node docs/part1/ch06-the-pipeline/labs/01-one-program-three-levels.mjs
```

The timings printed by the Chapter 4 labs are machine-specific. What should reproduce anywhere is the *structure*: a fixed cost that amortizes, a break-even size, transcendental functions costing an order of magnitude more than arithmetic, and fusion paying off in inverse proportion to how expensive the fused arithmetic is.
