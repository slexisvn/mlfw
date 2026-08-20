# Chapter 3 — A map of the codebase

Thirty-six thousand lines is too much to hold in your head, and you do not have to. What you need is a map with two layers: **what a program turns into**, and **which directory owns each turn**. This chapter builds both, and ends with two labs: one in which the compiler narrates its own run, and one that shows the program it produced.

Come back to this chapter whenever you lose your bearings. That is what it is for.

## 3.1 The one idea: a program is rewritten, level by level

A machine learning compiler is not one translation. It is a sequence of representations, each lower-level than the last, each one good at expressing something the previous one could not.

```
  user code            model.forward(x)
       │  tracing
       ▼
  Graph IR             %6 = dot(%0, %5) ; %7 = add(%6, %2) ; %10 = maximum(%7, %9)
       │  graph passes  (fusion, folding, layout, ...)
       ▼
  Graph IR (optimized) %7 = fusion(%5, %2, %6)   ← fewer, bigger operations
       │  lowering
       ▼
  TIR                  for i, j: C[i,j] = max(A[i,j] + b[j], 0)
       │  scheduling    (tile, vectorize, bind to threads, ...)
       ▼
  TIR (scheduled)      for io, ii, jo, ji: ...
       │  lowering
       ▼
  LIR                  C[i*8 + j] = ...   ← indices flattened, buffers assigned
       │  code generation
       ▼
  target source        JavaScript / WebAssembly / CUDA C / WGSL
```

Each arrow is a body of theory and a directory of code. The rest of the book walks down this diagram slowly. Right now, only one property matters:

> **Every level answers a different question.** The Graph IR knows *what mathematics the program performs* and nothing about loops. TIR knows *what loops compute the result* and nothing about how they will be laid out in memory. LIR knows *exactly which memory address is touched* and nothing about the mathematics. Compilers are structured this way because an optimization is easy to express at exactly one of these levels and awkward at every other.

Fusing an `add` into a `maximum` is a statement about operations — natural in the Graph IR, painful once loops exist. Tiling a loop for cache is a statement about loops — impossible in the Graph IR, natural in TIR. Reusing one buffer for two tensors is a statement about addresses — natural in LIR.

## 3.2 The five representations, and where to see each one

| Representation | Defined in | Printed by | Chapter |
|---|---|---|---|
| Eager `Tensor` | [`src/tensor/`](../../../src/tensor/) | `console.log(t.data)` | — (not a subject of this book) |
| **Graph IR** | [`src/compiler/ir/graph/`](../../../src/compiler/ir/graph/) | `printModule(graph)` | 8–13 |
| **TIR** (tensor IR) | [`src/compiler/ir/tensor/`](../../../src/compiler/ir/tensor/) | `printTensorIR(primFunc)`, internal | 32–37 |
| **LIR** (low-level IR) | [`src/compiler/ir/lir/`](../../../src/compiler/ir/lir/) | — | 53 |
| Target source | [`src/backend/`](../../../src/backend/) | `compiled.source()` | 54–58 |

You have already seen the second and the fifth, in Chapter 2's labs. The middle two arrive in Parts VI and X.

## 3.3 The directory tour

Sizes measured 2026-08-19. They tell you where the weight of the system is, which is a useful thing to know before you go exploring.

### `src/compiler/` — 36,166 lines

| Directory | Files | Lines | What lives there |
|---|---:|---:|---|
| [`ir/`](../../../src/compiler/ir/) | 51 | 9,035 | The three representations: their node types, verifiers, printers, parser |
| [`passes/`](../../../src/compiler/passes/) | 70 | 12,057 | Every transformation: fusion, simplification, lowering, memory planning, quantization |
| [`analysis/`](../../../src/compiler/analysis/) | 20 | 3,717 | Everything that computes facts without changing anything: use-def, liveness, dependence, index arithmetic |
| [`schedule/`](../../../src/compiler/schedule/) | 12 | 3,641 | The scheduling language — how a loop nest is reshaped, and the legality rules that constrain it |
| [`autotune/`](../../../src/compiler/autotune/) | 22 | 3,601 | Search over schedules: sketches, cost models, evolutionary search, the tuning database |
| [`ad/`](../../../src/compiler/ad/) | 17 | 2,636 | Automatic differentiation: VJP rules, backward graph construction, checkpointing |
| [`pipeline/`](../../../src/compiler/pipeline/) | 15 | 1,479 | The conductor: which passes run, in what order, with what verification and tracing |

If you read only one directory to understand how the whole thing is wired, read `pipeline/`. It is the smallest and it decides everything else.

### Outside the compiler

| Directory | Lines | Role |
|---|---:|---|
| [`src/backend/`](../../../src/backend/) | 5,431 | Four code generators (CPU, WASM, CUDA, WebGPU) plus target descriptions |
| [`src/runtime/`](../../../src/runtime/) | 4,512 | Loading and running generated kernels; device memory; the Node/browser split |
| [`src/tracing/`](../../../src/tracing/) | 1,841 | Turning eager user code into a Graph IR module; dynamic shapes; guards |
| [`src/dispatcher/`](../../../src/dispatcher/) | 1,616 | Choosing an implementation per operation and device, PyTorch-style |
| [`src/tensor/`](../../../src/tensor/), [`src/kernels/`](../../../src/kernels/) | 5,079 | Eager execution — the baseline the compiler is measured against |
| `src/nn/`, `src/optim/`, `src/lightning/`, `src/data/` | 6,974 | Model building and training. Callers of everything above; not subjects of this book |

## 3.4 Following one call all the way down

When you write

```js
const compiled = compile(model, [x], { target: CPUTarget() });
```

this happens:

1. [`compile()`](../../../src/tracing/compile.ts) in `src/tracing/compile.ts` traces `model.forward` with symbolic tensors and produces a `GraphModule`.
2. It constructs a `Compiler` and calls [`Compiler.compile()`](../../../src/compiler/pipeline/compiler.ts) — `src/compiler/pipeline/compiler.ts:262`.
3. `Compiler.compile()` runs a list of **phases**. The list is data, not control flow — `compiler.ts:285`:

```ts
    const phases = this._compilePhases();
    let relaunches = 0;
    for (let i = 0; i < phases.length; i++) {
      const phase = phases[i];
      if (phase.when && !phase.when(ctx)) continue;
      phase.run(ctx);
      ...
```

and each phase is an object with a name, an optional condition, and a body — `compiler.ts:329`:

```ts
      {
        name: 'partition',
        when: (ctx: CompileContext) => ctx.compiler.config.usePartition,
        run: (ctx: CompileContext) => ctx.compiler._runPartitioning(ctx.working, ctx.trace),
      },
```

The complete phase list, in order, is:

`verify:pre` → `calibrate` → `graphPasses` → `partition` → `split` → `verify:post` → `lowering` → `tirPasses` → `verify:tensor` → `lirLowering` → `lirPasses` → `verify:lir` → `codegen` → `relaunchOnSerialization` → `planBufferAssignment`

Read that list as a table of contents for Parts IV through X. Notice three things about it.

**Verification is interleaved, not appended.** `verify:pre`, `verify:post`, `verify:tensor`, `verify:lir` — the compiler checks that its own output is well-formed at every level boundary. Chapter 64 explains why this is the single highest-value habit in compiler engineering: it converts "the model produced garbage" into "pass X produced invalid IR", which is a debuggable statement.

**Some phases are conditional.** `calibrate` only runs when quantization needs statistics; `partition` only when the program is being split across multiple targets. The `when` field is how the pipeline stays one list instead of a nest of branches.

**One phase can rewind the pipeline.** `relaunchOnSerialization` exists because a GPU kernel can turn out, after code generation, to be unrunnable at its chosen launch geometry. Rather than fail, the compiler splits the graph and re-enters at `lowering`. Chapter 43 tells that story; note for now that the pipeline is a loop with a restart, not a straight line.

### Nothing is compiled unless you ask

This is worth stating plainly, because it shapes how you should read everything that follows. `model.forward(x)` does **not** invoke the compiler. It runs eagerly: each operation dispatches to a hand-written kernel, computes immediately, and returns a tensor. That path lives in `src/tensor/` and `src/kernels/`, and it is the baseline every measurement in this book is compared against.

The compiler runs only when you ask for it — by calling `compile(model, inputs)`, or by passing `compile: true` to a `Trainer`. Both give you back something callable that behaves like the model, and the two paths are expected to agree; §2.8 lists the tests that hold them to it.

So a machine learning framework contains two execution engines, not one, and part of the engineering is keeping them from drifting apart. Chapters 60 and 61 cover how the choice is made and how user code crosses from one to the other.

## 3.5 Labs — Watch the pipeline run, then read what it produced

```bash
node docs/part0/ch03-map-of-the-codebase/labs/01-watch-the-pipeline.mjs
```

The compiler can narrate its own execution. You pass it a trace level and a sink — a function that receives an event object for everything that happens ([`labs/01-watch-the-pipeline.mjs`](labs/01-watch-the-pipeline.mjs)):

```js
const compiled = compile(model, [x], {
  target: CPUTarget(),
  trace: {
    level: TraceLevel.VERBOSE,
    sink: (event) => {
      if (event.type === 'phase') { ... }
      else if (event.type === 'pass' && event.changed) { ... }
    },
  },
});
```

Compiling the two-layer network from Chapter 1 prints:

```
> phase compile
> phase graphPasses
    pass canonicalize: 10 ops -> 10 ops
    pass constant_fold: 10 ops -> 10 ops
    pass dce: 10 ops -> 7 ops
    pass PriorityFusionPass: 7 ops -> 6 ops
< phase graphPasses (11.08 ms)
> phase lowering
< phase lowering (4.06 ms)
> phase scheduling
< phase scheduling (0.20 ms)
> phase scheduling
< phase scheduling (0.14 ms)
> phase simplify
< phase simplify (1.26 ms)
> phase memoryScheduling
< phase memoryScheduling (3.32 ms)
> phase memoryPlanning
< phase memoryPlanning (4.49 ms)
> phase lirLowering
< phase lirLowering (2.71 ms)
> phase lirSimplify
< phase lirSimplify (0.40 ms)
> phase codegen
< phase codegen (2.58 ms)
< phase compile (40.83 ms)
```

This is the entire book in twenty lines. Read it slowly.

The op counts tell a complete story, and a second lab lets you see its ending:

```bash
node docs/part0/ch03-map-of-the-codebase/labs/02-see-the-optimized-graph.mjs
```

```
module @Sequential {
  func @Sequential(%0: tensor<2x2xf32>, %1: tensor<8x2xf32>, %2: tensor<8xf32>, %3: tensor<1x8xf32>, %4: tensor<1xf32>) -> (tensor<2x1xf32>) {
    %5 = dot(%0, %1) {lhs_batch = [], lhs_contracting = [1], rhs_batch = [], rhs_contracting = [1]} : tensor<2x8xf32>
    %6 = constant() {tensor_type = tensor<2x8xf32>, value = 0} : tensor<2x8xf32>
    %7 = fusion(%5, %2, %6) {fusion_kind = "kElementwise"} : tensor<2x8xf32>
    {
      ^bb(%8: tensor<2x8xf32>, %9: tensor<8xf32>, %10: tensor<2x8xf32>):
      %11 = add(%8, %9) : tensor<2x8xf32>
      %12 = maximum(%11, %10) : tensor<2x8xf32>
      yield(%12)
    }
    %13 = dot(%7, %3) {lhs_batch = [], lhs_contracting = [1], rhs_batch = [], rhs_contracting = [1]} : tensor<2x1xf32>
    %14 = add(%13, %4) : tensor<2x1xf32>
    return(%14)
  }
}
```

Compare it with the traced graph from Chapter 2 and every number in the log becomes concrete.

**The traced graph had ten operations:** two `transpose`, two `dot`, two `add`, one `constant`, one `broadcast_in_dim`, one `maximum`, one `return`. That is the `10` the log starts from.

**`canonicalize`: 10 → 10.** No operation disappeared, but look at `%5`. In the trace it was `dot(%0, %5)` with `rhs_contracting = [0]`, consuming a transposed weight. Here it is `dot(%0, %1)` with `rhs_contracting = [1]`, consuming the weight directly. The transpose was *absorbed into the operation that used it*, by changing one attribute. Nothing was deleted — the two `transpose` operations are still there at this point, now with no users.

**`constant_fold`: 10 → 10.** Similarly, `broadcast_in_dim` of a scalar zero became a `constant` that is already 2 × 8 (`%6`). The broadcast is now unused.

**`dce`: 10 → 7.** Now the deletions happen: two orphaned `transpose`s and one orphaned `broadcast_in_dim`. This is the standard rhythm of optimization — rewriting passes leave garbage behind on purpose, and one pass whose only job is deletion cleans up after all of them. Chapter 19 explains why "has no users" is a subtler condition than it looks.

**`PriorityFusionPass`: 7 → 6.** `add` and `maximum` were replaced by the single `fusion` operation at `%7`, whose *region* — the indented block with `^bb` — holds the two original operations. That region is what later becomes one loop nest, exactly the loop you read in Chapter 2's third lab. Fusion is decided here, at the graph level, long before loops exist. Chapters 22–24.

Regions are how this IR nests one program inside another. They are also how `if`, `while` and `scan` are represented. Chapter 9 introduces them properly.

**Op count is a crude measure of work.** Both `canonicalize` and `constant_fold` report `10 ops -> 10 ops`, and both changed the program substantially: one rewired a matrix multiply so that a transpose became unnecessary, the other replaced a broadcast with a constant of the final shape. Neither deleted anything, because deleting is not their job. Whenever you read a pass log, remember that the counter measures size, not effect — and that a pass which reports no change may still have done the decisive work.

**Every level has its own passes.** `scheduling`, `simplify`, `memoryScheduling`, `memoryPlanning` operate on TIR; `lirSimplify` operates on LIR. The three-level structure from §3.1 is visible directly in the timing log.

**Compilation cost is real.** 40 ms to compile a network that executes in microseconds. This is the fundamental tradeoff of compiled execution, and it is why Chapter 62's compilation cache exists, and why the autotuner in Part VIII is something you invoke deliberately rather than by default.

**Try this.** Raise the level to `TraceLevel.DEBUG` and drop the `event.changed` filter to see every pass, including the ones that did nothing. Then compile something bigger — a `TransformerEncoderLayer` — and watch which passes start to dominate the time.

## 3.6 Where to look when…

| You want to understand | Start at |
|---|---|
| What an operation means | [`src/compiler/ir/graph/ops/`](../../../src/compiler/ir/graph/ops/) — one file per family |
| What makes IR valid | [`ir/graph/verifier.ts`](../../../src/compiler/ir/graph/verifier.ts) |
| Which passes run and when | [`pipeline/graph_pipeline.ts`](../../../src/compiler/pipeline/graph_pipeline.ts), [`tir_pipeline.ts`](../../../src/compiler/pipeline/tir_pipeline.ts) |
| How a pass is written | [`passes/pass.ts`](../../../src/compiler/passes/pass.ts) and any file under `passes/simplify/` |
| Why two operations may fuse | [`passes/fusion/fusion_analysis.ts`](../../../src/compiler/passes/fusion/fusion_analysis.ts) |
| How an operation becomes loops | [`passes/lowering/rules/`](../../../src/compiler/passes/lowering/rules/) |
| Whether a loop transformation is legal | [`analysis/dependence.ts`](../../../src/compiler/analysis/dependence.ts), [`schedule/legality.ts`](../../../src/compiler/schedule/legality.ts) |
| How gradients are produced | [`ad/vjp_rules/`](../../../src/compiler/ad/vjp_rules/) |
| What code is emitted for a target | [`src/backend/<target>/codegen.ts`](../../../src/backend/) |
| How compiled code is actually called | [`src/runtime/runtime.ts`](../../../src/runtime/runtime.ts) |

## 3.7 How this repository is put together

A few facts worth knowing before you read source, all measured 2026-08-19:

- **TypeScript throughout, `strict` mode, no `any`.** Types are part of the documentation: a function signature usually tells you what an argument is without reading further.
- **Tests are the specification.** 302 test files, 5,131 tests for the default projects. Behaviour asserted by a test is behaviour you can rely on.
- **Almost no comments, by design.** 38 comment lines in 67,852 lines of source. The codebase explains itself through naming and structure rather than prose. That is a deliberate choice, and it is part of why this book exists: the *what* lives in the code, the *why* lives here.
- **Registries everywhere.** Operations, lowering rules, VJP rules, passes and code generators are all registered into tables rather than hard-wired. Chapter 11 explains the pattern; it is the reason the compiler can be extended without editing its core.

## 3.8 Read the tests

- [`tests/e2e/public-api.test.js`](../../../tests/e2e/public-api.test.js) — pins the public surface: what a user of the framework is allowed to import, and what the package promises to ship.
- [`tests/compiler/pipeline/`](../../../tests/compiler/pipeline/) — pins the pipeline itself: phase order, what each phase may assume about its input, and what happens when one of them fails.

## 3.9 How to use this map

You now have three things: a model of what a program turns into (§3.1), a directory for each stage (§3.3), and a way to make the compiler narrate itself (§3.5). That is enough to start.

From here the book descends the diagram. Part I asks why any of this machinery is justified. Part II builds the Graph IR properly — values, types, operations, verification. Part III builds the pass infrastructure that transforms it. Everything after that is a walk down the arrows.

---

**Next:** Part I — Why machine learning needs a compiler, which returns to Chapter 2's four speedups — 2.57× and 1.08×, then 14.54× and 4.57× for the same twelve operations with two `tanh` calls removed — and explains exactly where the time went.
