# Chapter 7 — Vocabulary

Every term this book uses, defined once, with the place in the code where it lives.

**This is a reference chapter, and it is the one chapter you should not read straight through.** Every other chapter earns its terms: it shows you a problem, then names the thing that solves it, so the word arrives attached to something you have already seen. A glossary inverts that. Reading sixty definitions before meeting any of the problems means memorizing sixty labels with nothing underneath them, which is both unpleasant and ineffective — and it is exactly the failure mode the six-beat structure in §1.5 exists to prevent.

So use it the way you would use a dictionary:

- **Now:** skim the section headings and the bolded terms, for five minutes, to know what is in here. Do not try to retain the definitions.
- **Later:** come back whenever a chapter uses a word you have half-forgotten. Each entry names the chapter that develops the idea properly, and that chapter is the real explanation; the entry is a pointer.

Terms are grouped by what they describe: data, programs, transformations, execution, and performance.

## 7.1 Data

**Tensor.** A multi-dimensional array with a uniform element type. Everything a model computes with is a tensor; a scalar is a tensor of rank 0. In this framework: [`src/tensor/core/tensor.ts`](../../../src/tensor/core/tensor.ts).

**Shape.** The size of a tensor along each dimension, as a list. `[2, 8]` is a 2 × 8 matrix. Written in IR as part of the type: `tensor<2x8xf32>`.

**Rank.** The number of dimensions — the length of the shape. `[2, 8]` has rank 2; a scalar has rank 0, written `tensor<f32>` — no dimensions to list, so nothing precedes the dtype.

**Element count** (`numel`). The product of the shape. A `[2, 8]` tensor has 16 elements.

**Dtype.** The element type: `f32`, `f16`, `i32`, `i64`, `bool`, and so on. Two tensors of the same shape and different dtypes are different types, and operations between them require an explicit or inferred conversion. Chapter 10.

**Dynamic dimension.** A dimension whose size is not known at compile time, written `?` in the IR. A layer traced with a dynamic batch dimension has a result type like `tensor<?x2xf32>`, and one kernel serves every batch size — at the cost of the compiler not knowing the bound. Chapter 62.

**Symbolic dimension.** A dynamic dimension that has a *name*, so that relationships between dimensions survive: if two tensors both have first dimension `n`, the compiler knows they match without knowing what `n` is. [`src/compiler/analysis/sym_int.ts`](../../../src/compiler/analysis/sym_int.ts).

**Strides.** The distance in elements between consecutive positions along each dimension of the underlying flat storage. A contiguous `[2, 8]` tensor has strides `[8, 1]`. Strides are what let a transpose be free at the eager level — you change the strides and touch no data.

**Layout.** How a tensor's logical dimensions map onto memory. The same 4-D activation can be stored as NCHW or NHWC; the choice changes which access patterns are contiguous, and therefore which kernels are fast. Chapter 25.

**Broadcasting.** The rule that lets an elementwise operation combine tensors of different shapes by treating any dimension of size 1 as if it repeated. `[4, 3] + [1, 3]` adds the single row to every row; `[4, 3] + scalar` adds the scalar to every element. No data is copied — the size-1 axis is simply read with the index `0` every time. Chapter 10 gives the dimension rule as an order; Chapter 34 shows that a broadcast never becomes a loop.

**Buffer.** A named region of memory with a shape and dtype, as seen inside TIR. The difference between a tensor and a buffer is the difference between a *value* and *the storage holding it* — the graph has tensors, TIR has buffers. [`src/compiler/ir/tensor/buffer.ts`](../../../src/compiler/ir/tensor/buffer.ts).

## 7.2 Programs

**IR** (intermediate representation). A data structure representing a program, designed to be analysed and transformed rather than read or executed. This compiler has three: Graph IR, TIR, LIR (Chapter 6).

**Operation** (op). One node of the graph: a name, operands, results, and attributes. `add`, `dot`, `reduce`, `fusion` are operations. Registered in [`src/compiler/ir/graph/ops/`](../../../src/compiler/ir/graph/ops/); 96 of them as of 2026-08-19.

**Operand / result.** The inputs and outputs of an operation. In `%7 = add(%6, %2)`, the operands are `%6` and `%2` and the single result is `%7`.

**Attribute.** A compile-time constant attached to an operation, as opposed to data flowing through it. In `transpose(%1) {permutation = [1, 0]}`, the permutation is an attribute: it is part of what this operation *is*, and it cannot vary at runtime.

**Value.** A single tensor produced by an operation or supplied as a function argument, named `%n` in the printed IR. Every value is produced exactly once — the SSA property, Chapter 8.

**Block.** An ordered list of operations. In TIR the word is also used for a *TensorIR block*, a unit of computation with declared iteration variables and declared read/write regions — the `block matmul_1 { ... }` you saw in Chapter 6. Context distinguishes them; Chapter 33 sorts it out properly.

**Region.** A block nested inside an operation, making that operation contain a program. Regions are how `fusion`, `scan`, `if` and `while` are represented without unrolling or flattening. Chapter 9.

**Region-carrying operation.** An operation that owns one or more regions — `fusion`, `scan`, `if`, `while`. Two things follow and both catch people: a traversal that uses `ops()` rather than `opsRecursive()` never sees inside one, and its side effects are whatever its contents' are, which is what the `RECURSIVE_MEMORY_EFFECTS` trait declares. Chapters 9 and 11.

**Lifted parameter.** A tensor that `forward` read from the model rather than receiving as an argument — `this.fc.weight` — and that tracing turned into a parameter of the traced function. Lifting is why a compiled model can still be trained: the artifact holds a live reference to the tensor's storage and sees an in-place update. A host *scalar* read the same way is not lifted; it is frozen into the graph as a constant. Chapter 5 §5.5.

**Graph break.** In a bytecode-capture system such as TorchDynamo, the seam where capture stops because the interpreter reached something it cannot compile, emits the graph so far, lets the host run the offending code, and starts a new graph after it. The program still runs; no optimization crosses the seam. This framework traces instead, so the same situation raises an error rather than producing a seam. Chapter 5 §5.2.

**Trait.** A declared property of an operation that passes can query instead of special-casing op names: *commutative*, *elementwise*, *terminator*, *reduction*. This is how a fusion pass can reason about operations it has never heard of. [`op_registry.ts`](../../../src/compiler/ir/graph/op_registry.ts), Chapter 11.

**Verifier.** A function that checks an IR's invariants and reports every violation. Not an optimization — a guard rail. Chapter 12.

## 7.3 Transformations

**Pass.** A transformation from an IR to an IR, reporting whether it changed anything. The unit of composition in a compiler. [`src/compiler/passes/`](../../../src/compiler/passes/); 31 concrete pass classes as of 2026-08-19 — 21 over the graph, 9 over TIR, 1 over LIR — plus the abstract bases they extend, which are not passes. Chapter 14.

**Analysis.** A computation over an IR that produces *facts* without changing it: which values are used where, which loops carry dependences, which buffers are live at which point. Analyses are cached and invalidated when passes modify the IR. Chapter 16.

**Fixed-point group.** A set of passes run repeatedly as a unit until none of them reports a change, or until an iteration cap is reached. It dissolves ordering *within* the group — any enabling relationship among its members is eventually exploited — and dissolves nothing between groups. The cap, not a monotone measure, is what guarantees termination. Chapters 6 §6.4 and 15.

**Pipeline.** The ordered list of passes for one level. [`graph_pipeline.ts`](../../../src/compiler/pipeline/graph_pipeline.ts), [`tir_pipeline.ts`](../../../src/compiler/pipeline/tir_pipeline.ts), [`lir_pipeline.ts`](../../../src/compiler/pipeline/lir_pipeline.ts).

**Canonicalization.** Rewriting operations into a preferred normal form, so that later passes have fewer shapes to recognize. Folding a transpose into a `dot` is canonicalization. Chapter 17.

**Constant folding.** Evaluating at compile time what does not depend on runtime data.

**DCE** (dead code elimination). Removing operations whose results cannot affect the output. Chapter 19.

**CSE** (common subexpression elimination). Replacing two identical computations with one.

**Fusion.** Merging several operations so their combined computation is performed in one pass over memory, rather than one pass per operation. The most valuable optimization in this domain, and the subject of Chapters 22–24.

**Lowering.** Translating to a more detailed, less abstract representation — graph to TIR, TIR to LIR, LIR to target source. Irreversible by construction (Definition 6.1).

**Schedule.** A description of *how* a computation is to be executed — loop order, tiling, vectorization, thread binding — as distinct from *what* it computes. Chapters 38–43.

**Schedule primitive.** One editing operation on a schedule. There are 22, and since the number is quoted throughout Parts VII and VIII, here is the list it refers to — the mutating methods of [`Schedule`](../../../src/compiler/schedule/schedule.ts), in the spelling the code uses:

> `split`, `fuseLoops`, `reorder`, `tile`, `unroll`, `vectorize`, `parallelize`, `bindThread`, `annotate`, `blockize`, `tensorize`, `cacheRead`, `cacheWrite`, `setScope`, `storageAlign`, `computeAt`, `reverseComputeAt`, `computeInline`, `computeInlineBlock`, `fuseConsumer`, `decomposeReduction`, `rfactor`

The class has more public methods than that — `getLoops`, `getBlock`, `getTrace`, `verify` and friends — but those *query* a schedule rather than edit one, and they are not counted. Chapters 40 and 41 work through the list; Chapter 38 is where the count matters, because a claim about all 22 is only as strong as the weakest one.

Beware of the snake_case names used in the TVM literature (`cache_read`, `compute_at`, `storage_align`). They denote the same primitives, and this book uses them when discussing the idea in general, but the identifiers in this codebase are camelCase.

**Legality.** Whether a transformation preserves the program's meaning. Not the same as *profitable*: a legal transformation may make the program slower, and an illegal one may make it faster and wrong. Chapter 42.

**Dependence.** A constraint on execution order arising because two operations touch the same memory. Read-after-write, write-after-read, write-after-write. The theory that decides legality for loops. Chapter 36.

**Autodiff / VJP.** Automatic differentiation; the vector-Jacobian product is the per-operation rule from which reverse-mode gradients are assembled. Part V.

## 7.4 Execution

**Eager execution.** Running each operation as it is called, returning a concrete tensor. The default in this framework, and the baseline against which compilation is measured. Chapter 4.

**Tracing.** Running a model with symbolic tensors to record the operations it performs, producing a graph. Chapter 5.

**Kernel.** A compiled function that performs one unit of work on a device — in this framework, one graph function's worth of computation. What `compiled.source()` prints is a kernel.

**Target.** A description of the machine being compiled for: its instruction set, memory hierarchy, thread model, and what it supports. `CPUTarget()`, `WasmTarget()`, `CUDATarget()`, `WebGPUTarget()`. [`src/backend/target.ts`](../../../src/backend/target.ts).

**Backend.** The code generator for a target, plus the runtime glue that loads and calls what it produced. [`src/backend/`](../../../src/backend/).

**Runtime.** What holds compiled kernels and executes them: argument marshalling, device memory, asynchrony. [`src/runtime/runtime.ts`](../../../src/runtime/runtime.ts), Chapter 59.

**Dispatch.** Choosing which implementation of an operation to run, based on device, dtype, and whether tracing is active. [`src/dispatcher/`](../../../src/dispatcher/), Chapter 60.

**Dispatch key.** One bit of the set every tensor carries that decides which implementation an operation resolves to — `CPU`, the autograd keys, `TRACING`. The highest key present wins, which is the entire mechanism by which a symbolic tensor diverts `add` away from the CPU kernel and into the tracer without the model's code knowing (Chapter 5 §5.3). Chapter 60.

**Guard.** A predicate checked before a compiled artifact is reused, ensuring the conditions it was compiled under still hold (Definition 5.5). Shapes are the most common guard.

**Specialization.** Compiling for specific known values — usually shapes — to produce better code, accepting recompilation when they change.

**Autotuning.** Searching over schedules by generating candidates, predicting or measuring their cost, and keeping the best. Part VIII.

## 7.5 Performance

**FLOP.** One floating-point operation. Used for counting work: an n × n matmul is 2n³ FLOP.

**Bandwidth.** Bytes per second moved between memory and the processor.

**Arithmetic intensity.** FLOP per byte of memory traffic (Definition 4.3). Elementwise operations sit near 0.08; matrix multiplication rises with size.

**Memory-bound / compute-bound.** Whether runtime is limited by moving data or by performing arithmetic (Theorem 4.4). Determines which optimizations can possibly help.

**Overhead-bound.** Limited by neither: the tensors are small enough that per-call framework cost dominates (Corollary 4.2).

**Occupancy.** On a GPU, how much of the hardware's parallel capacity a kernel actually uses. Chapter 43.

**ULP** (unit in the last place). The gap between adjacent representable floating-point numbers at a given magnitude. The natural unit for "how much did rearranging this computation change the answer" — Chapter 2 measured a difference of exactly one.

**Roofline.** The bound min(peak compute, intensity × bandwidth) on achievable throughput, and the mental model that goes with it (Theorem 4.4).

## 7.6 A note on terms this book avoids

Some words in circulation are ambiguous enough to be worth naming and setting aside.

- **"Graph mode"** is used by different frameworks to mean tracing, source parsing, or deferred execution. This book says *tracing* or *compilation* and means one of them.
- **"JIT"** describes both per-operation compilation (§4.2) and whole-model compilation. The book says which.
- **"Kernel fusion"** and **"operator fusion"** are the same thing as *fusion* here.
- **"Optimization"** in the machine learning sense (gradient descent) and in the compiler sense (program transformation) are different subjects. This book means the compiler sense, except in Part V where both appear and are distinguished explicitly.

---

**Part I ends here.** You have the motivation (Chapter 4), the mechanism that makes it possible (Chapter 5), the structure that organizes it (Chapter 6), and the vocabulary (Chapter 7).

**Next:** [Part II — Representing programs](../../part2/README.md), which builds the Graph IR properly: values and SSA, the object model, types, operations as a dialect, verification, and the textual format.
