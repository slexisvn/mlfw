# Part X — Code generation

Everything before this part decided things. Part IV decided which operations survive and which get fused; Part VI decided what loop nest computes them; Part VII decided the loops' shape; Part VIII searched for the best shape; Part IX decided where the bytes go. Nothing has been *written down* in a language a machine will run.

This part writes it down, four times, into four languages that agree on almost nothing — and it is the first part where the same program can come out *different* depending on the target, which makes it also the part that has to say exactly how different, and why.

| Chapter | Title | The question it answers |
|---|---|---|
| [53](ch53-lir-the-third-ir/README.md) | LIR: why a third IR | What should be computed once, before the fan-out to four backends? |
| [54](ch54-javascript-for-the-cpu/README.md) | Generating JavaScript for the CPU | What does a backend owe a target language that has no float32? |
| [55](ch55-webassembly/README.md) | WebAssembly | What does a backend do when it has to assemble the bytes itself? |
| [56](ch56-cuda/README.md) | CUDA | What does a backend do when the loops do not run? |
| [57](ch57-webgpu-and-wgsl/README.md) | WebGPU and WGSL | What does the same execution model cost under a much stricter language? |
| [58](ch58-someone-elses-kernel/README.md) | Calling someone else's kernel | What does a compiler give up to hand a piece of its program to a library? |

## The argument in one paragraph

Three of the four target languages need the same two things — a multidimensional subscript turned into a flat offset, and an accumulating loop turned into a register — so the compiler does both once, in a third IR that has neither subscripts nor accumulating stores, and hands the same object to every backend (Chapter 53). The CPU backend then renders that object as JavaScript, where the only real problem is that the language has no `f32` and a program that declares one expects it rounded after every operation, which costs an explicit rounding per node and buys back exactly one at the store (Chapter 54). The WebAssembly backend renders it as a stack machine with structured control flow over one flat linear memory, and then — because a browser accepts bytes and not text — assembles the module itself, LEB128 and section headers and all, with SIMD as an explicit transformation that is bit-identical on elementwise work and a reassociation on reductions (Chapter 55). The CUDA backend does something different in kind: it deletes the outer loops, because on a GPU they are the index space rather than a sequence, which means it must decide a launch geometry and must decide whether the program is safe to run all at once — and when a value crosses a block boundary it cannot repair it, so it sets the launch to one thread and produces a correct, catastrophic kernel (Chapter 56). WebGPU is the same execution model under a language with eight binding slots, no dynamic allocation and no scalar kernel arguments, so its backend spends most of its length placing things — buffers into binding groups, workgroup arrays into a pool, private arrays into reused slots, and thread-local arrays into single scalars — which is Part IX's packing problem solved three more times inside one kernel, bringing the count for one compilation to five (Chapter 57). And for the one operation the compiler will not win, there is a mechanism for emitting *no code at all*: a provider claims a function, an attribute carries the claim down the pipeline, and codegen produces a descriptor instead of a kernel — at the cost of a fusion barrier and a round trip through memory (Chapter 58).

## What Part X establishes for later parts

Part XI takes what this part produces. A `CompiledKernel` is a name, a source string, a target and a metadata object, and the runtime's whole job is against that pair: compile or assemble the source, allocate against the metadata, launch with the geometry it names. The metadata's shape is target-specific and Chapter 58 §58.6 lists it.

Part XII has to test it, and the thing it most needs from here is the list of places the four backends are *not* bit-identical. There are six, each named in the chapter that found it, and §58.6 collects them and sorts them by the *kind* of difference — because two of the six are reassociations no fixed tolerance covers, and one is a wrong value a tolerance cannot see at all.

## Labs

```bash
npm run build   # once, if you have not already

node docs/part10/ch53-lir-the-third-ir/labs/01-what-lir-throws-away.mjs
node docs/part10/ch53-lir-the-third-ir/labs/02-one-lir-four-texts.mjs
node docs/part10/ch54-javascript-for-the-cpu/labs/01-what-the-javascript-looks-like.mjs
node docs/part10/ch54-javascript-for-the-cpu/labs/02-the-shape-of-a-float-and-the-name-of-a-function.mjs
node docs/part10/ch55-webassembly/labs/01-a-stack-machine-and-a-binary.mjs
node docs/part10/ch55-webassembly/labs/02-four-lanes-at-a-time.mjs
node docs/part10/ch56-cuda/labs/01-the-loops-that-do-not-run.mjs
node docs/part10/ch56-cuda/labs/02-when-the-backend-refuses.mjs
node docs/part10/ch57-webgpu-and-wgsl/labs/01-eight-bindings-and-a-browser.mjs
node docs/part10/ch58-someone-elses-kernel/labs/01-a-kernel-with-no-source.mjs
```

Ten labs, and **none of them needs a GPU**. That is the useful property of a part about code generation: a backend's output is a text and a metadata object, and both can be produced, read and asserted on any machine. The CPU and WebAssembly labs also *run* what they emit — the WASM ones assemble the module with the compiler's own encoder and instantiate it with `WebAssembly.instantiate` — so the numbers they report are real. The CUDA and WebGPU labs read the emitted source and the launch metadata and stop there, which is exactly the boundary Chapters 56 §56.7 and 57 §57.6 draw.

The labs reach past the public surface, so they read the internal modules listed in [`docs/tools/internals-entry.ts`](../tools/internals-entry.ts) through [`_internals.mjs`](_internals.mjs), the same way Parts VII and VIII do: `npm run build` emits them as `dist/internals.node.js` beside the public bundle, and the labs refuse to run against a build older than `src/`.

## A note on what this part found

Part IX's findings were about the gap between a plan and a program. Part X's are about a different gap: **between what a mechanism was written to do and what the pipeline actually feeds it.** Four of the six chapters found the same shape.

The CUDA backend has a complete 256-thread shared-memory tree reduction that no compilation can reach, because Chapter 53's accumulator lowering removes the pattern it matches — so it fires on the path the unit tests take and never on the path a user takes (Chapter 56 §56.6). The WASM assembler silently drops an instruction it does not recognise and silently redirects a branch to a label it cannot find, both producing modules that *validate* (Chapter 55 §55.5). The WebGPU backend reports zero workgroup memory for a kernel that declares 8,228 bytes of it, because the figure counts only the buffers it did not itself allocate (Chapter 57 §57.5). And a CPU kernel whose model class is named `Math` emits `function Math(...)` and then calls `Math.tanh` inside it, which throws (Chapter 54 §54.6).

The rest are about semantics rather than reachability, and they share a cause: **an operation that reaches a backend as a *name* rather than as a node kind gets whatever that backend's table says.** `remainder` is floor-modulo in eager and on CPU and truncating on the other three (Chapter 53 §53.6); `erf` and its siblings come from one shared approximation on three backends and from the device math library on CUDA (Chapter 54 §54.3); `bf16` is two bytes everywhere except WGSL, where it is silently widened to four (Chapter 54 §54.3, Chapter 57 §57.7). And one is about a licence rather than a table: a vectorised reduction on WASM reassociates the sum, reachable from `scheduling: { enabled: true }` with no numerical licence asked for and nothing in the trace to say the summation order changed (Chapter 55 §55.6).

Each is carried into the outline's [Appendix E](../OUTLINE.md).

## A caution about this part's numbers

Every figure in these chapters is a property of *emitted text or metadata* unless a lab says it ran something. Line counts, instruction counts, binding counts, launch geometries, byte offsets and shared-memory figures are all read off the compiler's own output and are deterministic — they should reproduce exactly. The numbers that came from *running* a kernel are the CPU and WebAssembly ones, and they are marked as such where they appear.

**There are no timings anywhere in this part, and that is a choice with a cost.** The reasoning for it: a backend's job is to emit a correct program of a given *shape*, and whether that shape is fast was Parts VII and VIII's question, measured there against a cost model and a benchmark runner. A wall-clock number quoted here would attribute to code generation a result that mostly belongs to scheduling — and on three of the four targets it would also be a number from one machine, one driver and one browser, which [Chapter 48](../part8/ch48-reproducibility/README.md) spends a chapter explaining how to distrust.

The cost is real and worth stating plainly. Part I's method is measure first, explain second, and this part does not practise it: Chapter 55 argues that SIMD is worth having without saying what it is worth, and Chapter 58 argues that a library kernel is worth a fusion barrier without weighing the two against each other. Where a *ratio* would have settled a question, the chapters give a countable proxy instead — kernels launched, threads requested, instructions emitted, bytes allocated — and say so. A reader who wants the missing half should take a program from any of these labs to [Chapter 47](../part8/ch47-search-and-measurement/README.md)'s benchmark runner, which is where the apparatus for measuring it honestly already lives.

---

**Next:** [Chapter 53 — LIR: why a third IR](ch53-lir-the-third-ir/README.md).
