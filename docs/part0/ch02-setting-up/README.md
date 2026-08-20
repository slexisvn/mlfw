# Chapter 2 — Setting up

By the end of this chapter you will have run three programs. The first prints a compiler's intermediate representation. The second measures what compilation buys — including a case where it buys almost nothing. The third prints the machine code, or rather the JavaScript, that the compiler wrote.

Everything here takes about ten minutes.

## 2.1 What you need

| Tool | Version used for this book | Notes |
|---|---|---|
| Node.js | v22.19.0 | The bundle targets `node18`, so Node 18 or newer should work; everything here was verified on 22.19 |
| npm | 10.9.3 | Ships with Node |

No C++ compiler, no CUDA toolkit, no Python. The compiler is TypeScript, the default backend generates JavaScript, and the WebAssembly backend assembles its own binaries in process.

Two optional extras unlock two chapters much later in the book, and nothing before them:

- **An NVIDIA GPU with the CUDA runtime** — for the CUDA backend (Chapter 56) and its tests.
- **A Chrome or Chromium binary** — for the WebGPU backend (Chapter 57), which is tested by driving a real browser.

Without them, the corresponding tests skip themselves rather than fail. You will see this in §2.3.

## 2.2 Installing and building

From the root of a checkout of the repository:

```bash
npm install
```

Then build the distributable bundle. The labs import from it, so this step is not optional:

```bash
npm run build
```

The build prints something close to this:

```
ESM dist\index.browser.js 1.02 MB
ESM ⚡️ Build success in 4982ms
ESM dist\index.node.js 1.08 MB
ESM ⚡️ Build success in 5058ms
DTS ⚡️ Build success in 31348ms
DTS dist\index.d.ts 166.52 KB
```

Two bundles are produced because the framework has two runtime environments — Node and the browser — that differ in how they read files, spawn workers and reach the GPU. Chapter 59 explains the mechanism. For now, note only that the labs use `dist/index.node.js`.

> **Important.** `dist/` is a build artifact and can be stale. After you edit anything under `src/`, run `npm run build` again before running a lab, or you will be reading the behaviour of the code as it was, not as it is. This trips up everyone at least once.

## 2.3 Checking that it works

The framework's test suite is split into projects, because different parts need different hardware:

```bash
npm run test:unit
```

```
Test Files  287 passed (287)
      Tests  4185 passed (4185)
```

```bash
npm run test:e2e
```

The end-to-end project drives complete models through compilation and execution, and compares against eager execution. Together the two projects are 302 files and 5,131 tests, and they take about a minute on a laptop.

The remaining projects are hardware-gated:

| Command | Requires | If unavailable |
|---|---|---|
| `npm run test:cuda` | NVIDIA GPU + CUDA runtime | Test blocks skip |
| `npm run test:webgpu` | Chrome/Chromium | Test blocks skip |
| `npm run test:stress` | Time and memory | Slow; not part of the default check |
| `npm run test:perf` | Time | Asserts the *shape* of scaling, not absolute timings; Chapter 4 points you at it |

Prefer `npm run test:unit` and `npm run test:e2e` as your everyday check. Plain `npm test` also includes the WebGPU project, which will report failures on a machine with no browser available — a fact about your machine, not about the code.

*All test counts measured 2026-08-19.*

## 2.4 Lab 1 — Your first look at an IR

Run:

```bash
node docs/part0/ch02-setting-up/labs/01-first-look.mjs
```

The whole program is eight lines ([`labs/01-first-look.mjs`](labs/01-first-look.mjs)):

```js
import {
  tensor, Linear, ReLU, Sequential, trace, printModule, manual_seed,
} from '../../../../dist/index.node.js';

manual_seed(0);

const model = new Sequential(new Linear(2, 8), new ReLU(), new Linear(8, 1));
const x = tensor([[0.5, -1.5], [1.0, 2.0]]);

const graph = await trace((t) => model.forward(t), [x]);
console.log(printModule(graph));
```

`trace` runs the model with symbolic tensors instead of real ones: nothing is computed, but every operation the model performs is recorded. `printModule` renders the recording as text. Chapter 61 explains how tracing works; today we only want to look at the result.

```
module @traced {
  func @traced(%0: tensor<2x2xf32>, %1: tensor<8x2xf32>, %2: tensor<8xf32>, %3: tensor<1x8xf32>, %4: tensor<1xf32>) -> (tensor<2x1xf32>) {
    %5 = transpose(%1) {permutation = [1, 0]} : tensor<2x8xf32>
    %6 = dot(%0, %5) {lhs_batch = [], lhs_contracting = [1], rhs_batch = [], rhs_contracting = [0]} : tensor<2x8xf32>
    %7 = add(%6, %2) : tensor<2x8xf32>
    %8 = constant() {tensor_type = tensor<xf32>, value = 0} : tensor<xf32>
    %9 = broadcast_in_dim(%8) {broadcast_dimensions = [], result_shape = [2, 8]} : tensor<2x8xf32>
    %10 = maximum(%7, %9) : tensor<2x8xf32>
    %11 = transpose(%3) {permutation = [1, 0]} : tensor<8x1xf32>
    %12 = dot(%10, %11) {lhs_batch = [], lhs_contracting = [1], rhs_batch = [], rhs_contracting = [0]} : tensor<2x1xf32>
    %13 = add(%12, %4) : tensor<2x1xf32>
    return(%13)
  }
}
```

Read it line by line; the notation is not hard.

- `%0` … `%13` are **values**. Each is produced by exactly one line and never reassigned. `%0` is the input `x`; `%1` … `%4` are the two weight matrices and two bias vectors, which the trace turned from object properties into function parameters.
- `tensor<2x8xf32>` is a **type**: shape 2 × 8, element type 32-bit float. Every value carries one, and the compiler checks them.
- `transpose(%1) {permutation = [1, 0]}` is an **operation**: a name, operands in parentheses, attributes in braces, result type after the colon. Attributes are compile-time constants — `[1, 0]` is not data flowing through the program, it is part of what the operation *is*.
- `dot(%0, %5) {lhs_contracting = [1], rhs_contracting = [0]}` is matrix multiplication, stated generally: contract dimension 1 of the left operand against dimension 0 of the right. Chapter 11 explains why the general form is preferable to a dozen special cases.
- `%8` and `%9` are the ReLU's zero: a scalar constant, broadcast to 2 × 8, then `maximum(%7, %9)`. `ReLU` as an object no longer exists — it became arithmetic.

**Things to notice.**

Nothing here mentions layers, modules, or classes. The abstractions you programmed with have been dissolved into a flat list of tensor operations. That dissolution is not a loss — it is exactly what makes optimization possible, because an optimizer can now see across the boundaries that the object structure used to hide.

Also: order. Line `%6` uses `%5`, which is defined above it. Every use comes after its definition. Chapter 8 makes this precise and, perhaps surprisingly, shows that the *textual* order carries no meaning at all — only the use-def edges do.

**Try this.** Change the model to `new Sequential(new Linear(2, 8), new Sigmoid(), new Linear(8, 1))` and run again. `maximum` and its broadcast constant vanish, replaced by a single `sigmoid` operation. Ask yourself why ReLU is expressed as arithmetic while sigmoid gets its own operation — Chapter 21 is about exactly that choice.

## 2.5 Lab 2 — Measuring what compilation buys

```bash
node docs/part0/ch02-setting-up/labs/02-measure-the-gap.mjs
```

The lab ([`labs/02-measure-the-gap.mjs`](labs/02-measure-the-gap.mjs)) builds a chain of twelve elementwise operations:

```js
class Chain extends Module {
  forward(t) {
    return t.mul(2).add(1).relu().mul(0.5).add(3).tanh()
            .mul(1.5).add(0.25).relu().mul(0.8).add(2).tanh();
  }
}
```

and runs it two ways at three sizes: eagerly, one operation at a time, and compiled. Output from the machine this book was written on:

```
twelve operations, two `tanh` among them

  size      eager    compiled   ratio   max abs diff
    16       59.1 us     23.0 us    2.57x        6.0e-8
   128     1571.3 us   1154.4 us    1.36x        6.0e-8
   512    19794.2 us  18243.8 us    1.08x        6.0e-8
```

Two columns matter.

**`max abs diff` is 6 × 10⁻⁸ everywhere.** That number is not arbitrary: 2⁻²⁴ ≈ 5.96 × 10⁻⁸ is exactly one unit in the last place for a 32-bit float of magnitude between 0.5 and 1, which is where the final `tanh` leaves most of these values. The two programs disagree by the smallest amount a `float32` can express there, and no more. The compiled program computes the same thing as the eager one. It is not the *bitwise* same, and Chapter 20 explains why: the compiler is allowed to rearrange arithmetic in ways that change rounding, and it is decidedly not allowed to rearrange it in ways that change the answer. Learning which is which is a chapter of its own.

**The ratio falls as the tensors grow**, from 2.57× down to 1.08×. That fall is the most instructive number in this chapter, so it is worth being clear about what causes it.

At 16 × 16, each operation touches 256 numbers — a few microseconds of arithmetic at most. What dominates is everything *around* the arithmetic: allocating an output tensor, dispatching to the right kernel, checking shapes and dtypes, running the loop bookkeeping. Twelve operations pay that cost twelve times. The compiled version pays it once, because the twelve operations became one function over one set of buffers.

At 512 × 512, each operation touches 262,144 numbers, and the fixed per-call cost has become negligible. What remains is the arithmetic itself — and the compiler cannot make arithmetic disappear. It merges the twelve operations into one loop, so the eleven intermediate tensors are never written out at all (that merging is called *fusion*, and the kernel below shows it), but every element still has to go through the same twelve computations. Two of those twelve are `tanh`, which on this machine costs about ten times what an `add` costs. Removing the memory traffic around an expensive computation does not make the computation cheaper.

That last sentence is the whole of Chapter 4, and the lab's second half makes it concrete. Replace the two `tanh` calls with two multiplications — same twelve operations, same shapes, same compiler — and measure again:

```
the same twelve operations with the two `tanh` replaced by multiplications

  size      eager    compiled   ratio   max abs diff
    16       44.4 us      3.1 us   14.54x        1.9e-6
   128      830.3 us    130.2 us    6.38x        1.9e-6
   512     8057.6 us   1764.5 us    4.57x        1.9e-6
```

**1.08× became 4.57×, and 2.57× became 14.54×, because of two function calls.** Nothing else changed. Chapter 4 builds the cost model that predicts both tables and explains why the second one's speedup also falls with size, but from a much higher starting point.

(The larger `max abs diff` in the second table — 1.9 × 10⁻⁶ rather than 6 × 10⁻⁸ — is not a bug either. That chain ends in a multiplication rather than a `tanh`, so its outputs are much larger than 1, and one ulp at a larger magnitude is a larger absolute number. Relative error is what is comparable across the two tables, and it is unchanged.)

If you want to see exactly what replaced the twelve operations, add one line to the lab — `console.log(compiled.source())` — and the whole kernel appears. At the first size, 16 × 16:

```js
function Chain(buf_1, buf_3) {
  for (let i0_12 = 0; i0_12 < 16; i0_12++) {
    for (let i1_13 = 0; i1_13 < 16; i1_13++) {
      buf_3[((i0_12 * 16) + i1_13)] = Math.tanh(((Math.max(((Math.tanh(((Math.max(((buf_1[((i0_12 * 16) + i1_13)] * 2) + 1), 0) * 0.5) + 3)) * 1.5) + 0.25), 0) * 0.800000011920929) + 2));
    }
  }
}
```

Twelve operations, one loop nest, one expression, and — count them — zero intermediate buffers. Each element of the input is read once, carried through the entire chain as a single expression that is never materialized to memory in between, and written once. That is the whole of what fusion bought, made visible.

One detail rewards a second look: the constant `0.8` was emitted as `0.800000011920929`. That is `0.8` rounded to the nearest 32-bit float and printed back at full double precision. The compiler is being careful to compute what a `float32` program would compute, rather than what a `float64` program would. Chapter 20 is about exactly this kind of care.

> **An 8% win is a compiler barely earning its keep, and the book will show you more cases like it.** An optimization only pays when it removes what the program is actually spending its time on. Compilers earn their keep where per-operation overhead dominates (many small operations, as at size 16), where fusion removes traffic that is genuinely the bottleneck (the second table, where the same chain reaches 4.57× once the expensive arithmetic is taken out), where a target needs code that does not otherwise exist (Chapters 55–57 generate WebAssembly, CUDA and WGSL from this same program), and where the search space is too large for a human to explore (Part VIII). Reaching for a compiler when none of those apply is how people end up disappointed by compilers.

> **A word on benchmarking.** The lab discards the first 200 runs before measuring, then reports a *median* of repeated batches rather than a mean. Both matter. JavaScript engines optimize code as they observe it running, so early runs measure the wrong thing; and a mean is hostage to a single unlucky garbage collection. Numbers you produce will differ from the ones above — different machine, different Node version. What should reproduce is the *shape*: a large ratio when tensors are small, shrinking as they grow.

## 2.6 Lab 3 — Reading the generated code

```bash
node docs/part0/ch02-setting-up/labs/03-see-the-kernel.mjs
```

This compiles the two-layer network from Chapter 1 and prints what came out, via `compiled.source()`:

```js
function Sequential(buf_1, buf_3, buf_5, buf_7, buf_9, buf_11) {
  const buf_24 = new Float32Array(16);
  const buf_13 = new Float32Array(16);
  const buf_29 = new Float32Array(2);
  for (let di0_20 = 0; di0_20 < 2; di0_20++) {
    for (let di1_22 = 0; di1_22 < 8; di1_22++) {
      buf_13[((di0_20 * 8) + di1_22)] = 0;
    }
  }
  for (let ls0_14 = 0; ls0_14 < 2; ls0_14++) {
    for (let rs0_15 = 0; rs0_15 < 8; rs0_15++) {
      let _acc_0 = buf_13[((ls0_14 * 8) + rs0_15)];
      for (let c0_16 = 0; c0_16 < 2; c0_16++) {
        _acc_0 = (_acc_0 + (buf_1[((ls0_14 * 2) + c0_16)] * buf_3[((rs0_15 * 2) + c0_16)]));
      }
      buf_13[((ls0_14 * 8) + rs0_15)] = _acc_0;
    }
  }
  for (let i0_25 = 0; i0_25 < 2; i0_25++) {
    for (let i1_26 = 0; i1_26 < 8; i1_26++) {
      buf_24[((i0_25 * 8) + i1_26)] = Math.max((buf_13[((i0_25 * 8) + i1_26)] + buf_5[i1_26]), 0);
    }
  }
  ...
}
```

Four observations, each of which is a later chapter:

1. **Tensors became flat arrays with computed offsets.** `buf_13[(ls0_14 * 8) + rs0_15]` is the two-dimensional index `[i][j]` flattened by hand. That flattening happens in the third and lowest intermediate representation — Chapter 53.

2. **The transpose disappeared.** The IR in Lab 1 contained `transpose(%1)`. There is no transpose loop here. Instead the weight is read as `buf_3[(rs0_15 * 2) + c0_16]` — row and column swapped at the point of use. An entire pass over memory was removed by rewriting an index expression. This happens through a *canonicalization pattern* attached to the `dot` operation, and Chapter 17 covers that mechanism; Chapter 3 shows the exact moment it fires.

3. **`add` and `maximum` share a loop.** In the IR they were two operations; here they are one expression, `Math.max(buf_13[...] + buf_5[i1_26], 0)`, inside one loop nest. This is *fusion*, the single most valuable optimization in this domain, and Chapters 22–24 are devoted to it.

4. **Buffers are allocated once, up front, and zeroed where an accumulation needs it.** All three temporaries appear at the top of the function; `buf_13` is then zeroed by an explicit loop because the matrix multiply accumulates into it. Nothing is allocated inside a loop. Deciding how few buffers a program can get away with — and which ones may share storage — is memory planning, Part IX.

**Try this.** Recompile with fusion switched off:

```js
const compiled = compile(model, [x], { target: CPUTarget(), fusion: { enabled: false } });
```

The single loop nest splits in two, with a temporary buffer between them:

```js
  for (let i0_25 = 0; i0_25 < 2; i0_25++) {
    for (let i1_26 = 0; i1_26 < 8; i1_26++) {
      buf_24[((i0_25 * 8) + i1_26)] = (buf_13[((i0_25 * 8) + i1_26)] + buf_5[i1_26]);
    }
  }
  for (let i0_30 = 0; i0_30 < 2; i0_30++) {
    for (let i1_31 = 0; i1_31 < 8; i1_31++) {
      buf_29[((i0_30 * 8) + i1_31)] = Math.max(buf_24[((i0_30 * 8) + i1_31)], 0);
    }
  }
```

Count the memory traffic. The fused version reads `buf_13`, reads the bias, writes `buf_24` — one pass. The unfused version reads `buf_13`, writes `buf_24`, then reads `buf_24` back and writes `buf_29` — two passes plus an extra buffer that exists only to carry values from the first loop to the second. On a 2 × 8 tensor this is invisible. On a transformer activation it is the difference between a model that fits in cache and one that does not. Everything in Part IV exists to produce the first version rather than the second.

## 2.7 When something goes wrong

| Symptom | Cause | Fix |
|---|---|---|
| `Cannot find module '../../../../dist/index.node.js'` | Bundle not built | `npm run build` |
| A lab prints stale behaviour after you edited `src/` | `dist/` is a snapshot | `npm run build` again |
| `npm test` reports WebGPU failures | No Chrome available | Use `npm run test:unit` and `npm run test:e2e` |
| CUDA tests all skip | No GPU or CUDA runtime | Expected; nothing before Chapter 56 needs them |
| `ERR_UNSUPPORTED_ESM_URL_SCHEME` on Windows | Absolute path in an `import` | Use a relative path, as the labs do |

## 2.8 Read the tests

Lab 2 checked by hand that compiled output matches eager output. The test suite checks the same claim exhaustively, and those tests are worth opening now, because much of the book is an explanation of how the compiler manages to keep them passing:

- [`tests/e2e/differential.test.js`](../../../tests/e2e/differential.test.js) — compiled versus eager across a wide matrix of operations, on both the CPU and WebAssembly backends.
- [`tests/e2e/compile-contract.test.js`](../../../tests/e2e/compile-contract.test.js) — compiling a model and calling it behaves like calling the model.
- [`tests/e2e/differential-backward.test.js`](../../../tests/e2e/differential-backward.test.js) — compiled gradients versus eager autograd, and versus numerical finite differences as an independent oracle.

## 2.9 What to keep open

While reading, keep three things within reach:

- **`printModule`** — for any model, at any point, print the graph and look at it.
- **`compiled.source()`** — for any compiled model, read the generated code.
- **The test suite** — when the book says the compiler behaves a certain way, `tests/` contains the executable proof.

Between them, nothing in this book has to be taken on faith.

---

**Next:** [Chapter 3 — A map of the codebase](../ch03-map-of-the-codebase/README.md), which is the map you will return to whenever you lose your bearings.
