# Chapter 54 — Generating JavaScript for the CPU

The CPU backend has the strangest job in the compiler: its target language is the language the compiler is written in. It emits JavaScript source text, hands it to `new Function`, and lets the host's own optimizing JIT do the register allocation and the instruction selection. That sounds like a shortcut. It is, and understanding exactly *which* problems it shortcuts — and which it makes harder — is the content of this chapter.

## 54.1 The problem: the fastest loop a JIT will accept

A modern JavaScript engine compiles hot loops to machine code that is close to what a C compiler produces, on one condition: the loop must be *type-stable*. Every variable has to hold one kind of value on every iteration, every array access has to be to a typed array of known element type, and nothing in the loop may allocate. Give the engine that, and the arithmetic runs at native speed. Break any of it and the engine falls back to a generic path that is slower by an order of magnitude or more.

So this backend's brief is narrow and precise: produce a loop nest that the JIT will specialize. Concretely that means integer loop variables, `Float32Array`/`Int32Array` accesses at integer offsets, scalar locals, and no object allocation anywhere in the body. LIR hands it exactly that shape, so most of the backend is a direct rendering.

The one thing LIR does *not* hand it is a floating-point type. A JavaScript number is an IEEE 754 double. A program that declares `f32` and multiplies two `f32` values expects the product rounded to `f32`; JavaScript will give it the product rounded to `f64`. That gap is the chapter's real subject, because it is where a backend can be silently, slightly wrong.

## 54.2 Intuition: writing a program that writes a program

Everything here is string concatenation with an indent counter. `_emit(line)` appends `'  '.repeat(indent) + line` to an array of lines ([`codegen_base.ts:23`](../../../src/backend/codegen_base.ts)); a statement visitor walks the LIR and calls `_emit`; an expression walker turns a node tree into one string. At the end the lines are joined and the result is a program.

The awkward part is that a *statement* and an *expression* need different treatment. A statement produces lines; an expression produces a fragment that must be embedded in a line. So the backend has two walkers, and the boundary between them is exactly the boundary between "things that can contain a loop" and "things that cannot".

The second awkward part is that the emitter cannot see what it has written. A peephole optimization at this level is a rewrite over *text* — and this backend has one, which is both the pragmatic choice and the place a maintainer should look first when something odd appears in the output.

## 54.3 Theory

### The rounding contract

> **Definition 54.1 (Per-operation rounding).** **(stated here)** A backend *rounds per operation* for dtype τ when, for every arithmetic node in the program, the value it produces is the τ-rounding of the exact result of applying that operator to its (already τ-rounded) operands.

Per-operation rounding is what every hardware float unit does, and therefore what the eager operations, the CUDA backend, the WASM backend and the WGSL backend all do without being asked. It is also what [Definition 1.4](../../part0/ch01-what-this-book-is/README.md)'s level **N0** — bit-identical — means for a program in `f32`.

JavaScript offers `Math.fround(x)`, which is the `f32` rounding of a double, and a `Float32Array` store, which rounds on the way in. So a naive emitter that computed the whole expression in doubles and stored it once would round *once*, at the end.

> **Theorem 54.2 (One rounding is not per-operation rounding).** **(classical)** There exist `f32` values *a*, *b*, *c* for which fl₃₂(fl₃₂(a·b) + c) ≠ fl₃₂(a·b + c), where the inner product on the right is computed exactly.
>
> *Proof.* Take a = b = 1 + 2⁻¹². Then a·b = 1 + 2⁻¹¹ + 2⁻²⁴ exactly. At this magnitude the `f32` ulp is 2⁻²³, so 2⁻²⁴ is exactly half an ulp; round-to-nearest-even sends fl₃₂(a·b) to 1 + 2⁻¹¹, whose last bit is zero. Now take c = 2⁻²⁴. The left side is fl₃₂(1 + 2⁻¹¹ + 2⁻²⁴), another half-ulp tie, which rounds back to 1 + 2⁻¹¹. The right side is fl₃₂(1 + 2⁻¹¹ + 2⁻²³), which is exactly representable and is *not* 1 + 2⁻¹¹. The two differ by one ulp. ∎

So the emitter has to insert the roundings. `_roundToDtype` ([`cpu/codegen.ts:172`](../../../src/backend/cpu/codegen.ts)) wraps every `f32` arithmetic result in `Math.fround`. That is correct and it is not free: an *n*-operation expression carries *n* extra calls.

One of them can be removed, and exactly one.

> **Proposition 54.3 (The outermost rounding before a store is redundant).** **(stated here)** If an expression *e* of dtype `f32` is the value of a store into a `Float32Array`, then `Math.fround(e)` and `e` store the same bits.
>
> *Proof.* A `Float32Array` store applies the `f32` rounding to its argument. `Math.fround` is idempotent — it is that same rounding — so rounding before a store that rounds changes nothing. ∎

Proposition 54.3 is applied by `_stripOuterRound` ([`cpu/codegen.ts:176`](../../../src/backend/cpu/codegen.ts)), which scans the rendered string for a `Math.fround(` prefix whose matching parenthesis is the last character. The *outermost* one only: an inner `Math.fround` sits under an operator that will consume its value in double arithmetic, and removing it would be Theorem 54.2 happening.

### Exactness

There is a second class of expression that needs no rounding at all.

> **Definition 54.4 (Exact in τ).** **(stated here)** A function *g* is *exact in* τ when, for arguments already representable in τ, the exact value of *g* is itself representable in τ.

`max`, `min`, `abs`, `floor`, `ceil`, `round` and `sign` are exact in `f32` because each returns one of its arguments or an integer-valued neighbour of one. `sqrt` is exact in the weaker but sufficient sense that IEEE 754 requires it to be correctly rounded, and `Math.sqrt` of a value that came from an `f32` is the `f64` correctly-rounded square root, whose `f32` rounding is the `f32` correctly-rounded square root. `exp`, `log`, `tanh`, `pow` and the rest are none of these: `Math.tanh` is a `f64` approximation with its own error, and narrowing it needs an explicit rounding to be the `f32` answer.

The ninth name is `fmod`, exact for a reason the other eight do not need: a remainder is `a − n·b` for an integer *n*, and that subtraction is error-free, so the result is a value the operands' own format already holds. It is also the one member whose emitted form is not a single operation — the backend spells it as the same floor correction the `%` row of §54.3's table uses, three operations evaluated in doubles and narrowed once at the store.

The backend keeps Definition 54.4's answer as that literal set, `EXACT_EXTERNS` ([`cpu/codegen.ts:47`](../../../src/backend/cpu/codegen.ts)) — nine names, and membership in it is the whole difference between `Math.max(x, 0)` and `Math.fround(Math.tanh(x))` in the emitted source.

### Storage width and compute width

`f32` is not the only width the target language lacks. Half precision — `f16` and `bf16` — is a *storage* format on most hardware and a compute format on almost none, and JavaScript has neither. So the compiler splits the two questions.

> **Definition 54.5 (Storage width and compute width).** **(invariant)** Every buffer has a *storage width*, fixed by its dtype and used to size its allocation. Every value in a local has a *compute width*, given by `normalizeDtype` ([`lir/nodes.ts:174`](../../../src/compiler/ir/lir/nodes.ts)), which maps `f16` and `bf16` to `f32` and every integer narrower than 32 bits to `i32`. A load widens from storage to compute; a store narrows back.

The split costs something on two backends and nothing on the other two, for reasons that are entirely about the target language.

**On the CPU backend**, an `f16` buffer is a `Uint16Array` and the widening is a function call. `_wrapLoad` and `_wrapStoreVal` ([`cpu/codegen.ts:156`](../../../src/backend/cpu/codegen.ts) and [`:161`](../../../src/backend/cpu/codegen.ts)) wrap every access in `__mlfw_f16_to_f32` / `__mlfw_f32_to_f16` — so half precision on this backend costs *more* arithmetic than `f32`, not less, and buys only memory.

**On the WASM backend** there is no such call to make, so the codec is emitted inline as instructions: `i32.load16_u`, then a shift for `bf16` or a mask-shift-`reinterpret`-multiply sequence for `f16` ([`wasm/codegen.ts:244`](../../../src/backend/wasm/codegen.ts)), through a pair of scratch locals that `_ensureHalfScratch` declares only when the function needs them.

**On CUDA and WebGPU the type is real**, and the split disappears: `f16` is `__half` in the emitted C and `f16` in WGSL behind an `enable f16;` directive. Nothing is widened.

Which leaves one asymmetry worth naming. `bf16` is two bytes in the dtype table every other layer reads, and four in the WGSL table ([`dtype_map.ts:252`](../../../src/util/dtype_map.ts)), where it is mapped to `f32` because WGSL has no `bf16` at all. A `bf16` buffer therefore has a different size on WebGPU than the memory planner of [Chapter 50](../../part9/ch50-arena-allocation/README.md) assigned it — the widening is silent, and it is the one place in Part X where a dtype changes the *size* of a buffer rather than the width it computes in.

### Division

Definition 53.8 fixed the four integer operators. In JavaScript:

| LIR | Emitted | Why |
|---|---|---|
| `a // b` | `Math.floor(a / b)` | `/` is float division; `Math.floor` is the definition |
| `a % b` | `((a % b + b) % b)` | JavaScript's `%` truncates; this is the floor correction |
| `a tdiv b` | `((a / b) \| 0)` | `\|0` is ToInt32, which truncates toward zero |
| `a tmod b` | `(a % b)` | JavaScript's `%` already truncates |

The `|0` is worth a note: ToInt32 truncates *and* wraps to 32 bits, so `tdiv` is correct only while the quotient fits in an `i32`. For index arithmetic — which is what `tdiv` exists for, and which the simplifier only introduces where it has proved the operands non-negative — that is the intended domain.

## 54.4 In mlfw: 651 lines, two walkers and a text pass

[`src/backend/cpu/codegen.ts`](../../../src/backend/cpu/codegen.ts) is one class, `CPUCodegen`, with a `generate(func)` entry ([`cpu/codegen.ts:65`](../../../src/backend/cpu/codegen.ts)) that runs in four phases.

### 1. Decide the signature and the declarations

Parameters are the buffers in `bufferMap`, in order, followed by the shape parameters. Then, from the LIR metadata:

```ts
    for (const [bufName, buf] of usedBuffers) {
      if (zeroBuffers.has(bufName)) continue;
      if (constantBuffers.has(bufName)) continue;
      if (!paramBuffers.has(bufName) && !allocatedBuffers.has(bufName)) {
        const numel = buf.numel();
        if (numel > 0) {
          this._emit(`const ${bufName} = ${this._allocRhs(buf, numel)};`);
```

Three exclusions before anything is allocated. A buffer whose every store is the literal zero is skipped — loads of it will be folded to `0`. A buffer written one constant is skipped — loads of it become that constant. A buffer that is a parameter arrives from the caller, and one that has an `AllocateNode` is declared where the node is.

`_allocRhs` ([`cpu/codegen.ts:148`](../../../src/backend/cpu/codegen.ts)) is where Part IX arrives:

```ts
  _allocRhs(buf: Buffer, numel: number): string {
    const ty = jsTypedArray(buf.dtype);
    if (buf.poolByteOffset !== undefined && numel > 0) {
      return `new ${ty}(_mem_pool, ${buf.poolByteOffset}, ${numel})`;
    }
    return `new ${ty}(${numel})`;
  }
```

Under pool allocation a temporary is a *view* onto one `ArrayBuffer` at the offset Chapter 50 assigned. The backend does not compute the offset and does not check it; it reads `buffer.poolByteOffset` and trusts it.

### 2. Walk the statements

`_visitNode` ([`cpu/codegen.ts:190`](../../../src/backend/cpu/codegen.ts)) is a loop rather than a recursion for the node kinds that have exactly one continuation — `SeqNode`, `AllocateNode`, `LetStmtNode` — so a deep sequence does not grow the JavaScript stack. Everything else dispatches and returns.

Two statement cases carry real decisions.

`_visitForNode` ([`cpu/codegen.ts:228`](../../../src/backend/cpu/codegen.ts)) handles three shapes before emitting an ordinary loop: an extent-1 loop becomes an alias of its variable to `0` and no loop at all; an `UNROLLED` loop with a constant extent of at most 32 is emitted as that many braced blocks; and a redundant zero fill is dropped entirely.

`_visitLIRAccumulator` ([`cpu/codegen.ts:307`](../../../src/backend/cpu/codegen.ts)) writes out Theorem 53.5's three steps:

```ts
    this._emit('let ' + accVar + ' = ' + initExpr + ';');
    ...
    if (accOp === 'max') accRhs = 'Math.max(' + accVar + ', ' + accBody + ')';
    else if (accOp === 'min') accRhs = 'Math.min(' + accVar + ', ' + accBody + ')';
    else accRhs = '(' + accVar + ' ' + accOp + ' ' + accBody + ')';
    this._emit(accVar + ' = ' + accRhs + ';');
```

A `let`, a loop, and a store — and the accumulator is a plain JavaScript number, which is exactly the shape the JIT will keep in a register.

### 3. Render the expressions

`_exprToJS` ([`cpu/codegen.ts:428`](../../../src/backend/cpu/codegen.ts)) is an explicit work stack rather than a recursion, for the same reason: a fully fused elementwise chain is one expression tree, and its depth is the length of the chain. Each frame carries a `phase` counter that says which child to visit next, and completed children accumulate on a `vals` stack.

The `MathOpNode` case is where the arithmetic decisions live ([`cpu/codegen.ts:474`](../../../src/backend/cpu/codegen.ts)) — the identity folds `x - 0`, `x * 1`, `1 * x`, and, *for integer dtypes only*, `x + 0` and `x * 0`; then the four division operators of §54.3; then `_roundToDtype` for everything else. The dtype guard on the zero identities matters and it is not fussiness: `∞ × 0` and `NaN × 0` are `NaN`, not `0`, so folding them on floats would make this backend disagree with the other three.

The `CallExternNode` case ([`cpu/codegen.ts:522`](../../../src/backend/cpu/codegen.ts)) consults that set first, and then falls through a chain of seven reachable special cases for the functions JavaScript does not have: `rsqrt`, `fmod`, `exp2`, `erf`, `erfc`, `lgamma`, `gamma`. The last four are emitted as *inlined immediately-invoked arrow functions* carrying a Lanczos or Abramowitz–Stegun approximation, whose coefficients come from [`util/special_math.ts`](../../../src/util/special_math.ts).

That file is shared, and it does not reach all four backends. The WASM runtime imports `erfScalar` and its siblings from it directly ([`backend_registry.ts:85`](../../../src/runtime/backend_registry.ts)), and the WebGPU backend inlines the same `ERF_A`, `ERF_P` and Lanczos constants into WGSL. **CUDA does not**: `cMathFunc` maps `erf` to `erff` and `gamma` to `tgammaf`, so on that target the value comes from the device math library. §54.6 measures what that costs: Abramowitz–Stegun 7.1.26 carries a worst absolute error of 1.394 × 10⁻⁷ over [0, 3.2], about 1.17 `f32` ulps at magnitude 1, while `erff` is documented to within a couple of ulps of the true value. Three backends agree to the last bit; the fourth is close and not equal, and §58.6 counts it.

### 4. Clean up the text

`_cleanupSource` ([`cpu/codegen.ts:559`](../../../src/backend/cpu/codegen.ts)) makes one pass over the finished lines and removes two things:

```ts
    for (let i = 0; i < lines.length; i++) {
      if (allocName[i] !== null && counts.get(allocName[i] as string) === 1) continue; // dead alloc
      const line = lines[i];
      if (/^\s*\}\s*$/.test(line) && out.length > 0 && /^\s*for\s*\(.*\{\s*$/.test(out[out.length - 1])) {
        out.pop();
        continue;
      }
      out.push(line);
    }
```

A `const b = new Float32Array(n);` whose name occurs exactly once in the whole source is a buffer nothing reads — delete it. A `}` immediately after a `for (…) {` is an empty loop — delete both. Both are textual, and both are here because the decision that makes them dead is taken elsewhere: the allocation becomes dead when the memory planner aliases the buffer away, and the loop becomes empty when a zero fill is elided.

## 54.5 Lab — what the JavaScript looks like

```bash
node docs/part10/ch54-javascript-for-the-cpu/labs/01-what-the-javascript-looks-like.mjs
```

```
  program              lines loops arrays accs fround
  one elementwise op       7     2      0    0      0
  a chain of five          7     2      0    0      3
  two inputs               7     2      0    0      1
  a reduction             12     3      0    1      0
  matmul then relu        22     7      1    1      1
```

The first three rows are the same seven lines: two loops and one store. Fusion (Chapter 24) already collapsed the chain into one expression, so five operations cost the same loop structure as one, and the only thing that grows is the number of roundings inside the statement. The reduction adds an init nest and an accumulator; the matmul adds one temporary — the product — that the `relu` then reads.

The chain of five, in full:

```js
function Object(buf_1, buf_3) {
  for (let i0_8 = 0; i0_8 < 4; i0_8++) {
    for (let i1_9 = 0; i1_9 < 8; i1_9++) {
      buf_3[((i0_8 * 8) + i1_9)] = Math.fround(Math.tanh(Math.max(Math.fround(Math.fround(buf_1[((i0_8 * 8) + i1_9)] * 2) + 1), 0))) * 0.5;
    }
  }
}
```

Read it from the inside out and every decision of §54.3 and §54.4 is visible. The two scalar constants `2` and `1` are literals rather than buffer loads. `Math.max(…, 0)` is `relu`, emitted bare because `max` is exact. `Math.tanh` is wrapped, because it is not. And the outermost `* 0.5` has no rounding at all — that is Proposition 54.3, and the store on the left is what makes it sound.

The reduction shows what Theorem 53.5 buys, in the loop where it matters:

```js
function Object(buf_1, buf_3) {
  for (let si0_5 = 0; si0_5 < 4; si0_5++) {
    buf_3[si0_5] = 0;
  }
  for (let sa0_7 = 0; sa0_7 < 4; sa0_7++) {
    let _acc_0 = buf_3[sa0_7];
    for (let r0_9 = 0; r0_9 < 8; r0_9++) {
      _acc_0 = (_acc_0 + buf_1[((sa0_7 * 8) + r0_9)]);
    }
    buf_3[sa0_7] = _acc_0;
  }
}
```

The inner loop touches memory once per iteration, not three times. Note also what the accumulator does *not* have: a `Math.fround`. The addition is `f32 + f32` in doubles, and the result is not narrowed until the store — so the running sum is carried at `f64` precision and rounded once. That is deliberate and it is a *departure* from Definition 54.1, in the direction of accuracy rather than away from it; §54.8 returns to it.

The third section is Chapter 50 arriving in the emitted text:

```
  poolAllocation=false  pool bytes=   0  declared= 1  of which read=1  distinct addresses used=n/a
  poolAllocation=true   pool bytes= 324  declared= 8  of which read=4  distinct addresses used=5
```

The five-operation chain, compiled with fusion off so the temporaries are real, has four scalar constants and four full-size temporaries. **On the default path one array survives**: the constants are folded into the expressions, and the four temporaries were aliased onto one another by Chapter 51's in-place reuse, so `_cleanupSource` deleted the three names nothing read. **With the pool on, eight views are declared and four are read** — the aliasing is expressed as a shared byte offset rather than a shared name, so all four temporary names survive pointing at offset 0, and the four constants are given arena slots the emitted program never touches.

That is the two-numbers discipline of [Part IX](../../part9/README.md) showing up at the codegen boundary: the arena reserves 324 bytes and the program reads 128 of them.

## 54.6 Lab — the shape of a float, and the name of a function

```bash
node docs/part10/ch54-javascript-for-the-cpu/labs/02-the-shape-of-a-float-and-the-name-of-a-function.mjs
```

Theorem 54.2, executed on its own witness:

```
  a = b = 1 + 2^-12 = 1.000244140625
  c       = 2^-24   = 5.960464477539063e-8

  a*b in f64, exactly            1.0004883408546448
  rounded to f32 once, at the end  fround(a*b + c) = 1.0004884004592896
  rounded after every operation    fround(fround(a*b) + c) = 1.00048828125
  they differ by 1.1920928955078125e-7, which is one f32 ulp at this magnitude

  eager                            1.00048828125
  compiled                         1.00048828125
  the compiled kernel rounds per operation: true
```

One ulp, on a two-operation program, on values a user could plausibly hold. The compiled kernel matches eager because the emitter inserted the rounding, and a backend that had not would have been off by an ulp here and by more on a long chain.

The cost, counted:

```
  1 multiplies ->  0 roundings
  2 multiplies ->  1 roundings
  ...
  6 multiplies ->  5 roundings
```

*n* operations, *n*−1 roundings — Proposition 54.3 recovering exactly one of them, at the store.

And Definition 54.4 in the output, with each extern placed one operation away from the store so the outer strip does not hide the answer:

```
  relu (Math.max)  Math.max(x, 0) + 1
  sqrt             Math.sqrt(Math.abs(x)) + 1
  tanh             Math.fround(Math.tanh(x)) + 1
  exp              Math.fround(Math.exp(x)) + 1
```

Then Definition 54.4's sibling question — which functions the *language* does not have — with the answer measured:

```
  eager       0.0508555584   0.520500004   0.842700660   0.995322168
  cpu         0.0508555584   0.520500004   0.842700660   0.995322168
  wasm        0.0508555584   0.520500004   0.842700660   0.995322168
  cuda      emits erff(buf_1[v1_7]);
  webgpu    emits ((select(-1.0, 1.0, buf_1[v1_7] >= 0.0)) * (1.0 - (1.0 / (1.0 + 0.3275

  the shared approximation is Abramowitz & Stegun 7.1.26; against a Taylor
  reference its worst absolute error on [0, 3.2] is 1.394e-7 at x = 0.0451,
  which is 1.17 f32 ulps at magnitude 1.
```

Eager, CPU and WASM agree **to the last printed digit**, because all three read the same constants out of one file; WebGPU inlines those same constants into WGSL, visible in the `0.3275` that opens its expression. CUDA calls `erff`. So the sentence a differential test wants — "the four backends compute the same `erf`" — is true of three of them and false of the fourth by about an ulp, and no amount of tightening a tolerance turns that into agreement.

The last section is where the backend's one uncontrolled identifier bites. The emitted function's name is the traced model's name, unsanitised:

```js
function Math(buf_1, buf_3) {
  for (let i0_4 = 0; i0_4 < 2; i0_4++) {
    for (let i1_5 = 0; i1_5 < 3; i1_5++) {
      buf_3[((i0_4 * 3) + i1_5)] = Math.tanh(buf_1[((i0_4 * 3) + i1_5)]);
    }
  }
}
  it threw: TypeError: Math.tanh is not a function
```

A function declaration binds its own name in its own scope, so inside a kernel called `Math` the identifier `Math` is the kernel. Every `Math.*` the emitter relies on — nineteen, counting the sixteen routed through `isJSMathFunc` plus `fround`, `PI` and `LN10` — resolves to the wrong thing. The same program under any other class name runs with zero error.

**Try this.** Name the class `Infinity`, `Float32Array` or `undefined` and watch three different failure modes: a syntax error, a broken allocation, and a kernel that compiles and computes. Then note that the check the backend needs is one line against the reserved-word list plus the globals it emits, and that it belongs in the backend rather than in the tracer, because the constraint is a property of the emitted language.

## 54.7 The parallel path

Nothing above is parallel. `numCores` on `CPUTarget` is 8, and a `@parallel` annotation on a loop reaches this backend and is ignored — `_visitForNode` does not look at `ForKind.PARALLEL`, so a parallel loop is emitted as a serial one. The correct answer is produced; the annotation is a no-op.

That is a deliberate asymmetry with the WASM backend, which does act on `@parallel` (Chapter 55 §55.6) because its runtime owns a worker pool and its memory is one `SharedArrayBuffer` that workers can be handed slices of. The equivalent for this backend would be `worker_threads` over a shared arena, which is buildable on top of Chapter 50's pool and is not built.

## 54.8 Traps and limits

### A kernel's name goes into binding position in the language the kernel is written in

`this._emit('function ' + func.name + '(' + paramNames.join(', ') + ') {')` ([`cpu/codegen.ts:107`](../../../src/backend/cpu/codegen.ts)) takes a name that arrives from `model.constructor.name` ([`tracing/compile.ts:436`](../../../src/tracing/compile.ts)) and puts it in binding position in generated JavaScript. §54.6 executes the case where that shadows a global the body uses. The same channel reaches buffer names, which are compiler-generated and safe, and shape-parameter names, which are not user-controlled either — so the name is the one uncontrolled identifier in the emitted text.

### The rest

- **The accumulator carries `f64` precision that the rest of the expression does not.** `_visitLIRAccumulator` emits `_acc = (_acc + body)` with no `Math.fround`, so a reduction over *n* elements is summed in doubles and narrowed once. Every other backend accumulates in the declared width. The compiled CPU sum is therefore *more accurate* than eager rather than equal to it: over 8,192 standard normals, measured against a Kahan sum in `f64`, eager is off by 2.2 × 10⁻⁴ and the compiled kernel by 4.2 × 10⁻⁶. That is also why §54.5's reduction and matmul rows report `max err 5e-7` against eager while the elementwise rows report zero. It is a departure from Definition 54.1 that nothing records, and the argument for keeping it (accuracy) and the argument against (four backends should agree, and a differential test cannot then use a tight tolerance) are both real.
- **Two peepholes run on rendered strings.** `_cleanupSource`'s dead-allocation rule matches `const X = new YArray(<digits>);` and counts occurrences of `X` over the whole text with a `\w+` tokenizer; a pool view (`new Float32Array(_mem_pool, 0, 32)`) does not match the pattern, so under `poolAllocation` a dead temporary keeps its declaration — visible in §54.5's `declared=8, of which read=4`. `_stripOuterRound` scans for a `Math.fround(` prefix and walks parentheses to find its match. Neither can be wrong about *semantics* — the first only deletes unreferenced names, the second only removes a rounding a store will redo — but both are the kind of pass that belongs over the IR, where the LIR pass layer of Chapter 53 §53.4 is waiting for it.
- **The pre-LIR path detects only `+` accumulators.** `_detectReductionAcc` ([`cpu/codegen.ts:341`](../../../src/backend/cpu/codegen.ts)) requires `val.op !== '+'` to fail, so a `max` or `min` reduction handed to this backend as a raw `PrimFunc` gets a load and a store per iteration where the LIR path gets a register. The compiled pipeline always goes through LIR, so this is a difference between the tested path and the shipped one rather than a live defect.
- **`_visitBlockNode`'s init guard uses the innermost enclosing loop.** A block with an `initBody` is emitted as `if (reductionVar === 0) { … }` where `reductionVar` is the top of `_loopStack` ([`cpu/codegen.ts:369`](../../../src/backend/cpu/codegen.ts)). That is right when the block sits directly inside its reduction loop and is a guess otherwise, and there is no check that the loop it found is the reduction axis.
- **`@parallel` and `@vectorized` are both ignored.** §54.7 covers the first. The second is [Chapter 42](../../part7/ch42-legality/README.md)'s deliberate non-gap: the JIT vectorizes typed-array loops itself, and an emitted `Math.fround` per element is precisely the thing that stops it — so the rounding contract of §54.3 and the JIT's auto-vectorization are in tension, and neither is measured against the other here.
- **One branch of the extern chain is unreachable.** `isJSMathFunc` is tested first and `JS_MATH_FUNCS` contains `log10`, so the `else if (node.externName === 'log10')` at [`cpu/codegen.ts:539`](../../../src/backend/cpu/codegen.ts) — which would compute it as `Math.log(x) * (1/LN10)` — can never run. `Math.log10` is more accurate than that expansion, so the dead branch is the worse of the two and nothing is lost; what is lost is the signal, because the branch reads as though it were the implementation.
- **`i64` is emulated with `BigInt`.** `_wrapStoreVal` ([`cpu/codegen.ts:161`](../../../src/backend/cpu/codegen.ts)) wraps `i64` stores in `BigInt(...)` and `_zeroLit` returns `0n`. `BigInt` allocates, which is the one thing §54.1 said must not happen in the loop body, so an `i64` kernel on this backend is on the JIT's slow path by construction.

## 54.9 Read the tests

- [`tests/backend/cpu/codegen-generate.test.js`](../../../tests/backend/cpu/codegen-generate.test.js) — the emitted shape for each statement kind, and that a `PrimFunc` and the `LIRFunc` lowered from it produce equivalent source.
- [`tests/backend/cpu/codegen-internals.test.js`](../../../tests/backend/cpu/codegen-internals.test.js) — the pieces on their own: `_exprToJS` per node kind, the identity folds and their dtype guard, `_stripOuterRound`, and `_cleanupSource`'s two rules.
- [`tests/backend/cpu/kernel-source.test.js`](../../../tests/backend/cpu/kernel-source.test.js) — properties of the text: loop counts, temporary counts, and that no fused elementwise chain allocates.
- [`tests/backend/cpu/kernel-numerical.test.js`](../../../tests/backend/cpu/kernel-numerical.test.js) — the rounding contract, differentially against eager, which is where Theorem 54.2's consequence would show up if a rounding were dropped.
- [`tests/backend/kernel-audit.test.js`](../../../tests/backend/kernel-audit.test.js) — quality assertions over the emitted source: no arithmetic noise, no extent-1 loops, no redundant zero inits, no leftover modulos in index expressions.

---

**Next:** [Chapter 55 — WebAssembly](../ch55-webassembly/README.md), where there is no JIT on the other side to be generous, and the backend has to assemble the bytes itself.
