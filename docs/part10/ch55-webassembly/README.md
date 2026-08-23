# Chapter 55 — WebAssembly

Chapter 54's backend could lean on a JIT. This one cannot. WebAssembly is a machine, not a language with an optimizer behind it: the instructions you emit are close to the instructions that run, the memory you declare is the memory you get, and — because a browser accepts bytes and not text — this backend does not stop at generating source. It assembles.

That makes it the largest of the four (1,957 lines of emitter plus a 408-line assembler) and the most instructive, because everything a compiler back end normally delegates is visible here: stack discipline, structured control flow, a flat address space, a hand-built instruction table, and explicit SIMD.

## 55.1 The problem: three things that are not provided

**There are no named arrays.** A WebAssembly module has one *linear memory* — a flat, byte-addressed region — and instructions that load and store at a computed address. `buf_3[i]` does not exist. What exists is `f32.load` at address `base + i × 4`, and the `base` has to come from somewhere. Chapter 53's `memoryLayout` is that somewhere.

**There is no `goto`, and no expression syntax.** WebAssembly's control flow is *structured*: `block`, `loop`, `if`, and branches that name an enclosing label by its depth. Its computation is a *stack machine*: no parentheses, no precedence, no infix. Both of those are a change of shape rather than a change of meaning, but both have to be done exactly.

**There is no assembler.** `WebAssembly.compile` takes a `Uint8Array` in the binary format. The text format is a convenience for humans and for tools; nothing in a browser or in Node reads it. So either the compiler depends on an external assembler at run time — which would put a toolchain between a user and their kernel — or it emits the bytes itself. This one emits the bytes itself, from the text it just generated, with a tokenizer and a table of opcodes.

## 55.2 Intuition: postfix, and a stack of labels

Two pictures carry most of the chapter.

**An expression is a post-order walk.** `a * b + c` in a stack machine is: push `a`, push `b`, `mul` (which pops two and pushes one), push `c`, `add`. That is exactly the order a post-order traversal visits the nodes of the expression tree. Emitting an expression is therefore *the same recursion* the CPU backend does, with the parentheses replaced by the order of emission.

**A loop is two nested labels.** WebAssembly's `br L` and `br_if L` do not name a program point; they name how many enclosing structures to break out of. `block` and `loop` differ in where their label lands: branching to a `block` goes to its *end*, branching to a `loop` goes to its *start*. So a `for` loop is a `block` (whose end is "loop finished") wrapped around a `loop` (whose start is the back edge), with an inverted exit test at the top:

```wat
(block $break
  (loop $again
    <i >= extent?>  br_if $break
    <body>
    <i = i + 1>
    br $again))
```

Every `for` in every kernel this backend emits has that shape.

## 55.3 Theory

### The stack discipline

> **Definition 55.1 (Stack-machine emission).** **(classical)** Let *E* be an expression tree whose nodes are operators of known arity. The *postfix emission* of *E* is the sequence produced by visiting each node's children left to right and then the node itself.

> **Theorem 55.2 (Postfix emission is well-typed and balanced).** **(classical)** Executing the postfix emission of *E* on an initially empty operand stack leaves exactly one value — the value of *E* — and never pops an empty stack.
>
> *Proof.* Induction on *E*. A leaf pushes one value. For a node of arity *k*, the emission runs the *k* children's emissions in order and then the operator; by hypothesis each child leaves exactly one value, so when the operator runs there are exactly *k* values above the entry level, and it pops *k* and pushes 1. The stack therefore never goes below its entry level and ends one above it. ∎

The practical content of Theorem 55.2 is that the emitter never has to reason about the stack. If it emits children then operator, the module validates.

Two consequences worth separating from the theorem, because neither is needed for correctness and one is easy to get wrong.

**The depth is a property of the order, not of the tree.** Emitting children strictly left to right reaches a maximum depth of maxᵢ (dᵢ + i − 1) over the children — the *i*-th child runs with *i* − 1 finished siblings beneath it. That is *not* the Ershov (Sethi–Ullman) number, which is the minimum over *all* evaluation orders and is what a register allocator would want; this backend never reorders operands, so it pays the left-to-right figure. WebAssembly's operand stack is not a scarce resource the way registers are, so nothing here optimizes it.

**A value needed twice must be spilled.** WebAssembly has no stack-duplicating instruction; the idiom is `local.tee`, and this backend instead writes to a local and re-reads it. That is the `_vaddr_*` local the labs show.

### The memory

> **Definition 55.3 (Flat placement).** **(invariant)** Every buffer of a WASM kernel is assigned a byte offset in one linear memory by `computeMemoryLayout`, 16-byte aligned, in order: parameters first, then the rest. An access to buffer *B* at flat index *e* of a dtype of width *w* bytes is emitted as the address `offset(B) + e × w`.

The module then declares `⌈total / 65536⌉` pages, because WebAssembly memory is measured in 64 KiB pages.

This is the second place the compiler solves the packing problem of [Chapter 50](../../part9/ch50-arena-allocation/README.md), and it solves it more crudely: no lifetime analysis, no reuse, just a bump allocator over every buffer the function mentions. It can afford to, because a kernel's buffers are small and a page is cheap — with one exception, which §55.7 measures.

### Trapping

C's integer division by zero is undefined behaviour; JavaScript's is `Infinity`. WebAssembly's is defined and severe: `i32.div_s` with a zero divisor **traps**, which unwinds the whole module and surfaces as a `RuntimeError` in the host. So a backend that emitted the obvious instruction would turn a nonsense value into an aborted computation.

> **Proposition 55.4 (Guarded division).** **(invariant)** The WASM backend emits every integer division and remainder as: compute a *safe divisor* (the real one, or 1 when the real one is zero or when the pair would overflow), divide, apply the floor correction if the operator is `//` or `%`, then select 0 when the real divisor was zero.
>
> The result agrees with Definition 53.8 on every non-zero divisor, and is 0 rather than a trap on a zero divisor.

`_emitSafeDivisor` ([`wasm/codegen.ts:1215`](../../../src/backend/wasm/codegen.ts)) is the first half and `_emitZeroWhenDivisorZero` ([`wasm/codegen.ts:1229`](../../../src/backend/wasm/codegen.ts)) the second. The overflow clause is `INT_MIN / −1`, which also traps.

### Vectorization

WebAssembly's SIMD extension gives 128-bit vectors: four `f32` lanes, four `i32`, or two `f64`. Emitting them is a *transformation*, not a rendering, and therefore needs a legality argument.

> **Definition 55.5 (Lane variables).** **(stated here)** For a loop over *k*, the *lane variables* are the least set containing *k* and closed under: if a scope binding `v = e` has *e* mentioning a lane variable, then *v* is a lane variable.

> **Definition 55.6 (Vectorizable loop).** **(invariant)** A loop over *k* with constant extent *n* is vectorizable at width *w* when all five hold:
>
> 1. every lane variable is bound inside the loop body (`_vecLaneVarsResolvable`);
> 2. every guard in the body is lane-invariant (`_vecGuardIsLaneInvariant`);
> 3. every subscript is affine in the lane variables (`indicesAreLaneAffine`);
> 4. every guarded load stays in range for all *w* lanes (`guardedLoadsAreInRange`);
> 5. the body has no loop-carried dependence (`loopCarriedDependenceIn`).

> **Theorem 55.7 (Vectorizing an elementwise loop is N0).** **(stated here)** If a loop is vectorizable and its body contains no accumulation into a location the loop also reads, then the vectorised emission — ⌊n/w⌋ vector iterations followed by a scalar tail of n mod w — performs exactly the same operations on exactly the same operands as the scalar loop, and is therefore bit-identical.
>
> *Proof sketch.* Condition 3 makes lane *j* of vector iteration *m* correspond to scalar iteration *mw + j*; condition 5 says no scalar iteration reads what another wrote, so the reordering into groups of *w* is a permutation of independent work; conditions 1, 2 and 4 make each lane's operands the ones that scalar iteration would have used. Each lane therefore performs its scalar iteration's operations in its scalar iteration's order. ∎

The accumulator is the case Theorem 55.7 excludes, and it is genuinely different.

> **Proposition 55.8 (A vectorised reduction is N2).** **(stated here)** The vectorised accumulator maintains *w* partial sums, one per lane, and combines them at the end. The result is the same multiset of operands summed under a different association, which is level **N2** on Definition 1.4's ladder — reassociated — and is not in general bit-identical to the scalar loop.

That is the reason `_accumInstr` ([`wasm/codegen.ts:536`](../../../src/backend/wasm/codegen.ts)) refuses integer `max` and `min` outright rather than reassociating them quietly, and the reason the vector path is gated on the operator being `+`.

## 55.4 In mlfw: an emitter and an assembler

### The emitter

`WasmCodegen.generate` ([`wasm/codegen.ts:75`](../../../src/backend/wasm/codegen.ts)) builds a module in the order the text format wants it:

1. **Memory.** `memPages = max(1, ⌈totalMemBytes / 65536⌉)`, declared with a maximum of at least 256 pages so the runtime can grow it.
2. **Imports.** Every `CallExternNode` whose name is not one of the six WebAssembly has natively (`sqrt`, `abs`, `ceil`, `floor`, `min`, `max` — the `WASM_NATIVE_OPS` set, [`dtype_map.ts:111`](../../../src/util/dtype_map.ts)) becomes `(import "math" "exp" (func $math_exp (param f32) (result f32)))`. The runtime supplies an object of JavaScript functions.
3. **Parameters and locals.** Buffers arrive as `i32` parameters — although the emitter ignores their values and uses the static offsets — followed by shape parameters, followed, when a parallel loop was found, by `_par_start` and `_par_end`. Locals come from `metadata.locals` plus several prescans that add the vector temporaries.
4. **The body**, and then the closing parentheses.

Addresses are formed by `_emitFlatAddr` ([`wasm/codegen.ts:703`](../../../src/backend/wasm/codegen.ts)) exactly as Definition 55.3 describes, and that function carries the backend's one *reuse* optimization: in vector mode, when a loop body touches two or more buffers at the same offset expression, the scaled address is computed once into a `_vaddr_*` local and the per-buffer bases are added to it. That is common-subexpression elimination on addresses, performed by a backend, at the level of emitted instructions.

### Division

Proposition 55.4's three steps come out of `_emitIntDiv` ([`wasm/codegen.ts:1248`](../../../src/backend/wasm/codegen.ts)) in order:

```ts
    this._emit('(local.get $' + a + ')');
    this._emitSafeDivisor(prefix, a, b, true);
    this._emit(prefix + '.div_s');
    this._emit('local.set $' + r);

    if (floor) {
      ...
      this._emitSignsDifferAndNonZero(prefix, a, b, r);
      this._emit('select');
      this._emit(prefix + '.sub');
```

Truncating division, then — for the floor operators — subtract one when the signs differ and the remainder is non-zero, which is the standard correction. The `_idiv_a0`, `_idiv_b0`, `_idiv_r0` locals are indexed by an emission depth so a nested division does not clobber the outer one's operands.

### SIMD

Before it emits a lane, `_visitVectorizedFor` ([`wasm/codegen.ts:1363`](../../../src/backend/wasm/codegen.ts)) has to establish Theorem 55.7's hypotheses: it checks the width, the extent and the dtype, then Definition 55.6's five conditions, and falls back to `_emitForLoop` on any failure — so an unvectorizable loop is not an error, it is a scalar loop. The main loop runs `⌊n/w⌋ × w` iterations at stride *w*, and a tail loop finishes the rest.

The reassociation of Proposition 55.8 is visible in nine lines of `_visitVecAccumulator` ([`wasm/codegen.ts:606`](../../../src/backend/wasm/codegen.ts)):

```ts
    this._emit('(local.get $' + accLocal + ')');
    for (let l = 0; l < lanes; l++) {
      this._emit('(local.get $' + vaccLocal + ')');
      this._emit(simdEntry.extractLane + ' ' + l);
      this._emit(scalarAdd);
    }
    this._emit('local.set $' + accLocal);
```

A scalar accumulator seeded from the destination, a vector accumulator seeded from a zero splat, the main loop adding vectors, then the four lanes extracted and folded into the scalar one, then the tail. Four partial sums, combined at the end — which is the whole of Proposition 55.8, in nine lines.

There is one more thing this backend does that no other does: it re-derives dependence at codegen time. `loopCarriedDependenceIn` is called from `_vectorizationIsLegal` ([`wasm/codegen.ts:1587`](../../../src/backend/wasm/codegen.ts)) on the body the scheduler already annotated `@vectorized`. Chapter 42 explains why that is not redundant — a block's declared iteration-variable kinds can make the scheduler's legality check accept a loop whose dependence is real — and this backend is the one that checks independently and demotes.

### The assembler

[`wasm/wat_encoder.ts`](../../../src/backend/wasm/wat_encoder.ts) is 408 lines and three stages.

**Tokenize** ([`wat_encoder.ts:96`](../../../src/backend/wasm/wat_encoder.ts)): parentheses, quoted strings, line and block comments, everything else a bare token.

**Parse the module** ([`wat_encoder.ts:112`](../../../src/backend/wasm/wat_encoder.ts)): pull out the imports with their signatures, the memory limits, the exported function's name, its parameter and local types and names, and the body tokens.

**Encode.** Numbers are LEB128 — `uleb` and `sleb` at [`wat_encoder.ts:8`](../../../src/backend/wasm/wat_encoder.ts) — a variable-length little-endian base-128 encoding in which 127 is one byte and 128 is two. Sections are emitted in the order the specification requires: type, import, function, memory, export, code. The body is walked by `emitBlock` ([`wat_encoder.ts:227`](../../../src/backend/wasm/wat_encoder.ts)), which handles the folded forms (`(i32.const 4)`, `(local.get $x)`, `block`, `loop`, `if`) specially and otherwise looks the token up in `INSTR`, a map from mnemonic to opcode bytes.

`resolveBr` ([`wat_encoder.ts:321`](../../../src/backend/wasm/wat_encoder.ts)) turns a label name into the relative depth the binary format wants, by scanning the label stack from the top.

## 55.5 Lab — a stack machine and a binary

```bash
node docs/part10/ch55-webassembly/labs/01-a-stack-machine-and-a-binary.mjs
```

The first output is a complete module for a 2×3 elementwise multiply — the memory declaration, the export, the four locals, and two nested `block`/`loop` pairs. Then the layout it implies:

```
  pages: 1 x 65536 bytes
  buf_1    at byte    0
  buf_3    at byte   32
  buf_5    at byte   64
```

Six `f32` is 24 bytes, rounded to the 16-byte alignment gives 32 — Definition 55.3, as a bump allocator.

The postfix section takes a *fused* `a*b + a` and prints the statement:

```
  (local.get $v0_7)  (i32.const 4)  i32.mul  (i32.const 32)  i32.add            <- the store address
  (local.get $v0_7)  (i32.const 4)  i32.mul  f32.load                           <- a[i]
  (local.get $v0_7)  (i32.const 4)  i32.mul  (i32.const 16)  i32.add  f32.load  <- b[i]
  f32.mul
  (local.get $v0_7)  (i32.const 4)  i32.mul  f32.load                           <- a[i] again
  f32.add
  f32.store
```

The lab prints one token per line; the grouping above is that same sequence, folded so the operand groups are visible. It is the post-order walk of Theorem 55.2, with the store's address pushed first because `f32.store` pops address *then* value.

Note also what is *not* elided: `v0_7 * 4` is recomputed three times. The address CSE of §55.4 applies only in vector mode; in scalar mode each access recomputes its own address, and the engine's own optimizer is expected to notice.

Then the loop idiom, and then the binary:

```
  magic + version                  8 bytes
  section  1 type                     9 bytes
  section  3 function                 4 bytes
  section  5 memory                   7 bytes
  section  7 export                  21 bytes
  section 10 code                   118 bytes
  total                           167 bytes

  Sizes and indices are LEB128, a variable-length encoding:
         0 -> 00
       127 -> 7f
       128 -> 80 01
     65536 -> 80 80 04
```

Each section's figure includes its two-byte header — a one-byte id and the LEB128 length — so the column adds up. 167 bytes for a complete, runnable module. And it computes: `max err 0` against eager.

The last two sections are about the assembler, and both are places where it declines to complain.

```
  an unknown instruction: encoded to 53 bytes and validated.
  a branch to a label that was never opened: encoded and validated.
```

`emitBlock`'s dispatch is a chain of `if`/`else if` ending in `else if (INSTR.has(t))` with **no final else**, so a mnemonic the table does not know is silently skipped; and `resolveBr` returns `0` when it does not find the label, which is the innermost enclosing block. Both produce a module that *validates* — WebAssembly's validator checks types and stack balance, and a dropped instruction or a redirected branch can leave both intact — so the failure surfaces as a wrong number, not as an error.

The last section is the flat layout's one real limit:

```
   elements  max err vs a hand-computed reference
       4096  0
      16384  0
      16385  7.566e-1
      20000  8.357e+1
```

`DYNAMIC_BUFFER_SLAB_BYTES` is 65,536, so a dynamically-sized `f32` buffer is given room for 16,384 elements. Correct to exactly that, silently wrong past it, because the next buffer's base sits inside this one's range. WASM is the only backend that reads `memoryLayout`, so it is the only one that pays; the same program on CPU is correct at every size.

**Try this.** Give the module a second dynamic buffer and watch the threshold stay at 16,384 rather than halving — the slab is per buffer, so what fails is not the total but each individual buffer's extent.

## 55.6 Lab — four lanes at a time

```bash
node docs/part10/ch55-webassembly/labs/02-four-lanes-at-a-time.mjs
```

```
  program                  v128 ops v128.load extract_lane  vectorised?
  elementwise, two inputs        14         3            0  yes
  a longer chain                 24         1            4  yes
  a transcendental               18         1            4  yes
  a comparison                   12         2            0  yes
  a reduction                     0         0            0  no, scalar loop
  a 2-D reduction                17         1            4  yes
```

Two things to read here. **`extract_lane` inside a vectorised loop is the escape hatch:** WebAssembly has no `f32x4.exp`, so a transcendental loads a vector, pulls out the four lanes, calls the imported scalar function on each, and rebuilds the vector. The memory traffic is vectorised and the mathematics is not — which is still worth doing, because on an elementwise chain the loads and stores are the cost. **And the 1-D reduction is not vectorised at all**, because the rule policy does not annotate its loop; the 2-D one is.

Then the differential:

```
  elementwise, two inputs  scalar vs vectorised: bit-identical
  a longer chain           scalar vs vectorised: bit-identical
  a transcendental         scalar vs vectorised: bit-identical
  a comparison             scalar vs vectorised: bit-identical
  a reduction              scalar vs vectorised: bit-identical
  a 2-D reduction          scalar vs vectorised: max err 9.54e-7
```

Five of six are Theorem 55.7 — N0, bit-identical. **The sixth is Proposition 55.8 arriving through the default path**: `scheduling: { enabled: true }` annotated a reduction axis `@vectorized`, the backend took the vector accumulator, and the answer changed. Not by much, and not wrongly — but it is an N2 transformation that the trace reports as a scheduling decision rather than as a numerical one.

The next section makes the difference large enough to be unmistakable. Feed the reduction `x = [2²⁴, 1, 1, …]`:

```
  n=64   x = [2^24, 1, 1, ...]
         scalar loop      16777216
         four lanes       16777264   (differs by 48)
```

In `f32`, 2²⁴ + 1 rounds back to 2²⁴, so the scalar loop absorbs every one of the 63 ones and returns 2²⁴ exactly. The four-lane loop keeps four partial sums: one of them holds the big value and absorbs its share, and the other three are pure counts that survive to the combine. Same operands, same operator, different association, 48 apart.

The third section is about work the vector emitter does and then throws away:

```
  v0_7_vlet: 1 set, 0 get

  the 24 instructions that produce it:
    (local.get $i0_6_o_0)  f32.convert_i32_s  f32x4.splat
    (f32.const 4)  f32x4.splat  f32x4.mul
    (local.get $i0_6_i_1)  f32.convert_i32_s  f32x4.splat
    (local.get $i0_6_i_1)  (i32.const 1)  i32.add  f32.convert_i32_s  f32x4.replace_lane 1
    ... replace_lane 2 ... replace_lane 3 ...
    f32x4.add
    local.set $v0_7_vlet
```

(Again folded; the lab prints one token per line.)

The loop's scope binding is the flat index. In vector mode the emitter builds a *vector* of that index — splat the outer part, fill the four lanes with consecutive values — and does it in **float lanes**, because the vector mode's dtype is the one the arithmetic uses rather than the one an index needs. Then the actual addressing goes through the scalar `_vaddr_*` CSE, and the vector is set and never read: 24 dead instructions in the innermost loop of every vectorised kernel, against roughly a dozen live ones.

**Try this.** Vectorise a loop whose body genuinely uses the flat index as a *value* rather than only as an address — a `range`-like op, or an `arange` — and watch the binding become live. The emitter is not wrong to build it; it is missing the check that anything reads it, which is one liveness query over the body.

## 55.7 The parallel path

WASM is the one backend of the four that acts on `@parallel`. `_scanParallel` ([`wasm/codegen.ts:430`](../../../src/backend/wasm/codegen.ts)) finds the annotated loop; the module then takes two extra `i32` parameters, `_par_start` and `_par_end`, and the loop runs that half-open range instead of `0 … extent`. The runtime instantiates the module once per worker over one `SharedArrayBuffer` and hands each a slice.

`_isParallelSafe` ([`wasm/codegen.ts:203`](../../../src/backend/wasm/codegen.ts)) is the precondition: exactly one parallel loop, and every store in the function inside it. If some statement outside the parallel loop writes memory, every worker would run it, so the flag comes back false and the runtime declines to split.

## 55.8 Traps and limits

### Two silences in the assembler, and one in the vector emitter

**The assembler's instruction table has no failure mode.** `emitBlock`'s final `else if (INSTR.has(t))` has no else ([`wat_encoder.ts:315`](../../../src/backend/wasm/wat_encoder.ts)), so an unrecognised mnemonic is dropped. `resolveBr` ([`wat_encoder.ts:321`](../../../src/backend/wasm/wat_encoder.ts)) returns 0 for an unknown label rather than throwing. §55.5 executes both. The emitter and the table are edited by the same hands, so the live risk is a new instruction added to the emitter and not to the table — and the symptom is a wrong answer in one kernel, not a build failure. Two `throw`s would convert both into compile-time errors.

**The vectorised scope binding is dead work in every vectorised loop.** §55.6 measures it: 24 instructions set a `v128` local that nothing reads, in the innermost loop, computed in float lanes.

### The rest

- **A vectorised reduction reaches the default path without a licence.** Proposition 55.8 is an N2 transformation, and `scheduling: { enabled: true }` is enough to get it — there is no `fastMath` gate on the vector accumulator, and nothing in the trace says the summation order changed. Chapter 20's algebraic rewrites, which are the same kind of change, are gated. The narrow fix is to route the vector accumulator through the same licence; the broad one is for the annotation `@vectorized` on a *reduction* axis to be treated as the reassociation it is, wherever it is set.
- **The pre-LIR path imports three functions it cannot call.** The set of §55.4 is filtered in two places. On the compiled path the LIR scanner tests it directly ([`scanner.ts:71`](../../../src/compiler/ir/lir/scanner.ts)), so all six are kept out of `externCalls`. On the pre-LIR path `_scanMathImports` ([`wasm/codegen.ts:1851`](../../../src/backend/wasm/codegen.ts)) carries its own copy of the list, and that copy has three names — `sqrt`, `min`, `max`. So a raw `PrimFunc` using `floor` is emitted with `(import "math" "floor" …)` *and* the `f32.floor` instruction that makes the import unreachable, while the same function lowered to LIR gets the instruction alone. Like Chapter 54's `+`-only accumulator detection this is a difference between the tested path and the shipped one rather than a live defect, and where the path is exercised nothing breaks — the runtime resolves an import name it does not recognise against `Math` itself ([`backend_registry.ts:96`](../../../src/runtime/backend_registry.ts)). One set, written down twice, one copy three names behind.
- **Compound symbolic shapes do not compile.** `emitSymInt` throws for anything but a bare variable when the dialect is `wat` ([`codegen_utils.ts:48`](../../../src/backend/codegen_utils.ts)), so a dynamic shape that arrives as an expression — `n + 1`, `n * 2`, a `ceildiv` from a split — is a hard error on this backend and works on the other three. The reason is real (the emitter would have to build the expression on the stack in the middle of an address computation) and the message names the backend, so this one fails loudly.
- **`i64` and `f64` are partly supported and unevenly tested.** The instruction table carries the `i64` and `f64` opcodes and `_emitIntDiv` is parameterised by prefix, but `wasmSimdEntry` only has `f32x4`, `i32x4` and `f64x2`, and `_accumInstr` throws for an integer `max`/`min` reduction at any width.
- **The parallel decision is made twice.** `_isParallelSafe` runs over the function *after* the body has been emitted, and its result is metadata the runtime may or may not act on; the `_par_start`/`_par_end` parameters are added regardless, because `_hasParallel` was set by the scan. A module that is not `poolSafe` therefore has two parameters the runtime passes `0` and `extent` for.
- **The memory is bump-allocated with no reuse.** Definition 55.3 gives every buffer its own bytes, so a kernel with ten temporaries reserves all ten — where Chapter 50's arena, which ran earlier in the same compilation, may have proved that two of them can share. The two allocators do not talk: `poolByteOffset` is read by the CPU backend and ignored here.
- **Buffer parameters are declared and unused.** The emitted function takes one `i32` per buffer and never reads them; every address comes from the static layout. The parameters exist so the calling convention matches the other backends'.

## 55.9 Read the tests

- [`tests/backend/wasm/codegen.test.js`](../../../tests/backend/wasm/codegen.test.js) — the emitted text per node kind, the loop idiom, the import list, and the guarded division of Proposition 55.4 on the operands §53.6 tabulates.
- [`tests/backend/wasm/wat-encoder.test.js`](../../../tests/backend/wasm/wat-encoder.test.js) — the assembler on its own: LEB128, section order, label depth resolution, and round-trips through `WebAssembly.Module`.
- [`tests/backend/wasm/simd.test.js`](../../../tests/backend/wasm/simd.test.js) — Definition 55.6's five conditions one at a time, each with a loop that fails exactly one of them and must fall back.
- [`tests/backend/wasm/parallel.test.js`](../../../tests/backend/wasm/parallel.test.js) and [`parallel-quality.test.js`](../../../tests/backend/wasm/parallel-quality.test.js) — the range parameters, `_isParallelSafe`'s precondition, and that a split run agrees with an unsplit one.
- [`tests/backend/wasm/float-to-int-cast.test.js`](../../../tests/backend/wasm/float-to-int-cast.test.js) — the saturating truncation instructions, which are the other place WebAssembly traps where the other three languages do not.
- [`tests/backend/wasm/compile.test.js`](../../../tests/backend/wasm/compile.test.js) — end to end through `compile()`, instantiated and run.

---

**Next:** [Chapter 56 — CUDA](../ch56-cuda/README.md), where the loops do not run at all.
