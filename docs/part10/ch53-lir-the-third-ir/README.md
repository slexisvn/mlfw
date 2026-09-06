# Chapter 53 — LIR: why a third IR

Part IX finished with a plan: every buffer has a size, a lifetime and, under pool allocation, an address. Part VII finished with a loop nest whose shape had been chosen. Nothing is left to decide. What remains is to write the program down in a language a machine will run — and there are four such languages in this compiler, which is the reason for one more IR.

## 53.1 The problem: four backends, one piece of arithmetic

Here is a statement out of the TIR of a matmul, printed by [Chapter 33](../../part6/ch33-buffers-blocks-itervars/README.md)'s printer:

```
buf_5[vls0_9, vrs0_10] = (buf_5[vls0_9, vrs0_10] + (buf_1[vls0_9, vc0_11] * buf_3[vc0_11, vrs0_10]))
```

Three buffers, six subscripts, two dimensions each. Now write that in C. There is no `buf_5[i, j]` in C: there is `buf_5[i * 5 + j]`, and the `5` came from `buf_5`'s strides, and if `buf_5` were a slice of a larger tensor there would be a base offset too, and if its second dimension were dynamic the `5` would be a runtime shape parameter that has to be multiplied out of the remaining dimensions.

That is a real piece of work — small, but with edge cases — and it is *identical* for C, for JavaScript, for WebAssembly and for WGSL. Written once per backend, it is four chances to disagree about a stride, four places to fix when dynamic shapes arrive, and four implementations that must be tested separately.

There is a second thing every backend would otherwise rediscover. The loop above accumulates: each iteration of the `k` loop reads `buf_5[i, j]`, adds to it, and writes it back. A backend that emitted that literally would issue a load and a store to memory per iteration of the innermost loop of a matmul. Every backend wants instead to hold the running total in a register and store it once. Recognising when that is legal — the address must not move as the loop runs — is again the same analysis four times.

So the compiler runs both once, into a representation that has no subscripts and has accumulators. That is LIR.

## 53.2 Intuition: the last translation before the language

Think of the three IRs as three answers to "what is a program?"

- **Graph IR** (Part II): a program is a dataflow graph over whole tensors. `matmul(A, B)`.
- **TIR** (Part VI): a program is a loop nest over indexed buffers. `for i, j, k: C[i,j] += A[i,k] * B[k,j]`.
- **LIR**: a program is a loop nest over *flat memory*. `for i, j: acc = C[i*5+j]; for k: acc += A[i*6+k] * B[k*5+j]; C[i*5+j] = acc`.

Each step throws something away and gains something. TIR gave up the identity of the operation — nothing at that level knows this nest "is a matmul" — and gained loops to schedule. LIR gives up the *shape* of the buffers and gains addresses a machine can compute directly.

The thing to hold onto is that the third step is not a fifth of the work of the second. It is a rewriting so mechanical that its entire implementation is 700 lines, and its value is not in what it computes but in *where* it computes it: once, before the fan-out to four backends, instead of four times after.

### The three problems this back end delegates

A reader who has met a classical compiler will be looking for three things: instruction selection, register allocation, and instruction scheduling. None of the three is solved in the next five chapters, and it is worth saying why before they are hunted for in vain.

The third is quickest. **Instruction scheduling** — reordering machine instructions to hide latency and keep a pipeline full — is delegated wholesale to whatever consumes this part's output, and it is *not* what [Part VII](../../part7/README.md) meant by scheduling. Part VII scheduled *loops*: tiling, reordering, binding axes to threads. Those are decisions about the iteration space and they are made before TIR becomes LIR. The word collides; the problems do not overlap.

The other two are worth more than a sentence each.

**Instruction selection** is the problem of covering an IR expression with the machine instructions that exist. It is hard when the IR is finer-grained than the machine — when `a * b + c` should become one fused-multiply-add, or when an addressing mode can absorb a multiply — and the classical answers are tree-pattern matching and dynamic programming over tiles. Here it is a `switch`. LIR's operators are already at the granularity of the target languages' operators: `MathOpNode('+')` on `f32` is `f32.add`, `+` and `+` on the four targets, one node to one operator, no choice to make. The compiler bought that by choosing its IR's vocabulary to be the intersection of what four languages offer, and the price is paid where you would expect — `//` and `%` are not that intersection, and §53.6 is the four different expansions they need.

**Register allocation** is the problem of fitting an unbounded number of live values into a bounded number of machine registers, with spilling when they do not fit. It is not solved here at all: it is *delegated*, four times over, to whatever compiles the text this part emits. The JavaScript engine's JIT allocates for Chapter 54's output; the browser's WebAssembly compiler for Chapter 55's; NVCC and the WGSL compiler for Chapters 56 and 57. What this part does instead is decide *which values are candidates for a register* — Theorem 53.5's accumulator promotion, Chapter 57's scalarization — and then trust the downstream compiler to place them.

That division is normal for a compiler that emits source rather than machine code, and it has one real consequence worth carrying: **register pressure is decided upstream and is invisible downstream.** Chapter 56's register-blocked matmul declares `float rb_acc[16]` — sixteen accumulators per thread — and the budget check that allowed it ran in the *sketch* that chose the blocking, against `target.registersPerThread` ([`matmul_tiling.ts:189`](../../../src/compiler/schedule/matmul_tiling.ts)), long before any text existed. No backend in Part X reads that field at all. So a schedule that over-subscribes registers is caught, if at all, by the scheduler; the backend will emit whatever it is handed, and the symptom of a mistake is a kernel that spills to local memory and runs slowly, with nothing in the emitted text or the metadata to say so.

## 53.3 Theory

### Flattening

A buffer is a name for a range of memory plus an interpretation of that range as an array of some shape. The interpretation is the *layout*.

> **Definition 53.1 (Layout function).** **(classical)** Let a buffer have rank *n*, base offset *b*, and strides *s₀ … s₍ₙ₋₁₎*. Its *layout function* sends an index tuple to a scalar offset:
>
> λ(i₀, …, i₍ₙ₋₁₎) = b + Σₖ iₖ · sₖ.
>
> The layout is *row-major* when sₖ is the product of the extents after dimension *k*: s₍ₙ₋₁₎ = 1 and sₖ = s₍ₖ₊₁₎ · d₍ₖ₊₁₎.

> **Theorem 53.2 (Row-major flattening is injective).** **(classical)** Under a row-major layout with extents d₀ … d₍ₙ₋₁₎, λ is injective on the in-bounds index set ∏ₖ [0, dₖ).
>
> *Proof.* λ(i) − b is exactly the value of the digit string i₀ … i₍ₙ₋₁₎ read as a mixed-radix numeral with radices d₀ … d₍ₙ₋₁₎, because sₖ is the product of the radices to its right. Mixed-radix representation is unique when each digit is below its radix, which is what in-bounds means. Two distinct in-bounds tuples are therefore two distinct numerals with distinct values. ∎

Injectivity is what makes flattening *safe*: two elements that were distinct before flattening remain two distinct addresses after it, so no store can silently overwrite a value some other subscript was naming. It is also exactly the property that fails when a layout is not row-major — a broadcast buffer has a zero stride on purpose, and many index tuples share one address. That is legal only because a broadcast buffer is read and never written.

The interesting direction, though, is the other one.

> **Observation 53.3 (Flattening is not invertible from the syntax).** **(stated here)** Given the layout function, the tuple can be recovered — it is the mixed-radix expansion of Theorem 53.2. Given only the *offset expression* as a piece of syntax, it cannot: `i * 5 + j` and a variable holding that value are the same thing to every consumer downstream, and the compiler stops recording which factor came from which axis.
>
> This does not follow from Theorem 53.2 — injectivity says the map has an inverse, and this says the compiler discards what it would need to apply one. The two are about different objects: the mathematical function, and the IR node that stands for it.

That observation is the practical content of [Chapter 6](../../part1/ch06-the-pipeline/README.md)'s claim that lowering is irreversible, in its sharpest form. Everything that needs to know which loop variable indexed which axis — dependence testing ([Chapter 36](../../part6/ch36-dependence-analysis/README.md)), tiling ([Chapter 40](../../part7/ch40-loop-primitives/README.md)), buffer lifetimes ([Chapter 49](../../part9/ch49-buffer-lifetimes/README.md)) — has to have run already. **The position of the flattening step in the pipeline is a hard constraint, not a convention.**

### Accumulators

> **Definition 53.4 (Accumulating loop).** **(stated here)** A serial loop over variable *k* is *accumulating* into buffer *D* at offset expression *w* when its body is a single store `D[w] = D[w] ⊕ e` for an operator ⊕, the load and the store use the same *w*, and *w* does not mention *k*.

> **Theorem 53.5 (Accumulator promotion).** **(stated here)** Let a loop be accumulating in the sense of Definition 53.4. Then replacing it by
>
> `t ← D[w];  for k: t ← t ⊕ e;  D[w] ← t`
>
> where *t* is a fresh local, computes the same final contents of `D[w]` and touches no other location.
>
> *Proof.* Because *w* does not mention *k*, it evaluates to the same address ℓ on every iteration. The original loop's state after iteration *m* is therefore `D[ℓ] = v₀ ⊕ e₁ ⊕ … ⊕ eₘ`, the operations applied left to right, where v₀ is the value before the loop. The promoted form computes the same left-to-right fold in *t* and writes it to ℓ once. The only difference in the trace is the intermediate stores to ℓ, and by hypothesis the loop body is the only statement in the loop, so nothing reads ℓ between them. ∎

Two remarks, and both matter for what follows.

**Associativity is not needed for the theorem as stated, only for reordering.** The proof keeps the operations in their original order; the promotion is level **N1** on [Definition 1.4](../../part0/ch01-what-this-book-is/README.md)'s ladder — the same operations in the same order. It is Chapter 55's *vectorised* accumulator that reassociates and lands at N2. Getting that boundary right is the difference between an optimization that applies unconditionally and one that needs a licence.

**The hypothesis "*w* does not mention *k*" is the whole condition,** and it is why the promotion is checked per loop rather than per buffer.

> **Counterexample 53.6.** `for k: D[k] = D[k] + A[k]` is a store whose load and store subscripts agree — but *w* is `k`, so the address moves. The promoted form of Theorem 53.5 is not merely wrong here, it is not even well formed: `t ← D[w]` and `D[w] ← t` sit outside the loop, where `k` is not in scope, so there is no address for them to name. Pick one anyway — evaluate *w* at the first iteration for the load and the last for the store — and the loop computes `D[n−1] = D[0] + A[0] + … + A[n−1]`, collapsing *n* independent updates into one and leaving every other element untouched. Nothing about *liveness* distinguishes this case from the legal one; the distinction is entirely in whether the address is loop-invariant.

## 53.4 In mlfw: 700 lines and a metadata block

LIR lives in [`src/compiler/ir/lir/`](../../../src/compiler/ir/lir/) — four files, 702 lines — and is produced by one pass, [`tensor_to_lir.ts`](../../../src/compiler/passes/lowering/tensor_to_lir.ts).

### Five node kinds

`LirNode` ([`nodes.ts:225`](../../../src/compiler/ir/lir/nodes.ts)) is a union of five; the rest of TIR passes through unchanged.

| Node | Replaces | Carries |
|---|---|---|
| `LIRFunc` | `PrimFunc` | the same signature, plus `metadata` |
| `LIRFlatLoadNode` | `BufferLoadNode` | a buffer and one `offsetExpr` |
| `LIRFlatStoreNode` | `BufferStoreNode` | a buffer, one `offsetExpr`, a value |
| `LIRAccumulatorNode` | an accumulating `ForNode` and its `BlockNode` | a local name, an operator, an init load, a loop, a flush store |
| `LIRBindingsNode` | a `BlockNode`'s `iterVars` | a list of `name = expr`, and a body |

`LIRBindingsNode` is the small one and the easiest to overlook. A `BlockNode` declares iteration variables bound to expressions over the enclosing loops — `bind v0_9 = i0_7` — and a backend has to do *something* with those bindings before it emits the body. Making them a node of their own is what lets each backend choose what, and the four choose differently: CUDA emits `const int v0_9 = ...;`, WebGPU a `let`, WASM a `local.set`, and the CPU backend emits nothing at all — it records the rendered expression in an alias table and substitutes it at every use ([`cpu/codegen.ts:299`](../../../src/backend/cpu/codegen.ts)). That is why §53.5's CPU output has no binding variables in it and §53.6's CUDA output does.

### Building the offset expression

Definition 53.1's sum is built term by term at [`flatten.ts:9`](../../../src/compiler/ir/lir/flatten.ts), with three shortcuts that exist to keep the emitted text small:

```ts
  if (indices.length === 0) return new IntImmNode(baseOffset);
  if (indices.length === 1) {
    if (baseOffset === 0) return indices[0];
    return new MathOpNode('+', indices[0], new IntImmNode(baseOffset));
  }

  const terms: TirNode[] = [];
  if (baseOffset !== 0) terms.push(new IntImmNode(baseOffset));
  for (let i = 0; i < indices.length; i++) {
    const idx = indices[i];
    if (idx.type === 'IntImmNode' && (idx as IntImmNode).value === 0) continue;

    const stride = buffer.strides[i];
    if (stride === 1) {
      terms.push(idx);
    } else if (typeof stride === 'number' && stride >= 0) {
      terms.push(new MathOpNode('*', idx, new IntImmNode(stride as number)));
    } else {
      const dynStride = computeDynamicStride(buffer, i, shapeParamMap);
      terms.push(new MathOpNode('*', idx, dynStride));
    }
  }
```

A rank-0 buffer is its base offset. A stride of 1 needs no multiply. An index that is the literal zero contributes nothing. And a stride that is not a non-negative number is *dynamic*: `computeDynamicStride` ([`flatten.ts:39`](../../../src/compiler/ir/lir/flatten.ts)) rebuilds it as the product of the extents to its right, reading each unknown extent either from the symbolic dimension itself or from the function's shape-parameter map. Definition 53.1's sₖ = ∏_{j>k} dⱼ, emitted as an expression rather than evaluated as a number.

### Accumulator detection

`detectAccumulator` ([`accumulator.ts:59`](../../../src/compiler/passes/lowering/accumulator.ts)) walks Definition 53.4's clauses in order, cheapest first. The body must be one `BufferStoreNode`; its value must be a `MathOpNode` whose operator is in `ACCUMULATOR_OPS` (`+`, `*`, `max`, `min`); one side must be a load of the buffer being stored; and then:

```ts
  const storeKey = indicesKey(store.indices);
  const loadKey = indicesKey(loadSide.indices);
  if (storeKey !== loadKey) return null;
  if (storeKey.includes('?')) return null;
  ...
  const resolvedKey = indicesKey(outerIndices);
  if (resolvedKey.includes('?')) return null;
  if (resolvedKey.includes('$' + loopVarName)) return null;
```

`indicesKey` renders the subscripts as a canonical string, with `?` standing for any node kind it does not model. The first test is "the load and the store name the same element". The last is Counterexample 53.6: after substituting the block's bindings back to the enclosing loop variables, the address must not mention the loop variable. A `?` anywhere is treated as failure — an unmodelled index shape is refused rather than guessed at, which is how a legality test whose false positive is a wrong answer ought to fail.

`lowerAccumulator` ([`tensor_to_lir.ts:217`](../../../src/compiler/passes/lowering/tensor_to_lir.ts)) then builds the node: a fresh local `_acc_N` registered in the metadata's `locals` table, an `initLoad` at the flattened outer address, the value side with the block's bindings substituted in, and a `flushStore` at the same address whose `value` is `null` — the backend supplies the accumulator local.

### The metadata block

`scanMetadata` ([`scanner.ts:11`](../../../src/compiler/ir/lir/scanner.ts)) walks the function once and answers the questions every backend asks before it emits its first line.

| Field | The question it answers | Read by |
|---|---|---|
| `locals` | which scalar variables must be declared, and at what type | WASM; CPU, via the accumulator's dtype |
| `threadBindings` | which loops became thread indices, at what extents | CUDA, WebGPU |
| `sharedBuffers` | which allocations are in shared scope | CUDA, WebGPU |
| `memoryLayout` | where each buffer sits in one flat address space | WASM |
| `externCalls` | which math functions must be imported | WASM |
| `zeroBuffers`, `constantBuffers` | which buffers are only ever written one constant | CPU |
| `usedBuffers`, `allocatedBuffers`, `paramBuffers` | which buffers must be declared, and which arrive as arguments | CPU |

Two are worth pausing on. `memoryLayout` ([`scanner.ts:131`](../../../src/compiler/ir/lir/scanner.ts)) assigns every buffer an offset in one flat address space, aligned to 16 bytes — Chapter 50's arena, computed a second time, at a different granularity, for the one target that has no other way to name a buffer. And `detectZeroBuffers` ([`scanner.ts:160`](../../../src/compiler/ir/lir/scanner.ts)) records buffers all of whose writes are the literal zero, so the CPU backend can skip both the allocation and the stores and fold the loads to `0`.

### The pass layer over LIR

There is one, and it has one pass. `buildLirPipeline` ([`lir_pipeline.ts:6`](../../../src/compiler/pipeline/lir_pipeline.ts)) opens a `pre` phase for registered passes, runs `FlatIndexSimplifyPass`, and opens a `post` phase. The simplify pass re-runs [Chapter 37](../../part6/ch37-proving-things-about-indices/README.md)'s arithmetic simplifier over the newly built offset expressions, because flattening is exactly the step that creates `i * 1`, `+ 0` and `(i // c) * c + (i % c)` in bulk.

Then `verify:lir` runs `verifyLIR` ([`verifier.ts:30`](../../../src/compiler/ir/lir/verifier.ts)), whose one real check is scope: every `VariableNode` must be bound by an enclosing loop, `LetStmtNode`, `LIRBindingsNode`, accumulator local, or shape parameter. A narrow contract, and the one that catches the mistake this pass can actually make — an accumulator or a binding that captures a variable out of scope.

### The backend contract

Everything from here to the end of Part X is five implementations of one interface, so it is worth stating the interface once.

> **Definition 53.7 (The backend contract).** **(invariant)** A backend is a function from an `LIRFunc` and a `TargetFeatures` to a `CompiledKernel`: a **name**, a **source** string, the **target**, and a **metadata** object. `BackendPipeline.compile` ([`pipeline.ts:14`](../../../src/backend/pipeline.ts)) selects the backend by `target.kind` from a registry, and every consumer downstream — the runtime, the executor plan, the trace — sees only that four-field object.

The name and the target are bookkeeping. The other two fields are where the five implementations differ, and they differ in different ways.

**The source is a string, and it need not be a program.** Chapter 54 puts JavaScript in it, Chapter 55 a WAT module, Chapters 56 and 57 CUDA C and WGSL. Chapter 58 puts the empty string in it, because there is no program — the kernel is a library call the metadata describes. Nothing downstream branches on which.

**The metadata is target-shaped**, and it is where each backend states what the runtime has to do that the text cannot say:

| Target | What the metadata carries | Why the source cannot say it |
|---|---|---|
| CPU | `paramCount` | nothing else is needed; the text is a complete function |
| WASM | `memoryPages`, `bufferOffsets`, `imports`, `params`, `parallel` | the module needs a memory sized and math functions supplied before it can be instantiated |
| CUDA | `blockDim`, `gridDim`, `sharedMemBytes`, `params`, `outputIndices`, `scratch`, `launchDiagnosis` | a launch geometry is an argument to the launch, not a statement in the kernel |
| WebGPU | `workgroupSize`, `dispatchSize`, `sharedMemBytes`, `params`, `bindings`, `launchDiagnosis` | the binding table must be reproduced on the host side to build a bind group |
| external | the provider's descriptor and `outputIndices` | there is no kernel to describe |

**And the constant buffers are the one thing the contract adds on the way out.** If the function carries folded weights, `BackendPipeline.compile` attaches them to the metadata as `constBuffers`, and refuses outright when the target's `supportsConstBuffers` is false ([`pipeline.ts:48`](../../../src/backend/pipeline.ts)) — the one place in this part where a backend's *capabilities* rather than its output are checked.

Definition 53.7 is what Part XI's runtime is built against, and Chapter 58 §58.6 returns to it once the five implementations exist.

## 53.5 Lab — what lowering to LIR throws away

```bash
node docs/part10/ch53-lir-the-third-ir/labs/01-what-lir-throws-away.mjs
```

The first table is the whole pass in one view — a census of node kinds on the same matmul before and after:

```
  node kind               TIR  LIR
  BlockNode                 2    0
  BufferLoadNode            3    0
  BufferStoreNode           2    0
  ForNode                   5    4
  IntImmNode                5   10
  LIRAccumulatorNode        0    1
  LIRBindingsNode           0    1
  LIRFlatLoadNode           0    3
  LIRFlatStoreNode          0    2
  MathOpNode                2   11
  VariableNode             15   10

  unchanged: FloatImmNode SeqNode
```

Read it as three separate facts. **Every access node changed kind** — five subscripted accesses became five flat ones. **`MathOpNode` went from 2 to 11**: the nine new ones are the address arithmetic that used to be implicit in the subscripts, now written down. And **one `ForNode` disappeared** while an `LIRAccumulatorNode` appeared — the `k` loop is inside the accumulator now, not beside it.

`VariableNode` falls from 15 to 10, which is the `LIRBindingsNode` story. Two blocks declared six iteration variables between them; the accumulator substituted the matmul block's three away entirely, and the init block's survive as a binding node.

The second section runs `detectAccumulator` on each of the five loops:

```
  for di0_12    extent  4  not an accumulator
  for di1_14    extent  5  not an accumulator
  for ls0_6     extent  4  not an accumulator
  for rs0_7     extent  5  not an accumulator
  for c0_8      extent  6  accumulator over '+'
```

Exactly one fires, and it is the contraction axis. The two `ls`/`rs` loops fail Definition 53.4's body clause — their bodies are not a single store, they contain the inner loop — and the two `di` loops of the zero-init nest fail the operator clause, since a store of the constant `0` is not a `MathOpNode` at all.

Then the metadata:

```
  locals            11 (_acc_0:f32)
  buffer offsets    buf_1@0  buf_3@96  buf_5@224
  total bytes       304 (alignment 16)
  extern calls      (none)
  zero buffers      (none)
  param buffers     buf_1 buf_3 buf_5
```

Ten of the eleven locals are `i32` loop and binding variables; the eleventh is the accumulator, and its dtype was inferred from the buffer it accumulates into. The offsets are the flat layout: 4×6 floats is 96 bytes, so `buf_3` starts at 96; 6×5 floats is 120 bytes, and the next 16-byte boundary after 96 + 120 is 224.

Finally, the statement itself, before and after:

```
  TIR:  buf_5[vls0_9, vrs0_10] = (buf_5[vls0_9, vrs0_10] + (buf_1[vls0_9, vc0_11] * buf_3[vc0_11, vrs0_10]))
  LIR:  buf_5[((ls0_6 * 5) + rs0_7)] = _acc_0
        _acc_0 = (_acc_0 + (buf_1[((ls0_6 * 6) + c0_8)] * buf_3[((c0_8 * 5) + rs0_7)]))   over c0_8 < 6
```

Two loads of `buf_5` per iteration and one store became one load before the loop and one store after it. The three strides — 5, 6, 5 — are now literals in the program rather than properties of a buffer object.

**Try this.** Change `matmul` to `a.matmul(b).relu()` and watch the accumulator survive while a second, non-accumulating nest appears beside it. Then try `a.sum()` and note that the accumulator fires on a reduction to a scalar too, with the flush store at the constant offset `0`.

## 53.6 Lab — one LIR, four texts

```bash
node docs/part10/ch53-lir-the-third-ir/labs/02-one-lir-four-texts.mjs
```

The first table hands the *same* `LIRFunc` to all four backends and asks where the row stride 6 ended up:

```
  target   lines  where the row stride 6 shows up
  cpu          7  buf_3[((i0_4 * 6) + i1_5)] = buf_1[((i0_4 * 6) + i1_5)] * buf_1[((i0_4 *
  wasm        67  6 x i32.mul, 6 x i32.add, over 4 x (i32.const 6)
  cuda         9  buf_3[((v0_6 * 6) + v1_7)] = (buf_1[((v0_6 * 6) + v1_7)] * buf_1[((v0_6
  webgpu      13  buf_3[((v0_6 * 6) + v1_7)] = (buf_1[((v0_6 * 6) + v1_7)] * buf_1[((v0_6
```

Three of the four render the offset in near-identical infix syntax, and the fourth flattens it onto a stack. None of them computed it: they were handed a `MathOpNode` tree and walked it.

The second half is the integer division contract, which is the piece of *semantics* this chapter fixes for the rest of the part.

> **Definition 53.8 (The division operators).** **(invariant)** TIR and LIR carry four integer operators, and every backend implements them to these definitions:
>
> - `a // b` is **floor division**: the largest integer *q* with *q* ≤ a/b.
> - `a % b` is **floor modulo**: `a − (a // b) · b`, which takes the sign of *b*.
> - `a tdiv b` is **truncating division**: rounds toward zero.
> - `a tmod b` is **truncating modulo**: `a − (a tdiv b) · b`, which takes the sign of *a*.
>
> The scalar definitions live in one file, [`util/divmod.ts`](../../../src/util/divmod.ts), shared by `SymInt`, by TIR constant folding and by the four backends. `tdiv`/`tmod` are introduced by the simplifier only where it has proved the operands make the two agree, so choosing floor as the default costs nothing on the index arithmetic that dominates.

The lab shows how each backend spells them:

```
  op     cpu                              cuda                               wasm
  //     Math.floor(a[i] / b[i])          (((a[i]) - ((((a[i]) % (b[i])) +.. 28 integer instructions
  %      ((a[i] % b[i] + b[i]) % b[i])    ((((a[i]) % (b[i])) + (b[i])) % .. 22 integer instructions
  tdiv   ((a[i] / b[i]) | 0)              (a[i] / b[i])                      19 integer instructions
  tmod   (a[i] % b[i])                    (a[i] % b[i])                      15 integer instructions
```

Three languages, three amounts of work for one meaning. C's `%` truncates, so floor modulo costs the `((a % b) + b) % b` correction; JavaScript's `%` truncates too, and its `/` is floating-point, so floor division is a `Math.floor` rather than a correction. WebAssembly's `i32.div_s` truncates *and traps*, so the emitter needs a guard as well as a correction, and the price is 28 instructions where truncation costs 19.

And then the numbers, on the operands where the two conventions differ:

```
    a   b     // js   // wasm    % js    % wasm tdiv js tdiv wasm tmod js tmod wasm
    7   3         2         2       1         1       2         2       1         1
   -7   3        -3        -3       2         2      -2        -2      -1        -1
    7  -3        -3        -3      -2        -2      -2        -2       1         1
   -7  -3         2         2      -1        -1       2         2      -1        -1
    7   0         0         0       0         0       0         0       0         0

  operators on which the two disagree anywhere above: (none)
```

Row two is the point of fixing a convention: `−7 // 3` is `−3` and `−7 % 3` is `2` on both backends, because both implement Definition 53.8 rather than their host language's default.

The last rows have a zero divisor, and both columns read `0` for different reasons. JavaScript evaluates `Math.floor(7/0)` to `Infinity` and the `i32` store flattens it; WebAssembly *cannot* let the division happen, because `i32.div_s` on a zero divisor traps and aborts the module, so the emitter selects a safe divisor and then selects `0` back over the result. **Definition 53.8 fixes the sign convention and says nothing about dividing by nothing**, and the two backends arrive at the same answer by accident rather than by contract.

The last section is where the contract is not held. `remainder` on floats does not go through `MathOpNode` at all — it lowers to a `CallExternNode` named `fmod` — and each backend maps that name onto its own language:

```
  source      -7 % 3    7 % 3 -7.5 % 2  7.5 % 2
  eager            2        1      0.5      1.5
  cpu              2        1      0.5      1.5
  wasm            -1        1     -1.5      1.5
  cuda     emits fmodf(buf_1[v1_9], buf_3[v1_9])
  webgpu   emits (buf_1[v1_9] % buf_3[v1_9])
```

Eager and CPU agree, and they agree with the operation's definition. WASM's import is JavaScript's `%`; CUDA's `fmodf` and WGSL's float `%` both truncate. **One operation, two answers, reachable from the public `compile()`** — and the divergence is not in the division logic, which four backends implement from one definition, but in the road the operation took to get there.

**Try this.** Replace `remainder` with `x.floor_divide(y)` and watch all four agree — that operator goes through `MathOpNode('//')` and is therefore covered by Definition 53.8. The divergence is not about division; it is about which of two routes an operation takes to a backend.

## 53.7 Traps and limits

### One operation reaches the backends as a name, and four tables disagree about it

Definition 53.8 is enforced because `//` and `%` are *node kinds* every backend must handle, and each backend's handling was written against one shared definition. `fmod` is a *string* in a `CallExternNode`, resolved per backend against a per-backend table — `C_MATH_BASES` for CUDA, `WGSL_MATH_FUNCS` for WebGPU, a JavaScript object literal in the runtime for WASM, and an explicit special case in the CPU emitter. Nothing checks that the four tables agree, and for `fmod` they do not: the CPU emitter special-cases it to floor semantics ([`cpu/codegen.ts:533`](../../../src/backend/cpu/codegen.ts)) to match the eager operation, and the other three take their language's primitive. The fix is either to route `remainder` through `MathOpNode('%')`, which already carries the definition, or to give the extern table a semantics column the backends are checked against.

### The rest

- **A dynamically-sized buffer is given a fixed 64 KiB slab, and a bigger one overruns it silently.** `computeMemoryLayout` ([`scanner.ts:131`](../../../src/compiler/ir/lir/scanner.ts)) reserves `max(DYNAMIC_BUFFER_SLAB_BYTES, staticLowerBound × elementSize)` bytes, where the slab is 65,536 ([`scanner.ts:129`](../../../src/compiler/ir/lir/scanner.ts)), for any buffer with a non-static extent, because a flat layout has to be decided before the extents are known. WASM is the only backend that reads `memoryLayout`, so it is the only one that pays: a `dynamic_shapes` compilation is correct up to 16,384 `f32` elements per dynamic buffer and silently wrong above it. Chapter 55 §55.7 measures it.
- **The metadata scan re-implements the tree walk.** `walkTree` ([`scanner.ts:25`](../../../src/compiler/ir/lir/scanner.ts)) pushes children by a hand-written list of slot names — `body`, `stmts`, `value`, `thenBody`, `elseBody`, `initBody`, `condBody`, `loopBody`, `condition`, `a`, `b`, `expr`, `args`, `indices` — while the same file imports the schema-driven `collect` from [`ir_visitor.ts`](../../../src/compiler/ir/ir_visitor.ts) for `detectZeroBuffers` three functions later. A node kind with a child slot not on that list is scanned as a leaf, and the symptom would be a missing local or a missing extern import rather than an error.
- **`normalizeDtype` defaults to `f32` for anything it does not recognise** ([`nodes.ts:174`](../../../src/compiler/ir/lir/nodes.ts)), and its table maps `f16` and `bf16` to `f32` on purpose — half-precision values are held expanded in locals and narrowed at the store. `inferDtype` has the same default ([`nodes.ts:189`](../../../src/compiler/ir/lir/nodes.ts)). Both are reasonable, and both mean a dtype the compiler has never seen produces a working kernel that computes in the wrong width rather than an error.
- **Every backend still has a second front end.** `CodegenFunc` is `PrimFunc | LIRFunc` ([`codegen_utils.ts:8`](../../../src/backend/codegen_utils.ts)) and each backend opens with `const isLIR = func.type === 'LIRFunc'`, taking the metadata from the LIR or re-deriving it by scanning otherwise. The pre-LIR path is what the backend unit tests and Parts VII–VIII's labs exercise; the compiled path is LIR. That is two implementations of the metadata scan per backend, and a third implementation of index flattening: `flattenRowMajorIndex` ([`index_emit.ts:7`](../../../src/backend/index_emit.ts)) is Definition 53.1 again, at the level of strings, and it is what runs when a backend is handed a raw `PrimFunc`.
- **The accumulator's `initLoad` reads the destination.** Theorem 53.5 starts the fold from `D[w]`'s existing contents, which is correct and which makes the promotion depend on the zero-init nest having run first. The CPU backend has a rule that elides redundant zero fills ([`cpu/codegen.ts:584`](../../../src/backend/cpu/codegen.ts)); it protects itself by checking that the buffer is not read elsewhere, and the accumulator's init load is exactly such a read.
- **The LIR pass layer has one pass.** `LirFuncPass`, `LirPassManager` and the `pre`/`post` phase registry ([`lir_pipeline.ts`](../../../src/compiler/pipeline/lir_pipeline.ts)) are a full pass infrastructure carrying `FlatIndexSimplifyPass`. Target-independent peephole work on addresses belongs there and is not there: the address CSE Chapter 55 §55.4 finds inside the WASM backend, and the constant-buffer elision inside the CPU backend, are both LIR-level rewrites currently written three and four times over.

## 53.8 Read the tests

- [`tests/compiler/ir/lir/flatten.test.js`](../../../tests/compiler/ir/lir/flatten.test.js) — Definition 53.1 term by term: rank 0, rank 1 with and without a base offset, unit strides elided, a literal-zero index dropped, and the dynamic-stride product.
- [`tests/compiler/ir/lir/scanner.test.js`](../../../tests/compiler/ir/lir/scanner.test.js) — what each metadata field records, including the 16-byte alignment of the flat layout and the zero-buffer detection.
- [`tests/compiler/ir/lir/verifier.test.js`](../../../tests/compiler/ir/lir/verifier.test.js) — the scope contract: a variable used outside its binder is an error, and each of the five binders is checked.
- [`tests/compiler/passes/lowering/tensor-to-lir.test.js`](../../../tests/compiler/passes/lowering/tensor-to-lir.test.js) — the pass end to end, including the accumulator cases and Counterexample 53.6's shape.
- [`tests/compiler/pipeline/lir-pipeline.test.js`](../../../tests/compiler/pipeline/lir-pipeline.test.js) — that the LIR phase runs where the phase list says it does, and that a registered `pre`/`post` pass is picked up.

---

**Next:** [Chapter 54 — Generating JavaScript for the CPU](../ch54-javascript-for-the-cpu/README.md), the first backend, and the one whose target language is the language the compiler is written in.
