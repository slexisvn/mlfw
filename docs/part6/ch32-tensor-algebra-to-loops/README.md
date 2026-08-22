# Chapter 32 — From tensor algebra to loop nests

Every program in this book so far has been a graph of whole-tensor operations. `add` takes two tensors and produces a tensor; nothing in the IR says in what order the elements are computed, or where they live while it happens.

No machine executes that. A CPU executes loops over addresses. So somewhere between the graph and the backend there is a step that invents an order and invents the addresses — and it is the point of no return in the pipeline, because everything Parts III to V do is expressed over values and every decision Parts VII to IX make is expressed over iterations.

This part is about the representation on the other side of that step.

## 32.1 The problem: `add` is not a program

Take the smallest possible graph:

```
%2 = add(%0, %1) : tensor<2x2xf32>
```

This is a true statement about three tensors. To run it, a backend needs answers to questions the statement does not contain:

- **In what order** are the four elements computed? Row by row, column by column, all at once on four threads?
- **Where** do they live? At which offset in which buffer, with what stride between rows?
- **How many times** is each input read, and can a read be shared with a neighbouring computation?

You could answer those questions inside each backend. That is what an eager runtime does, and Chapter 4 measured what it costs: one kernel per operation, one round trip to memory per kernel. It also does not scale as a compiler design. The registry holds 96 operations; the book ships four backends; and fusion (Chapter 24) invents new composite operations at compile time that no backend could have a hand-written kernel for.

The alternative is to answer the questions **once**, in a representation that can express the answers, and let each backend translate that representation instead of the graph. That representation is the second IR.

## 32.2 Intuition: two levels because there are two kinds of statement

Look at what the earlier parts actually said about programs.

*"These two operations read the same value, so compute it once."* (CSE, Chapter 19.) *"This chain of five elementwise operations touches memory five times, so make it one operation."* (Fusion, Chapter 22.) *"The derivative of `exp` is `exp` times the incoming cotangent."* (Chapter 28.) Every one of these is a statement about **values**: which value feeds which, what a value is equal to, which value the derivative needs.

Now look at what the later parts want to say. *"Split this loop into tiles of 32 so the working set fits in L1."* *"Bind this axis to `threadIdx.x`."* *"Vectorise the innermost eight iterations."* *"This buffer's last read is here, so its memory can be reused from here on."* Every one of these is a statement about **iterations and addresses**.

Neither vocabulary can express the other. A graph has no loops to tile. A loop nest has no dataflow edges to common-subexpression. So the compiler uses one IR for each, and the map between them runs one way.

> Not every compiler splits here, and the split is not free. A single IR carrying both — MLIR's tensor and affine dialects in one module, say — lets a pass see across the boundary. The cost is that every pass then has to be written against a much larger language. This compiler takes the simpler trade: two small IRs and a lowering step, at the price that after Chapter 32 nothing can look back.

## 32.3 Theory

Fix a program as a finite set of statements, each writing one scalar.

> **Definition 32.1 (Iteration domain).** The *iteration domain* of a statement `S` is the set of integer vectors `(i₁,…,i_d)` for which `S` executes, one component per enclosing loop. In this compiler every domain is a rectangle: `0 ≤ i_k < e_k` for extents `e_k` that are integer literals or symbolic expressions in the function's shape parameters.

> **Definition 32.2 (Loop nest).** A *loop nest* for `S` is an ordered list of `d` loops whose bodies contain `S`, together with an assignment of each loop variable to a component of the iteration domain. The nest fixes the execution order: *lexicographic* in `(i₁,…,i_d)` — the order a dictionary sorts words in, so the outermost loop varies slowest and the innermost fastest, exactly as nested `for` loops run.

The graph fixes neither of these. Lowering picks both.

> **Definition 32.3 (Lowering).** *Lowering* is a map `L` from a graph function to a `PrimFunc`: it assigns a buffer to every value, and replaces every operation with statements over those buffers wrapped in a loop nest.

Two properties of `L` decide how the rest of the book is organised.

> **Theorem 32.4 (Lowering is not injective, stated here).** There exist distinct graph functions `F ≠ G` with `L(F) = L(G)`.

*Proof.* By exhibition. `add(x, y)` where `x` has shape `[4,3]` and `y` has shape `[1,3]` broadcasts implicitly: the elementwise rule reads the size-1 axis with a literal `0`. `add(x, broadcast_in_dim(y, [4,3], [0,1]))` is a two-operation graph, and its broadcast is folded into the same index expression by the alias path of §32.4. The two `PrimFunc`s are character-identical, which §32.7 checks. ∎

> **Corollary 32.5 (Lowering is irreversible).** No inverse `L⁻¹` exists. In particular, given only `L(F)`, no procedure can determine which of the graph functions in `L⁻¹(L(F))` produced it.

**Read the corollary as the precise statement it is, not as the stronger one it invites.** "The information is gone forever" is the natural gloss and it is too strong in two directions worth separating.

*What genuinely cannot be done.* Recover the graph *uniquely* from the TIR alone. Theorem 32.4 exhibits two graphs with character-identical output, so no procedure taking only that output can tell you which it was. That is a proof, and everything the pipeline order in the next paragraph rests on follows from it.

*What is not ruled out.* First, **inference is not determination**: a heuristic that guesses "this nest was probably a broadcast add" can be right nearly always, and pattern-matching a loop nest back to a high-level operation is exactly what a library-offload pass does — Chapter 58's external codegen recovers `dot` from its shape and index structure for precisely this reason. Such a pass is unsound in the sense of Theorem 32.4 and useful anyway, because being wrong costs a missed optimization rather than a wrong answer. Second, **extra metadata changes the problem entirely**: nothing stops a lowering rule from *recording* what it came from, and this compiler already does this in places — block names carry the originating operation, and `FuncAttr` entries carry facts the IR cannot express. A fact that was deliberately preserved is not a fact recovered from the mapping; it is a fact carried alongside it.

So the useful form of the corollary is: **the mapping preserves what the rules chose to preserve, and nothing downstream can appeal to the mapping for anything else.** If a later pass needs a graph-level fact, the answer is to carry it, not to reconstruct it — and the reason to know Theorem 32.4 is to recognize when you are about to try the second.

This is Definition 6.1 arriving where it bites. It is also why the pipeline order is what it is: differentiation (Part V) before lowering because the chain rule needs dataflow; fusion (Part IV) before lowering because a fusion decision is a claim about which values are internal; layout (Chapter 25) before lowering because a layout choice becomes a stride, and a stride is not a choice any more.

> **Definition 32.6 (Lowering rule).** A *lowering rule* for an operation `o` is a function taking `o`, the buffers assigned to its operands, and the buffers assigned to its results, and returning a statement.

Note what Definition 32.6 does **not** take: the rest of the program. A rule sees one operation. The driver is what sees the function, and §32.4 shows the two decisions it makes that no rule could.

## 32.4 In mlfw: one function, five passes over the graph

[`passes/lowering/graph_to_tensor.ts`](../../../src/compiler/passes/lowering/graph_to_tensor.ts), 206 lines, is the whole driver. `lowerGraphToPrimFunc` ([`graph_to_tensor.ts:75`](../../../src/compiler/passes/lowering/graph_to_tensor.ts)) walks the function five times, and each walk exists for a reason worth naming.

**One — parameters.** Every graph argument gets a variable and a buffer:

```ts
  for (const arg of graphFunc.args) {
    const v = ctx.allocVar('arg');
    params.push(v);
    bufferMap.set(v, ctx.getOrAllocBuffer(arg));
  }
```

**Two — results, and the aliasing problem.** A `PrimFunc` has no return value; outputs are parameters, written in place. So the returned values become parameters too — unless a returned value is also an input, or is returned twice, in which case writing it in place would corrupt it ([`graph_to_tensor.ts:99`](../../../src/compiler/passes/lowering/graph_to_tensor.ts)):

```ts
    const srcBuf = ctx.getOrAllocBuffer(retOp.getOperand(i));
    if (inputBuffers.has(srcBuf) || usedReturnBuffers.has(srcBuf)) {
      const outBuf = ctx.allocFreshBuffer(retOp.getOperand(i));
      bufferMap.set(v, outBuf);
      copyPairs.push({ src: srcBuf, dst: outBuf });
```

A function that returns its own input compiles to a copy loop, emitted at the end. That is the first thing no rule could have done: it is a fact about the function's signature, not about any operation.

**Three — constants first.** Constant operations are lowered ahead of everything else, in graph order, and skipped when the main loop reaches them. They are separated because `lowerConstant` ([`lowering_registry.ts:476`](../../../src/compiler/passes/lowering/lowering_registry.ts)) does not always emit a statement: a weight folded into the graph becomes a `constBuffer` attached to the function rather than a store, and a *scalar* constant is given a rank-matched size-1 buffer tagged for broadcast, which its consumers then read with a literal `0` in every axis. Both need the buffer registered before any rule asks for it.

**Four — the main loop**, in topological order ([`graph_to_tensor.ts:122`](../../../src/compiler/passes/lowering/graph_to_tensor.ts)):

```ts
  for (const op of topologicalOps(graphFunc)) {
    if (isTerminatorOp(op.opName)) continue;
    if (isConstantOp(op.opName)) continue;

    if (op.opName === 'fusion') {
      if (canLowerAsElementwiseFusion(op)) {
        stmts.push(lowerFusion(ctx, op));
      } else {
        lowerFusionAsIndividualOps(ctx, op, stmts);
      }
      continue;
    }
```

then the broadcast special case below, then:

```ts
    const rule = getLoweringRule(op.opName, target, context);
    if (!rule) throw new Error(`No lowering rule defined for op: ${op.opName}`);

    const inputs: Buffer[] = new Array(op.numOperands);
    for (let i = 0; i < op.numOperands; i++) inputs[i] = ctx.getOrAllocBuffer(op.getOperand(i));
    const outputs: Buffer[] = new Array(op.numResults);
    for (let i = 0; i < op.numResults; i++) outputs[i] = ctx.getOrAllocBuffer(op.getResult(i));

    const stmt = rule(ctx, op, inputs, outputs);
    if (stmt) stmts.push(stmt);
```

`getOrAllocBuffer` is memoised on the `Value` ([`lowering_registry.ts:118`](../../../src/compiler/passes/lowering/lowering_registry.ts)), so the buffer an operation writes is the same object its consumer later reads. **The graph's dataflow edges become buffer identity.** That is where the use-def graph of Chapter 8 stops existing and a set of shared mutable buffers takes its place — and it is why Part IX has to reconstruct liveness from scratch instead of reading it off the IR.

**Five — shape parameters.** Symbolic dimensions collected along the way become extra trailing parameters, and a symbol with no input dimension to bind it to is a hard error ([`graph_to_tensor.ts:195`](../../../src/compiler/passes/lowering/graph_to_tensor.ts)):

```ts
  for (const [name, node] of ctx.symVars) {
    if (!seenSp.has(node.name)) {
      throw new Error(`Symbolic dimension '${name}' has no input dimension to bind it to at runtime`);
    }
  }
```

### The decision no rule could make

Between the fusion case and the rule lookup sits the second driver-level decision ([`graph_to_tensor.ts:135`](../../../src/compiler/passes/lowering/graph_to_tensor.ts)):

```ts
    if (isBroadcastOp(op.opName)
        && !returnedValues.has(op.getResult(0))
        && op.getOperand(0).getUsers().length === 1
        && op.getResult(0).getUsers().every((u) => broadcastViewSafeForUser(op.getResult(0), u))) {
      const srcBuf = ctx.getOrAllocBuffer(op.getOperand(0));
      ...
      srcBuf.broadcastDims = broadcastDims;
      ctx.bufferMap.set(op.getResult(0), srcBuf);
      continue;
    }
```

A broadcast whose every consumer can absorb it becomes **no statement at all**: the result value is mapped to the *source* buffer, tagged with the axes it expands along, and the consumers' rules read it with a literal `0` in those axes. The condition is a question about the operation's users, which Definition 32.6 forbids a rule from asking, so the driver asks it.

`broadcastViewSafeForUser` ([`graph_to_tensor.ts:37`](../../../src/compiler/passes/lowering/graph_to_tensor.ts)) recurses through `fusion` regions down to the block arguments, because a fused consumer is only safe if everything inside it is.

### Where it is called from

[`compiler.ts:467`](../../../src/compiler/pipeline/compiler.ts), one function at a time, into a fresh `TirModule`:

```ts
    this._eachFunc(graphModule, 'lowering', trace, errors, failed, resilient, (func) => {
      const ft0 = performance.now();
      const primFunc = lowerGraphToPrimFunc(func, this.config.target as unknown as null, this.context as unknown as null);
```

and immediately afterwards the TIR pipeline of [`tir_pipeline.ts:13`](../../../src/compiler/pipeline/tir_pipeline.ts) runs over the result: scheduling (Part VII), loop partitioning, simplification (Chapter 37), memory planning (Part IX).

## 32.5 The language, in one table

`TirNode` ([`ir/tensor/nodes.ts:426`](../../../src/compiler/ir/tensor/nodes.ts)) is a union of 21 classes. That is the entire second IR.

| Group | Nodes | What they say |
|---|---|---|
| Function | `PrimFunc` | parameters, a buffer map, shape parameters, one body |
| Control | `ForNode`, `IfThenElseNode`, `WhileNode`, `SeqNode` | order |
| Structure | `BlockNode`, `BlockRealizeNode` | an isolated iteration domain and its bindings (Chapter 33) |
| Memory | `BufferStoreNode`, `BufferLoadNode`, `AllocateNode`, `LetStmtNode`, `VecCopyNode` | where |
| Scalar expression | `MathOpNode`, `CompareNode`, `CastNode`, `CallExternNode`, `VariableNode`, `IntImmNode`, `FloatImmNode` | what to compute per element |
| Other | `EvaluateNode`, `SyncThreadsNode` | side effects, barriers |

Twenty-one node kinds against the graph IR's 96 registered operations. That ratio is the point of the level change: everything specific to *what a tensor operation means* is spent during lowering, and what comes out is written in a language with almost no vocabulary — which is exactly what makes four backends and a search space of schedules tractable.

## 32.6 Lab — one program, three levels

```bash
node docs/part6/ch32-tensor-algebra-to-loops/labs/01-one-op-three-levels.mjs
```

Chapter 6's lab showed the same three levels for a whole `Sequential` model, as evidence that the pipeline exists. This one shows them for the smallest program there is, because the middle picture is now the subject rather than the evidence:

```
=== 1. graph IR — whole tensors, no indices ===
  module @Object {
    func @Object(%0: tensor<2x2xf32>, %1: tensor<2x2xf32>) -> (tensor<2x2xf32>) {
      %2 = add(%0, %1) : tensor<2x2xf32>
      return(%2)
    }
  }

=== 2. TIR — loops, buffers, one scalar store ===
  prim_func Object(arg_0, arg_2, ret_4) {
    buf_1 = buffer_map(arg_0, shape=[2,2], dtype=f32)
    buf_3 = buffer_map(arg_2, shape=[2,2], dtype=f32)
    buf_5 = buffer_map(ret_4, shape=[2,2], dtype=f32)
    for i0_6 in 0..2 {
      for i1_7 in 0..2 {
        block add_block_0 {
          bind v0_8 = i0_6
          bind v1_9 = i1_7
          reads([buf_1[...], buf_3[...]])
          writes([buf_5[...]])
          buf_5[v0_8, v1_9] = (buf_1[v0_8, v1_9] + buf_3[v0_8, v1_9])
        }
      }
    }
  }

=== 3. emitted CPU source — flat offsets ===
  function Object(buf_1, buf_3, buf_5) {
    for (let i0_6 = 0; i0_6 < 2; i0_6++) {
      for (let i1_7 = 0; i1_7 < 2; i1_7++) {
        buf_5[((i0_6 * 2) + i1_7)] = (buf_1[((i0_6 * 2) + i1_7)] + buf_3[((i0_6 * 2) + i1_7)]);
      }
    }
  }
```

Follow the result through the three levels. In the graph it is `tensor<2x2xf32>` — a type. In the TIR it is a buffer with a shape and a rank-2 subscript. In the emitted source the rank is gone and the subscript is `i*2 + j`. Each step spends information for something the next step needs.

Note also the third parameter. `ret_4` is an output *parameter*: §32.4's second walk, visible.

### The running example, as a loop nest

Chapter 1 promised that the two-layer network would be shown at every level, and this is the level it has been waiting for. `Sequential(Linear(2,8), ReLU(), Linear(8,1))` on a 2 × 2 input, lowered with fusion on:

```
prim_func Sequential(arg_0, arg_2, arg_4, arg_6, arg_8, ret_10) {
  buf_1 = buffer_map(arg_0, shape=[2,2], dtype=f32)
  buf_3 = buffer_map(arg_2, shape=[8,2], dtype=f32)
  buf_5 = buffer_map(arg_4, shape=[8], dtype=f32)
  buf_7 = buffer_map(arg_6, shape=[1,8], dtype=f32)
  buf_9 = buffer_map(arg_8, shape=[1], dtype=f32)
  buf_11 = buffer_map(ret_10, shape=[2,1], dtype=f32)
  buf_12[0, 0] = 0
  for di0_20 in 0..2 {
    for di1_22 in 0..8 {
      block matmul_init_0 {
        ...
        buf_13[div0_21, div1_23] = 0
      }
    }
  }
  for ls0_14 in 0..2 {
    for rs0_15 in 0..8 {
      for c0_16 in 0..2 {
        block matmul_1 {
          ...
          buf_13[vls0_17, vrs0_18] = (buf_13[vls0_17, vrs0_18] + (buf_1[vls0_17, vc0_19] * buf_3[vrs0_18, vc0_19]))
        }
      }
    }
  }
  for i0_25 in 0..2 {
    for i1_26 in 0..8 {
      block fusion_block_2 {
        ...
        buf_24[v0_27, v1_28] = max((buf_13[v0_27, v1_28] + buf_5[v1_28]), buf_12[0, 0])
      }
    }
  }
  ... the second layer, identically shaped ...
}
```

Six things to read off it, each of which you have met before under a different name.

**The layers are six buffers.** `buf_3` and `buf_5` are the first `Linear`'s weight and bias; `buf_7` and `buf_9` the second's. What was `this.l1.weight` in Chapter 1 became a graph argument in Chapter 2 and is now a named region of memory with a shape and a dtype.

**The `dot` became four loops and two blocks.** One nest zeroes the accumulator, one accumulates. That is the two-block shape §34.6 catalogues for every contraction, and the reason the output buffer is written before it is read.

**The transpose is still absent.** `buf_3[vrs0_18, vc0_19]` reads the weight with its indices swapped — the canonicalization from Chapter 2 §2.6, now visible as a subscript rather than as a missing operation.

**The ReLU's zero is a rank-2 buffer with one element.** `buf_12[0, 0] = 0` at the top, read as `buf_12[0, 0]` inside the loop. That is §32.4's scalar-constant path: a size-1 buffer tagged for broadcast rather than a 2 × 8 tensor of zeros.

**`add` and `maximum` are one statement** inside `fusion_block_2` — the fusion decided in Chapter 3's log, arriving here as a single expression in a single nest.

**`buf_13` and `buf_24` are new.** Neither is a parameter. They are the intermediates the graph carried as values and TIR must carry as storage, and deciding how few of them a program needs is Part IX.

### The size of the change

```
=== the same program, counted at each level ===

  fusion   graph ops   TIR loops   TIR blocks   TIR lines   source lines
  false           6          16            7          90             44
  true            7          14            6          79             38
```

Six operations become sixteen loops. That expansion is what this part is about, and it is also why the earlier parts had to run first: a pass before lowering sees six things, and the same pass after lowering would see ninety lines.

The fusion row is worth a second look. Fusion *adds* a graph operation — the `fusion` op wrapping a region — and *removes* two loops and one block. Chapter 22's argument about memory traffic, denominated in loop nests.

## 32.7 Lab — the rule is not the whole story

```bash
node docs/part6/ch32-tensor-algebra-to-loops/labs/02-lowering-reads-the-users.mjs
```

The same `broadcast_in_dim`, with the same operand and the same attributes, twice:

```
=== broadcast_in_dim -> mul   (consumer is elementwise) ===
  graph:
    %1 = broadcast_in_dim(%0) {broadcast_dimensions = [0, 1], result_shape = [4, 3]} : tensor<4x3xf32>
    %3 = mul(%1, %2) : tensor<4x3xf32>
  TIR:
    buf_3[v0_7, v1_8] = (buf_1[0, v1_8] * buf_4[])
  distinct buffers: 3   loop nests: 1

=== broadcast_in_dim -> sum   (consumer is a reduction) ===
  graph:
    %1 = broadcast_in_dim(%0) {broadcast_dimensions = [0, 1], result_shape = [4, 3]} : tensor<4x3xf32>
    %3 = reduce(%1, %2) {dimensions = [0], reduce_type = "sum"} : tensor<3xf32>
  TIR:
    buf_5[v0_8, v1_9] = buf_1[0, v1_9]
    ...
  distinct buffers: 4   loop nests: 3
```

One buffer more and two loop nests more, for an identical operation. Twelve elements are written and read back that the first version never touches. `reduce` is not on `BROADCAST_VIEW_SAFE_EXTRA` ([`graph_to_tensor.ts:31`](../../../src/compiler/passes/lowering/graph_to_tensor.ts)), and it could not be: absorbing the broadcast would mean rewriting the reduction's extent, not merely its subscript.

And Theorem 32.4, executed:

```
=== two graphs, one loop nest ===

  x.add(row)                                  -> add_block
  x.add(broadcast_in_dim(row, [4,3], [0,1]))  -> add_block

  identical TIR: true
```

A one-operation graph and a two-operation graph produce character-identical output. Downstream, nothing can distinguish them, and nothing should need to.

**Try this.** Add a second consumer to the broadcast in the first case — `b.mul(2.0).add(b.sum(0))` — and watch the view collapse: `getUsers().every(...)` fails, the broadcast materialises, and both consumers read the expanded buffer.

## 32.8 Traps and limits

- **Lowering is a phase, not a pass.** It runs outside the pass manager of Chapter 15, so it has no CHANGED/UNCHANGED verdict, no analysis invalidation, and no per-pass verification. `_eachFunc` catches its exceptions and records a `CompilationError` ([`compiler.ts:452`](../../../src/compiler/pipeline/compiler.ts)); that is the whole error contract.
- **A missing rule is a runtime throw, not a diagnostic.** [`graph_to_tensor.ts:155`](../../../src/compiler/passes/lowering/graph_to_tensor.ts) throws when `getLoweringRule` returns nothing, so a module with three unlowerable operations reports one. There is a `hasLoweringRule` predicate ([`lowering_registry.ts:68`](../../../src/compiler/passes/lowering/lowering_registry.ts)) that could check a whole module up front and name all three; nothing uses it that way. The same shape as Chapter 31's `findUnsupportedGradOps`.
- **The order of `stmts` is the order of execution, and it is fixed here.** Topological order is *an* order; nothing later reorders whole statements except `MemorySchedulePass` (Chapter 52). Every scheduling primitive in Part VII works inside one nest.
- **There is no parser.** Chapter 13's round-trip property — print, edit by hand, parse, get the same module — has no analogue here. `ir/tensor/` contains a printer and no parser, so the text form is a report, not a format, and every TIR test constructs nodes directly.
- **The printer is not total.** `TensorIRPrinter` dispatches on the node's type name and falls back to `[UnknownNode: ...]` ([`printer.ts:36`](../../../src/compiler/ir/tensor/printer.ts)). It implements 17 visitors for 21 node kinds; `WhileNode`, `SyncThreadsNode` and `VecCopyNode` have none, so any function containing a barrier — every lowered `scan`, for one — prints with a hole in it. `BlockRealizeNode` is the fourth missing visitor and is not a gap: `visitBlockNode` prints it inline.
- **A unary operator prints as nothing.** `visitMathOpNode` emits the operator only when the node has a second operand ([`printer.ts:181`](../../../src/compiler/ir/tensor/printer.ts)), so `neg` prints as `(x)` and so does `logical_not`. The node is correct and the backends read the node, so this costs nothing at runtime and costs a reader of §34.7's table a double take.
- **`target` and `context` are passed through untyped.** `lowerGraphToPrimFunc(func, this.config.target as unknown as null, this.context as unknown as null)` ([`compiler.ts:473`](../../../src/compiler/pipeline/compiler.ts)) casts two real arguments to `null` to satisfy a signature that declares them nullable with a narrower type. The values arrive intact; the cast means the type checker is not checking this call.

## 32.9 Read the tests

- [`tests/compiler/passes/lowering/registry.test.js`](../../../tests/compiler/passes/lowering/registry.test.js) — rule registration, target overrides, and the priority order that resolves them.
- [`tests/compiler/passes/lowering/coverage.test.js`](../../../tests/compiler/passes/lowering/coverage.test.js) — which registered operations have a rule, the executable form of §34.5's table.
- [`tests/compiler/ir/tensor/tir-module.test.js`](../../../tests/compiler/ir/tensor/tir-module.test.js) — `TirModule` and the verification it runs over each `PrimFunc`.

---

**Next:** [Chapter 33 — Buffers, blocks, iteration variables](../ch33-buffers-blocks-itervars/README.md), which is the vocabulary this chapter kept borrowing: what a buffer knows about its own memory, and why the statement inside the loops is wrapped in a `block` at all.
