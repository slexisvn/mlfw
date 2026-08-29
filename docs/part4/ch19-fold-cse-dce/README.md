# Chapter 19 — Constant folding, CSE, and dead code elimination

Part III built the machinery. This is the first chapter that uses it for something.

Three passes, all of them in the fixed-point group you have watched run since Chapter 3, all of them in the same directory, and all of them describable in one sentence each. Constant folding computes what can be computed now. Common subexpression elimination keeps one copy of a computation that appears twice. Dead code elimination removes what nobody reads.

Two of those sentences are easy to implement. The third is not, and the reason it is not is the subject of the second half of the chapter.

## 19.1 The problem: the graph contains work nobody asked for

Nobody writes redundant code on purpose. It arrives anyway, from three directions.

**The user writes it without noticing.** A term appears twice in an expression; a helper is called twice with the same arguments; a debugging line survives a refactor.

**The framework generates it.** Chapter 21's decomposition turns one `softmax` into nine operations, and two `softmax` calls on the same tensor become eighteen operations of which nine are duplicates. Every `ReLU` emits a scalar zero constant and a broadcast of it, and a network with sixteen ReLUs emits sixteen of each.

**Earlier passes leave it behind.** Chapter 11's canonicalization rewrites a `dot`'s contracting dimensions so the `transpose` feeding it becomes unnecessary — but canonicalization does not delete the transpose, it merely stops using it. Somebody has to sweep.

The last one is the important one, and it is a design principle rather than an accident: **a rewrite pass should make things unnecessary, not remove them.** A pattern that both rewires an operation and cleans up after itself has to reason about whether the operand it just orphaned is used elsewhere — which is a global question inside a local rewrite. Leave the orphan, and a pass whose whole job is that global question will collect it.

## 19.2 Intuition: three questions about one operation

Stand at an operation and ask three things.

- *Are all my operands known constants?* Then I am a constant too — compute me now and replace me with my answer. **Folding.**
- *Has an operation identical to me already run?* Then use its result instead of computing again. **Common subexpression elimination.**
- *Does anybody read my result, and do I do anything besides producing it?* If neither, I can go. **Dead code elimination.**

Each of the three needs a notion of *sameness* or *observability* that is more subtle than it first looks, and each gets that notion from Chapter 11's registry rather than inventing one.

## 19.3 Theory

> **Definition 19.1 (Constant-foldable).** **(stated here)** An operation is *constant-foldable* if it is pure, has no regions, every operand is a compile-time constant, and its registry entry supplies a `fold` function. Folding replaces the operation with a constant holding `fold`'s result.

> **Definition 19.2 (Redundant).** **(stated here)** Two operations are *redundant* if they have the same opcode, the same attributes, and operands that are pairwise the same SSA values — modulo operand order when the operation is commutative (Chapter 11) — and both are pure.

Redundancy is a syntactic relation, and that is the whole reason canonicalization runs first (Chapter 17): two operations computing the same value in different spellings are not redundant until a normalizer has made their spellings identical.

> **Definition 19.3 (Dead).** **(stated here)** An operation is *dead* if it is not a terminator, none of its results has a use, and it has no side effect.

The three clauses are not symmetric in difficulty. "Not a terminator" is a lookup. "No uses" is `O(1)` on the intrusive use list from Chapter 8. "No side effect" is the hard one, because it is a claim about the world outside the dataflow graph.

> **Theorem 19.4 (Soundness of DCE).** **(stated here)** Deleting a dead operation preserves the meaning of the program, where meaning is the tuple of function results together with every effect on state outside the function.

*Proof sketch.* By SSA (Chapter 8), a value's only route to the outside is through the operations that use it. A result with no uses reaches nothing, so removing its producer cannot change any other value. The second clause of the meaning — effects on external state — is preserved exactly because the third clause of Definition 19.3 excludes operations that have any. ∎

The proof makes the dependency explicit: **DCE is sound only to the extent that the side-effect information is correct.** An operation wrongly declared pure is deleted and its effect is lost — a miscompile. An operation wrongly declared effectful is kept forever, along with everything feeding it — dead work, not a wrong answer. The asymmetry is why the analysis errs upward, and §19.7 measures what that costs.

And one more consequence, which the transitive worklist in §19.4 exists for:

> **Corollary 19.5 (DCE is a fixed point).** **(stated here)** Deleting a dead operation can make its operands' producers dead. Hence DCE is a fixed-point computation, not a single sweep.

## 19.4 In mlfw: three files, 354 lines

### Folding

[`simplify/constant_fold.ts`](../../../src/compiler/passes/simplify/constant_fold.ts). The entry condition is Definition 19.1, checked field by field ([`constant_fold.ts:88`](../../../src/compiler/passes/simplify/constant_fold.ts)):

```ts
      const def = registry.get(op.opName);
      if (!def || op.regions.length > 0) continue;
      if (def.hasSideEffects) continue;
      if (def.getMemoryEffects && def.getMemoryEffects(op).length > 0) continue;
      if (!def.fold) continue;
      if (op.numOperands === 0) continue;
```

Then each operand is resolved to a constant *recursively* ([`constant_fold.ts:41`](../../../src/compiler/passes/simplify/constant_fold.ts)), so a chain of foldable operations collapses in one visit rather than one per round, with a memo table keyed by value so a shared subexpression is resolved once.

One check is easy to miss and is the interesting one — the guard that decides whether a computed answer may be written into the graph at all ([`constant_fold.ts:22`](../../../src/compiler/passes/simplify/constant_fold.ts)):

```ts
function coerceFoldResult(value: AttrValue | undefined, dtype: string): AttrValue | undefined {
  if (value === undefined || typeof value !== 'number') return value;
  if (isIntType(dtype as ScalarDType)) {
    return Number.isInteger(value) && Number.isSafeInteger(value) ? value : undefined;
  }
  return roundToDtype(dtype, value);
}
```

Folding happens in JavaScript, whose only number type is a double. For an integer operation whose true result exceeds 2⁵³, the double has already lost the answer, and writing that answer into the graph would be a miscompile that no later stage could detect. So the fold is discarded and the operation survives to run at the target's real precision. This is the general shape of a folding hazard: **the compiler's arithmetic and the target's arithmetic are not the same arithmetic**.

The float branch is there for the same reason and was added later, which is worth knowing because the earlier version of this guard is the one most compilers ship. It read, in full:

```ts
// the earlier version — int-only, and no longer what the file contains
function isFoldResultRepresentable(value: AttrValue, dtype: string): boolean {
  if (!isIntType(dtype as ScalarDType)) return true;
  if (typeof value !== 'number') return true;
  return Number.isInteger(value) && Number.isSafeInteger(value);
}
```

**Every float dtype was waved through by that first line.** The check stopped an `i64` fold from silently losing precision to a double, and it did nothing at all for `f32`, where the same hazard is present for the same reason, only earlier: an `f32` has 24 bits of significand against a double's 53, so a fold that a double computes exactly can be a value the target rounds away.

> **Counterexample 19.6.** `16777216f + 1f` is `16777217` in a double and `16777216` in `f32`, since `2²⁴ + 1` is not representable at single precision. Folding the pair without rounding puts a constant in the graph that the machine could not have produced, and nothing downstream re-rounds a constant.

Both hazards — the integer one and the float one — are now resolved in the one function above, and a fold cannot escape either. Note the asymmetry between its two branches, which is not arbitrary: the integer path **refuses** the fold, because no rounding makes an out-of-range integer right; the float path **rounds**, because rounding is exactly what the target would have done.

`roundToDtype` ([`half.ts`](../../../src/tensor/utils/half.ts)) is `Math.fround` for `f32`, a round-trip through the half encoders for `f16` and `bf16`, and the identity for `f64` — the same single source the backends use for storage coercion.

One detail matters more than it looks. The pass folds in two places — the main loop, and the recursive resolver that collapses a chain of foldable operations in one visit — and rounding only the first would let a multi-step chain accumulate `f64` precision internally before rounding once at the end. Both go through one `foldOperation` helper, so every intermediate in a folded chain rounds exactly where execution would round it.

**And be careful about what does *not* demonstrate this.** The obvious repro is a chain like `t.add(16777216).add(1).add(-16777216)`, which does disagree with eager execution — but it never reaches the folder at all, because `t` is a runtime argument and nothing in it is all-constant. What it exercises is fusion carrying intermediates at a wider precision, which is §19.8's separate entry. Demonstrating a folder defect needs a fold: two constants, no runtime operand.

### Elimination

[`simplify/cse.ts`](../../../src/compiler/passes/simplify/cse.ts) is a scoped hash table ([`cse.ts:10`](../../../src/compiler/passes/simplify/cse.ts)) over the structural hash from Chapter 11. Scopes are pushed and popped around regions ([`cse.ts:97`](../../../src/compiler/passes/simplify/cse.ts)), which is Chapter 9's region-scope isolation being respected: an operation inside a `fusion` body may reuse a result computed outside it, but not the other way round.

Eligibility is again a registry question ([`cse.ts:46`](../../../src/compiler/passes/simplify/cse.ts)):

```ts
function isRedundancyCandidate(op: Operation): boolean {
  if (op.regions.length > 0) return false;
  const def = registry.get(op.opName);
  if (!def) return true;
  if (def.hasSideEffects) return false;
  if (def.getMemoryEffects && def.getMemoryEffects(op).length > 0) return false;
  return true;
}
```

Two operations with regions are never merged — Chapter 11's `structuralEquals` refusal, restated here so the table never even holds them.

### Elimination of the dead

[`simplify/dce.ts`](../../../src/compiler/passes/simplify/dce.ts) is Definition 19.3 and Corollary 19.5 in thirty lines. The predicate ([`dce.ts:81`](../../../src/compiler/passes/simplify/dce.ts)):

```ts
  _isDead(op: Operation, memEffects: MemoryEffectResult): boolean {
    if (isTerminatorOp(op.opName)) return false;

    for (let i = 0; i < op.numResults; i++) {
      if (op.getResult(i).hasUses) return false;
    }

    return !memEffects.hasSideEffect(op);
  }
```

and the fixed point ([`dce.ts:35`](../../../src/compiler/passes/simplify/dce.ts)):

```ts
    while (worklist.length > 0) {
      const op = worklist.pop() as Operation;
      if (!op.parentBlock) continue;
      if (!this._isDead(op, memEffects)) continue;

      const operandDefs: Operation[] = [];
      for (const consumed of this._valuesReadBy(op)) {
        const defOp = consumed.definingOp;
        if (defOp && defOp.parentBlock) operandDefs.push(defOp);
      }

      this._eraseRecursively(op);
```

The producers of everything the erased operation read go back on the list. `_valuesReadBy` ([`dce.ts:67`](../../../src/compiler/passes/simplify/dce.ts)) descends into regions, so erasing a dead `fusion` correctly releases the values its *body* was reading — a detail that is invisible until you have a region, and wrong in a way that leaks operations if you forget it.

### The analysis DCE leans on

[`analysis/memory_effect.ts`](../../../src/compiler/analysis/memory_effect.ts) is the analysis from Chapter 16 — the one pass with a preservation declaration. Effects come from four bits ([`op_registry.ts:7`](../../../src/compiler/ir/graph/op_registry.ts)):

```ts
export const SideEffectKind = Object.freeze({
  NONE: 0,
  READ: 1,
  WRITE: 2,
  ALLOCATE: 4,
  CONTROL: 8
});
```

Across all 96 operations, **four declare a non-zero mask**: `call` (`CONTROL`), `custom_call` (`WRITE`), `scatter` (`WRITE`), and `transfer` (`READ | WRITE`). Everything else is pure, which is what a functional tensor IR should look like.

The fifth case is not declared at all but computed ([`memory_effect.ts:74`](../../../src/compiler/analysis/memory_effect.ts)):

```ts
  static _foldRecursiveEffects(opKinds: Map<Operation, SideEffectMask>): void {
    const nestedKind = (op: Operation): SideEffectMask => {
      let mask = SideEffectKind.NONE;
      for (const region of op.regions) {
        for (const block of region.blocks) {
          for (const inner of block.ops()) {
            mask |= opKinds.get(inner) ?? SideEffectKind.NONE;
            mask |= nestedKind(inner);
          }
        }
      }
      return mask;
    };
```

An operation carrying `RECURSIVE_MEMORY_EFFECTS` — `if`, `while` and `scan` ([`ops/control_flow.ts:44`](../../../src/compiler/ir/graph/ops/control_flow.ts)) — inherits the union of the effects inside it. Without this, a `while` loop whose body writes would look pure, and DCE would delete the loop. Chapter 11 introduced that trait as a promise; this function is the promise being kept.

## 19.5 Lab 1 — Three passes on one program

```bash
node docs/part4/ch19-fold-cse-dce/labs/01-three-passes.mjs
```

The program is written to give each pass something to do: a duplicated subexpression, a `ReLU` whose broadcast folds, and a branch nobody reads.

```js
class Wasteful extends Module {
  forward(a) {
    const scaled = relu.forward(a.mul(2).add(1));
    const again = relu.forward(a.mul(2).add(1));
    const unused = a.exp().log();
    return scaled.add(again);
  }
}
```

Eighteen operations go in. With fusion switched off so the picture stays clean:

```
=== what each pass did ===
  round 1  constant_fold        18 -> 18 ops   foldedCount=2
  round 1  cse                  18 -> 11 ops   eliminated=7
  round 1  dce                  11 -> 8 ops   erasedCount=3
```

and eight come out:

```
module @Wasteful {
  func @Wasteful(%0: tensor<2x2xf32>) -> (tensor<2x2xf32>) {
    %1 = constant() {tensor_type = tensor<f32>, value = 2} : tensor<f32>
    %2 = mul(%0, %1) : tensor<2x2xf32>
    %3 = constant() {tensor_type = tensor<f32>, value = 1} : tensor<f32>
    %4 = add(%2, %3) : tensor<2x2xf32>
    %5 = constant() {tensor_type = tensor<2x2xf32>, value = 0} : tensor<2x2xf32>
    %6 = maximum(%4, %5) : tensor<2x2xf32>
    %7 = add(%6, %6) : tensor<2x2xf32>
    return(%7)
  }
}
```

Three things are worth reading carefully.

**Folding changed the graph without changing its size.** `18 -> 18`, `foldedCount=2`. Each `broadcast_in_dim` of a scalar zero became a `constant` of the full `2x2` shape — one operation replaced by one operation. And note what that means for a real tensor: folding a broadcast *materializes* the broadcast result. On a `2x2` that is 16 bytes; on the activation of a transformer layer it is megabytes of constant data in the compiled artifact. The pass has no size limit, and Chapter 61's weight folding is where the framework confronts the same question deliberately.

**CSE removed nearly half the graph.** Seven operations, because the duplicate `mul`, `add`, `maximum` and their four constants all collapsed. The result is `add(%6, %6)` — one value used twice, which is the same shape Chapter 11's commutativity lab produced.

**DCE removed three, not two.** `exp` and `log` were the dead operations the program wrote; the third is a constant left orphaned by CSE. That is Corollary 19.5 in the smallest possible instance.

**Try this.** Add a third identical `relu.forward(a.mul(2).add(1))` and predict the three counts before running. Then reorder the passes with `passContext` (Chapter 14) so `dce` runs before `cse`, and see whether the final graph differs.

## 19.6 Lab 2 — What DCE may not remove

```bash
node docs/part4/ch19-fold-cse-dce/labs/02-what-dce-may-not-remove.mjs
```

Two programs of the same shape. Each computes something and throws it away.

```js
class DeadPureChain extends Module {
  forward(a) {
    const dead = a.exp().log().mul(3).add(7);
    return a.add(1);
  }
}
```

```
=== a dead chain of pure operations ===
traced: exp, log, constant, mul, constant, add, constant, add, return
dce erased 6 operation(s)
module @DeadPureChain {
  func @DeadPureChain(%0: tensor<2x2xf32>) -> (tensor<2x2xf32>) {
    %1 = constant() {tensor_type = tensor<f32>, value = 1} : tensor<f32>
    %2 = add(%0, %1) : tensor<2x2xf32>
    return(%2)
  }
}
```

Four operations and two constants, gone in one pass — the worklist walking backwards from `add` to `mul` to `log` to `exp`, each becoming dead only after its consumer was removed.

Now the same shape with one operation swapped:

```js
class DeadSideEffect extends Module {
  forward(a, i) {
    const dead = ops.scatter_add(a, 0, i, a);
    return a.add(1);
  }
}
```

```
=== a dead operation that writes ===
traced: convert, reshape, iota, reshape, concat, scatter, constant, add, return
dce erased 0 operation(s)
```

**Zero.** Not only does the `scatter` survive — its entire index-computation subgraph survives with it, because `scatter` still uses those values and they are therefore not dead. Six operations and a kernel, computing a result the program never reads.

This is Theorem 19.4's dependency made visible. DCE is not looking at the `scatter` and concluding that its write matters; it is looking at `hasSideEffect` and finding `true`, which is the end of the conversation. The scatter is opaque to constant folding for the same reason and to CSE for the same reason: all three ask the same registry field.

**Try this.** Replace `scatter_add` with `gather` — an indexing operation with no declared effect — and watch all six operations disappear.

## 19.7 What a conservative declaration costs

It is worth asking whether the `scatter` in §19.6 *is* effectful, because the answer decides whether that output is a demonstration or a bug report.

The graph operation is functional. Its type inference returns a fresh tensor of the input's shape ([`ops/shape.ts:369`](../../../src/compiler/ir/graph/ops/shape.ts)), and its lowering rule copies the operand into a *new* output buffer and then writes the updates into that same output buffer ([`lowering/rules/linalg.ts:147`](../../../src/compiler/passes/lowering/rules/linalg.ts)):

```ts
    const copyBlock = new BlockNode(ctx.blockName('scatter_copy'), copyNest.loopBinds, [{ buffer: operandBuf }], [{ buffer: outBuf }], copyStore);
```

Writing to your own output buffer is what every operation in the IR does. On the evidence of the lowering rule, `scatter` mutates nothing its consumers can observe, and the `sideEffects: 2` on its registration ([`ops/shape.ts:368`](../../../src/compiler/ir/graph/ops/shape.ts)) is conservative rather than required.

Being conservative here is the *safe* direction — Theorem 19.4's asymmetry — and it is not free: a dead scatter survives, a duplicated scatter is never merged, and a scatter over constant indices is never folded. Whether the declaration is deliberate is not recorded anywhere, which is the actual lesson. A trait or an effect mask is a claim with a cost, and a claim nobody wrote down the reason for is a claim nobody can safely revisit.

## 19.8 Traps and limits

- **CSE never merges two operations with regions.** [`cse.ts:47`](../../../src/compiler/passes/simplify/cse.ts) refuses them outright. Two identical `fusion` bodies, two identical `scan` loops, two identical `if`s all survive in duplicate. Comparing region bodies is graph isomorphism, so this is the right call, and it means CSE must run *before* fusion to be useful — which is exactly where the pipeline puts it.
- **A fused float chain rounds its intermediates, and that costs something.** This is not a folding matter and is easy to confuse with one. Once fusion merges a chain of elementwise operations into a single expression (Chapter 22), the generated CPU kernel is one line of JavaScript, and every intermediate in it is a JavaScript number — a `double`. Eager execution materializes each operation into a `Float32Array` and therefore rounds at every step. Left alone, the two disagree: `t.add(16777216).add(1).add(-16777216)` gives `0` eagerly and `1` compiled, and the CPU backend would disagree with WebAssembly, whose `f32` operations round natively.

  So the CPU backend rounds each `f32` intermediate as it is produced. Two refinements keep the cost off the common path. A store into a `Float32Array` already rounds, so the outermost round of a stored expression is stripped — which means a single-operation kernel, the shape the eager JIT compiles, is byte-for-byte what it was before. And operations that return one of their operands or an exact result — `min`, `max`, `abs`, `floor`, `ceil`, `round`, `sqrt` — are not rounded at all.

  What remains is a real cost on genuinely fused arithmetic, and §22.5 is where it shows up: fusing a four-operation chain is worth about 1.9× with the intermediates rounded, against roughly 2.7× without. **About a third of the apparent fusion win is fusion computing in a wider precision than the program asked for.** That is worth knowing when reading any framework's fusion benchmark, including this one's.

- **Constant folding is scalar-shaped.** `FoldFn` takes and returns `AttrValue`s (Chapter 11), so what gets folded is an operation whose *whole result* is one attribute value — a scalar, or a constant tensor produced by broadcasting one. There is no elementwise folding of two large constant tensors; a graph containing one will keep it and compute it at run time.
- **`fold` runs inside a `try` and failures are silent at `INFO`.** [`constant_fold.ts:110`](../../../src/compiler/passes/simplify/constant_fold.ts) catches, emits a `pass_detail` at `DEBUG`, and moves on. A fold rule that throws on every call costs you nothing visible and buys you nothing at all.
- **The effect mask has four bits and the analysis uses two.** `ALLOCATE` and `CONTROL` are defined, `CONTROL` is declared by `call`, and every consumer in the compiler asks only `hasSideEffect(op)` — *is the mask non-zero*. Nothing distinguishes a read from a write when deciding what may be deleted or reordered. That is sufficient for DCE and insufficient for anything that wants to move two effectful operations past each other, which is why Chapter 36's dependence analysis works on buffers rather than on this.
- **Fixed-point cost.** All three passes are in the same group (Chapter 15), so each of them runs once more than it needs to, and on a large graph "once more" is a full traversal each. The group's convergence round is cheap only because these three passes are cheap.

- **All three passes are exactly as correct as the registry's metadata, and none of them can check it.** This is worth stating as the chapter's closing thought, because it is the same sentence three times. Definition 19.1 folds an operation if its `fold` callback is present and the operation is *pure*; Definition 19.2 merges two operations if they are *pure* and their opcodes, attributes and operands match, modulo order when the operation is declared *commutative*; Definition 19.3 deletes an operation with no users if it has *no side effect*. Every italicised word is a lookup in the `OpDef`, and Chapter 12 §12.6 established that a declaration is checked only when a verifier exists for it — which, for the algebraic traits, none does and none could.

  So the failure modes are declaration bugs rather than pass bugs, and they present as miscompiles rather than as errors:

  | Wrong declaration | What the pass does | Symptom |
  |---|---|---|
  | `COMMUTATIVE` on an operation that is not | CSE merges `f(a,b)` with `f(b,a)` | one of the two computations silently becomes the other |
  | side-effect mask left empty on an effectful op | DCE deletes it when its result is unused | the effect never happens |
  | `fold` that disagrees with the backend's arithmetic | constant folding writes the wrong constant | Counterexample 19.6 |

  Theorem 19.4 is honest about this in its own terms — its proof leans entirely on Definition 19.3's third clause, and that clause is a declaration. The theorem is true; the hypothesis is unverified. When a CSE or DCE bug is reported, the first place to look is not the pass.

## 19.9 Read the tests

- [`tests/compiler/passes/simplify/`](../../../tests/compiler/passes/simplify/) — the three passes individually, including the fold-representability guard and the CSE region refusal.
- [`tests/compiler/analysis/memory-effect.test.js`](../../../tests/compiler/analysis/memory-effect.test.js) — the effect masks, and the recursive folding that makes a `while` with an effectful body effectful.

---

**Next:** [Chapter 20 — Algebraic simplification meets IEEE 754](../ch20-algebra-and-ieee754/README.md), which takes the fourth pass in the same group and asks which of the identities you learned in school are true of the numbers a computer actually has.
