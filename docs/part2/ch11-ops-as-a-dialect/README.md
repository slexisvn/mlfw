# Chapter 11 — Ops as a dialect

There are 96 operations in this IR. There are 31 passes. If every pass had to know about every operation, that would be three thousand pieces of knowledge to keep consistent, and adding a new operation would mean editing thirty-one files.

Nobody does that, and this chapter is about what they do instead.

## 11.1 The problem: the pass that has never heard of your operation

Here is a pass you already watched run. Common subexpression elimination finds two operations that compute the same thing and keeps one. To do that it must decide when two operations are "the same".

For `add`, "the same" has a wrinkle: `add(a, b)` and `add(b, a)` compute the same value. For `sub` they do not. So CSE needs to know something about `add` that is not true of `sub`.

The tempting implementation writes it down:

```js
if (op.opName === 'add' || op.opName === 'mul' || op.opName === 'maximum' || ...) {
```

That works, and it is wrong in a specific way. The list is now in the CSE pass, and the same list — or a subtly different one — will appear in the algebraic simplifier, the fusion cost model, and the canonicalizer. Adding `logical_and` means finding all of them. Missing one means a silently weaker optimizer, or a silently wrong one.

This is not a hypothetical failure mode in this codebase; it is a documented one. An architectural review of this compiler found that op semantics leaking into passes as name-switches and hard-coded `Set`s was the single largest structural gap in the system, and the migration away from it — `ELEMENTWISE`, `CONSTANT`, and the broadcast-safety predicate moving from `Set`s in pass files into declared traits — is why the code you are about to read looks the way it does.

## 11.2 Intuition: put the knowledge with the operation

Move the fact to where it belongs. Instead of CSE knowing which operations are commutative, let each operation *declare* that it is, and let CSE ask.

That inverts the dependency. A pass now depends on a *vocabulary of properties* — commutative, elementwise, terminator, reduction — rather than on a list of names. Add a new operation tomorrow, declare it commutative, and CSE handles it correctly without being recompiled, because CSE never knew about operations in the first place.

This is what MLIR calls a *dialect*: a set of operations, each with declared traits, verification, type inference and folding behaviour, registered into a table that passes query. The vocabulary is the interface; the operation list is data.

> **Definition 11.1 (Op registry).** An *op registry* is a mapping from operation name to a description carrying: arity, attribute schema, declared traits, side-effect kind, a type inference rule, a verification rule, a folding rule, and a set of canonicalization patterns.

## 11.3 In mlfw: what an operation definition holds

[`op_registry.ts:109`](../../../src/compiler/ir/graph/op_registry.ts):

```ts
export class OpDef {
  name: string;
  numOperands: number;
  numResults: number;
  attrs: readonly AttrSpec[];
  sideEffects: SideEffectMask;
  traits: Set<OpTraitValue>;
  inferResultTypes: InferResultTypesFn | null;
  propagateSymbolicShapes: PropagateSymbolicShapesFn | null;
  verify: VerifyFn | null;
  getMemoryEffects: MemoryEffectsFn | null;
  fold: FoldFn | null;
  getCanonicalizationPatterns: CanonicalizationPatternsFn | null;
  getFlops: FlopsFn | null;
  hasRegions: boolean;
  numRegions: number;
  regionSpecs: readonly RegionSpec[] | null;
  genericAttrs: Map<string, unknown>;
```

Every field is a question some pass wants answered, and every one of them is answered here rather than in the pass:

| Field | Answers | Used by |
|---|---|---|
| `numOperands` / `numResults` / `attrs` | Is this operation well-formed? | The verifier, Chapter 12 |
| `traits` | What algebraic properties hold? | CSE, canonicalization, fusion |
| `sideEffects` | May I delete this if nobody uses it? | DCE, Chapter 19 |
| `inferResultTypes` | What type does this produce? | Tracing, verification, every rewrite |
| `verify` | Are *these particular* operands legal? | The verifier |
| `fold` | Can I compute this now? | Constant folding, Chapter 19 |
| `getCanonicalizationPatterns` | What is this operation's preferred form? | Canonicalize, Chapter 17 |
| `getFlops` | How much work is this? | Fusion cost model, Chapter 24 |
| `genericAttrs` | Anything a later subsystem needed | Layout, partitioning, GPU launch |

The last row is the extensibility hatch, and it is what keeps this file from growing. `OpAttrKey` ([`op_registry.ts:15`](../../../src/compiler/ir/graph/op_registry.ts)) names the keys — `gpuCapable`, `launchBoundary`, `inferLayout`, `layoutSensitivity` — and a subsystem that needs a new per-operation fact adds a key and calls `registry.registerOpAttr(...)` from its own file. `OpDef` does not learn about it.

### The trait vocabulary

[`op_registry.ts:24`](../../../src/compiler/ir/graph/op_registry.ts) is the whole vocabulary — fifteen properties:

```ts
export const OpTrait = Object.freeze({
  COMMUTATIVE: 'commutative',
  ASSOCIATIVE: 'associative',
  IDEMPOTENT: 'idempotent',
  ELEMENTWISE: 'elementwise',
  SAME_OPERAND_AND_RESULT_TYPE: 'same_type',
  SAME_OPERAND_AND_RESULT_SHAPE: 'same_shape',
  TERMINATOR: 'terminator',
  CONSTANT: 'constant',
  BROADCAST: 'broadcast',
  REDUCTION: 'reduction',
  VIEW: 'view',
  INJECTIVE: 'injective',
  OUT_EWISE_FUSABLE: 'out_ewise_fusable',
  OPAQUE: 'opaque',
  RECURSIVE_MEMORY_EFFECTS: 'recursive_memory_effects'
});
```

Read them as claims a pass may rely on. `COMMUTATIVE` licenses swapping two operands. `ELEMENTWISE` licenses fusing into a neighbouring loop nest. `VIEW` says no data moves. `RECURSIVE_MEMORY_EFFECTS` says "look inside my regions before deciding I am pure" — Chapter 9's warning, made checkable.

A trait is only useful if something consumes it, and a trait nobody consumes is worse than none: it looks like a guarantee and is not. Eight of the fifteen also have a *verifier* — a function that checks the declaration is true of the operation carrying it — and Chapter 12 is where those live.

### Registering an operation

[`ops/arithmetic.ts:22`](../../../src/compiler/ir/graph/ops/arithmetic.ts):

```ts
export function register(registry: OpRegistry) {
  registry.register(new OpDef({
    name: 'add',
    numOperands: 2,
    numResults: 1,
    traits: commBinaryArithTraits,
    inferResultTypes: inferBinaryElementwise,
    verify: verifyBinaryElementwise,
    getCanonicalizationPatterns() { return [new pat.AddZero()]; },
    fold: scalarBinaryFold((a, b) => a + b)
  }));
```

That is the entire definition of `add`. Nine lines, four of which are shared helpers used by every binary arithmetic operation. Compare `sub` immediately below it in the same file: same arity, same type inference, same verification, with three fields differing — its own fold, its own canonicalization patterns, and `traits: binaryArithTraits` where `add` has `commBinaryArithTraits`. That last one is a single word, and it is the whole of what makes the two operations behave differently in CSE.

> **Now unfold that single word, because it contains a claim that is false for most of the dtypes it covers.** [`ops/helpers.ts:69`](../../../src/compiler/ir/graph/ops/helpers.ts):
>
> ```ts
> export const commBinaryArithTraits: readonly OpTraitValue[] =
>   [...binaryArithTraits, OpTrait.COMMUTATIVE, OpTrait.ASSOCIATIVE];
> ```
>
> `add` and `mul` therefore declare **`ASSOCIATIVE` unconditionally** — for every dtype the operation accepts, including `f32` and `f64`. Floating-point addition is not associative: `(2⁵³ + 1) + (−2⁵³)` is `0` in `f64` while `2⁵³ + (1 + (−2⁵³))` is `1`. The declaration is simply untrue for the dtypes these operations spend almost all their time carrying.
>
> The trait is *declared per operation*, and an operation is not per-dtype — there is one `add` in the registry, serving `i32` (where the claim holds) and `f32` (where it does not), so the data model as it stands has nowhere to put a dtype-conditional trait.
>
> That would be a wording problem if nothing read it. Something does: canonicalization registers a reassociation pattern for any operation that is commutative, associative and foldable ([`canonicalize.ts:19`](../../../src/compiler/passes/canonicalize/canonicalize.ts)), which is exactly `add` and `mul`. Chapter 20 §20.8 is what that pattern has to do about the false half of the declaration.

So the accurate reading of the trait list is: **a trait is a claim about the operation, asserted for every dtype it accepts.** Where a property is dtype-dependent — and associativity is the important case — a trait cannot express it, and **the check has to move into the pattern that consumes the trait**. That is where it lives: `AddZero` and `AssociativeConstantReassoc` both ask for the operand's dtype at match time and decline on floats without a fast-math licence. The trait over-claims; every consumer of it re-checks.

Whenever you see a trait consulted without a dtype test nearby, that is the question to ask — and Chapter 17 §17.7 is why a *generated* pattern makes the question harder to spot than a hand-written one does.

Operations are grouped into fourteen dialect files under [`ops/`](../../../src/compiler/ir/graph/ops/) by family, alongside a `helpers.ts` of shared inference and verification functions — arithmetic, unary, comparison, shape, reduction, linalg, data, control flow, layout, quantization, composite, transfer, pooling, resize — and assembled once ([`ops.ts:19`](../../../src/compiler/ir/graph/ops.ts)):

```ts
export const DIALECTS: readonly DialectRegistrar[] = [
  registerArithmetic, registerUnary, registerComparison, registerShape,
  registerReduction, registerLinalg, registerData, registerControlFlow,
  registerLayout, registerQuantization, registerComposite, registerTransfer,
  registerPooling, registerResize,
];

export function buildRegistry(dialects: readonly DialectRegistrar[] = DIALECTS): OpRegistry {
  const reg = new OpRegistry();
  for (const register of dialects) register(reg);
  return reg;
}
```

`buildRegistry` takes the dialect list as a parameter with a default. That is not decoration: it means a test can build a registry containing three operations and nothing else, which is how the verifier tests in Chapter 12 construct deliberately broken IR without fighting the real vocabulary.

## 11.4 How a trait is actually consumed

Follow `COMMUTATIVE` from declaration to use. It lands in three methods on `Operation`, and CSE uses all three.

[`operation.ts:167`](../../../src/compiler/ir/graph/operation.ts):

```ts
  hasInterchangeableOperands(): boolean {
    if (this.operands.length !== 2) return false;
    const def = registry.get(this.opName);
    return def !== null && def.isCommutative;
  }
```

A registry lookup, not a name comparison. Then the hash ([`operation.ts:178`](../../../src/compiler/ir/graph/operation.ts)):

```ts
    if (this.hasInterchangeableOperands()) {
      const a = this.operands[0].id, b = this.operands[1].id;
      h = ((h ^ (a < b ? a : b)) * 0x01000193) & 0x7fffffff;
      h = ((h ^ (a < b ? b : a)) * 0x01000193) & 0x7fffffff;
    } else {
      for (let i = 0; i < this.operands.length; i++) {
        h = ((h ^ this.operands[i].id) * 0x01000193) & 0x7fffffff;
      }
    }
```

For a commutative operation the operands are hashed in sorted order, so `add(a, b)` and `add(b, a)` land in the same hash bucket. That is a requirement, not an optimization: a hash-based CSE that hashed them differently would never even compare them. And the equality check ([`operation.ts:205`](../../../src/compiler/ir/graph/operation.ts)):

```ts
    if (this.hasInterchangeableOperands()) {
      const sameOrder = this.operands[0] === other.operands[0] && this.operands[1] === other.operands[1];
      const swapped = this.operands[0] === other.operands[1] && this.operands[1] === other.operands[0];
      if (!sameOrder && !swapped) return false;
```

Three places, one trait, zero operation names. Declare a new operation commutative and all three follow.

## 11.5 Lab 1 — A trait is data

```bash
node docs/part2/ch11-ops-as-a-dialect/labs/01-a-trait-is-data.mjs
```

The lab writes the same shape of program twice, once with `add` and once with `sub`, computing each expression in both operand orders.

```
=== add(a, b) and add(b, a) ===
module @traced {
  func @traced(%0: tensor<2x2xf32>, %1: tensor<2x2xf32>) -> (tensor<2x2xf32>) {
    %2 = add(%0, %1) : tensor<2x2xf32>
    %3 = add(%1, %0) : tensor<2x2xf32>
    %4 = mul(%2, %3) : tensor<2x2xf32>
    return(%4)
  }
}
passes that changed something: cse: 4 -> 3, PriorityFusionPass: 3 -> 2
module @Commutative {
  func @Commutative(%0: tensor<2x2xf32>, %1: tensor<2x2xf32>) -> (tensor<2x2xf32>) {
    %2 = fusion(%0, %1) {fusion_kind = "kElementwise"} : tensor<2x2xf32>
    {
      ^bb(%3: tensor<2x2xf32>, %4: tensor<2x2xf32>):
      %5 = add(%3, %4) : tensor<2x2xf32>
      %6 = mul(%5, %5) : tensor<2x2xf32>
      yield(%6)
    }
    return(%2)
  }
}
```

Two `add`s went in; one came out. Look at the fused body: `mul(%5, %5)` — the same value, used twice. CSE recognized `add(%0, %1)` and `add(%1, %0)` as the same computation and rewired the second's users onto the first, using the `replaceAllUsesWith` from Chapter 8.

Now the counterexample, which is the half that proves the mechanism is real:

```
=== sub(a, b) and sub(b, a) ===
passes that changed something: PriorityFusionPass: 4 -> 2
module @NotCommutative {
  func @NotCommutative(%0: tensor<2x2xf32>, %1: tensor<2x2xf32>) -> (tensor<2x2xf32>) {
    %2 = fusion(%0, %1) {fusion_kind = "kElementwise"} : tensor<2x2xf32>
    {
      ^bb(%3: tensor<2x2xf32>, %4: tensor<2x2xf32>):
      %5 = sub(%3, %4) : tensor<2x2xf32>
      %6 = sub(%4, %3) : tensor<2x2xf32>
      %7 = mul(%5, %6) : tensor<2x2xf32>
      yield(%7)
    }
    return(%2)
  }
}
```

CSE does not appear in the pass list at all — nothing changed. Both `sub`s survive, in both orders, and correctly so: `a - b` and `b - a` are different numbers.

**The two programs are structurally identical.** Same shapes, same operand wiring, same number of operations. The only difference between "CSE merges these" and "CSE must not merge these" is one entry in a `Set` on an `OpDef`. That is what "a trait is data" means, and it is worth internalizing before Part IV, where the fusion engine makes far larger decisions on the same basis.

**Try this.** `maximum` is also commutative, and `div` is not. Predict what happens for `a.maximum(b).mul(b.maximum(a))` and for `a.div(b).mul(b.div(a))`, then run them.

## 11.6 Folding and canonicalization: two ways to be simpler

Two more `OpDef` fields do work you have already seen without being told what it was.

**`fold`** answers "can I compute this at compile time?" Its signature ([`op_registry.ts:79`](../../../src/compiler/ir/graph/op_registry.ts)) takes the constant values of the operands and returns a constant or `undefined`:

```ts
export type FoldFn = (
  constValues: readonly AttrValue[],
  attrs: OpAttrMap,
  constOps: readonly Operation[],
) => AttrValue | undefined;
```

Returning `undefined` means "not foldable with these operands", which is the normal case. Eleven operations define one.

**`getCanonicalizationPatterns`** answers "what is this operation's preferred form?" It returns pattern objects — 27 classes live in [`patterns.ts`](../../../src/compiler/ir/graph/patterns.ts) — each of which matches a shape of IR and rewrites it. `AddZero` on `add`, `MulOne` and `MulZero` on `mul`, and the pattern that folds a `transpose` into the `dot` that consumes it.

The distinction between them is worth keeping straight: **folding replaces a computation with its answer; canonicalization replaces a computation with a better-shaped computation.** Folding needs constants; canonicalization does not.

## 11.7 Lab 2 — Both, on three small programs

```bash
node docs/part2/ch11-ops-as-a-dialect/labs/02-fold-and-canonicalize.mjs
```

**First, the identity function written the long way.** `t.add(0).mul(1)`:

```
traced:
    %1 = constant() {tensor_type = tensor<xf32>, value = 0} : tensor<xf32>
    %2 = add(%0, %1) : tensor<2x2xf32>
    %3 = constant() {tensor_type = tensor<xf32>, value = 1} : tensor<xf32>
    %4 = mul(%2, %3) : tensor<2x2xf32>
    return(%4)

passes: canonicalize: 5 -> 4, dce: 4 -> 3
after graph passes:
module @Identity {
  func @Identity(%0: tensor<2x2xf32>) -> (tensor<2x2xf32>) {
    %1 = constant() {tensor_type = tensor<xf32>, value = 0} : tensor<xf32>
    %2 = add(%0, %1) : tensor<2x2xf32>
    return(%2)
  }
}
```

Five operations to three, and **one of the two identities survived**. `MulOne` rewrote `mul(%2, %3)` to just `%2` and DCE swept the orphaned `1`; the `add` of zero is still there.

That asymmetry is the whole point of the example, and it is not about traits. Both patterns are registered the same way, on the same kind of declaration, and the canonicalizer tried both. `MulOne` fired because `x × 1 = x` is true of the numbers a machine actually has. `AddZero` declined because `x + 0 = x` is *not*: for `x = −0`, IEEE 754 gives `(−0) + (+0) = +0`, so rewriting the add away would change the sign bit of a zero. The pattern asks for the operand's dtype, sees `f32`, finds no fast-math licence, and refuses.

Chapter 20 is where that distinction gets its name and its four counterexamples. What Chapter 11 shows is the mechanism underneath it: a canonicalization pattern is not a syntactic rewrite that always applies, it is a rewrite *plus a condition*, and the condition can consult the type the registry attached to the value. Change the dtype to an integer and the same `AddZero` fires, because two's complement has one zero.

**Try this.** Re-run the lab with `optimization: { fastMath: true }` in the compile options and watch the graph collapse to `return(%0)` — that is the licence being granted, nine chapters early.

**Second, the transpose you have been watching since Chapter 1.** A `Linear` layer traces as `transpose` then `dot`:

```
traced:
    %3 = transpose(%1) {permutation = [1, 0]} : tensor<2x3xf32>
    %4 = dot(%0, %3) {lhs_batch = [], lhs_contracting = [1], rhs_batch = [], rhs_contracting = [0]} : tensor<2x3xf32>

passes: canonicalize: 4 -> 4, dce: 4 -> 3
after graph passes:
    %3 = dot(%0, %1) {lhs_batch = [], lhs_contracting = [1], rhs_batch = [], rhs_contracting = [1]} : tensor<2x3xf32>
```

`rhs_contracting` went from `[0]` to `[1]` and the `transpose` disappeared. The pattern did not delete anything — note `canonicalize: 4 -> 4` — it rewrote one attribute so that the `dot` reads the untransposed weight along the other axis, leaving the `transpose` with no users for DCE to collect.

This is the concrete answer to a question Chapter 2 raised and deferred: *why is `dot` defined with explicit contracting-dimension attributes instead of as plain matrix multiplication?* Because a general form has somewhere to absorb the transpose. A `matmul(a, b)` operation with fixed semantics would have no attribute to change, and the transpose would have to survive into the loop nest, where removing it costs a full pass over memory.

**Third, constants that were already folded before the compiler saw them.** `t.mul(2 * 3).add(10 - 4)`:

```
traced:
    %1 = constant() {tensor_type = tensor<xf32>, value = 6} : tensor<xf32>
    %2 = mul(%0, %1) : tensor<2x2xf32>
    %3 = constant() {tensor_type = tensor<xf32>, value = 6} : tensor<xf32>
    %4 = add(%2, %3) : tensor<2x2xf32>

passes: cse: 5 -> 4, PriorityFusionPass: 4 -> 3
```

JavaScript evaluated `2 * 3` and `10 - 4` long before tracing began, so the compiler never sees the arithmetic — it sees two separate constants that happen to both be 6. CSE merges them, and the fused region ends up reading a single `%4` twice.

That is worth noticing for what it says about where the boundary is. The tracer records tensor operations; host arithmetic on plain numbers happens at trace time and arrives as a literal. Constant *folding* in the compiler is for constants the graph itself produces — a `broadcast_in_dim` of a scalar becoming a constant of the final shape, which is exactly what Chapter 3's `constant_fold: 10 -> 10` line was doing.

## 11.8 Traps and limits

- **The registry is process-global.** [`ops.ts:32`](../../../src/compiler/ir/graph/ops.ts) constructs one `registry` at module load and everything imports it. `register` throws on a duplicate name, so two dialects cannot both define `add` — deliberate, but it means there is no namespacing and no way to have two dialects coexist with overlapping names. MLIR's `dialect.op` prefixes exist for exactly this and are not reproduced here.
- **A trait with no verifier is a promise nobody checks — and there is no verifier that could check the algebraic ones.** Fifteen traits are declared; eight have verifiers (Chapter 12). `ASSOCIATIVE`, `INJECTIVE`, `REDUCTION`, `BROADCAST`, `OPAQUE`, `OUT_EWISE_FUSABLE` and `RECURSIVE_MEMORY_EFFECTS` are consumed by passes but not validated, so declaring one wrongly produces a miscompile rather than an error. Be clear about *why* the split falls where it does: the eight that are verified are **structural** claims — "same operand and result type", "terminator is last" — which the verifier decides by inspecting the operation in front of it. The seven that are not are **semantic** claims, quantified over all inputs. Verifying `ASSOCIATIVE` would mean establishing `f(f(a,b),c) = f(a,f(b,c))` for every triple of tensors the operation accepts, which no verifier is going to do by inspection, and which is *false* for the operations that declare it. So this is not a gap somebody forgot to fill; it is a limit on what a declaration-plus-verifier design can enforce. The vocabulary lets an operation assert a law, and the compiler's only defence against a wrong assertion is review. When you add an operation, the traits are the part to get right first.
- **`structuralEquals` refuses operations with regions.** [`operation.ts:204`](../../../src/compiler/ir/graph/operation.ts) returns `false` if either side has a region, so CSE never merges two identical `fusion`s or two identical `scan`s. That is conservative and safe — comparing two region bodies for equality is a graph isomorphism problem — but it does mean a program with genuinely duplicated loops keeps both.
- **Commutativity is declared over the operation, not the dtype — and so is associativity.** `add` on floats is genuinely commutative: IEEE 754 addition returns the same result for `a + b` and `b + a`, NaN payloads aside, so `COMMUTATIVE` is a true claim. `ASSOCIATIVE` on the same operation is a false one, for the reason §11.3 spells out, and the two travel together inside `commBinaryArithTraits`. Do not read either as licence to reassociate: the first does not imply it, and the second asserts it wrongly — which is why every consumer of `ASSOCIATIVE` has to re-check the dtype itself.
- **`fold` operates on scalars.** The `FoldFn` signature takes `AttrValue`s, and the arithmetic folds are written as scalar functions. Folding a whole constant tensor elementwise is not what this hook does; that is the constant-folding pass's job, using the hook per element where it applies.

## 11.9 Read the tests

- [`tests/compiler/ir/graph/operation.test.js`](../../../tests/compiler/ir/graph/operation.test.js) — `structuralHash` and `structuralEquals`, including the commutative cases from §11.4 and the region refusal from §11.8.
- [`tests/compiler/ir/rewrite/pattern.test.js`](../../../tests/compiler/ir/rewrite/pattern.test.js) — canonicalization patterns matching and rewriting, in isolation from the pipeline.
- [`tests/compiler/passes/simplify/`](../../../tests/compiler/passes/simplify/) — folding and canonicalization as passes, including the cases where one enables another.

---

**Next:** [Chapter 12 — What "valid IR" means](../ch12-valid-ir/README.md), which takes the `verify` and trait fields you just met and asks who calls them, and when.
