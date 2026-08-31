# Chapter 10 — The type system

Every value in the IR carries a type. Chapters 8 and 9 kept deferring what that type is; this chapter fills it in, and then spends most of its length on one question that turns out to be subtler than it looks: **when is one shape usable where another was expected?**

The answer is not "when they are equal", and the relation that gives the right answer is not an equivalence relation. Getting that wrong is how a compiler either rejects valid programs or accepts invalid ones.

## 10.1 The problem: what has to be decided before the program runs?

A tensor compiler makes a promise: by the time code is generated, it knows enough about every intermediate value to allocate storage for it, choose loop bounds, and pick instructions. That knowledge has to come from somewhere, and it comes from types propagated forward from the inputs.

So a type has to answer, for every value:

- How many elements, and arranged how? — **shape**
- How wide is each element, and is it a float or an integer? — **dtype**
- Where does element `[i][j]` live relative to element `[i][j+1]`? — **layout**

And it has to have an answer for the case where the honest answer is "not yet" — a batch size that varies per call. That case is what makes the rest of the chapter interesting.

## 10.2 In mlfw: four kinds of type, one of which matters

[`types.ts:7`](../../../src/compiler/ir/graph/types.ts):

```ts
export type IRType = TensorType | TupleType | TokenType | FunctionType;
```

Three of those four are rare. `TupleType` groups results (`split` produces one); `TokenType` orders side effects without carrying data; `FunctionType` types a callee. Ninety-nine values in a hundred carry the remaining one, `TensorType` ([`types.ts:282`](../../../src/compiler/ir/graph/types.ts)):

```ts
export class TensorType {
  readonly shape: readonly Dim[];
  readonly dtype: ScalarDType;
  readonly layout: Layout;
  private _hash: number | null;

  constructor(shape: Shape, dtype: ScalarDType, layout: Layout | null = null) {
    this.shape = Object.freeze([...shape]);
    this.dtype = dtype;
    this.layout = layout || Layout.rowMajor(shape.length);
```

Three `readonly` fields and a frozen shape array. **Types are immutable**, and every "modification" is a new object — `withShape`, `withDtype`, `withLayout` at [`types.ts:279`](../../../src/compiler/ir/graph/types.ts) all construct rather than mutate. This is not fastidiousness: types are shared freely between values, and a mutable type would let a rewrite of one value silently retype another.

### Dimensions come in three kinds

[`types.ts:4`](../../../src/compiler/ir/graph/types.ts):

```ts
export type Dim = number | SymIntValue;
```

and [`types.ts:63`](../../../src/compiler/ir/graph/types.ts):

```ts
export const DYNAMIC = -1;
```

So a dimension is one of:

| Kind | Representation | Prints as | Means |
|---|---|---|---|
| **Static** | a non-negative `number` | `4` | exactly four |
| **Dynamic** | the number `-1` | `?` | not known at compile time, and unrelated to any other unknown |
| **Symbolic** | a `SymInt` | `[n]` | not known, but *named*, so two dimensions can be known equal without knowing the value |

The distinction between the last two is the whole reason `SymInt` exists. If a function takes two tensors that both have first dimension `?`, the compiler knows nothing about whether they match. If both have first dimension `n`, it knows they match — and that fact is enough to fuse them, or to prove an index in bounds — without ever knowing what `n` is. Chapter 62 is where symbolic dimensions earn their keep; here we mostly meet `?`.

### Dtype, and the promotion lattice

[`types.ts:173`](../../../src/compiler/ir/graph/types.ts) declares eleven scalar types, and [`types.ts:173`](../../../src/compiler/ir/graph/types.ts) says what happens when two meet:

```ts
export function promoteDtype(a: ScalarDType, b: ScalarDType): ScalarDType | null {
  if (a === b) return a;
  const ai = isIntType(a), af = isFloatType(a);
  const bi = isIntType(b), bf = isFloatType(b);
  if (af && bf) {
    return scalarBytes(a) >= scalarBytes(b) ? a : b;
  }
  if (ai && bi) {
    return scalarBytes(a) >= scalarBytes(b) ? a : b;
  }
  if (af && bi) return a;
  if (ai && bf) return b;
  return null;
}
```

Read the rules: within a kind, wider wins; across kinds, float beats integer; and any *mixed* pair involving `bool` or `index` returns `null`, meaning "no promotion exists, this is an error" — identical dtypes short-circuit on the first line, so `bool` with `bool` is still `bool`. It is a small join semilattice with an explicit failure, and returning `null` rather than guessing is the right call — a silent bool-to-float promotion would hide a real mistake.

### Layout is a permutation

[`types.ts:188`](../../../src/compiler/ir/graph/types.ts):

```ts
export class Layout {
  readonly order: readonly number[];
```

`order` is the sequence in which dimensions vary from slowest to fastest. Row-major is `[0, 1, 2, ...]`; column-major is the reverse; NHWC against NCHW is a permutation of four. Its one job is [`types.ts:235`](../../../src/compiler/ir/graph/types.ts):

```ts
  computeStrides(shape: Shape): number[] {
    const n = shape.length;
    const strides = new Array(n);
    let stride = 1;
    for (let i = n - 1; i >= 0; i--) {
      const dim = this.order[i];
      strides[dim] = stride;
      if (shape[dim] === DYNAMIC || shape[dim] instanceof SymInt) {
        stride = DYNAMIC;
```

Note the propagation: once a dimension is unknown, every stride outside it is unknown too. That is correct and it is the practical cost of a dynamic dimension — Chapter 62's "generated code quality falls" is, concretely, this.

## 10.3 Lab 1 — What a type knows

```bash
node docs/part2/ch10-type-system/labs/01-what-a-type-knows.mjs
```

```
value  produced by        shape      dtype  rank  numel  bytes  strides
%0     argument          [2, 2]     f32    2     4      16     [2, 1]
%1     argument          [8, 2]     f32    2     16     64     [2, 1]
%2     argument          [8]        f32    1     8      32     [1]
%3     argument          [1, 8]     f32    2     8      32     [8, 1]
%4     argument          [1]        f32    1     1      4      [1]
%5     transpose         [2, 8]     f32    2     16     64     [8, 1]
%6     dot               [2, 8]     f32    2     16     64     [8, 1]
%7     add               [2, 8]     f32    2     16     64     [8, 1]
%8     constant          []         f32    0     1      4      []
%9     broadcast_in_dim  [2, 8]     f32    2     16     64     [8, 1]
%10    maximum           [2, 8]     f32    2     16     64     [8, 1]
%11    transpose         [8, 1]     f32    2     8      32     [1, 1]
%12    dot               [2, 1]     f32    2     2      8      [1, 1]
%13    add               [2, 1]     f32    2     2      8      [1, 1]
```

Everything the compiler needs to allocate this program is in that table, and none of it was written by the user. It was inferred, one operation at a time, from the five argument types — which is Chapter 11's `inferResultTypes`.

Two rows repay a second look.

`%8` has shape `[]`, rank 0, and one element. A scalar is a tensor, not a special case; there is no separate scalar type anywhere in the IR. That uniformity is why `constant` needs no variant for scalars and why `broadcast_in_dim` can lift it to `[2, 8]` with an ordinary operation rather than a rule.

`%5` is the transposed weight: shape `[2, 8]`, strides `[8, 1]`. Compare `%1`, the weight itself: shape `[8, 2]`, strides `[2, 1]`. The transpose here produces a genuinely new type with row-major strides, not a restrided view of the original — because at the graph level there is no storage yet to alias, only values. Restriding is a decision made much later, and Chapter 2 watched the whole `transpose` vanish before it ever got there.

## 10.4 Equality is the wrong question

Now the part that matters. There are three relations on types in this file, and they are not the same relation.

```ts
  equals(other: unknown): boolean            // types.ts:283
  shapeEquals(other: unknown): boolean       // types.ts:294
  shapeCompatible(other: TensorType): boolean // types.ts:304
```

`equals` compares dtype, every dimension, and layout. `shapeEquals` drops the layout requirement. `shapeCompatible` is the interesting one ([`types.ts:342`](../../../src/compiler/ir/graph/types.ts)):

```ts
  shapeCompatible(other: TensorType): boolean {
    if (this.shape.length !== other.shape.length) return false;
    for (let i = 0; i < this.shape.length; i++) {
      const a = this.shape[i], b = other.shape[i];
      if (a === DYNAMIC || b === DYNAMIC) continue;
      if (dimEquals(a, b)) continue;
      if (typeof a === 'number' && typeof b === 'number') return false;
    }
    return true;
  }
```

Rank must match. Then a dimension pair passes if either side is unknown, or if they are equal. Two *known, different* sizes are the only rejection — and *known* means a plain number, so `[n, 3]` and `[4, 3]` are compatible. The compiler declines to assert an inequality it cannot prove.

To say what that relation is, order dimensions by how much they claim:

> **Definition 10.1 (Specificity order).** **(stated here)** Write `d ⊑ e` when `d` is `?` or `d = e`. This is a partial order on dimensions: reflexive, antisymmetric, transitive, with `?` as least element. Extend it componentwise to shapes of equal rank. A symbolic dimension sits with `?` rather than with the numbers: two symbols are related only when they are structurally equal, and a symbol is never provably different from anything.

> **Definition 10.2 (Compatibility).** **(stated here)** Two shapes of equal rank are *compatible* when they have a common upper bound under ⊑ — that is, when there is a shape at least as specific as both. Equivalently, and this is the reading the code implements: when no dimension pair can be *proved* different.

`shapeCompatible` decides exactly Definition 10.2, and the least such upper bound — the **join** under ⊑, since the order runs from unknown up towards specific — is what type inference propagates: `[?, 3]` joined with `[4, ?]` is `[4, 3]`, and every unknown that a caller resolves makes the whole function more specific. (Keep *join* and *meet* apart here. Both orders in this chapter are used for their joins; a meet, the greatest lower bound, would go the other way and is never what inference wants — the greatest lower bound of `[4, 3]` and `[9, 3]` is `[?, 3]`, which throws away everything the caller told you.)

It is worth naming what this is *not*, because the obvious guess is wrong. Compatibility is not unification. Definition 10.1 puts a symbolic dimension next to `?`, and it does so at each occurrence independently — nothing records that the two `n`s in `[n, n]` are the same `n`. So `[n, n]` is compatible with `[4, 5]`: neither pair is two known numbers, so neither is rejected, even though no value of `n` could satisfy both. A unifier would answer no here; `shapeCompatible` answers "I cannot prove otherwise", and those are different claims. Deciding the first would need a solver over the symbolic layer of Chapter 37, and nothing in the type system reaches for one.

Now the trap.

> **Theorem 10.3 (Compatibility is not transitive).** **(stated here)** `shapeCompatible` is reflexive and symmetric but not transitive, hence not an equivalence relation and not a partial order.
>
> *Proof.* `[4, 3]` is compatible with `[?, 3]` because the first dimension of the second side is unknown. `[?, 3]` is compatible with `[9, 3]` for the same reason. But `[4, 3]` and `[9, 3]` are two known, different sizes, so they are incompatible. ∎

Lab 2 prints exactly this, on real types:

```
=== a static type and a dynamic one, compared ===
  static  [4, 3]
  dynamic [-1, 3]   (-1 is what '?' prints as)
  s.equals(d)          = false
  s.shapeCompatible(d) = true   <- a dynamic dimension is compatible with anything
  d.shapeCompatible(s) = true

  a [9, 3] input
  s.shapeCompatible(other) = false   <- two known, different sizes are not
  d.shapeCompatible(other) = true
```

The consequence is a rule for using the function, and it is worth stating flatly: **`shapeCompatible` is a check of an actual against an expected, never a way to group types.** Because it is not transitive, you cannot use it to build equivalence classes, deduplicate a set of types, or key a cache. The compiler obeys this. The verifier calls it in exactly two places — result type against inferred type, and return operand against declared output ([`verifier.ts:100`](../../../src/compiler/ir/graph/verifier.ts) and [`verifier.ts:297`](../../../src/compiler/ir/graph/verifier.ts)) — and both are actual-against-expected. Caching and deduplication use `equals` and `hash`.

## 10.5 Broadcasting is a different order

Elementwise operations do not require compatible shapes; they require *broadcast-compatible* ones, and that is a second, independent relation ([`types.ts:272`](../../../src/compiler/ir/graph/types.ts)):

```ts
export function broadcastDim(a: Dim, b: Dim): Dim | null {
  if (dimEquals(a, b)) return a;
  if (a === 1) return b;
  if (b === 1) return a;
  if (a === DYNAMIC) return b === DYNAMIC ? DYNAMIC : b;
  if (b === DYNAMIC) return a;
  if (a instanceof SymInt || b instanceof SymInt) return DYNAMIC;
  return null;
}
```

> **Definition 10.4 (Broadcast order).** **(stated here)** Write `d ⊴ e` when `d = 1` or `d = e`. **On known dimensions**, `broadcastDim` computes the least upper bound of `d` and `e` under ⊴ when one exists, and returns `null` when it does not. Its last three lines leave that order rather than extending it, and Counterexample 10.5 is what they do instead.

So on known dimensions broadcasting is a join — a genuine one, in a lattice whose bottom is `1` rather than `?`. `TensorType.broadcastShape` ([`types.ts:328`](../../../src/compiler/ir/graph/types.ts)) lifts it to whole shapes, right-aligning ranks so that a `[8]` bias joins a `[2, 8]` activation. That is precisely what happened at `%7` in Lab 1: `add([2,8], [8]) : [2,8]`.

Two orders, then, doing different jobs: **specificity** decides whether a type may be used where another was expected, and **broadcast** decides what shape an elementwise operation produces. `1` is special in one and not the other; `?` is special in the other and not the one. Conflating them produces a compiler that thinks `[1, 8]` and `[4, 8]` are interchangeable everywhere, which they are not.

> **Counterexample 10.5.** `broadcastDim(DYNAMIC, 4)` returns `4`, not `DYNAMIC`. Read as a claim, that says: *if this dimension is not 4 at run time, it must be 1.* That is an assumption, not a deduction — the unknown could turn out to be 7, and then the operation is simply invalid. The alternative, propagating `DYNAMIC`, would be sound and would also make every downstream shape unknown, which is to say useless. So the compiler bets on the useful answer.
>
> Two symbols err the opposite way: `broadcastDim(n, m)` returns `DYNAMIC`, which is no more an upper bound under ⊴ than `4` was, but discards both claims rather than betting on one. So on `?` and on symbols the function is not computing a join at all. It picks the shape the program most likely meant.

That bet is not checked anywhere later, which §10.7 traces precisely — it is worth knowing before you rely on a dynamic dimension broadcasting against a known one.

## 10.6 Lab 2 — Static, dynamic, and what changes

```bash
node docs/part2/ch10-type-system/labs/02-static-dynamic-symbolic.mjs
```

The lab traces one small model three ways.

```
=== every dimension known ===
  func @traced(%0: tensor<4x3xf32>, %1: tensor<2x3xf32>, %2: tensor<2xf32>) -> (tensor<4x2xf32>) {
    ...
  input  isFullyStatic true  hasDynamic false numel 12  sizeInBytes 48
  output isFullyStatic true  hasDynamic false numel 8  sizeInBytes 32

=== dimension 0 dynamic ===
  func @traced(%0: tensor<?x3xf32>, %1: tensor<2x3xf32>, %2: tensor<2xf32>) -> (tensor<?x2xf32>) {
    ...
  input  isFullyStatic false hasDynamic true  numel -1  sizeInBytes -1
  output isFullyStatic false hasDynamic true  numel -1  sizeInBytes -1
```

The unknown propagates: one dynamic input dimension makes the output dynamic, and `numel` and `sizeInBytes` both return `-1`, meaning "cannot say". Every downstream decision that wanted a byte count now cannot have one, and that is the cost Chapter 5 charged for avoiding recompilation.

Two smaller observations from the same run.

**Only the marked dimension goes unknown.** `dynamic_shapes: [new Set([0])]` produced `tensor<?x3xf32>` — dimension 1 stayed at 3, because it was not marked. `dynamic_shapes: [true]` is the shorthand for "every dimension of this input", and produced `tensor<?x?xf32>`. Per-dimension control is the more useful form and the one to reach for when you mean "batch size varies".

**The body did not change.** Compare the two printed functions: same seven operations, same attributes, only the types differ. Making a dimension dynamic is a change to the types, and everything structural about the program is untouched. That is why one traced graph can serve every batch size.

**Try this.** Trace with `dynamic_shapes: [true]` on a model that reduces over the dynamic axis — `t.sum(0)` — and look at what the result type becomes. Then ask whether the compiler could have known the answer, and what it would have needed to know it. Chapter 37 is about the machinery for that question.

## 10.7 Traps and limits

**Nothing checks Counterexample 10.5's bet.** It is tempting to finish that counterexample with "…and the guard catches it at run time", because that is what Definition 5.5 exists for and it would make the design sound. It is not what happens, and the gap is the difference between *deferring* an obligation and *dropping* one.

Here is the entire set of guards this framework ever records. A **static** dimension gets an equality — `produceShapeSpec` ([`shape_env.ts:53`](../../../src/tracing/shape_env.ts)) emits `sym == 4` for each dimension not marked dynamic. A **dynamic** dimension gets exactly one guard, `sym > 0`, added by `createInput` ([`tracer.ts:65`](../../../src/tracing/tracer.ts)). Beyond that, `specialize` pins a symbol to its hint when a later stage demands a concrete value, and `guardDivisible` exists but is called from nowhere outside its own file. That is the list — and **no step of type inference contributes to it.** `broadcastDim` is a pure function over `Dim`s in the IR's type layer; it holds no reference to the `ShapeEnv` and could not add a constraint if it wanted to. So when it decides a `?` is 4, that decision is recorded in the result *type* and nowhere in the *guard set*, which is what `evaluateGuards` consults on every call (§5.6).

The honest statement of what dynamic shapes buy is consequently narrower than "one kernel, many shapes, checked at the boundary":

| Case | Inferred | Guarded | Sound? |
|---|---|---|---|
| static dim vs static dim | join under ⊴, or `null` | equality per dim | yes — a mismatch is a compile-time error |
| `?` vs `1` | `?` | `> 0` | yes — anything broadcasts against 1 |
| `?` vs known `k` | `k` | `> 0` only | **no** — assumes the unknown is `k` or 1, and never checks |
| symbol vs symbol | `?` | `> 0` on each | information discarded, not unsound |

Only the third row is a defect, and it is a real one: the type system asserts something about run time that neither the type system nor the runtime verifies. Two designs would close it — have inference emit a `sym ∈ {1, k}` guard when it makes this choice, or propagate `?` and accept the loss of downstream information — and the compiler currently does neither. Until it does, treat "broadcasting a dynamic dimension against a known one" as an assertion *you* are making, and one the framework will not catch you getting wrong.

- **`numel()` returns `-1` for "unknown", not `null`.** [`types.ts:262`](../../../src/compiler/ir/graph/types.ts) uses the same sentinel as `DYNAMIC`, so a caller that forgets to check will allocate a negative number of elements or, worse, silently compute a nonsensical byte count. `symbolicNumel()` is the version that keeps a `SymInt` instead of collapsing to the sentinel.
- **A negative dimension that is not `-1` prints as itself.** `dimToString` maps `-1` to `?` and anything else through `String`, so a shape that acquires, say, `-3` through a bad inference prints as `tensor<-3xf32>`. That rendering is deliberate — it shows you the corrupt value instead of disguising it as "unknown" — and the verifier is what refuses to let such a shape cross a phase boundary: every result type and every declared input and output is checked for a negative extent other than `DYNAMIC` (Chapter 12). If you see a negative number other than `-1` in a printed type, you have found a shape-inference bug, not an unusual tensor.
- **Layout is carried but rarely varied.** Every type in Lab 1 has the identity layout, because layout selection is an optional pass (Chapter 25) that is off by default on this target. The field exists so that the pass has somewhere to put its answer; do not read its presence as evidence that anything uses it yet.
- **`equals` includes layout, `shapeEquals` does not.** Two types with the same shape and dtype but different layouts describe the same values in different memory arrangements. Which comparison a pass should use depends on whether it is about to touch memory, and getting it backwards produces either a missed optimization or a wrong answer.
- **`hash()` is lossy on purpose.** [`types.ts:315`](../../../src/compiler/ir/graph/types.ts) folds each dimension to sixteen bits and hashes only the first character of the dtype. It is a hash, so collisions are fine — but it means two types that hash equal must still be compared with `equals`, and any code that treats the hash as an identity is wrong.

## 10.8 Read the tests

- [`tests/compiler/ir/graph/symbolic-shape.test.js`](../../../tests/compiler/ir/graph/symbolic-shape.test.js) — `SymInt` dimensions, `dimEquals` on symbolic expressions, and broadcasting where one side is named.
- [`tests/compiler/ir/graph/reshape-inferred-dim.test.js`](../../../tests/compiler/ir/graph/reshape-inferred-dim.test.js) — what a `-1` in a shape means on each side of the boundary: the placeholder an operation resolves against its operand, and the extent no type may carry.
- [`tests/compiler/ir/graph/comparison.test.js`](../../../tests/compiler/ir/graph/comparison.test.js) — the three relations of §10.4, pinned against each other.
- [`tests/e2e/dynamic-shapes.test.js`](../../../tests/e2e/dynamic-shapes.test.js) — the same types, end to end: one kernel serving many batch sizes.

---

**Next:** [Chapter 11 — Ops as a dialect](../ch11-ops-as-a-dialect/README.md), which explains where those inferred types actually come from.
