# Chapter 37 — Proving things about indices

Chapter 35 needed `0 ≤ f_R ≤ c−1` before it could split a division. Chapter 36 needed `|distance| < extent` before it could call a dependence real. Chapter 34's `pad` rule emitted four bounds tests and only two of them are ever false. Every one of those is the same question: **given the loop extents, what range can this expression take?**

The component that answers it is 141 lines long, and the interesting thing about it is not what it proves. It is the precise shape of what it is allowed to get wrong.

## 37.1 The problem: a compiler that guesses is a compiler that miscompiles

A guard costs a comparison per element. Deleting one that is always true is free performance; deleting one that is sometimes false is a buffer overrun.

Consider `pad`, which shifts an index by the low padding and tests whether the result landed inside the source. For a `[4,3]` input padded by one row at each end, the rule emits this condition — §37.5's lab prints it verbatim:

```
((1 * (((v0_7 + -1) >= 0) * ((v0_7 + -1) < 4))) * (((v1_8 + 0) >= 0) * ((v1_8 + 0) < 3)))
```

Four comparisons: two on the row axis, which was padded, and two on the column axis, which was not. The column pair is the interesting one. `v1_8 + 0` is exactly the column loop variable, running over `[0, 3)`, so both of its tests are constants. Removing them removes two comparisons from every element of the output.

But nothing in the expression says so. The compiler has to *derive* that `v1_8` is in `[0,2]`, propagate that through `+ 0`, and compare against `0` and `3`. And whatever machinery does that will meet expressions it cannot handle — a subscript loaded from another buffer, a divisor that is a runtime shape, a product of two variables. What it does then is the whole design.

## 37.2 Intuition: replace every number with the interval it might be in

Run the program with intervals instead of integers. A loop variable of extent 4 is `[0,3]`. Adding `−1` gives `[−1,2]`. Comparing `[−1,2] ≥ 0` gives *maybe*. Comparing `[0,2] < 3` gives *definitely*.

Three properties make this work as a compiler component.

**It is cheap.** One pass over the expression, constant work per node.

**It is sound in one direction.** The computed interval always *contains* the true set of values. It may be too wide — `[−7,7]` for an expression that is really always `0` — and it is never too narrow.

**Too wide degrades to "I don't know", never to a wrong answer.** Because the interval contains the truth, a test that comes out definitely-true really is always true. A test that comes out *maybe* might have been provable by a stronger analysis, and the cost of that is a guard that survives, not a program that breaks.

The trade is asymmetric on purpose: the analysis is allowed to be imprecise and is not allowed to be unsound.

## 37.3 Theory

> **Definition 37.1 (Interval abstraction).** **(classical)** An *abstract value* is a pair `[lo, hi]` with `lo ∈ ℤ ∪ {−∞}` and `hi ∈ ℤ ∪ {+∞}`. Its *concretisation* `γ([lo,hi])` is `{n ∈ ℤ : lo ≤ n ≤ hi}`.

> **Definition 37.2 (Sound abstraction).** **(classical)** An abstract operation `f#` is *sound* for a concrete operation `f` if for all abstract values `A, B`, `{f(a,b) : a ∈ γ(A), b ∈ γ(B)} ⊆ γ(f#(A,B))`.

> **Theorem 37.3 (Soundness of interval arithmetic; Moore, 1966, with the soundness criterion of Cousot and Cousot, 1977).** **(classical)** With `[a₁,a₂] + [b₁,b₂] = [a₁+b₁, a₂+b₂]`, `−[a₁,a₂] = [−a₂,−a₁]`, `[a₁,a₂]·[b₁,b₂] = [min P, max P]` for `P` the four endpoint products, and division by a positive constant `c` given by `[⌊a₁/c⌋, ⌊a₂/c⌋]`, every operation is sound.

*Proof sketch.* Addition and negation are monotone in each argument, so the extremes of the result are attained at the extremes of the arguments. Multiplication is monotone in each argument for a fixed sign of the other, so the extremes are attained at some pair of endpoints, and taking the min and max over all four covers every sign case. Floor division by a positive constant is monotone. ∎

> **Theorem 37.4 (One-sided decidability).** **(classical)** Let `E` be an expression and `B` its computed interval. If `B ⊆ [0, ∞)` then `E ≥ 0` on every concrete state. The converse fails: `E ≥ 0` everywhere does not imply `B ⊆ [0,∞)`.

*Proof.* The forward direction is Definition 37.2 applied inductively over the expression. For the converse, take `E = i − j` where `i` and `j` are two names for values that happen always to be equal: `B = [−e, e]`, and `E` is identically zero. ∎

The converse's counterexample names the essential limitation:

> **Definition 37.5 (Non-relational domain).** **(classical)** An abstract domain is *non-relational* if it assigns each variable an abstract value independently. Such a domain cannot express any relationship between variables — `i ≤ j`, `i + j = n`, `i ≡ 0 mod 4` — and therefore loses precision at every point where two occurrences of related quantities meet.

Intervals are non-relational. Relational domains exist (octagons, convex polyhedra) and cost more; this compiler uses none of them, and §37.7 lists what that costs.

### And one theorem that is not about intervals

The interval analysis has a second customer, and it is the one that settled a question spanning the whole book.

> **Theorem 37.6 (Floor and truncation agree on non-negative dividends).** **(classical)** For integers `a ≥ 0` and `c > 0`, `⌊a/c⌋ = trunc(a/c)` and `a − c⌊a/c⌋ = a mod c` under either convention.

*Proof.* For `a ≥ 0` the real quotient `a/c` is non-negative, and for a non-negative real, floor and truncation are the same function. The remainders then agree because both are `a − c·q` for the same `q`. ∎

This matters because the two conventions disagree for negative dividends — `⌊−7/2⌋ = −4` while `trunc(−7/2) = −3` — and every backend's native integer division truncates. The compiler's rule is: `//` and `%` **mean floor** everywhere, from one definition ([`util/divmod.ts`](../../../src/util/divmod.ts)), and the truncating pair `tdiv`/`tmod` is introduced by the simplifier **only where the dividend is provably non-negative**, which by Theorem 37.6 is exactly where the two agree. Chapters 54 to 58 hold every backend to it.

So the interval analysis is not only an optimisation. It is the licence for a semantic substitution, and if it were unsound the compiler would compute wrong numbers rather than merely slow ones.

## 37.4 In mlfw: 141 lines and a bound of `EVERYTHING`

### The domain

[`analysis/analyzer.ts`](../../../src/compiler/analysis/analyzer.ts) is Definition 37.1:

```ts
export class IntBound {
  min: number;
  max: number;
  ...
  isConst(): boolean {
    return this.min === this.max && isFinite2(this.min);
  }
}

const EVERYTHING = new IntBound(NEG_INF, POS_INF);
```

`EVERYTHING` is the interval `[−∞, +∞]`: the answer that excludes nothing and therefore proves nothing. Every abstract domain has one — the *top* of its lattice, the value meaning "could be anything" — and returning it is how this analysis says "I don't know" without ever saying something false. It is returned from six places in this file. Reading those six is the fastest way to know what this compiler cannot prove: a non-symbolic expression, an unbound variable, an unrecognised operator, a multiplication with an infinite endpoint, and a division or modulo whose divisor is not a positive constant.

`bind` is how facts get in ([`analyzer.ts:61`](../../../src/compiler/analysis/analyzer.ts)), and there are four call sites in the whole compiler. All four say the same thing: a loop `for i in 0..e` with a literal `e` binds `i` to `[0, e−1]` ([`simplify_tir.ts:33`](../../../src/compiler/passes/simplify/simplify_tir.ts)):

```ts
function bindLoopVar(ctx: SimplifyCtx, name: string, extentNode: TirNode): VarBound {
  const prev = ctx.analyzer.getVarBound(name);
  const imm = extentNode as IntImmNode;
  if (extentNode && extentNode.type === 'IntImmNode' && imm.value > 0) {
    ctx.analyzer.bind(name, 0, imm.value - 1);
  } else {
    ctx.analyzer.setVarBound(name, null);
  }
  return prev;
}
```

Note the `else`: a non-literal extent *unbinds* the variable rather than leaving a stale bound from an outer scope. That is the single most important line in the file, because the alternative is unsound.

Block iteration variables get bounds too, but derived rather than asserted: `setVarBound(name, irBound(analyzer, binding))` ([`simplify_tir.ts:62`](../../../src/compiler/passes/simplify/simplify_tir.ts)) evaluates the binding in the current environment, which is Chapter 33's iteration map feeding Chapter 37. So every bound in the system traces back to a literal loop extent.

There is a fifth entry point, `bindShape` ([`analyzer.ts:81`](../../../src/compiler/analysis/analyzer.ts)), which would bind a whole shape environment at once. It has one caller and it is a test.

### The transfer functions

`constIntBound` ([`analyzer.ts:160`](../../../src/compiler/analysis/analyzer.ts)) is Theorem 37.3:

```ts
      case 'add':
        return new IntBound(boundLo(a.min + b.min), boundHi(a.max + b.max));
      case 'sub':
        return new IntBound(boundLo(a.min - b.max), boundHi(a.max - b.min));
      case 'neg':
        return new IntBound(-a.max, -a.min);
      case 'mul':
        return this._mulBound(a, b);
      case 'max':
        return new IntBound(Math.max(a.min, b.min), Math.max(a.max, b.max));
      case 'min':
        return new IntBound(Math.min(a.min, b.min), Math.min(a.max, b.max));
```

`boundLo` and `boundHi` map `NaN` to the corresponding infinity, which is what makes `∞ − ∞` degrade to "unknown" instead of poisoning the lattice.

Three operations refuse rather than approximate:

```ts
  _mulBound(a: IntBound, b: IntBound): IntBound {
    if (!isFinite2(a.min) || !isFinite2(a.max) || !isFinite2(b.min) || !isFinite2(b.max)) {
      return EVERYTHING;
    }
    ...
  _divBound(a: IntBound, b: IntBound, rounder: (x: number) => number): IntBound {
    if (b.isConst() && b.min > 0) { ... }
    return EVERYTHING;
  }

  _modBound(a: IntBound, b: IntBound): IntBound {
    if (b.isConst() && b.min > 0) {
      return new IntBound(0, b.min - 1);
    }
    return EVERYTHING;
  }
```

`_modBound` is worth pausing on: when the divisor is a positive constant, the bound is `[0, c−1]` **regardless of the dividend**, which is only valid because `%` is floor-mod. Under truncating semantics a negative dividend would give a negative remainder and this line would be unsound. Definition and analysis agree because the definition was chosen to make them agree.

### From a bound to a decision

The three `canProve` methods reduce everything to a subtraction ([`analyzer.ts:278`](../../../src/compiler/analysis/analyzer.ts)):

```ts
  canProveGreaterEqual(expr: SymExpr, value: SymExpr): boolean {
    const bound = this.constIntBound(SymInt.sub(expr, value));
    return bound.min >= 0;
  }
```

and [`analysis/ir_arith.ts`](../../../src/compiler/analysis/ir_arith.ts) lifts that to TIR nodes. `irToSymInt` ([`ir_arith.ts:22`](../../../src/compiler/analysis/ir_arith.ts)) is the bridge, and like `toLinearForm` in Chapter 35 its `null` cases define the frontier: a `BufferLoadNode`, a `CastNode`, a `CompareNode`, an extern call other than `max`/`min`, and a division whose divisor is not a positive literal all return `null`, which means no bound at all.

Then two tables and two predicates ([`ir_arith.ts:79`](../../../src/compiler/analysis/ir_arith.ts)):

```ts
const CMP_TRUE: ComparePredicates = {
  lt: (d) => d.max < 0,
  le: (d) => d.max <= 0,
  gt: (d) => d.min > 0,
  ge: (d) => d.min >= 0,
  eq: (d) => d.min === 0 && d.max === 0,
  ne: (d) => d.min > 0 || d.max < 0,
};
```

Six comparison directions, each decided by one endpoint of the difference. `CMP_FALSE` is the mirror image, and the two are not complements — an interval that straddles satisfies neither, which is the third answer.

### The consumers

**Guard elision.** `simplifyStmt` folds an `IfThenElseNode` whose condition is decided ([`simplify_tir.ts:73`](../../../src/compiler/passes/simplify/simplify_tir.ts)):

```ts
      const cond = simplifyExpr(ite.condition, ctx);
      if (proveTrue(ctx.analyzer, cond)) return simplifyStmt(ite.thenBody, ctx);
      if (proveFalse(ctx.analyzer, cond)) return ite.elseBody ? simplifyStmt(ite.elseBody, ctx) : new SeqNode([]);
```

and `RewriteSimplify._simplifyCompare` folds a decided comparison to `0` or `1` inside an expression, which is how a four-way product of comparisons loses two of its factors.

**The `tdiv` substitution.** Theorem 37.6, implemented ([`ir_arith.ts:165`](../../../src/compiler/analysis/ir_arith.ts)):

```ts
function nonNegativeDivMod(analyzer: Analyzer, node: MathOpNode): MathOpNode | null {
  if (node.op !== '//' && node.op !== '%') return null;
  const divisor = node.b as IntImmNode | null;
  if (!divisor || divisor.type !== 'IntImmNode' || divisor.value <= 0) return null;
  if (!boundWithin(analyzer, node.a, 0, Infinity)) return null;
  return new MathOpNode(node.op === '//' ? 'tdiv' : 'tmod', node.a, node.b as TirNode);
}
```

It is tried last, after Corollary 35.6 and after Theorem 35.5, so a division that could disappear entirely is never merely made cheaper.

**The dependence tests of Chapter 36**, which use extents rather than the `Analyzer` directly, and **`classifyBufferIndex`** ([`legality.ts:59`](../../../src/compiler/schedule/legality.ts)), which turns a bound into a three-valued in-bounds answer for the schedule validator:

```ts
  if (b.min >= 0 && b.max <= dimExtent - 1) return 'in';
  if (b.min > dimExtent - 1 || b.max < 0) return 'oob';
  return 'unknown';
```

Definition 37.1 and Theorem 37.4 in five lines: `in`, `oob`, and the third answer that is not a failure.

### What happens when there is no bound at all

An index loaded from a buffer has no expression to abstract. The compiler emits no guard and removes none — and the safety question is answered outside the kernel, on the host, once per call. [`analysis/index_bounds.ts`](../../../src/compiler/analysis/index_bounds.ts) walks the *graph* for indexed-table operations ([`index_bounds.ts:42`](../../../src/compiler/analysis/index_bounds.ts)):

```ts
  for (const op of func.ops()) {
    const spec = INDEXED_TABLE_OPS.get(op.opName);
    if (!spec) continue;
    const idx = traceToArg(op.getOperand(spec.indices), argIndex);
    if (idx === undefined) continue;
```

`traceToArg` ([`index_bounds.ts:17`](../../../src/compiler/analysis/index_bounds.ts)) follows the index operand backwards through `reshape`, `transpose`, `reverse` and `broadcast_in_dim` — the operations that move values without changing them — to a function argument. If it reaches one, the table's extent along the indexed axis becomes a precondition checked before every call.

The reasoning is complementary to everything else in this chapter: the analyser cannot bound the index, so the bound is turned into a runtime obligation on the caller and checked at the one place the values are available.

## 37.5 Lab — guards that disappear

```bash
node docs/part6/ch37-proving-things-about-indices/labs/01-guards-that-disappear.mjs
```

```
  pad rows only    [4,3] -> [6,3]
    lowered  (4 comparisons): ((1 * (((v0_7 + -1) >= 0) * ((v0_7 + -1) < 4))) * (((v1_8 + 0) >= 0) * ((v1_8 + 0) < 3)))
    emitted  (2 comparisons): ((((i0_5 + -1) >= 0) * ((i0_5 + -1) < 4)) ? buf_1[(((i0_5 + -1) * 3) + i1_6)] : 0);

  pad columns only [4,3] -> [4,5]
    lowered  (4 comparisons): ((1 * (((v0_7 + 0) >= 0) * ((v0_7 + 0) < 4))) * (((v1_8 + -1) >= 0) * ((v1_8 + -1) < 3)))
    emitted  (2 comparisons): ((((i1_6 + -1) >= 0) * ((i1_6 + -1) < 3)) ? buf_1[((i0_5 * 3) + (i1_6 + -1))] : 0);

  pad both         [4,3] -> [6,5]
    lowered  (4 comparisons): ((1 * (((v0_7 + -1) >= 0) * ((v0_7 + -1) < 4))) * (((v1_8 + -1) >= 0) * ((v1_8 + -1) < 3)))
    emitted  (4 comparisons): (((((i0_5 + -1) >= 0) * ((i0_5 + -1) < 4)) * (((i1_6 + -1) >= 0) * ((i1_6 + -1) < 3))) ? buf_1[(((i0_5 + -1) * 3) + (i1_6 + -1))] : 0);

  pad nothing      [4,3] -> [4,3]
    no pad op reached lowering: canonicalisation removed it
    emitted  (0 comparisons): buf_1[((i0_4 * 3) + i1_5)];
```

The rule emits four comparisons every time; the analyser decides which of them are constants. Work the second row by hand. The column loop has extent 5, so `v1` is bound to `[0,4]`; `v1 − 1` has bound `[−1,3]`, which straddles zero, so neither test is decided. The row loop has extent 4, so `v0` is bound to `[0,3]`; `v0 + 0 − 0` has bound `[0,3]` with `min ≥ 0`, so `≥ 0` is proved, and `v0 + 0 − 4` has bound `[−4,−1]` with `max < 0`, so `< 4` is proved. Two constants, folded to `1`, and `1 * x` folded away by `mathOp`.

The pattern is worth stating plainly: **the rule is allowed to be careless because the analysis is reliable.** `pad`'s author did not have to check which axes were padded. A rule that tried to be clever about it would be a second implementation of the same reasoning, in a place with less information.

The fourth row is Part IV finishing the job from the other end: a pad of zero on every axis never reaches this phase at all.

The last section shows what is left on the table:

```
  function Object(buf_1, buf_3) {
    for (let i0_5 = 0; i0_5 < 6; i0_5++) {
      for (let i1_6 = 0; i1_6 < 3; i1_6++) {
        buf_3[((i0_5 * 3) + i1_6)] = ((((i0_5 + -1) >= 0) * ((i0_5 + -1) < 4)) ? buf_1[(((i0_5 + -1) * 3) + i1_6)] : 0);
      }
    }
  }
```

The surviving test is false for exactly two of the six rows. Split the outer loop into `0..1`, `1..5` and `5..6` and each piece needs no test at all. That is loop partitioning; `LoopPartitionPass` exists ([`passes/loop_partition/loop_partition.ts`](../../../src/compiler/passes/loop_partition/loop_partition.ts)), matches only the guard shape that a split with a non-dividing extent produces (Chapter 40), and is off by default.

## 37.6 Lab — what the analyser cannot see

```bash
node docs/part6/ch37-proving-things-about-indices/labs/02-what-the-analyzer-cannot-see.mjs
```

Three degrees of knowledge, in one table:

```
=== 1. an index the analyser can bound completely ===
  x.transpose(1,0)             buf_1[((i1_5 * 3) + i0_4)];

=== 2. an index the analyser bounds well enough for tdiv, not enough to fold ===
  x.reshape([12]) from [4,3]   buf_1[((((i0_4 / 3) | 0) * 3) + (i0_4 % 3))];

=== 3. an index the analyser cannot bound at all ===
  embedding lookup             buf_3[(((buf_1[i0_6] | 0) * 3) + i1_7)];
```

Case 2 is Theorem 37.6 visible in the output: `/ 3 | 0` is truncating division and `% 3` is truncating modulo, and both appear because `i0_4 + 0` was proved non-negative. Case 3 is `irToSymInt` returning `null` on a `BufferLoadNode`: no expression, no bound, no guard either way.

And the answer for case 3, from the other side of the compiler:

```
  in range  [0,2,4] -> shape [3,3]
  out of range [0,2,9] -> gather: compiled input 0: index 9 at position 2 is out of range for a table of 5 row(s); valid indices are 0..4
```

One host-side check per call, derived at compile time by the graph walk of §37.4. The compiler could not prove it, so it arranged for someone who can to be asked.

**Try this.** Feed the embedding a negative index. The same check catches it, because `assertIndicesInRange` bounds on both sides — and note that the kernel would not have: a negative index in JavaScript reads `undefined` and produces `NaN`, silently.

### The guard covers arguments, not indices

That last parenthesis is the whole of §37.6's exposure, so it is worth following. The guard is not derived from the *access*; it is derived from an **argument**. `collectArgIndexBounds` walks the graph for indexed-table operations and then does this ([`index_bounds.ts:52`](../../../src/compiler/analysis/index_bounds.ts)):

```ts
    const idx = traceToArg(op.getOperand(spec.indices), argIndex);
    if (idx === undefined) continue;
```

`traceToArg` follows the index operand backwards, looking for a function argument. If it finds one, a bound is recorded against that argument's index and checked on every call. **If it finds anything else, the operation is skipped and no bound is recorded at all.**

"Anything else" decides the coverage, so it is worth seeing what falls outside it:

> **Counterexample 37.7.** Two models, same table of five rows, same input `[0, 2, 4]`:
>
> ```js
> class Direct   extends Module { forward(idx, t) { return ops.embedding(t, idx); } }
> class Computed extends Module { forward(idx, t) { return ops.embedding(t, idx.add(1)); } }
> ```
>
> If `traceToArg` gives up at the `add`, the computed model reads row 5 of a five-row table and nothing throws: the kernel indexes past the end of the typed array, JavaScript returns `undefined`, and the arithmetic turns it into `NaN`. On a backend without JavaScript's bounds-checked arrays it is an out-of-bounds read.

### Carrying the offset back to the argument

The answer follows the shape of the problem. An index of the form `arg + c` is still a statement *about `arg`*: the access is in range exactly when `0 ≤ arg + c < limit`, which is `−c ≤ arg < limit − c`. So `traceToArg` walks through `add` and `sub` with a scalar constant operand, accumulating the offset, and the recorded bound becomes a **range on the argument** rather than a table extent:

```ts
    const lo = -traced.offset;
    const hi = limit - traced.offset;
```

`assertIndicesInRange` takes `lo` and `hi` instead of a single `limit`, which also lets the error name the interval the caller must satisfy. The check runs on the host, once per call, over an argument — nothing is added to the kernel.

Both directions matter, and the second shows the bound is a real range rather than a stricter limit: a model indexing `idx.sub(1)` *accepts* `idx = limit`, which a bare table-extent check would reject.

So the contract is:

| Index expression | Compile-time proof | Runtime guard |
|---|---|---|
| affine in loop variables | yes — the analyser bounds it | not needed |
| a function argument, possibly reshaped | no | **yes** — `assertIndicesInRange` |
| an argument plus or minus a constant | no | **yes** — the range is shifted to compensate |
| genuinely data-dependent (a lookup, an `argmax`, an index scaled by a runtime value) | no | **no** |

The last row is what remains, and it is the irreducible part: when the index is a value the compiler cannot relate to any argument, there is nothing on the host to check, and the only general answer is to emit the comparison in the kernel — a branch per access, and what Chapter 43's GPU backends would need anyway. **Treat a data-dependent index into a table as unchecked**, and clamp it yourself if it can go out of range.

## 37.7 Traps and limits

- **The domain is non-relational, and that is the ceiling.** Definition 37.5. Two loop variables of extent 8 give `i − j ∈ [−7,7]`, so nothing that depends on `i ≤ j` is provable: no triangular iteration space, no "the tail of this tile is shorter than the head". The compiler never builds such nests today, which is why the ceiling has not been hit — and it is exactly what a relational domain would be for.
- **A symbolic divisor loses the modulo bound entirely.** `_modBound` requires a constant positive divisor ([`analyzer.ts:271`](../../../src/compiler/analysis/analyzer.ts)), so `x % n` with `n` a shape parameter gets `EVERYTHING`, even though `0 ≤ x % n < n` holds for every positive `n` under floor semantics. The fact is expressible in this domain — the upper bound would be symbolic, which `IntBound` cannot hold, since its fields are `number`. The consequence is that dynamic shapes lose guard elision on every subscript involving a modulo.
- **Multiplication refuses when any endpoint is infinite.** `_mulBound` returns `EVERYTHING` ([`analyzer.ts:88`](../../../src/compiler/analysis/analyzer.ts)) rather than case-splitting, so `0 · unknown` is `unknown` where it could be `[0,0]`. Sound, and coarser than necessary.
- **Only loop extents are ever bound.** Nothing binds a variable from a surrounding `if`. A body under `if (i < 4)` does not know `i < 4`; the condition is tested and discarded. Path-sensitive bounds would make loop partitioning largely unnecessary, and there is no mechanism for them.
- **`canProveEqual` gets its strength from structural equality, not from the domain.** It first tries `SymInt.equals` ([`analyzer.ts:143`](../../../src/compiler/analysis/analyzer.ts)), and `SymInt.sub` already folds `a − a` to `0`. So `i − i = 0` is decided before an interval is ever computed, and the credit belongs to the symbolic layer.
- **`boundWithin(analyzer, node, 0, Infinity)` is a one-sided test wearing a two-sided signature.** `b.max <= Infinity` is vacuously true, so both call sites in `affineDivMod` and `nonNegativeDivMod` are really "is the minimum non-negative". Harmless, and it reads as if an upper bound were being checked.
- **The bridge is narrower than the domain.** `irToSymInt` handles `IntImmNode`, `VariableNode`, seven `MathOpNode` operators and two extern calls. A `CastNode` is `null`, so an index that is an integer by construction but reached through a cast has no bound — which is why a gather's subscript is unbounded twice over, once for the buffer load and once for the `cast<i32>` around it.
- **Nothing checks the analysis against reality.** There is no assertion mode that evaluates a guard the analyser deleted and compares. The protection is [`tests/compiler/passes/lowering/guard-elision.test.js`](../../../tests/compiler/passes/lowering/guard-elision.test.js) plus the differential tests of Chapter 65, which would catch an unsound elision as a wrong number rather than as a bad proof.

## 37.8 Read the tests

- [`tests/compiler/analysis/analyzer.test.js`](../../../tests/compiler/analysis/analyzer.test.js) — every transfer function including the refusals, and the `canProve` family.
- [`tests/compiler/analysis/ir-arith.test.js`](../../../tests/compiler/analysis/ir-arith.test.js) — `proveTrue`/`proveFalse` over the six directions, and the `tdiv`/`tmod` substitution with its non-negativity precondition.
- [`tests/compiler/passes/lowering/guard-elision.test.js`](../../../tests/compiler/passes/lowering/guard-elision.test.js) — the guards that must disappear, as an executable specification of §37.5.
- [`tests/compiler/passes/simplify/simplify-tir.test.js`](../../../tests/compiler/passes/simplify/simplify-tir.test.js) — the statement-level folding, including the unbinding of a variable whose extent is not literal.
- [`tests/compiler/analysis/sym-int.test.js`](../../../tests/compiler/analysis/sym-int.test.js) — the symbolic layer whose structural simplifications the analyser inherits.

---

**Part VI ends here.** A graph of whole-tensor operations went in; a `PrimFunc` of loops, buffers and scalar stores came out, with every access carrying the region it covers, every pair of accesses answerable for dependence, and every expression answerable for range.

Be careful with the word *exact*, because this part has used it about two different things and only one of them earns it. **Affine analysis is exact for a single-index-variable subscript with equal coefficients** — Theorem 36.5's strong SIV test computes a distance, not an over-approximation, and that case covers most subscripts a lowering rule emits. It is **not** exact in general. The moment a subscript involves two or more loop variables, Chapter 36 falls back to the GCD test, which decides only divisibility and reports a possible dependence whenever the gcd divides the offset — §36.8's `A[i + 64j]` example is a false positive that no bound-aware refinement in this implementation removes. Interval analysis is likewise one-sided by design (Theorem 37.4): it may answer "unknown" where a stronger domain would answer "safe", and §37.7 lists three routine cases where it does.

So the machinery is: **exact on the single-variable affine case, sound-but-imprecise on the multi-variable one, and a refusal outside affine forms entirely.** Where a `reshape` left a `//` and a `%` behind or a `gather` reads its index out of a buffer, `toLinearForm` returns `null` and every question above it is answered `unknown`, in the conservative direction. That last clause is the load-bearing one, and Chapter 36 §36.7 is where it turns out to have an exception.

What has not happened is any decision about *how* to run those loops. Every nest in this part is serial, in the order the rule wrote it, at the size the tensor happened to be.

**Next:** [Part VII — Scheduling](../../part7/README.md), which is the part where that order becomes a choice, and where every legality question is one of the three this part just built.
