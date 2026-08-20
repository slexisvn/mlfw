# Chapter 20 — Algebraic simplification meets IEEE 754

`x + 0 = x`. `x × 0 = 0`. `x − x = 0`. `x ÷ x = 1`.

Four identities every reader learned before they were ten. In a compiler for real numbers, all four are rewrite rules. In a compiler for floating-point numbers, **one of them is true**, one is true only under an assumption, and two are false.

This chapter is about which is which, why the distinction is a flag rather than a fact, and what it looks like when the flag is not consulted.

## 20.1 The problem: the compiler's algebra is not the machine's

Chapter 17 gave you a rewrite system: rules as objects, a worklist, a benefit ordering. Nothing in it asked whether a rule is *true*. A pattern that matches `mul` with a zero operand and rewrites to zero is mechanically identical to one that matches `transpose(transpose(x))` and rewrites to `x`, and the machinery cannot tell them apart.

The trouble is that a float is not a real number. It is a finite set of values plus three things the reals do not have:

- **Two zeros.** `+0.0` and `−0.0` are distinct bit patterns that compare equal. They are distinguishable by division: `1/(+0) = +∞` and `1/(−0) = −∞`.
- **Infinities.** `+∞` and `−∞` are values, not errors, and arithmetic on them is defined.
- **NaN.** Not-a-Number is produced by `0×∞`, `∞−∞`, `0/0`, and it propagates: any operation with a NaN operand returns NaN. It compares unequal to everything, including itself.

Each of these breaks at least one schoolbook identity, and the breakages are not obscure corners. A model that produces an `∞` in an intermediate — an overflowing `exp`, a division by a zero variance — and then multiplies it by a mask of zeros is relying on `0 × ∞ = NaN` to tell it something went wrong. A compiler that rewrites that multiply to `0` deletes the evidence.

## 20.2 Intuition: two algebras, and a switch between them

There are two consistent positions.

**Position one: the program means what IEEE 754 says it means.** Every rewrite must produce bit-identical results for every input, including the strange ones. Under this rule `x − x → 0` is illegal, because if `x` is `∞` the true answer is NaN.

**Position two: the program means what it would mean over the real numbers**, and the floats are an approximation the compiler may re-approximate. Now `x − x → 0` is fine, and so is reassociating a sum, and so is turning a division into a multiplication by a reciprocal.

Neither is wrong. Position one is what you want when a NaN is a signal you are debugging. Position two is what you want when you know your data is finite and you would like the compiler to be aggressive. What is wrong is *not choosing* — applying position-two rewrites while telling users they are getting position one.

So the design is: rules that are valid under position one are always on; rules that need position two are gated behind a flag the user sets. Everything then depends on every such rule being correctly classified, and §20.6 is about what happens when one is not.

## 20.3 Theory: which identities survive

Write `≡` for "produces the same bits for every input, in every rounding mode".

> **Theorem 20.1 (Identities valid over floats).** For IEEE 754 binary arithmetic with default rounding:
> - `x × 1 ≡ x`
> - `x ÷ 1 ≡ x`
> - `x − 0 ≡ x`
> - `−(−x) ≡ x`

*Proof sketch.* Multiplication and division by exactly 1 are exact and preserve sign and payload, so they are the identity on every finite value, on both infinities, and on NaN. `x − 0`: for finite non-zero `x` the result is `x` exactly; for `x = +0` it is `+0`; for `x = −0` it is `−0 − +0 = −0` — subtraction of a positive zero preserves a negative zero, which is precisely the case that breaks the addition version. Negation flips one bit. ∎

> **Theorem 20.2 (Identities not valid over floats).** Each of the following fails on at least one input:
> - `x + 0 ≢ x`. **Counterexample:** `x = −0`. IEEE addition gives `(−0) + (+0) = +0`; the rewrite yields `−0`. The two differ under `1/x` and under `copysign`.
> - `x × 0 ≢ 0`. **Counterexamples:** `x = ∞` gives NaN, not 0. `x = NaN` gives NaN. `x = −0` gives `−0`, not `+0`.
> - `x − x ≢ 0`. **Counterexamples:** `x = ∞` gives NaN. `x = NaN` gives NaN.
> - `x ÷ x ≢ 1`. **Counterexamples:** `x = 0` gives NaN. `x = ∞` gives NaN. `x = NaN` gives NaN.

Note the shape of these. `x + 0` fails only on signed zero — a quiet, single-bit difference that surfaces only if something downstream divides by the result or inspects its sign. The other three fail loudly, turning a NaN into a number, which is worse: a NaN is a diagnostic, and erasing it converts a detectable failure into a plausible wrong answer.

> **Definition 20.3 (Fast-math licence, stated here).** A *fast-math licence* is a user assertion that the program's values are finite and that signed zero is not observed. Under it, the compiler may apply identities that hold over the reals, and results may differ from IEEE 754 in bits, in NaN propagation, and in signed zero.

The important word is *assertion*. The compiler does not verify finiteness — it cannot, in general — so the licence transfers responsibility rather than establishing a fact. That is exactly how `-ffast-math` works in C compilers, and it is why the flag is per-compilation and not per-operation.

Two things that are **not** licensed by any flag in this compiler, and are worth naming so they are not assumed:

- **Reassociation.** `(a + b) + c ≢ a + (b + c)` even over finite floats, because rounding happens at each step. Chapter 11's `ASSOCIATIVE` trait exists and no pass uses it to reassociate float arithmetic.
- **Distribution.** `a×b + a×c ≢ a×(b + c)`, same reason.

## 20.4 In mlfw: the gate is a constructor argument

[`simplify/algebraic.ts`](../../../src/compiler/passes/simplify/algebraic.ts) builds two pattern sets, once, at module load ([`algebraic.ts:9`](../../../src/compiler/passes/simplify/algebraic.ts)):

```ts
function buildAlgebraicPatterns(fastMath: boolean): PatternSet {
  const set = new PatternSet();
  set.add(new pat.AddZero());
  set.add(new pat.SubZero());
  set.add(new pat.SubSelf(fastMath));
  set.add(new pat.MulOne());
  set.add(new pat.MulZero(fastMath));
  set.add(new pat.DivOne());
  set.add(new pat.DoubleNeg());
  set.add(new pat.TransposeTranspose());
  set.add(new pat.ReshapeReshape());
  set.add(new pat.MulNegNeg());
  set.add(new pat.AddNegToSub());
  set.add(new pat.SubNegToAdd());
  set.add(new pat.DoubleConvert());
  if (fastMath) {
    set.add(new pat.DivSelf(fastMath));
    set.add(new pat.ExpLog(fastMath));
    set.add(new pat.LogExp(fastMath));
  }
  return set;
}
```

Three of the sixteen are only *present* under fast-math; two more are present always and *decide for themselves*. `SubSelf` ([`patterns.ts:199`](../../../src/compiler/ir/graph/patterns.ts)):

```ts
  override match(op: Operation): boolean {
    if (op.getOperand(0) !== op.getOperand(1)) return false;
    return isDtypeInt((op.getResult(0).type as TensorType).dtype) || this.fastMath;
  }
```

`x − x → 0` is applied unconditionally **for integers** — where it is simply true, there being no NaN and no signed zero in two's complement — and otherwise only under the licence. `MulZero` ([`patterns.ts:235`](../../../src/compiler/ir/graph/patterns.ts)) is the same shape. This is the right design: the licence is consulted per rule, and the dtype does half the work, so integer code gets the aggressive rewrites for free.

The flag reaches the pass from the compiler configuration, in one line of the pipeline builder ([`graph_pipeline.ts:41`](../../../src/compiler/pipeline/graph_pipeline.ts)):

```ts
    new AlgebraicSimplificationPass({ fastMath: config.optimization.fastMath }),
```

`optimization.fastMath` defaults to `false` ([`compiler.ts:150`](../../../src/compiler/pipeline/compiler.ts)), so out of the box the compiler is in position one.

## 20.5 Lab 1 — The identities, tested on the hard inputs

```bash
node docs/part4/ch20-algebra-and-ieee754/labs/01-identities-under-ieee754.mjs
```

Six identities, applied to a tensor containing exactly the values that break them, run eagerly (which is IEEE 754, one operation at a time, no compiler in the way) and compiled, in both modes.

```
=== default: fastMath off ===
input     [NaN, Infinity, -Infinity, -0]
x + 0    eager [NaN, Infinity, -Infinity, 0]  compiled [NaN, Infinity, -Infinity, -0] <-- DIFFERENT
x - 0    eager [NaN, Infinity, -Infinity, -0] compiled [NaN, Infinity, -Infinity, -0] 
x * 1    eager [NaN, Infinity, -Infinity, -0] compiled [NaN, Infinity, -Infinity, -0] 
x * 0    eager [NaN, NaN, NaN, -0]            compiled [0, 0, 0, 0]                   <-- DIFFERENT
x - x    eager [NaN, NaN, NaN, 0]             compiled [NaN, NaN, NaN, 0]             
x / x    eager [NaN, NaN, NaN, NaN]           compiled [NaN, NaN, NaN, NaN]           
```

Read the eager column first: it is Theorems 20.1 and 20.2 as measurements. `x − 0` preserves `−0`; `x + 0` turns it into `+0`; `∞ × 0` and `∞ − ∞` and `∞/∞` are all NaN. Nothing here is surprising if you know IEEE 754, and all of it is surprising if you only know algebra.

Now the compiled column, with fast-math **off**:

- **`x − 0` and `x × 1` agree.** Theorem 20.1's rules applied, results identical. This is the system working.
- **`x − x` and `x ÷ x` agree.** The patterns are present but declined, because the dtype is `f32` and the licence was not granted. The compiled kernel actually performs the subtraction and the division. This is the gate working.
- **`x + 0` differs**, on `−0`.
- **`x × 0` differs**, on everything but the zero.

And with fast-math **on**:

```
x - x    eager [NaN, NaN, NaN, 0]             compiled [0, 0, 0, 0]                   <-- DIFFERENT
x / x    eager [NaN, NaN, NaN, NaN]           compiled [1, 1, 1, 1]                   <-- DIFFERENT
```

`x − x` and `x ÷ x` now diverge — which is not a bug, it is the licence being exercised. The user asserted their values are finite; the compiler took them at their word and replaced two elementwise kernels with a constant. That is the whole trade, visible in four numbers.

Which leaves two rows that differ in *both* modes.

## 20.6 Lab 2 — Where the rewrite happened

```bash
node docs/part4/ch20-algebra-and-ieee754/labs/02-where-the-rewrite-happened.mjs
```

The two anomalies have different causes, and the lab separates them by following one program through three representations.

```
=== x + 0 ===
traced:      constant -> add -> return
after passes: return(%0)
kernel:
   buf_3[i1_5] = buf_1[i1_5];

=== x * 0 ===
traced:      constant -> mul -> return
after passes: %1 = constant() {tensor_type = tensor<xf32>, value = 0} : tensor<xf32> | %2 = mul(%0, %1) : tensor<1x2xf32> | return(%2)
kernel:
   buf_3[i1_6] = 0;
```

**`x + 0` is a graph-level rewrite.** After the graph passes the function is `return(%0)` — the add is gone. The pattern responsible is `AddZero` ([`patterns.ts:167`](../../../src/compiler/ir/graph/patterns.ts)):

```ts
export class AddZero extends Pattern {
  constructor() { super('add_zero', 5); this.rootOpName = 'add'; }
  override match(op: Operation): boolean {
    return isConstantVal(op.getOperand(1).definingOp, 0) || isConstantVal(op.getOperand(0).definingOp, 0);
  }
```

Compare it to `SubZero` immediately below, and to `SubSelf` and `MulZero`, all of which take a `fastMath` argument. `AddZero` does not. By Theorem 20.2 it needs one: `(−0) + (+0) = +0` and the rewrite yields `−0`. The neighbouring rule for subtraction is sound and needs no gate; the addition rule is not and has none. On the evidence, this is an oversight rather than a decision — the rule is one line away from three rules that got it right.

**`x × 0` is not a graph-level rewrite at all.** The graph after every pass still contains `mul(%0, %1)` — `MulZero` was present, checked the dtype, found `f32`, found no licence, and correctly declined. Lowering keeps the multiply, and so does scheduling; the TensorIR reads `buf_3[v0_7, v1_8] = (buf_1[0, v1_8] * buf_4[])`. The multiply survives every layer the flag is visible in.

Then the CPU code generator emits its expression, and hits this ([`backend/cpu/codegen.ts:472`](../../../src/backend/cpu/codegen.ts)):

```ts
              else if (node.op === '*' && (a === '0' || b === '0')) { vals.push('0'); }
```

`a` and `b` here are *rendered strings*. There is no dtype at this point, no operation registry, and no compiler configuration — just two pieces of JavaScript text, one of which happens to read `0`. The identity Theorem 20.2 forbids is applied by a peephole that could not have consulted the flag even if it wanted to.

The CUDA backend has no such peephole; its `MathOpNode` case emits `(${a} ${node.op} ${b})` and stops ([`backend/cuda/codegen.ts:492`](../../../src/backend/cuda/codegen.ts)). So the same program, compiled for two targets, produces `0` on one and `NaN` on the other for a non-finite input. That is a differential-testing finding in the sense of Chapter 65 — the two backends disagree, and the test that would catch it is one that feeds non-finite inputs to both.

Both anomalies are recorded in the outline's Appendix E. Neither is fixed in the code this book describes, because a book that quietly patches its subject is a book you cannot check.

**Try this.** Run the same two programs against `WasmTarget()`. WebAssembly's code generator has no zero peephole either — predict which of the two anomalies survives the change of target, and which does not.

## 20.7 The general lesson: an identity has a level

The two anomalies are different bugs with the same shape. An algebraic identity can be applied at four levels of this compiler:

| Level | Where | Knows the dtype? | Knows the fast-math flag? |
|---|---|---|---|
| Graph patterns | [`patterns.ts`](../../../src/compiler/ir/graph/patterns.ts) | yes | yes |
| TIR construction | [`mathOp`](../../../src/compiler/ir/tensor/nodes.ts) | integer-only by construction | no |
| LIR simplification | [`flat_index_simplify.ts`](../../../src/compiler/passes/simplify/flat_index_simplify.ts) | index arithmetic only | no |
| Backend text emission | `backend/*/codegen.ts` | no | no |

Only the top row is in a position to be careful. The TIR folder is safe by construction — it folds only when an operand is an `IntImmNode`, an integer immediate from index arithmetic ([`nodes.ts:393`](../../../src/compiler/ir/tensor/nodes.ts)) — and that is not an accident, it is the same reasoning as `SubSelf`'s dtype check, applied structurally.

The rule that falls out: **an algebraic rewrite belongs at the highest level that can see the types.** Every level below that is doing arithmetic on syntax, and syntax does not carry the information the rewrite needs.

## 20.8 Traps and limits

- **`AddZero` is not fast-math gated and should be** ([`patterns.ts:167`](../../../src/compiler/ir/graph/patterns.ts)). §20.6 is the demonstration. The fix is the one-line dtype-or-licence check its three neighbours already have.
- **The CPU backend folds `x * 0`, `x + 0`, `x - 0`, `x * 1` on rendered strings** ([`codegen.ts:470`](../../../src/backend/cpu/codegen.ts)). Three of the four are sound by Theorem 20.1; the multiply-by-zero is not, and no other backend does it.
- **`fastMath` is one global flag.** There is no per-operation, per-function or per-region control, so a model with one numerically delicate layer either gives up fast-math everywhere or accepts it everywhere. This is the same limitation C compilers have, and the same workaround applies: compile the delicate part separately.
- **The licence is asserted and never checked.** Nothing verifies finiteness, and nothing warns when a fast-math rewrite fires on a graph whose inputs the compiler has actually seen. The optimization gate (Chapter 62) *does* verify candidate optimizations numerically before keeping them — that machinery exists, and fast-math does not use it.
- **Integer division is not simplified at all.** `DivSelf` is fast-math-only, so `x / x` for integers is left alone even though the only exception is `x = 0`, which is a trap on most hardware rather than a wrong value. Conservative, and inconsistent with `SubSelf`'s treatment of integers.
- **Reassociation is absent, and that is deliberate.** `ASSOCIATIVE` is declared on `add` and `mul` (Chapter 11), and the only pattern that uses it, `AssociativeConstantReassoc`, reassociates *constants* — it requires both reassociated operands to be constant operations and folds them ([`patterns.ts:491`](../../../src/compiler/ir/graph/patterns.ts)). No pass reassociates a float sum of runtime values, under any flag.

## 20.9 Read the tests

- [`tests/compiler/passes/simplify/`](../../../tests/compiler/passes/simplify/) — the algebraic patterns, including which ones require `fastMath`.
- [`tests/compiler/ir/rewrite/pattern.test.js`](../../../tests/compiler/ir/rewrite/pattern.test.js) — the applicator behaviour these rules rely on, from Chapter 17.
- [`tests/e2e/`](../../../tests/e2e/) — the differential tests that compare compiled against eager. Note what they feed: ordinary finite tensors. A test that would have caught §20.6 is one that feeds `NaN` and `±∞`, and Chapter 65 argues for exactly that.

---

**Next:** [Chapter 21 — Decomposition](../ch21-decomposition/README.md), which goes the other way: instead of making the graph smaller, it deliberately makes it much bigger, and then relies on the next four chapters to make it small again.
