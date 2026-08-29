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

Write `≡` for "produces the same bits for every input", with two exclusions stated once and relied on below. A *signalling* NaN is excluded, because every arithmetic operation quiets one, so `x × 1` returns a quiet NaN where the rewrite returns the signalling one it started with. NaN *payloads* are excluded too, because WebAssembly permits an implementation to return a canonical NaN instead of propagating an operand's bits. Admit either and every identity in Theorem 20.1 fails except negation, which is why the relation is defined to stand outside them: neither is reachable from a tensor program this compiler accepts.

> **Theorem 20.1 (Identities valid over floats).** **(classical)** For IEEE 754 binary arithmetic with default rounding:
> - `x × 1 ≡ x`
> - `x ÷ 1 ≡ x`
> - `x − 0 ≡ x`
> - `−(−x) ≡ x`

*Proof sketch.* Multiplication and division by exactly 1 are exact and preserve sign and payload, so they are the identity on every finite value, on both infinities, and on NaN. `x − 0`: for finite non-zero `x` the result is `x` exactly; for `x = +0` it is `+0`; for `x = −0` it is `−0 − +0 = −0` — subtraction of a positive zero preserves a negative zero, which is precisely the case that breaks the addition version. Negation flips one bit. ∎

> **Theorem 20.2 (Identities not valid over floats).** **(classical)** Each of the following fails on at least one input:
> - `x + 0 ≢ x`. **Counterexample:** `x = −0`. IEEE addition gives `(−0) + (+0) = +0`; the rewrite yields `−0`. The two differ under `1/x` and under `copysign`.
> - `x × 0 ≢ 0`. **Counterexamples:** `x = ∞` gives NaN, not 0. `x = NaN` gives NaN. `x = −0` gives `−0`, not `+0`.
> - `x − x ≢ 0`. **Counterexamples:** `x = ∞` gives NaN. `x = NaN` gives NaN.
> - `x ÷ x ≢ 1`. **Counterexamples:** `x = 0` gives NaN. `x = ∞` gives NaN. `x = NaN` gives NaN.

Note the shape of these. `x + 0` fails only on signed zero — a quiet, single-bit difference that surfaces only if something downstream divides by the result or inspects its sign. The other three fail loudly, turning a NaN into a number, which is worse: a NaN is a diagnostic, and erasing it converts a detectable failure into a plausible wrong answer.

> **Definition 20.3 (Fast-math licence).** **(stated here)** A *fast-math licence* is a user assertion about every value the program computes, intermediates included and not only its inputs and outputs: that none is infinite or NaN, that signed zero is never observed, and that every operation is applied inside its domain — no division by zero, no logarithm of a non-positive number. Under it, the compiler may apply identities that hold over the reals, and results may differ from IEEE 754 in bits, in NaN propagation, and in signed zero.

Each clause pays for a different rule, and the third is the one that is easy to leave out. Finiteness alone licenses `x − x → 0` and `x × 0 → 0`; the signed-zero clause licenses `x + 0 → x`; but `x ÷ x → 1` needs `x ≠ 0`, and zero is perfectly finite, while `exp(log x) → x` needs `x > 0`. A licence phrased only in terms of finiteness would gate those three rules behind an assertion that does not imply them.

The important word is *assertion*. The compiler does not verify finiteness — it cannot, in general — so the licence transfers responsibility rather than establishing a fact. That is exactly how `-ffast-math` works in C compilers, and it is why the flag is per-compilation and not per-operation.

Two things that are **not** licensed by any flag in this compiler, and are worth naming so they are not assumed:

- **Reassociation.** `(a + b) + c ≢ a + (b + c)` even over finite floats, because rounding happens at each step. Chapter 11's `ASSOCIATIVE` trait exists and is declared unconditionally on `add` and `mul`, so the trait alone cannot be trusted here; §20.8 is where the rewrite that consumes it gets its dtype test.
- **Distribution.** `a×b + a×c ≢ a×(b + c)`, same reason.

## 20.4 In mlfw: the gate is a constructor argument

[`simplify/algebraic.ts`](../../../src/compiler/passes/simplify/algebraic.ts) builds two pattern sets, once, at module load ([`algebraic.ts:9`](../../../src/compiler/passes/simplify/algebraic.ts)):

```ts
function buildAlgebraicPatterns(fastMath: boolean): PatternSet {
  const set = new PatternSet();
  set.add(new pat.AddZero(fastMath));
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

Three of the sixteen are only *present* under fast-math; three more are present always and *decide for themselves*. `SubSelf` ([`patterns.ts:201`](../../../src/compiler/ir/graph/patterns.ts)):

```ts
  override match(op: Operation): boolean {
    if (op.getOperand(0) !== op.getOperand(1)) return false;
    return isDtypeInt((op.getResult(0).type as TensorType).dtype) || this.fastMath;
  }
```

`x − x → 0` is applied unconditionally **for integers** — where it is simply true, there being no NaN and no signed zero in two's complement — and otherwise only under the licence. `MulZero` ([`patterns.ts:237`](../../../src/compiler/ir/graph/patterns.ts)) and `AddZero` are the same shape, the last of them only since §20.6. This is the right design: the licence is consulted per rule, and the dtype does half the work, so integer code gets the aggressive rewrites for free.

The flag reaches the pass from the compiler configuration, in one line of the pipeline builder ([`graph_pipeline.ts:43`](../../../src/compiler/pipeline/graph_pipeline.ts)):

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
x + 0    eager [NaN, Infinity, -Infinity, 0]  compiled [NaN, Infinity, -Infinity, 0]  
x - 0    eager [NaN, Infinity, -Infinity, -0] compiled [NaN, Infinity, -Infinity, -0] 
x * 1    eager [NaN, Infinity, -Infinity, -0] compiled [NaN, Infinity, -Infinity, -0] 
x * 0    eager [NaN, NaN, NaN, -0]            compiled [NaN, NaN, NaN, -0]            
x - x    eager [NaN, NaN, NaN, 0]             compiled [NaN, NaN, NaN, 0]             
x / x    eager [NaN, NaN, NaN, NaN]           compiled [NaN, NaN, NaN, NaN]           
```

Read the eager column first: it is Theorems 20.1 and 20.2 as measurements. `x − 0` preserves `−0`; `x + 0` turns it into `+0`; `∞ × 0` and `∞ − ∞` and `∞/∞` are all NaN. Nothing here is surprising if you know IEEE 754, and all of it is surprising if you only know algebra.

Now the compiled column, with fast-math **off**: every row agrees.

- **`x − 0` and `x × 1` agree** because Theorem 20.1's rules applied and the results are identical. This is the system working.
- **`x + 0`, `x × 0`, `x − x` and `x ÷ x` agree** because all four patterns are present and all four declined: the dtype is `f32` and the licence was not granted. The compiled kernel actually performs the addition, the multiplication, the subtraction and the division. This is the gate working.

Two of those four rows read `DIFFERENT` in the first draft of this chapter, and §20.6 is the account of why. Both are now fixed, and the rows above are what the fixes are worth: on fast-math-off, a compiled program and an eager one agree bit for bit on `−0`, `±∞` and NaN.

And with fast-math **on**:

```
x + 0    eager [NaN, Infinity, -Infinity, 0]  compiled [NaN, Infinity, -Infinity, -0] <-- DIFFERENT
x * 0    eager [NaN, NaN, NaN, -0]            compiled [0, 0, 0, 0]                   <-- DIFFERENT
x - x    eager [NaN, NaN, NaN, 0]             compiled [0, 0, 0, 0]                   <-- DIFFERENT
x / x    eager [NaN, NaN, NaN, NaN]           compiled [1, 1, 1, 1]                   <-- DIFFERENT
```

Four rows diverge — which is not a bug, it is the licence being exercised. The user asserted their values are finite; the compiler took them at their word and replaced four elementwise kernels with a constant or a copy. That is the whole trade, and the two columns are now a clean statement of it: the identities Theorem 20.2 forbids fire exactly when they are licensed, and never otherwise.

## 20.6 Lab 2 — Where the rewrite happened

```bash
node docs/part4/ch20-algebra-and-ieee754/labs/02-where-the-rewrite-happened.mjs
```

The two identities that used to break had different causes, and the lab separates them by following one program through three representations. Here is the fixed pipeline:

```
=== x + 0 ===
traced:      constant -> add -> return
after passes: %1 = constant() {tensor_type = tensor<f32>, value = 0} : tensor<f32> | %2 = add(%0, %1) : tensor<1x2xf32> | return(%2)
kernel:
   buf_3[i1_6] = (buf_1[i1_6] + 0);

=== x * 0 ===
traced:      constant -> mul -> return
after passes: %1 = constant() {tensor_type = tensor<f32>, value = 0} : tensor<f32> | %2 = mul(%0, %1) : tensor<1x2xf32> | return(%2)
kernel:
   buf_3[i1_6] = (buf_1[i1_6] * 0);
```

Both operations survive every layer, and the kernel performs both. Read that as the target state, then read what each row used to say.

**`x + 0` was a graph-level rewrite.** The graph after the passes read `return(%0)` — the add was gone, and the kernel was a copy. The pattern responsible is `AddZero` ([`patterns.ts:167`](../../../src/compiler/ir/graph/patterns.ts)), which took no `fastMath` argument and consulted no dtype, while `SubZero` immediately below it and `SubSelf` and `MulZero` below that all did. By Theorem 20.2 it needs one: `(−0) + (+0) = +0` and the rewrite yields `−0`. It now carries the same check its three neighbours already had:

```ts
export class AddZero extends Pattern {
  fastMath: boolean;
  constructor(fastMath = false) { super('add_zero', 5); this.rootOpName = 'add'; this.fastMath = fastMath; }
  override match(op: Operation): boolean {
    if (!isDtypeInt((op.getResult(0).type as TensorType).dtype) && !this.fastMath) return false;
    return isConstantVal(op.getOperand(1).definingOp, 0) || isConstantVal(op.getOperand(0).definingOp, 0);
  }
```

The default is the sound behaviour, integers only, which matters because `add`'s canonicalization list constructs the pattern with no arguments ([`ops/arithmetic.ts:29`](../../../src/compiler/ir/graph/ops/arithmetic.ts)) and canonicalize has no licence to pass it. The algebraic pass, which does, forwards its `fastMath` flag ([`simplify/algebraic.ts:11`](../../../src/compiler/passes/simplify/algebraic.ts)).

**`x × 0` was never a graph-level rewrite at all.** The graph after every pass still contains `mul(%0, %1)` — `MulZero` was present, checked the dtype, found `f32`, found no licence, and correctly declined. Lowering keeps the multiply, and so does scheduling; the TensorIR reads `buf_3[v0_7, v1_8] = (buf_1[0, v1_8] * buf_4[])`. The multiply survived every layer the flag is visible in, and then the CPU code generator flattened it in the last one ([`backend/cpu/codegen.ts:470`](../../../src/backend/cpu/codegen.ts)):

```ts
              else if (node.op === '*' && (a === '0' || b === '0')) { vals.push('0'); }
```

`a` and `b` here are *rendered strings*, and the branch fired on any operand that happened to render as `0`. But this peephole does not only render values: the same `_exprToJS` renders every index expression in every CPU kernel, and there `+ 0` and `× 0` are integer arithmetic, where Theorem 20.1 does hold. Deleting the branch outright is therefore the wrong fix — it leaves index expressions reading `buf_3[((0 * 2) + i1_6)]` and costs a dozen kernel-quality tests. What the branch was missing is the check `MulZero` already has, and the dtype is available after all: `inferDtype` ([`ir/lir/nodes.ts:190`](../../../src/compiler/ir/lir/nodes.ts)) walks a `MathOpNode` down to its leaves, which is how the WASM backend already picks its numeric prefix. The zero identities are now gated on it:

```ts
              const dtype = inferDtype(node);
              const foldsZero = isDtypeInt(dtype) && (a === '0' || b === '0');
              ...
              else if (foldsZero && node.op === '+') { vals.push((b === '0' ? a : b) as string); }
              else if (foldsZero && node.op === '*') { vals.push(this._zeroLit(dtype)); }
```

`x - 0`, `x * 1` and `1 * x` stayed ungated, because Theorem 20.1 makes them sound for floats too. And `_zeroLit` replaces the bare `'0'`, which was a second latent bug in the same branch: an `i64` kernel folding `x * 0` used to emit a `Number` zero into BigInt arithmetic.

The CUDA backend never had the peephole; its `MathOpNode` case emits `(${a} ${node.op} ${b})` and stops ([`backend/cuda/codegen.ts:492`](../../../src/backend/cuda/codegen.ts)). So the same program, compiled for two targets, produced `0` on one and `NaN` on the other for a non-finite input. That is a differential-testing finding in the sense of Chapter 65 — the two backends disagreed, and the test that catches it is one that feeds non-finite inputs to both. That test now exists, in the `SPECIAL` block of [`tests/e2e/differential.test.js`](../../../tests/e2e/differential.test.js), together with a comparison that can tell `−0` from `+0`. The old one could not, which is half the reason the row survived this long.

**Try this.** Run the same two programs against `WasmTarget()`. WebAssembly's code generator never had a zero peephole — predict which of the three representations differs from the CPU listing above, and confirm the two targets now agree on all four inputs.

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

- **An identity's soundness is a property of the dtype, and two of the six rules had lost track of it** ([`patterns.ts:167`](../../../src/compiler/ir/graph/patterns.ts), [`codegen.ts:470`](../../../src/backend/cpu/codegen.ts)). §20.6 is the account of both. Neither was a hard bug to fix once located, and neither was locatable without a test that feeds `−0` and `±∞`; the general trap is that a rule written for integers is one copy-paste away from a rule that fires on floats.
- **The zero identities in the CPU backend are gated on an inferred dtype, not a declared one.** `inferDtype` walks a `MathOpNode` to its leaves and falls back to `'f32'` for any node kind it does not recognize ([`ir/lir/nodes.ts:190`](../../../src/compiler/ir/lir/nodes.ts)). The fallback is the safe direction — an unrecognized node is treated as float and the identity declines — but it means a new integer-valued LIR node kind silently loses the index-arithmetic folding until it is added to the switch.
- **`fastMath` is one global flag.** There is no per-operation, per-function or per-region control, so a model with one numerically delicate layer either gives up fast-math everywhere or accepts it everywhere. This is the same limitation C compilers have, and the same workaround applies: compile the delicate part separately.
- **The licence is asserted and never checked.** Nothing verifies finiteness, and nothing warns when a fast-math rewrite fires on a graph whose inputs the compiler has actually seen. The optimization gate (Chapter 61) *does* verify candidate optimizations numerically before keeping them — that machinery exists, and fast-math does not use it.
- **Integer division is not simplified at all.** `DivSelf` is fast-math-only, so `x / x` for integers is left alone even though the only exception is `x = 0`, which is a trap on most hardware rather than a wrong value. Conservative, and inconsistent with `SubSelf`'s treatment of integers.
- **Reassociation is gated in the pattern, not in the trait.** `ASSOCIATIVE` is declared unconditionally on `add` and `mul` (Chapter 11 §11.3), and canonicalization registers a reassociation pattern for every operation that is commutative, associative and foldable ([`canonicalize.ts:19`](../../../src/compiler/passes/canonicalize/canonicalize.ts)). `AssociativeConstantReassoc` ([`patterns.ts:493`](../../../src/compiler/ir/graph/patterns.ts)) rewrites `(x ⊕ c₁) ⊕ c₂` into `x ⊕ (c₁ ⊕ c₂)`. Both *constants* are constant, which is the sense in which the pattern is "about constants" — but `x` is a runtime value, and re-bracketing a three-term float expression around it is a reassociation whatever the operands are called.

  So the pattern applies the same test `AddZero` does: integers unconditionally, floats only under a licence. Without it the rewrite fires at N1 and changes results:

  > **Counterexample 20.4.** With `fastMath: false`, eager execution of `(1 + 10¹⁶) + (−10¹⁶)` yields `0`, because `1` is lost to rounding when added to `10¹⁶`. Reassociated, the compiler folds `10¹⁶ + (−10¹⁶)` to `0` first and the answer is `1`. Note that the reassociated program is *more* accurate here, which is exactly why this class of rewrite survives review — but "differs from eager execution, in a direction we did not choose, on an input we did not anticipate" is the problem, and Definition 1.4's point is that the level is the user's to pick.

  The gate has one structural consequence worth knowing: `CanonicalizePass` has to be fast-math aware, so it caches one pattern set per setting rather than one globally. [`traits.test.js`](../../../tests/compiler/passes/canonicalize/traits.test.js) pins all three cases — fires on `i32`, refused on `f32`, fires again on `f32` under fast-math. The alternative, dropping `ASSOCIATIVE` from float-capable arithmetic, was rejected because the trait is *true* for the integer instantiations of those same operations.

## 20.9 Read the tests

- [`tests/compiler/passes/simplify/`](../../../tests/compiler/passes/simplify/) — the algebraic patterns, including which ones require `fastMath`.
- [`tests/compiler/ir/rewrite/pattern.test.js`](../../../tests/compiler/ir/rewrite/pattern.test.js) — the applicator behaviour these rules rely on, from Chapter 17.
- [`tests/e2e/differential.test.js`](../../../tests/e2e/differential.test.js) — the differential tests that compare compiled against eager. Most of them feed ordinary finite tensors; the `SPECIAL` block feeds `NaN`, `±∞` and `−0` through each identity in §20.5 on both CPU and WASM, and its comparison distinguishes `−0` from `+0`. That block is what holds §20.6 closed, and Chapter 65 argues for more of it.

---

**Next:** [Chapter 21 — Decomposition](../ch21-decomposition/README.md), which goes the other way: instead of making the graph smaller, it deliberately makes it much bigger, and then relies on the next four chapters to make it small again.
