# Chapter 35 — Index arithmetic

Every rule in Chapter 34 built a subscript. `add` built `[i, j]`. `transpose` built `[j, i]`. `slice` built `[1 + i, 0 + j]`. `reshape` built this:

```
b0[((i1 + (i0 * 6)) // 3), ((i1 + (i0 * 6)) % 3)]
```

That is `[4,3] -> [2,6]`, printed by §35.5's lab with the buffers and loop variables renumbered.

Those expressions are now the program. The data movement is fixed — a copy is a copy — and everything the compiler can still do is arithmetic on coordinates: proving two accesses cannot collide (Chapter 36), proving a guard is unnecessary (Chapter 37), proving a loop can be split (Chapter 40).

All of that rests on one representation and three theorems — a flattening is a bijection, a division sometimes splits exactly, and a subscript sometimes covers its range exactly once — and this chapter is those.

## 35.1 The problem: an index expression is a program the compiler has to understand

Take the reshape above and ask three questions about it.

*Is it in bounds?* The subscripts are a quotient and a remainder; the remainder is obviously below 3, the quotient obviously is not obviously anything.

*Do two different iterations ever touch the same element?* You cannot answer that by looking at the syntax. You have to solve `i₁ + 6i₀ = i₁' + 6i₀'` over the loop bounds, and conclude something about the quotient and remainder from the answer.

*Can it be cheaper?* The expression contains a division and a modulo, and integer division by a **runtime-variable** divisor is among the slowest integer instructions on the processors this book targets — typically tens of cycles against one for a multiply, and often not pipelined, so a dependent chain of them serializes.

That claim needs its qualifiers, because the unqualified version ("integer division is the slowest operation on every target") is false in ways that matter to this chapter:

- **A constant divisor is not a division.** Every compiler in the stack below this one — the JavaScript engine, the WebAssembly runtime, `nvcc` — strength-reduces division by a compile-time constant into a multiply-and-shift. That is exactly the case a lowered index expression is usually in, since extents are constants after specialization. So the instruction the chapter is trying to remove is often already gone.
- **The cost is architecture-specific and moving.** Division latency differs by an order of magnitude across the targets here, and recent CPUs have narrowed the gap substantially.
- **"Slowest" is the wrong axis anyway.** Chapter 4 established these kernels are memory-bound. An index division inside a loop whose body waits on a cache miss costs nothing observable; the same division in a register-resident inner loop costs everything. Which regime you are in decides whether §35.5's transformation is worth anything.

So the honest motivation for what follows is narrower and still sufficient: **a division whose divisor the compiler cannot see is expensive enough to be worth eliminating when the surrounding loop is not memory-bound**, and §35.6 measures one case where it is worth 1.10× on a pure copy — a workload chosen precisely because nothing else is competing for the time.

A compiler that treats index expressions as opaque trees can answer none of these. One that puts them in a normal form can answer all three with linear algebra — as long as the expressions stay inside the class the normal form covers. Choosing that class is the design decision, and it is the same one every loop compiler makes: **affine**.

## 35.2 Intuition: coordinates are numerals

A row-major tensor of shape `(4, 3)` stores element `(i, j)` at offset `3i + j`. Read that as a two-digit number where the low digit has base 3 and the high digit has base 4. Element `(2, 1)` is offset 7, and 7 in "base (4,3)" is the digit pair `2,1`.

Everything in this chapter follows from taking that seriously.

- **Flattening a coordinate is writing a numeral.** Multiply each digit by the product of the bases below it, and add.
- **Unflattening an offset is reading a numeral.** Repeatedly take the remainder modulo the lowest base and divide it out.
- **A reshape is a change of base.** The offsets are unchanged; the digits are regrouped.
- **A division by `c` throws away the digits below `c`, and a modulo keeps exactly those digits** — provided `c` falls on a digit boundary.

That last clause is the whole of §35.3's second theorem, and the whole of why some reshapes compile to a bare index and some compile to a division.

## 35.3 Theory

> **Definition 35.1 (Affine form).** **(classical)** An *affine form* over variables `x₁,…,x_n` is an expression `Σ aₖxₖ + b` with all `aₖ` and `b` integers. Affine forms are closed under addition, negation, and multiplication by an integer constant, and under nothing else.

The class is chosen because it is exactly the class closed under the operations loop transformations perform. Splitting a loop replaces `i` by `8·i_o + i_i`, which is affine in the new variables. Fusing two loops replaces `i, j` by `k // n, k % n`, which is *not* — and that asymmetry is why fusion of loops is the primitive that most often defeats later analysis.

### Flattening

> **Definition 35.2 (Row-major flattening).** **(classical)** For a shape `(n₁,…,n_r)` with all `nₖ > 0`, define strides `sₖ = ∏_{j>k} n_j` and `φ(i₁,…,i_r) = Σ iₖ sₖ`.

> **Theorem 35.3 (Flattening is a bijection).** **(classical)** `φ` is a bijection from `∏ₖ [0, nₖ)` onto `[0, N)` where `N = ∏ₖ nₖ`, and its inverse is the mixed-radix decomposition

> ```
> i_r = f mod n_r,   f₁ = ⌊f / n_r⌋
> i_{r-1} = f₁ mod n_{r-1},   f₂ = ⌊f₁ / n_{r-1}⌋,   …
> ```

*Proof.* Injectivity and surjectivity together follow from existence and uniqueness of the mixed-radix representation, which is the division algorithm applied `r−1` times. For existence, run the displayed recurrence: each step produces a digit in `[0, nₖ)` by definition of the remainder, and `f₁ < ∏_{j<r} n_j` because `f < N`, so the recursion stays in range. For uniqueness, suppose `φ(i) = φ(i')`. Reducing mod `n_r` kills every term except `i_r`, since every other stride is a multiple of `n_r`; so `i_r = i'_r`. Subtract, divide by `n_r`, and induct. ∎

Theorem 35.3 is the reason `reshape` has a lowering rule at all. A reshape does not move data; it re-reads the same offsets under different bases. So the rule flattens the output coordinate with `φ_out` and unflattens with `φ_in⁻¹`, and the displayed recurrence is the code.

### When the division disappears

> **Definition 35.4 (Divisor split).** **(stated here)** For an affine form `f = Σ aₖxₖ + b` and an integer `c > 0`, the *divisor split* of `f` by `c` is the pair `(f_D, f_R)` where `f_D` collects the terms whose coefficient is a multiple of `c` and `f_R` collects the rest, together with the offset.

> **Theorem 35.5 (Exact split).** **(stated here)** Let `(f_D, f_R)` be the divisor split of `f` by `c`, and let `//` and `%` be floor division and floor modulo. If `0 ≤ f_R ≤ c−1` at every point of the iteration domain, then on that domain
> ```
> f // c = f_D / c        and        f % c = f_R
> ```
> and both right-hand sides are affine.

*Proof.* Every coefficient in `f_D` is a multiple of `c`, so `f_D / c` is an affine form with integer coefficients and `f_D = c·(f_D/c)`. Then `f = c·(f_D/c) + f_R` with `0 ≤ f_R < c`. Floor division is exactly the Euclidean division that produces a remainder in `[0, c)`, and that decomposition is unique, so `f_D/c` is the quotient and `f_R` the remainder. ∎

The hypothesis is one range fact, supplied by the interval analysis of Chapter 37 — which is why this theorem lives in an arithmetic simplifier that carries an `Analyzer` with it. Note that the sign of `f` itself does not enter: the theorem holds for a negative `f` because floor division puts the remainder in `[0, c)` whatever the dividend does. §35.4 shows the implementation testing for it anyway.

There is a degenerate case worth naming separately because it fires constantly:

> **Corollary 35.6 (The degenerate split).** **(stated here)** If `0 ≤ f ≤ c−1` on the domain, then `f // c = 0` and `f % c = f`.

That is the split with `f_D` empty, and it is what makes a reshape of a size-1 leading axis compile to a bare copy.

### Recognising a flattening

> **Definition 35.7 (Mixed-radix form).** **(stated here)** An affine form `Σ aₖxₖ + b` over variables with ranges `[mₖ, mₖ + eₖ)` is in *mixed-radix form* if, after sorting the terms by coefficient, `a₍₁₎ = 1` and `a₍ᵢ₊₁₎ = a₍ᵢ₎·e₍ᵢ₎` for every `i`.

> **Theorem 35.8 (Mixed-radix forms are exact covers).** **(stated here)** If `f` is in mixed-radix form with total extent `E = ∏ eₖ`, then as the variables range over their domain `f` takes every value in `[b', b' + E)` exactly once, where `b' = b + Σ aₖmₖ`.

*Proof.* This is Theorem 35.3 read backwards: the sorted coefficients are exactly the strides of a shape whose dimensions are the extents, so `f` is `φ` for that shape, shifted by `b'`. ∎

Theorem 35.8 is what lets the compiler answer *"which region of this buffer does this access touch?"* with an exact interval rather than a conservative one, and it is what lets the dependence test of Chapter 36 conclude "these two accesses coincide only when every index agrees" for a multi-variable subscript.

## 35.4 In mlfw: 153 lines that everything else is built on

[`analysis/iter_map.ts`](../../../src/compiler/analysis/iter_map.ts) is the whole representation.

### The form

```ts
export class LinearForm {
  offset: number;
  terms: Map<string, number>;
```

A map from variable name to coefficient, plus a constant. `add`, `negate` and `scale` are the three closure operations of Definition 35.1, and `add` deletes a term whose coefficient reaches zero ([`iter_map.ts:29`](../../../src/compiler/analysis/iter_map.ts)) so that cancellation is visible in the representation rather than hidden in it.

`toLinearForm` ([`iter_map.ts:57`](../../../src/compiler/analysis/iter_map.ts)) is the parser, and its shape is the definition of what the compiler considers analysable:

```ts
      switch (math.op) {
        case '+': return a.add(b);
        case '-': return a.add(b.negate());
        case '*':
          if (a.isConstant) return b.scale(a.offset);
          if (b.isConstant) return a.scale(b.offset);
          return null;
        default: return null;
      }
```

Four cases and two `null`s. A product of two non-constants is not affine; a division, a modulo, a cast, a comparison, and a buffer load are all `default`. Every "unknown" answer in Chapters 36 and 37 traces back to one of those two `null`s.

### Substitution

`composeForm` ([`iter_map.ts:86`](../../../src/compiler/analysis/iter_map.ts)) substitutes a form for each variable:

```ts
export function composeForm(form: LinearForm | null | undefined, varForms: ReadonlyMap<string, LinearForm>): LinearForm | null {
  if (!form) return null;
  let result = LinearForm.constant(form.offset);
  for (const [name, coeff] of form.terms) {
    const bound = varForms.get(name);
    if (!bound) return null;
    result = result.add(bound.scale(coeff));
  }
  return result;
}
```

This is the *iteration map* of Chapter 33 made concrete. A subscript is written in terms of block iteration variables; each of those is bound to a form over loop variables; composing gives the subscript in terms of the loops that actually execute. `collectBufferAccesses` calls it once per subscript ([`buffer_access.ts:152`](../../../src/compiler/analysis/buffer_access.ts)) and stores both versions: the raw form for computing the region, the composed one for dependence testing.

Note that composition is where a split loop is handled. After `split`, the binding of an axis is `i_o·8 + i_i`, and every subscript containing that axis becomes affine in the two new variables without anything else in the compiler knowing a split happened.

### The two theorems as code

Theorem 35.5's hypothesis test is `splitByDivisor` ([`iter_map.ts:97`](../../../src/compiler/analysis/iter_map.ts)):

```ts
  for (const [name, coeff] of form.terms) {
    if (coeff % divisor === 0) divisible.set(name, coeff);
    else remainder.set(name, coeff);
  }
```

and its conclusion is `affineDivMod` ([`ir_arith.ts:146`](../../../src/compiler/analysis/ir_arith.ts)):

```ts
  const quotient = () => formToNode(parts.divisible.scale(1 / divisor));

  if (parts.remainder.isConstant && parts.remainder.offset === 0) {
    return { quotient: quotient(), remainder: new IntImmNode(0) };
  }

  const remainderNode = formToNode(parts.remainder);
  if (!boundWithin(analyzer, remainderNode, 0, divisor - 1)) return null;
  if (!boundWithin(analyzer, node, 0, Infinity)) return null;

  return { quotient: quotient(), remainder: remainderNode };
```

Two `boundWithin` calls. The first is Theorem 35.5's hypothesis, `0 ≤ f_R ≤ c−1`. The second, `f ≥ 0`, is not — and under the floor semantics this compiler guarantees it is not needed, because the theorem's proof never uses the sign of `f`. It is a conservatism: `−4i + j` with `i ∈ [0,2]`, `j ∈ [0,3]` and `c = 4` splits exactly to `f // 4 = −i` and `f % 4 = j`, and `affineDivMod` refuses it because `f` ranges over `[−8,3]`. No lowering rule in this compiler emits a subscript of that shape today, so the refusal currently costs nothing measurable — but it is an extra condition, not a second hypothesis.

Theorem 35.8 is `mixedRadixDecomposition` ([`iter_map.ts:122`](../../../src/compiler/analysis/iter_map.ts)):

```ts
  factors.sort((x, y) => x.coeff - y.coeff);
  let stride = 1;
  for (const factor of factors) {
    if (factor.coeff !== stride) return null;
    stride *= factor.extent;
  }

  return { offset, extent: stride, factors };
```

Sort by coefficient, walk, and require each coefficient to equal the running product. Definition 35.7 in five lines. Its two consumers are `coverRangeOfForm`, which turns it into the `[offset, extent)` region recorded on every access, and the dependence test's exact-coincidence case ([`dependence.ts:131`](../../../src/compiler/analysis/dependence.ts)).

### And where the subscripts come from

The reshape rule ([`rules/shape.ts:59`](../../../src/compiler/passes/lowering/rules/shape.ts)) is Theorem 35.3 written out twice — once forwards, once backwards:

```ts
      let flatIndex: TirNode = outIndices[outBuf.shape.length - 1];
      let stride = 1;
      for (let i = outBuf.shape.length - 2; i >= 0; i--) {
        stride *= outBuf.shape[i + 1] as number;
        flatIndex = mathOp('+', flatIndex, mathOp('*', outIndices[i], new IntImmNode(stride)));
      }
      inIndices = new Array(inBuf.shape.length);
      let cur: TirNode = flatIndex;
      for (let i = inBuf.shape.length - 1; i >= 0; i--) {
        if (i === 0) { inIndices[i] = cur; }
        else {
          inIndices[i] = mathOp('%', cur, new IntImmNode(inBuf.shape[i] as number));
          cur = mathOp('//', cur, new IntImmNode(inBuf.shape[i] as number));
        }
      }
```

The first loop is `φ_out`. The second is the displayed recurrence of Theorem 35.3, and the `i === 0` case is the observation that the last quotient needs no further division because `f < n₁·s₁` already.

`mathOp` ([`nodes.ts:391`](../../../src/compiler/ir/tensor/nodes.ts)) folds the obvious identities as it builds — `x + 0`, `x * 1`, `x % 1`, and two integer literals — which is why the emitted subscript for a rank-1 output has no `+ 0` in it.

## 35.5 Lab — flatten and unflatten

```bash
node docs/part6/ch35-index-arithmetic/labs/01-flatten-and-unflatten.mjs
```

Six reshapes. The `TIR` line is what the rule emitted; the `JS` line is what reached the backend after Chapter 37's simplifier ran.

```
  [1,6] -> [6]
    TIR : b0[(i0 // 6), (i0 % 6)]
    JS  : buf_1[i0_4]     <- no division left
  [4,3] -> [2,2,3]
    TIR : b0[(((i2 + (i1 * 3)) + (i0 * 6)) // 3), (((i2 + (i1 * 3)) + (i0 * 6)) % 3)]
    JS  : buf_1[(((i1_5 + (i0_4 * 2)) * 3) + i2_6)]     <- no division left
  [2,2,3] -> [4,3]
    TIR : b0[(((i1 + (i0 * 3)) // 3) // 2), (((i1 + (i0 * 3)) // 3) % 2), ((i1 + (i0 * 3)) % 3)]
    JS  : buf_1[(((((i0_4 / 2) | 0) * 6) + ((i0_4 % 2) * 3)) + i1_5)]
  [4,3] -> [12]
    TIR : b0[(i0 // 3), (i0 % 3)]
    JS  : buf_1[((((i0_4 / 3) | 0) * 3) + (i0_4 % 3))]
  [4,3] -> [2,6]
    TIR : b0[((i1 + (i0 * 6)) // 3), ((i1 + (i0 * 6)) % 3)]
    JS  : buf_1[(((((i1_5 + (i0_4 * 6)) / 3) | 0) * 3) + ((i1_5 + (i0_4 * 6)) % 3))]
  [4,3] -> [3,4]
    TIR : b0[((i1 + (i0 * 4)) // 3), ((i1 + (i0 * 4)) % 3)]
    JS  : buf_1[(((((i1_5 + (i0_4 * 4)) / 3) | 0) * 3) + ((i1_5 + (i0_4 * 4)) % 3))]
```

Take them in order against the theorems.

**Row 1 is Corollary 35.6.** `i0` ranges over `[0,5]` and the divisor is 6, so `f // 6 = 0` and `f % 6 = f`. Both operations vanish; the reshape is a copy, and the backend flattens `buf_1[0, i0]` to `buf_1[i0]`.

**Row 2 is Theorem 35.5 firing.** The flat index is `i2 + 3i1 + 6i0` and the divisor is 3. Split it: `6i0` and `3i1` are multiples of 3, `i2` is not. So `f_D = 3i1 + 6i0` and `f_R = i2`, and `i2` ranges over `[0,2] ⊆ [0,2]`. Hypothesis met, so `f // 3 = i1 + 2i0` and `f % 3 = i2`, exactly — and the emitted subscript `(i1 + 2i0)*3 + i2` contains no division.

**Row 3 is the same theorem applied to the inner pair only.** `(i1 + 3i0) // 3` splits cleanly to `i0` and `(i1 + 3i0) % 3` to `i1`; the outer `// 2` and `% 2` on `i0` have `f_D` empty and `f_R = i0` with range `[0,3] ⊄ [0,1]`, so they stay.

**Rows 4, 5 and 6 are the hypothesis failing.** In row 5 the divisor is 3, `f_R = i1` with `i1 ∈ [0,5]`, and `[0,5] ⊄ [0,2]`. No exact split exists, and a division and a modulo per element reach the kernel.

The second half of the lab is the contrast case:

```
  transpose(1,0)           b1[i0, i1] = b0[i1, i0]
  slice rows 1..3          b1[i0, i1] = b0[(1 + i0), (0 + i1)]
  x + x                    b1[i0, i1] = (b0[i0, i1] + b0[i0, i1])
```

A permutation is a relabelling; a slice is one addition per axis. Neither leaves the affine class or needs a division, because neither crosses a stride boundary. Only reshape does, and only reshape pays.

## 35.6 Lab — what the index costs

```bash
node docs/part6/ch35-index-arithmetic/labs/02-what-the-index-costs.mjs
```

Two reshapes of 98,304 elements each. One satisfies Theorem 35.5, one does not.

```
  A  [8192,12] -> [8192,4,3]   98304 elements
     buf_1[((i0_4 * 12) + (i2_6 + (i1_5 * 3)))];
  B  [32768,3] -> [16384,6]    98304 elements
     buf_1[(((((i1_5 + (i0_4 * 6)) / 3) | 0) * 3) + ((i1_5 + (i0_4 * 6)) % 3))];

  arithmetic per element
  case    div  mod  mul  add
  A         0    0    2    2
  B         1    1    3    3

  median of 9 runs of 20 calls (machine-specific):
    A  0.271 ms
    B  0.296 ms      1.10x
```

Ten per cent, for a kernel that is otherwise a pure memory copy — which is the right order of magnitude, and the right lesson. Index arithmetic is not usually the bottleneck; it becomes one when the loop body is small and the trip count is large, which describes every layout transform, every reshape, and every gather in a model.

> **Provenance, since a bare ratio does not travel.** Node 24.9 on the CPU backend, 2026-08-21, 98,304 `f32` elements, median of 9 rounds of 20 calls. Two things about this number are worth separating. The **arithmetic-per-element table above it is exact and portable** — it is counted from the generated source, not measured, and it will be the same on your machine. The **1.10× is neither**: it depends on how your engine strength-reduces `/ 3` (§35.1), on whether the copy is bandwidth-bound at this size on your cache hierarchy, and on the JIT's mood. What should reproduce is the *sign* — case B is slower than case A — and the order of magnitude. If you measure 1.02× or 1.25×, nothing is wrong; if you measure A slower than B, something is.

The last block of the lab is the honest part:

```
  B computes  (f / 3 | 0) * 3 + f % 3  where f = i1 + i0 * 6
```

Truncating division appears in that expression because the simplifier **proved `f ≥ 0`** (Chapter 37). For a non-negative `f`, `(f tdiv c)·c + (f tmod c)` is exactly `f`. So the compiler holds a proof that the entire subscript equals the flat index it started from, and emits the division anyway: the identity is not in the rewrite set. The proof that licenses the cheap division is the same proof that would license deleting it.

**Try this.** Add `[8192,12] -> [8192,2,2,3]` to the first lab's table and predict, before you run it, whether the division survives. The input's inner extent is 12, so the divisor is 12; the flat index is `i3 + 3i2 + 6i1 + 12i0`; the remainder part is `i3 + 3i2 + 6i1`, ranging over `[0,11]`. Then try `[2,6] -> [4,3]`, where the divisor is 6 and the remainder part is `i1 + 3i0` over `[0,11]`, and see the other answer.

## 35.7 Traps and limits

- **`(f // c) * c + (f % c)` is not simplified.** §35.6. `RewriteSimplify._simplifyMathOp` handles `//` and `%` and never looks at a sum of a scaled quotient and a remainder ([`ir_arith.ts:210`](../../../src/compiler/analysis/ir_arith.ts)). Every reshape that fails Theorem 35.5 pays a division, a modulo, a multiply and an add per element for an expression the compiler could prove is the identity. This is the largest single finding of Part VI.
- **A reshape is lowered as a copy even when it is a relabelling.** Independently of the arithmetic: `reshape` always emits a nest and a store. When input and output shapes are identical the rule takes a fast path ([`rules/shape.ts:51`](../../../src/compiler/passes/lowering/rules/shape.ts)) and still emits the copy; only fusion can remove it, and only when a consumer absorbs it.
- **`mixedRadixDecomposition` requires a *perfect* radix chain.** A gap in the coefficients — `x + 4y` where `x` has extent 3 — returns `null`, even though the form is still injective. The test is `factor.coeff !== stride`, an equality, so it recognises exact covers and nothing weaker. That is deliberate (it is what makes Theorem 35.8's conclusion "exactly once" rather than "at most once") and it means a strided access is never given an exact region.
- **Coefficients must be positive.** `mixedRadixDecomposition` rejects `coeff <= 0` outright ([`iter_map.ts:131`](../../../src/compiler/analysis/iter_map.ts)), so a reversed axis — `(n−1) − i`, which `reverse` emits — never gets an exact cover, and the accesses of a `reverse` are treated as unknown regions.
- **`affineDivMod` tests one condition more than Theorem 35.5 needs.** §35.4. The extra `f ≥ 0` would be required if `//` truncated; it does not, so the test refuses exact splits whose divisible part is negative. Nothing in this compiler produces such a subscript today, which is why it has never cost anything — and it is worth knowing which of the two tests is the theorem and which is the belt.
- **`splitByDivisor` ignores the offset's divisibility.** The constant is always placed in the remainder part ([`iter_map.ts:106`](../../../src/compiler/analysis/iter_map.ts)), so `(3i + 6) // 3` does not split to `i + 2`; it splits to `f_D = 3i`, `f_R = 6`, and then fails the range test when `c ≤ 6`. A constant that is a multiple of the divisor could be moved to the divisible side and is not.
- **Non-integer literals fall out of the class immediately.** `toLinearForm` requires `Number.isInteger` on an `IntImmNode` ([`iter_map.ts:61`](../../../src/compiler/analysis/iter_map.ts)) — a defensive check, since an `IntImmNode` holding a non-integer would already be a bug elsewhere.
- **The form is keyed by variable *name*, not by node identity.** `LinearForm.terms` is a `Map<string, number>`. Two distinct `VariableNode`s with the same name are the same variable to this analysis, which is exactly why the verifier's "already bound" check ([`verifier.ts:48`](../../../src/compiler/ir/tensor/verifier.ts)) is load-bearing rather than stylistic: shadowing would silently merge two variables.
- **`linearFormToNode` emits terms in `Map` iteration order.** That is insertion order, which is deterministic but arbitrary, so the reconstructed expression can differ syntactically from the one that was parsed. It matters for CSE on index expressions (Chapter 19 does not run at this level) and for reading the output.

## 35.8 Read the tests

- [`tests/compiler/analysis/iter-map.test.js`](../../../tests/compiler/analysis/iter-map.test.js) — `LinearForm` arithmetic, `toLinearForm`'s rejection cases, `splitByDivisor`, and the mixed-radix recogniser including the near-miss forms it must reject.
- [`tests/compiler/analysis/ir-arith.test.js`](../../../tests/compiler/analysis/ir-arith.test.js) — `affineDivMod` and the two range hypotheses, plus the `tdiv`/`tmod` substitution.
- [`tests/compiler/ir/tensor/mathop-simplify.test.js`](../../../tests/compiler/ir/tensor/mathop-simplify.test.js) — the identities `mathOp` folds while the IR is being built.
- [`tests/compiler/passes/lowering/shape.test.js`](../../../tests/compiler/passes/lowering/shape.test.js) — the reshape recurrence, and the rank-1 special case.

---

**Next:** [Chapter 36 — Dependence analysis](../ch36-dependence-analysis/README.md). Given two subscripts in this form, decide whether they can ever name the same element — which is the question every loop transformation in Part VII has to ask first.
