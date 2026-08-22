# Chapter 33 — Buffers, blocks, iteration variables

Chapter 32 produced a loop nest with a store inside it, and the store was surrounded by something not yet explained:

```
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
```

The loops are the schedule. The `block` is the computation. Keeping them apart is the single design decision that makes Part VII possible, and this chapter is about what each of the three nouns in the title actually holds.

**First, a word that now means two things.** In the graph IR of Part II, a *block* is an ordered list of operations — the body of a region, with arguments, ending in a terminator (Chapter 9). A **TensorIR block** is not that. It is a unit of computation wrapped around a statement, carrying its own iteration variables and their bindings. The two share a name and nothing else: a graph block holds operations and has no iteration space; a TIR block holds one statement and is *made of* its iteration space. From here to the end of Part VI, "block" means the TIR sense unless the graph IR is named explicitly.

## 33.1 The problem: after you tile it, what was the program?

Suppose a scheduler splits `i1_7` into an outer loop of 8 and an inner loop of 8, then swaps the outer one with `i0_6`, then vectorises the inner one. The loops are now four, differently named, in a different order, two of them annotated. What in that nest still says *"this is a two-dimensional elementwise addition of `buf_1` and `buf_3` into `buf_5`"*?

If the answer is "the loops", then every scheduling primitive has to re-derive the computation from its transformed form, and every subsequent primitive has to re-derive it again. If the answer is "nothing", then the compiler has lost the ability to check whether the transformation was legal.

So the answer has to be a piece of IR that transformations move but do not rewrite. That is the block.

## 33.2 Intuition: the block is a declaration, the loops are a plan

Two ways to say the same thing about a nest.

**The loops say:** run these iterations, in this order, on these units of hardware.

**The block says:** this computation is defined over a `d`-dimensional index space; here is how each of its `d` axes is obtained from the surrounding loop variables; these are the buffers it reads and writes; and each axis is either *spatial* (iterations are independent) or *reducing* (iterations combine into one location).

A scheduler is then allowed to change the first without touching the second, provided it fixes up the bindings. Split a loop and the binding for that axis becomes `outer*8 + inner` instead of a bare variable. Reorder two loops and the bindings do not change at all. The block never notices.

This is TVM's TensorIR design, and the reason to name the lineage is that the pay-off is precisely stated: **the legality question moves from "is this loop transformation safe?" to "is this axis spatial?"**, and the second question has a stored answer.

## 33.3 Theory

> **Definition 33.1 (Buffer).** **(stated here)** A *buffer* is a name, a shape `(n₁,…,n_r)`, an element type, a memory scope, and a stride vector `(s₁,…,s_r)`. The element at logical index `(i₁,…,i_r)` lives at flat offset `Σ i_k s_k` from the buffer's base.

Strides are the whole of layout at this level. Chapter 25's NCHW-versus-NHWC choice arrives here as a permutation of `s`, and nothing downstream knows it was ever a choice.

> **Definition 33.2 (Block).** **(stated here)** A *block* is a name, a list of iteration variables `v₁,…,v_d` each with a *binding* — an expression over the enclosing loop variables — and a *kind*, together with a body and declared read and write sets. The block's iteration domain is the image of the enclosing loops' domain under the bindings.

> **Definition 33.3 (Iteration variable kind).** **(invariant)** Every axis of a block carries one of two tags: `DataPar`, the *spatial* axes, and `CommReduce`, the *reducing* ones. The tag is data on the node; §33.4 shows where it is set and Definition 33.5 says what setting it asserts.

The kinds are the contract, and what Part VII is allowed to do is defined against them:

> **Definition 33.4 (Block abstraction).** **(stated here)** A transformation *respects the block abstraction* if it preserves each block's body and read/write sets, changes only the bindings, and moves a loop variable only across axes whose kinds permit that movement.

Now the part that is easy to get wrong. The obvious reading of `DataPar` — "distinct values of this axis write distinct locations" — is the one to avoid. It rules out two iterations *writing* the same element and does **not** rule out one iteration reading what another wrote: a block writing `A[i]` and reading `A[i−1]` has distinct writes per iteration and a genuine loop-carried dependence. The tag has to claim something about the whole block, not about its writes.

> **Definition 33.5 (What `DataPar` claims).** **(stated here)** Declaring an axis `DataPar` asserts that any two iterations of the block differing only in that axis are *independent*: neither reads a location the other writes, and they do not write the same location. `CommReduce` asserts the weaker property that they write the same location under an operator that is associative and commutative **over the reals**, and read nothing else that either writes.

**"Over the reals" is the qualification the tag cannot make, and it is the one that matters.** `CommReduce` is what licenses a scheduler to reorder a reduction axis, split it across accumulators, or run it in parallel — and every one of those is an N2 transformation under Definition 1.4, because it changes the order in which the partial results are combined. Floating-point addition is commutative but not associative, so the tag on an `f32` sum is asserting something true of the mathematics and false of the arithmetic.

That is not a reason to remove the tag: without it no reduction could ever be parallelised, and every framework in this space makes the same trade. It is a reason to be exact about what the tag *buys*, and the honest statement is:

> `CommReduce` licenses transformations that preserve the **value over the reals** and may change the **floating-point result**. It is a declaration that the program's author accepts N2 equivalence on that axis.

Two consequences follow. A schedule that reorders a `CommReduce` axis is not "semantics-preserving" in the N0 or N1 sense that the rest of the scheduling vocabulary uses, and Chapter 41's `rfactor` is the primitive where that becomes a measurable difference — a serial `3` and a four-way-split `6` on the same eight values. And because the tag is per-axis rather than per-dtype, an integer reduction (where associativity genuinely holds, modulo overflow) and a float one carry the identical declaration; nothing downstream can distinguish the case where reordering is exact from the case where it is not.

That is a claim, not a derivation, and it is the point:

> **Proposition 33.6 (Kind-based legality).** **(stated here)** If a loop variable feeds only `DataPar` axes of every block beneath it, and every such declaration is true, then running its iterations in any order — including concurrently — computes the same result.

*Proof.* Immediate from Definition 33.5: two iterations of the loop differ only in axes it feeds, all of which are `DataPar`, so by the claim they are independent, and independent computations commute. ∎

The content is entirely in the hypothesis "and every such declaration is true". Nothing derives it and nothing checks it. What makes the design work in practice is a property of Chapter 34's rules rather than of anything in this chapter: in every block the lowering rules emit, a same-buffer read is at the *same* subscript as the write — an accumulator — and every other read is of a different buffer. So the declarations happen to be true, and Proposition 33.6 happens to apply.

The pay-off is speed: the expensive alternative is to solve for dependences (Chapter 36), and the block lets the scheduler skip that whenever the kinds are available. The compiler does exactly this, and the ordering is worth reading carefully ([`schedule/legality.ts:40`](../../../src/compiler/schedule/legality.ts)):

```ts
export function loopCarriedDependence(state: ScheduleState, loop: ForNode, allowedKinds: ReadonlySet<string>): string | null {
  const { info, deps } = state.nestAnalysis(loop);
  const dep = carriesDependence(deps, loop, (buffer) => isPrivateToLoop(info, buffer, loop));
  if (!dep) return null;
  if (blockAbstractionPermits(state, loop, [loop.loopVar.name], allowedKinds, info.byBlock)) return null;
  return `loop '${loop.loopVar.name}' carries a ${dep.kind} dependence on buffer '${dep.buffer.name}'`;
}
```

Read line 4 and line 5 together. A dependence was **found**, and the declaration **overrules it**. That is not a bug; it is what a contract is for — the analysis is conservative and a true declaration is exact. But it does mean the kinds are trusted, and §33.7 is about what checks them.

> **Corollary 33.7 (The declaration is load-bearing).** **(invariant)** A block whose axis kinds are wrong can be transformed into a program that computes something else, without any pass reporting an error, because the analysis that would have caught it is skipped precisely when the declaration is present.

## 33.4 In mlfw

### The buffer

[`ir/tensor/buffer.ts`](../../../src/compiler/ir/tensor/buffer.ts), 79 lines. The constructor computes row-major strides when none are given ([`buffer.ts:30`](../../../src/compiler/ir/tensor/buffer.ts)):

```ts
    if (strides) {
      this.strides = strides;
    } else {
      this.strides = new Array(shape.length);
      let s = 1;
      for (let i = shape.length - 1; i >= 0; i--) {
        this.strides[i] = s;
        if (s === DYNAMIC) continue;
        if (typeof shape[i] === 'number') s *= shape[i] as number;
        else s = DYNAMIC;
      }
    }
```

and the strides that *are* given come from the graph type's layout ([`lowering_registry.ts:124`](../../../src/compiler/passes/lowering/lowering_registry.ts)): `const strides = t.layout ? t.layout.computeStrides(shape) : null;`. One line, and Chapter 25 becomes arithmetic.

Beyond shape and strides the buffer carries: `scope`, from a four-member enumeration ([`tensor_types.ts:1`](../../../src/compiler/ir/tensor/tensor_types.ts)); `alignment`, default 64; `offset` and `poolByteOffset`, filled in by Chapter 50's allocator; `broadcastDims`, set by the alias path of §32.4; `symbolicShape`, for dynamic dimensions; and `storageAlign`, from the scheduling primitive of the same name.

### The block

`BlockNode` ([`nodes.ts:126`](../../../src/compiler/ir/tensor/nodes.ts)) is six fields: `name`, `iterVars`, `reads`, `writes`, `body`, `initBody`. The iteration variables are `BlockRealizeNode`s ([`nodes.ts:154`](../../../src/compiler/ir/tensor/nodes.ts)) — three fields, and the third is Definition 33.3:

```ts
export class BlockRealizeNode extends TensorNode {
  declare type: 'BlockRealizeNode';
  iterVar: VariableNode;
  binding: TirNode;
  kind: IterVarKindValue;
```

with

```ts
export const IterVarKind = Object.freeze({ DATA_PAR: 'DataPar', COMM_REDUCE: 'CommReduce' });
```

The default is `DATA_PAR`, so a rule that says nothing declares every axis parallel. Rules that produce a reduction opt in by calling `markCommReduce` ([`lowering_registry.ts:217`](../../../src/compiler/passes/lowering/lowering_registry.ts)):

```ts
export function markCommReduce(ivs: BlockRealizeNode[]): BlockRealizeNode[] {
  for (const iv of ivs) iv.kind = IterVarKind.COMM_REDUCE;
  return ivs;
}
```

Five call sites, in the rules that emit an accumulation: `reduce` ([`rules/reduction.ts:53`](../../../src/compiler/passes/lowering/rules/reduction.ts)), `argmax` and `argmin` ([`rules/reduction.ts:129`](../../../src/compiler/passes/lowering/rules/reduction.ts)), the contraction axes of `dot` ([`lowering_registry.ts:557`](../../../src/compiler/passes/lowering/lowering_registry.ts)), the input-channel and kernel axes of `conv` ([`lowering_registry.ts:311`](../../../src/compiler/passes/lowering/lowering_registry.ts)), and the window axes of `pool2d` ([`rules/pooling.ts:64`](../../../src/compiler/passes/lowering/rules/pooling.ts)). Everything else in the compiler is spatial by omission.

`initBody` is the second half of a reduction expressed inside one block: the statement that runs once per spatial point before the reduction axes begin. The lowering rules in this compiler do not use it — `reduce` emits an init block and an accumulation block as *siblings* — and Chapter 41's `decompose_reduction` is the primitive that converts between the two forms.

### The consumer of the kinds

`BlockAccessInfo.iterKindsOfLoopVar` ([`analysis/buffer_access.ts:80`](../../../src/compiler/analysis/buffer_access.ts)) is how a loop variable is turned into a set of kinds:

```ts
  iterKindsOfLoopVar(name: string): Set<string | undefined> | null {
    if (!this.affineBinding || !this.typedIterVars || this.directBodyVars.has(name)) return null;
    const kinds = new Set<string | undefined>();
    for (const binding of this.bindings) {
      if (binding.form && binding.form.terms.has(name)) kinds.add(binding.kind);
    }
    return kinds.size === 0 ? null : kinds;
  }
```

Three ways to answer `null`, meaning "the declaration does not apply, go and analyse": a binding that is not affine, an untyped iteration variable, or — the interesting one — a loop variable used **directly in the block body** rather than through a binding. That last guard is what keeps the abstraction honest: a body that reaches around the bindings to touch a loop variable is not described by its bindings, so its kinds say nothing.

### Scopes and allocation

`AllocateNode` ([`nodes.ts:225`](../../../src/compiler/ir/tensor/nodes.ts)) introduces a buffer whose lifetime is its body. It is how a rule asks for scratch space, and the argmax rule is the one place in the lowering rules that does ([`rules/reduction.ts:117`](../../../src/compiler/passes/lowering/rules/reduction.ts)):

```ts
      const bestValBuf = new Buffer('_argval_' + ctx.varCounter, [1], inBuf.dtype, 'local');
```

A one-element `local` buffer, allocated inside the spatial loop, holding the running best value while the output buffer holds the running best index. §33.6 shows what the CPU backend does with `local`.

### The verifier

[`ir/tensor/verifier.ts`](../../../src/compiler/ir/tensor/verifier.ts), 154 lines, run by `TirModule.verify` ([`module.ts:63`](../../../src/compiler/ir/tensor/module.ts)) and by the pass manager between TIR passes. It checks exactly four things:

| Check | Line |
|---|---|
| a loop variable is not already bound | [`verifier.ts:48`](../../../src/compiler/ir/tensor/verifier.ts) |
| a block iteration variable is not already bound | [`verifier.ts:58`](../../../src/compiler/ir/tensor/verifier.ts) |
| a store's and a load's subscript count equals the buffer's rank | [`verifier.ts:92`](../../../src/compiler/ir/tensor/verifier.ts), [`verifier.ts:116`](../../../src/compiler/ir/tensor/verifier.ts) |
| every variable used is in scope | [`verifier.ts:142`](../../../src/compiler/ir/tensor/verifier.ts) |

Scoping and rank. That is the TIR analogue of Chapter 12's four invariants, and it is a much shorter list.

**So sort a block's properties into two piles, because the chapter has been describing them in one voice and they are not the same kind of thing.**

| Property | Status |
|---|---|
| subscript count matches buffer rank | **checked** — verifier |
| every variable is in scope; no variable bound twice | **checked** — verifier |
| the iteration domain is the image of the loops under the bindings | **derived** — computed from the bindings, not declared |
| a binding is affine in the enclosing loop variables | **declared** — assumed by every analysis in Chapters 35–37, checked by none |
| the declared read set covers what the body reads | **declared** — §33.6 exhibits one that does not |
| the declared write set covers what the body writes | **declared** |
| an axis declared `DataPar` carries no dependence | **declared** — and Chapter 42's Counterexample 42.9 buys an illegal permutation with a false one |
| the region touched inside a buffer | **absent** — nothing constructs it (§33.7) |

The middle five rows are the ones to be careful with. They read like facts about the program — "this axis is a reduction axis", "this block reads these buffers" — and they are *assertions made at construction time by whichever lowering rule built the block*. Nothing revisits them. A scheduler consults `DataPar` to decide that a permutation is legal without running dependence analysis, which is the entire performance argument for declaring it (§33.3); the cost is that a wrong declaration is a miscompile with no diagnostic.

This is Chapter 12 §12.6's split showing up one level down, and for the same reason: the verifier can decide **structural** questions by inspecting the node in front of it, and cannot decide **semantic** ones without an analysis it deliberately does not run. Chapter 37 explains why in-range is undecidable in general, and Chapter 42 is where a false `DataPar` produces a wrong answer.

## 33.5 Lab — the anatomy of a block

```bash
node docs/part6/ch33-buffers-blocks-itervars/labs/01-anatomy-of-a-block.mjs
```

A reduction is the smallest program with two blocks over one buffer:

```
for si0_5 in 0..2 {
  block reduce_init_0 {
    bind siv0_6 = si0_5
    reads([buf_4[...]])
    writes([buf_3[...]])
    buf_3[siv0_6] = buf_4[]
  }
}
for sa0_7 in 0..2 {
  for r0_9 in 0..3 {
    block reduce_acc_1 {
      bind sav0_8 = sa0_7
      bind rv0_10 = r0_9
      reads([buf_1[...]])
      writes([buf_3[...]])
      buf_3[sav0_8] = (buf_3[sav0_8] + buf_1[sav0_8, rv0_10])
    }
  }
}
```

Two axes in `reduce_acc_1`, and the whole reduction is the statement that the first is `DataPar` and the second is `CommReduce`. Nothing in the printed text says so:

```
=== what a block header carries ===

  reduce_init_0
    iter vars : siv0_6=si0_5
    reads     : buf_4[...]
    writes    : buf_3[...]
  reduce_acc_1
    iter vars : sav0_8=sa0_7, rv0_10=r0_9
    reads     : buf_1[...]
    writes    : buf_3[...]
```

Two things the header does not carry, and both are on the node:

- **The region touched inside each buffer.** `BufferRegionLike` has optional `min` and `extent` fields ([`buffer.ts:67`](../../../src/compiler/ir/tensor/buffer.ts)), and a `BufferRegion` class with required ones ([`buffer.ts:69`](../../../src/compiler/ir/tensor/buffer.ts)). Nothing in the compiler constructs either, so every declared region is "this buffer, unspecified", printed `[...]`. Consumers that need the region compute it by walking the body (Chapter 36).
- **The kind of each axis** — the field Proposition 33.6 is entirely about.

The contraction shows the same structure one size up, and shows an empty read set: `matmul_init_0` declares `writes([buf_5[...]])` and no reads, because zeroing an output reads nothing.

## 33.6 Lab — buffers, scopes, and a declaration that is not true

```bash
node docs/part6/ch33-buffers-blocks-itervars/labs/02-buffers-and-scopes.mjs
```

The `local` scratch buffer, in TIR and in the emitted kernel:

```
    for ai0_5 in 0..2 {
      allocate _argval_4[1] (local) {
        block arg_init_0 {
          bind aiv0_6 = ai0_5
          writes([_argval_4[...], buf_3[...]])
          _argval_4[0] = -Infinity
          buf_3[aiv0_6] = 0
        }
        for ar_7 in 0..3 {
          block arg_acc_1 {
            bind aiv0_6 = ai0_5
            bind arv0_8 = ar_7
            reads([buf_1[...], _argval_4[...]])
            writes([_argval_4[...], buf_3[...]])
            buf_3[aiv0_6] = if ((buf_1[aiv0_6, arv0_8] > _argval_4[0])) {
              arv0_8
            } else {
              buf_3[aiv0_6]
            }
            _argval_4[0] = if ((buf_1[aiv0_6, arv0_8] > _argval_4[0])) {
              buf_1[aiv0_6, arv0_8]
            } else {
              _argval_4[0]
            }
          }
        }
      }
    }
```

The lab prints the emitted kernel next to it, with the loop lines kept so the nesting is readable:

```
  for (let ai0_5 = 0; ai0_5 < 2; ai0_5++) {
  const _argval_4 = new Float32Array(1);
  _argval_4[0] = -Infinity;
  for (let ar_7 = 0; ar_7 < 3; ar_7++) {
```

`local` on the CPU backend is a fresh typed array per outer iteration. On CUDA the same node becomes a thread-local array, and on WebGPU a `var<private>`. The scope is a portable statement of intent that each backend spends differently — which is why the enumeration has four members even though the lowering rules use exactly one of them beyond `global`, and `register` is never used at all.

How many buffers a real model needs:

```
=== how many buffers does a model need? ===

  fusion=false  buffers: 11   bound to a parameter: 6   internal: 5
  fusion=true   buffers: 10   bound to a parameter: 6   internal: 4
```

Six of them are the function's own parameters — two inputs, two weights, two biases, one output, minus the sharing — and the rest are intermediates that Chapter 50 will have to place. Fusion removes one, which is the same intermediate the fused loop no longer materialises.

And the part of the block header that is decoration:

```
=== the declared read set is not the read set ===

  reduce_init_0   declared reads: {buf_4}  actually read: {buf_4}
  reduce_acc_1    declared reads: {buf_1}  actually read: {buf_3, buf_1}  MISSING: buf_3
```

Every accumulation block in this compiler reads its own output and does not declare it. `bufRefs(inputs)` ([`lowering_registry.ts:421`](../../../src/compiler/passes/lowering/lowering_registry.ts)) builds the read set from the *operation's operands*, and the accumulator is not an operand — it is the result. Nothing breaks today, because everything that must be right walks the body instead: `collectBufferAccesses` (Chapter 36) ignores the declaration entirely, and `buffer_liveness.ts:153` touches the declared sets *and then* walks the body. What consumes the declaration alone is the autotuner's workload key and block DAG (Chapters 45 and 47), where a missing buffer changes a cache key rather than a result.

**Try this.** Replace `x.sum(1)` with `x.matmul(y)` in the lab's last section and read the same column for `matmul_1`: it declares its two inputs and omits the accumulator, for the same reason.

## 33.7 Traps and limits

- **Nothing checks the axis kinds.** Corollary 33.7 is a live exposure, not a hypothetical: `loopCarriedDependence` skips a real dependence when the kinds permit, and the verifier of §33.4 does not look at kinds at all. The protection is that the five call sites of `markCommReduce` are the five rules that build accumulations, and every other rule builds a genuinely elementwise store. A new rule that accumulates and forgets the call would produce a block whose reduction axis is declared parallel, and the first scheduler to parallelise it would be silently wrong.
- **The declared read set is incomplete by construction.** §33.6. It also duplicates: an operation using one operand twice declares that buffer twice, because the set is built per operand rather than per buffer.
- **The declared regions do not exist.** `BufferRegion` ([`buffer.ts:69`](../../../src/compiler/ir/tensor/buffer.ts)) is exported and never constructed anywhere in `src/` or `tests/`. The optional `min` and `extent` on `BufferRegionLike` are never set. Region-level reasoning is therefore always recomputed from the body — correct, and O(body) every time it is asked.
- **`initBody` is unused by every lowering rule.** The field exists ([`nodes.ts:126`](../../../src/compiler/ir/tensor/nodes.ts)), the verifier visits it, the printer prints it, and no rule sets it. Reductions are emitted as two sibling blocks instead, which is the form `decompose_reduction` produces rather than the form it consumes.
- **`MemoryScope` is a constant object that the rules do not use.** `MemoryScope.GLOBAL` has four call sites; `SHARED`, `LOCAL` and `REGISTER` have none — the rules and sketches that want those scopes write `'shared'` and `'local'` as string literals, and `'register'` appears nowhere. The type is `string`, so nothing catches a typo.
- **A block's name is a hint, not an identity.** `ctx.blockName('add_block')` appends a counter ([`lowering_registry.ts:110`](../../../src/compiler/passes/lowering/lowering_registry.ts)), and that name is what the scheduler's `explain` output, the autotuner's per-block cache and Part VIII's tuning database all key on. Two structurally identical functions produce the same names only because the counter is deterministic and the traversal order is fixed.
- **Buffer identity is the only aliasing information.** Two buffers either are the same object or are assumed disjoint. There is no partial-overlap representation, which is what makes §32.4's return-value copy necessary and what Chapter 51's in-place analysis has to work around.

## 33.8 Read the tests

- [`tests/compiler/ir/tensor/buffer.test.js`](../../../tests/compiler/ir/tensor/buffer.test.js) — stride computation, including the dynamic-dimension case where the running product becomes `DYNAMIC` and stays there.
- [`tests/compiler/ir/tensor/tir-module.test.js`](../../../tests/compiler/ir/tensor/tir-module.test.js) — the four verifier checks, one failing function each.
- [`tests/compiler/passes/lowering/blockname.test.js`](../../../tests/compiler/passes/lowering/blockname.test.js) — block naming, which is load-bearing for everything in Part VIII that caches per block.
- [`tests/compiler/analysis/tir-queries.test.js`](../../../tests/compiler/analysis/tir-queries.test.js) — the queries that walk a `PrimFunc` for buffers and loops rather than trusting the declarations.

---

**Next:** [Chapter 34 — Lowering rules](../ch34-lowering-rules/README.md), which is the other half of Chapter 32's driver: the 68 operations lowering can handle, how a rule is selected, and what happens to the 28 that have none.
