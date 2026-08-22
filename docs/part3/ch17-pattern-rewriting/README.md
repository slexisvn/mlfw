# Chapter 17 — Pattern rewriting

Two of the five passes in the fixed-point group are built the same way: a set of small rules, each of which finds a shape of IR and replaces it with a better one. `x + 0` becomes `x`. `transpose(transpose(x))` becomes `x`. `add(constant, x)` becomes `add(x, constant)`.

There are a few dozen of these rules and there will be more. This chapter is about the machinery that lets them be written one at a time, by different people, without any of them knowing about the others — and about the two properties such a system needs that this one does not prove.

> **Which passes this chapter is about.** It is easy to read "the simplification passes are pattern rewriting" and assume all five of them share one engine. They do not, and knowing which is which saves an hour the first time you go looking for a rule that is not there:
>
> | Pass | Driven by |
> |---|---|
> | `canonicalize` | `PatternApplicator` over a `PatternSet` ([`canonicalize.ts:49`](../../../src/compiler/passes/canonicalize/canonicalize.ts)) |
> | `algebraic_simplify` | `PatternApplicator` over a `PatternSet` ([`algebraic.ts:45`](../../../src/compiler/passes/simplify/algebraic.ts)) |
> | `constant_fold` | its own traversal, calling each op's `fold` ([`constant_fold.ts`](../../../src/compiler/passes/simplify/constant_fold.ts)) |
> | `cse` | its own hash-and-compare walk over the block |
> | `dce` | its own reachability walk from the terminator |
>
> The bottom three are not pattern sets and adding a `Pattern` will not make them fire. They are *rewrites*, in the loose sense that they change the IR, but each is a single global algorithm rather than a collection of local rules — which is the right shape for them, because "is this operation reachable" is not a question you answer by looking at one operation. The worklist machinery in this chapter drives the top two only.

## 17.1 The problem: where does a rewrite rule live?

Write the first one inside the pass:

```js
for (const op of func.ops()) {
  if (op.opName === 'add' && isZero(op.getOperand(1))) {
    op.replaceAllResultsWith([op.getOperand(0)]);
    op.erase();
  }
}
```

Now write the twentieth. The loop body is a 200-line `if`/`else if` chain; two rules that both match `mul` are separated by ninety lines; nobody can tell whether the order of the branches matters, and it does. Adding a rule means editing a function that everyone else's rules live in, and testing it means running all of them.

Worse, the loop is wrong in a way that is easy to miss. After `AddZero` fires on `add(x, 0)` and rewires its users, some *other* operation — the one that consumed the add — now has a different operand, and may itself have become rewritable. A single forward pass over `func.ops()` will not revisit it. So the pass gets run again by the fixed-point group, and again, and the work that a two-line change could have done locally is paid for by re-running five passes over the whole function.

Both problems have the same shape as Chapter 11's: knowledge that belongs to one operation is being stored in a place that is shared by all of them.

## 17.2 Intuition: a rule is an object, and the driver is a worklist

Split a rewrite rule in two halves and make each an object with a method.

- **match**: given an operation, do I apply here? No side effects, no commitment.
- **rewrite**: apply. Edit the IR through a builder, and report whether you actually did.

Collect the rules in a set. Give the set to a driver that maintains a **worklist**: pull an operation, find the first rule that matches it, apply, and then push back onto the worklist everything the rewrite could have made newly rewritable — the operation's users, its operands' producers, and whatever the rewrite created. Repeat until the worklist is empty.

That last step is the part that turns twenty independent rules into a cascade. `AddZero` fires, the consumer is re-enqueued, `MulOne` fires on it, its consumer is re-enqueued, and a chain of four identities collapses in one pass without anyone having written a chain-of-four rule.

## 17.3 Theory

This is term rewriting, and the vocabulary is a hundred years old.

> **Definition 17.1 (Rewrite rule).** A *rule* is a pair `(m, r)` where `m` is a predicate on operations and `r` is a partial function editing the IR at a matched operation. A rule is *applicable* at `op` if `m(op)` holds and `r` succeeds.

> **Definition 17.2 (Normal form).** IR is in *normal form* with respect to a rule set if no rule in the set is applicable anywhere in it.

Two properties decide whether a rule set is any good, and they are independent.

**Termination.** No infinite chain of rewrites. This is not automatic: two rules that undo each other loop forever. The classical way to get it is a *well-founded reduction order* — a measure on terms that every rule strictly decreases *(classical, term rewriting)*. Most of the identities here obviously decrease operation count, and `CommutativeConstantRight` — which moves a constant from the left operand to the right and changes nothing else — obviously does not.

**Confluence.** If two different rules apply, the order does not matter: the results can be rewritten to a common form. Confluence is what makes "the canonical form" a meaningful phrase rather than "one of the canonical forms, depending on which rule the applicator happened to try first".

> **Theorem 17.3 (Newman's Lemma).** *(Newman, 1942.)* A terminating rewrite system that is locally confluent is confluent. In a terminating and confluent system, every term has a unique normal form.

Local confluence is the weaker and checkable version: whenever two rules apply to the same term, the two one-step results can each be rewritten to a common term. Newman's Lemma is what upgrades it to the global property, and the global property is what a canonicalizer is for.

Now the honest position of this compiler, which is worth stating plainly because it is the position of most production compilers including MLIR's:

- **Termination is not proved. It is bounded.** The applicator carries a step budget ([`passes/rewrite/pattern.ts:30`](../../../src/compiler/passes/rewrite/pattern.ts)) and gives up when it is exhausted, logging that it did so.
- **Confluence is not proved either.** Nothing checks that two patterns matching the same operation agree. When they disagree, the winner is decided by an integer called *benefit*, which is a priority, not an argument.

So "canonical form" here means "the form this rule set with these priorities happens to produce", and §17.6 tests that empirically on four inputs rather than proving it on all of them. That is worth knowing before you rely on canonicalization to make two graphs comparable.

## 17.4 In mlfw: a rule, a set, and a driver

### The rule

[`ir/rewrite/pattern.ts:4`](../../../src/compiler/ir/rewrite/pattern.ts) — the whole base class:

```ts
export class Pattern {
  name: string;
  benefit: number;
  rootOpName: string | null;

  constructor(name: string, benefit = 1) {
    this.name = name;
    this.benefit = benefit;
    this.rootOpName = null;
  }

  match(op: Operation): boolean { return false; }
  rewrite(op: Operation, builder: IRBuilder): boolean { return false; }
}
```

Three fields, two methods, and every one of the 27 pattern classes in [`patterns.ts`](../../../src/compiler/ir/graph/patterns.ts) is a subclass. `rootOpName` is an index key — a pattern that only ever matches `add` says so, and never gets asked about anything else. `benefit` is the priority when several match.

The split between `match` and `rewrite` is not decoration, and `AddZero` shows why ([`patterns.ts:167`](../../../src/compiler/ir/graph/patterns.ts)):

```ts
export class AddZero extends Pattern {
  fastMath: boolean;
  constructor(fastMath = false) { super('add_zero', 5); this.rootOpName = 'add'; this.fastMath = fastMath; }
  override match(op: Operation): boolean {
    if (!isDtypeInt((op.getResult(0).type as TensorType).dtype) && !this.fastMath) return false;
    return isConstantVal(op.getOperand(1).definingOp, 0) || isConstantVal(op.getOperand(0).definingOp, 0);
  }
  override rewrite(op: Operation, builder: IRBuilder): boolean {
    const keep = isConstantVal(op.getOperand(1).definingOp, 0) ? op.getOperand(0) : op.getOperand(1);
    if (keep.type.equals(op.getResult(0).type)) {
      op.replaceAllResultsWith([keep]);
      op.erase();
      return true;
    }
    return false;
  }
}
```

`match` asks the cheap question: is one operand a zero constant? `rewrite` asks the expensive one: does the surviving operand have the *same type* as the result? It might not — `add(tensor<2x8>, zero-scalar)` produces a `2x8` result from a scalar operand, and replacing the add with the scalar would silently change the program's shape. So the rewrite declines, returns `false`, and the applicator moves on to the next pattern. **A pattern that matches is not a pattern that fires**, and the two-method shape is what lets a rule express that without a half-completed edit.

### The set

[`ir/rewrite/pattern.ts:19`](../../../src/compiler/ir/rewrite/pattern.ts). A `PatternSet` keeps two collections — patterns indexed by `rootOpName`, and generic ones with no root — and merges them on demand ([`pattern.ts:61`](../../../src/compiler/ir/rewrite/pattern.ts)):

```ts
  getForOp(opName: string): Pattern[] {
    this._ensureSorted();
    const specific = this._byOp.get(opName);
    if (!specific) return this._generic;
    if (this._generic.length === 0) return specific;
    const merged = new Array<Pattern>(specific.length + this._generic.length);
    let si = 0, gi = 0, mi = 0;
    while (si < specific.length && gi < this._generic.length) {
      if (specific[si].benefit >= this._generic[gi].benefit) {
```

Both lists are kept sorted by descending benefit, so the merge is linear rather than a re-sort, and the returned order is the order the applicator will try. For a set of 30 patterns and an `add`, this hands back the two or three that could possibly apply.

The benefits in use are a coarse scale rather than a fine ranking:

| Benefit | Patterns | What they do |
|---|---|---|
| 20 | `quantize_dequantize_identity` | delete a `quantize` that undoes the `dequantize` feeding it |
| 15 | `constant_quantize`, `dequantize_fold_into_dot` | fold a quantization boundary into the constant or the `dot` beside it |
| 10 | `transpose_transpose`, `fold_trivial_reshape`, `fold_transpose_into_dot`, `idempotent_self`, … | remove an operation outright |
| 5–6 | `add_zero`, `mul_one`, `div_one`, `commutative_constant_right`, … | algebraic identities and normalizations |
| 4 | `add_neg_to_sub`, `associative_constant_reassoc`, … | reshuffles that enable other rules |

Read top-down that is a policy: prefer deleting to rewriting, prefer rewriting to reassociating, and put the quantization rules above everything because a quantize left in place blocks all of it (Chapter 26). The bottom three rows are the whole of [`patterns.ts`](../../../src/compiler/ir/graph/patterns.ts); the top two come from the quantization pattern file and reach the canonicalizer through `quantize`'s and `dot`'s registry entries. It is a reasonable policy and it is not derived from anything.

### The driver

[`passes/rewrite/pattern.ts:17`](../../../src/compiler/passes/rewrite/pattern.ts) is §17.2 in code. The worklist starts as every operation, including those nested inside regions:

```ts
    const worklist = [...func.opsRecursive()];
    let head = 0;
    const queued = new Set<Operation>(worklist);
    const enqueue = (op: Operation | null | undefined): void => {
      if (!op || !op.parentBlock || queued.has(op)) return;
      queued.add(op);
      worklist.push(op);
    };
```

`enqueue` refuses an operation with no `parentBlock` — an erased one — which is how a rewrite that deletes an operation avoids re-queuing a corpse. The `queued` set keeps the list free of duplicates.

The re-enqueue after a successful rewrite is the cascade ([`pattern.ts:60`](../../../src/compiler/passes/rewrite/pattern.ts)):

```ts
        totalRewrites++;
        for (const a of affected) enqueue(a);
        let cur = prevOp ? prevOp._next : block._head;
        let guard = block._size + 2;
        while (cur && cur !== nextOp && guard-- > 0) { enqueue(cur); cur = cur._next; }
        enqueue(op);
        break;
```

Three groups get re-queued. `affected` was captured *before* the rewrite: every user of the operation's results and every producer of its operands, because those are the neighbours whose surroundings just changed. The `while` loop walks from the operation's old predecessor to its old successor, which picks up anything the rewrite *inserted* in between. And `enqueue(op)` re-queues the operation itself, so a second pattern can fire on the same operation — harmlessly skipped if the rewrite erased it.

And the bound ([`pattern.ts:30`](../../../src/compiler/passes/rewrite/pattern.ts)):

```ts
    const safetyBudget = Math.max(maxIterations, 1) * Math.max(worklist.length, 1) * 4 + 1000;
```

Proportional to the function size, so it scales, and reported when hit:

```ts
        trace.emit({
          type: 'pass_detail', passName: 'PatternApplicator',
          message: `pattern rewriting hit safety budget (${safetyBudget}) without converging`,
```

This is the same shape as Chapter 15's `max-iter` line, at a different level of the hierarchy: the system cannot prove it will stop, so it counts, and it tells you when the count ran out.

### Where the canonicalizer's rules come from

[`passes/canonicalize/canonicalize.ts:15`](../../../src/compiler/passes/canonicalize/canonicalize.ts) builds its set from the op registry rather than from a list:

```ts
function traitPatternsFor(def: OpDef): Pattern[] {
  const patterns: Pattern[] = [];
  if (def.isCommutative) {
    patterns.push(new CommutativeConstantRight(def.name));
    if (def.isAssociative && def.fold) patterns.push(new AssociativeConstantReassoc(def.name));
  }
  if (def.hasTrait(OpTrait.IDEMPOTENT)) patterns.push(new IdempotentSelf(def.name));
  return patterns;
}
```

This is Chapter 11's argument arriving at its destination. Nobody wrote "`maximum` may have its constant moved to the right". `maximum` declares `COMMUTATIVE`, and a pattern instance is *generated* for it, rooted at its name so the index still works. Register a new commutative operation tomorrow and it gets a canonicalization rule with no further code. The rest of the set comes from `def.getCanonicalizationPatterns()` — the per-operation rules the op author wrote — and the whole thing is built once and cached.

The other pattern pass builds its set from an explicit list instead ([`simplify/algebraic.ts:9`](../../../src/compiler/passes/simplify/algebraic.ts)), 13 patterns, plus three more when `fastMath` is on.

The tempting story is that the division of labour is by *licence* — that `algebraic_simplify` holds the rules needing an assumption the user opts into (Chapter 20) and `canonicalize` holds the unconditional ones. Print both sets and that story does not survive:

| | |
|---|---|
| In **both** sets (8) | `AddZero`, `SubZero`, `SubSelf`, `MulOne`, `MulZero`, `DivOne`, `DoubleNeg`, `ReshapeReshape` |
| Only in `algebraic_simplify` (5) | `TransposeTranspose`, `MulNegNeg`, `AddNegToSub`, `SubNegToAdd`, `DoubleConvert` |

All five of the algebraic-only rules are unconditionally true; none of them needs a licence. Of the thirteen, only `SubSelf` and `MulZero` consult `fastMath` at all, and both default to firing on integers only, where the identity holds outright. The genuinely licensed rules are the *three* that are absent from the set entirely unless `fastMath` is on — `DivSelf`, `ExpLog`, `LogExp` — and the licence gate is the `if (fastMath)` in the builder, not the choice of pass.

So the real division is: **`canonicalize` gets whatever an operation declares in its registry entry, and `algebraic_simplify` gets whatever somebody added to a list**, with an eight-rule overlap where both happened. §17.7 returns to what that costs.

### Matching more than one operation

A `Pattern` matches a single operation and reaches its operands by hand. For rules that need to see a subgraph, there is a small combinator language ([`ir/rewrite/dfpattern.ts:90`](../../../src/compiler/ir/rewrite/dfpattern.ts)):

```ts
export const wildcard = (): DFPattern => new AnyPattern();
export const isOp = (name: string, ...operandPatterns: DFPattern[]): DFPattern => new OpPattern(name, operandPatterns);
export const hasAttr = (inner: DFPattern, key: string, value?: AttrValue): DFPattern => new AttrPattern(inner, key, value);
export const alt = (...patterns: DFPattern[]): DFPattern => new AltPattern(patterns);
export const capture = (name: string, inner: DFPattern = new AnyPattern()): DFPattern => new CapturePattern(name, inner);
```

so `exp(log(x))` is `isOp('exp', isOp('log', wildcard()))`, and `matchPattern` returns a bindings object or `null`. `OpPattern.match` walks `op.getOperand(i).definingOp` ([`dfpattern.ts:23`](../../../src/compiler/ir/rewrite/dfpattern.ts)) — it matches *up the dataflow graph*, which is the direction Chapter 8's use-def edges point, and the reason this is a data-flow pattern rather than a tree pattern. Five patterns use it, and all five are the same two-deep shape ([`patterns.ts:11`](../../../src/compiler/ir/graph/patterns.ts)):

```ts
const TRANSPOSE_TRANSPOSE_PAT = isOp('transpose', isOp('transpose', wildcard()));
const RESHAPE_RESHAPE_PAT = isOp('reshape', isOp('reshape', wildcard()));
const DOUBLE_NEG_PAT = isOp('neg', isOp('neg', wildcard()));
const EXP_LOG_PAT = isOp('exp', isOp('log', wildcard()));
const LOG_EXP_PAT = isOp('log', isOp('exp', wildcard()));
```

## 17.5 Lab 1 — A rewrite cascade

```bash
node docs/part3/ch17-pattern-rewriting/labs/01-a-rewrite-cascade.mjs
```

Four identities stacked on one input — `transpose(transpose(a)) + 0) * 1` — on an `i32` tensor, because `x + 0` is an identity on integers and not on floats (Chapter 20). Traced to seven operations:

```
module @traced {
  func @traced(%0: tensor<2x2xi32>) -> (tensor<2x2xi32>) {
    %1 = transpose(%0) {permutation = [1, 0]} : tensor<2x2xi32>
    %2 = transpose(%1) {permutation = [1, 0]} : tensor<2x2xi32>
    %3 = constant() {tensor_type = tensor<xi32>, value = 0} : tensor<xi32>
    %4 = add(%2, %3) : tensor<2x2xi32>
    %5 = constant() {tensor_type = tensor<xi32>, value = 1} : tensor<xi32>
    %6 = mul(%4, %5) : tensor<2x2xi32>
    return(%6)
  }
}
```

At `DEBUG`, the applicator reports what it did:

```
=== what the pattern applicator did ===
  round 1  canonicalize         7 -> 5 ops   2 rewrite(s) from a set of 30 patterns
  round 1  algebraic_simplify   5 -> 5 ops   1 rewrite(s) from a set of 13 patterns
  round 1  dce                  5 -> 2 ops   dce reports erasedCount=3
  round 2  canonicalize         2 -> 1 ops   1 rewrite(s) from a set of 30 patterns

=== after graph passes ===
module @LongWayRound {
  func @LongWayRound(%0: tensor<2x2xi32>) -> (tensor<2x2xi32>) {
    return(%0)
  }
}
```

**Two rewrites in one run of canonicalize.** That is the cascade: `AddZero` fired on `%4`, which re-enqueued its user `%6`, on which `MulOne` then fired. The fixed-point group did not run twice to achieve that; the worklist did it inside a single pass, which is precisely the work §17.1 said a plain forward loop would waste.

**One rewrite in algebraic simplification, and the op count does not move** (`5 -> 5`). `TransposeTranspose` deletes nothing: it composes the two permutations into a *new* transpose — here `[1, 0]` after `[1, 0]`, which is the identity `[0, 1]` — and erases the outer one ([`patterns.ts:67`](../../../src/compiler/ir/graph/patterns.ts)). One operation created, one erased, and the inner transpose left with no users for DCE to collect. Same phenomenon as Chapter 15's `CHANGED 4 -> 4`, seen from the pattern's side.

**And then round 2 needs canonicalize again**, because the identity transpose that algebraic simplification *produced* is folded away by `FoldTrivialTranspose`, which lives in canonicalize's set and had already run. No worklist can fix that: the two rules are in different passes, so the only mechanism that connects them is the fixed-point group. Chapters 15 and 17 are two halves of one design — the worklist reaches a local fixed point inside a pass, the group reaches a global one across passes, and both are needed because the rule set is split across passes for reasons of licence.

Note the last line of the pass detail: `dce reports erasedCount=3`. Each pass emits its own diagnostic shape into the same event type; the driver does not impose a schema. Chapter 18 is about that stream.

## 17.6 Lab 2 — Canonical form is a normal form

```bash
node docs/part3/ch17-pattern-rewriting/labs/02-canonical-form.mjs
```

Definition 17.2, tested. Four different programs, all computing `a * a`, written to require different rules to see it:

```
=== a * a ===
  traced:      2 operations
  canonical:
    %1 = mul(%0, %0) : tensor<2x2xi32>
    return(%1)
=== transpose(transpose(a)) * a ===
  traced:      4 operations
  canonical:
    %1 = mul(%0, %0) : tensor<2x2xi32>
    return(%1)
=== (a + 0) * (a * 1) ===
  traced:      6 operations
  canonical:
    %1 = mul(%0, %0) : tensor<2x2xi32>
    return(%1)
=== reshape(reshape(a)) * (a - 0) ===
  traced:      6 operations
  canonical:
    %1 = mul(%0, %0) : tensor<2x2xi32>
    return(%1)

4 spellings collapsed to 1 canonical form(s).
```

Two, four, six and six operations in; the same two lines out, character for character. Four different rules did the work — `TransposeTranspose`, `AddZero`, `MulOne`, `ReshapeReshape`, `SubZero` — and none of them knows about any of the others.

This is the property everything downstream leans on. CSE (Chapter 19) merges two operations when they are *structurally* equal, which means a program that computes the same thing twice in two different spellings is only deduplicated if canonicalization has already made the two spellings identical. Fusion compares operation kinds. The tuning database (Chapter 47) keys schedules by a graph signature. Every one of those is a comparison of syntax that is only meaningful because a normalization ran first.

And it is worth being precise about what the lab does and does not show, because "four inputs, one output" is a weaker result than it looks.

**What it shows.** Four textually different modules, each computing the same term, converge to byte-identical printed output. That is genuine evidence of confluence *on this term*, and it is the property a canonicalizer exists to provide.

**What it does not show, in increasing order of importance.**

*It does not check idempotence.* The lab prints the canonical form; it does not feed that form back through `canonicalize` and assert `UNCHANGED`. A rule set can map four inputs to one output and still not be stable there — `A → B → A` maps everything to `A` or `B` and never settles. The check costs one more pass run and Definition 17.2 is the reason to want it: a normal form is defined by *no rule being applicable*, not by two runs agreeing.

*It does not check that no pattern still matches.* Even an idempotent result is not necessarily a normal form. A pattern whose `match` succeeds and whose `rewrite` returns `false` leaves the IR unchanged while remaining applicable, and the applicator would report no change. Distinguishing "nothing matches" from "nothing changed" requires asking the `PatternSet` directly, which the lab does not do.

*It generalizes over one term.* Theorem 17.3 says that *if* the rule set is terminating and locally confluent, then all inputs computing any term converge — but neither hypothesis is established anywhere in this codebase, and §17.7 lists the reasons both are hard. Four inputs computing one term is one data point about a system with a few dozen interacting rules.

That gap is real, and it is the difference between a canonicalizer you can test and one you can trust. **Try this.** Add the idempotence check: re-run `canonicalize` on the parsed canonical form and print the verdict. It should report `UNCHANGED`, and if it ever does not, you have found either a non-terminating rule or a confluence failure — both worth knowing about, and neither visible in the table above.

**Try this.** Add a fifth spelling — `a.mul(a).add(0).mul(1)` — and check it lands in the same place. Then try one that should *not*: `a.mul(a.add(1))`.

## 17.7 Traps and limits

- **A trait-derived pattern inherits the trait's mistakes.** §17.4's `traitPatternsFor` builds patterns from declarations: any commutative operation gets `CommutativeConstantRight`, and any commutative *and associative* one that can fold gets `AssociativeConstantReassoc` ([`canonicalize.ts:19`](../../../src/compiler/passes/canonicalize/canonicalize.ts)). That is the design working — no operation names anywhere. It is also the design's exposure: a generated pattern is exactly as sound as the trait behind it, and Chapter 11 §11.3 shows `ASSOCIATIVE` is declared unconditionally on `add` and `mul`, including on floats where it is false.

  A hand-written pattern can compensate by testing the dtype at match time, as `AddZero` does ([`patterns.ts:171`](../../../src/compiler/ir/graph/patterns.ts)). A pattern *generated from a trait* has only the trait to consult, so the generator has to hand it the missing context:

  ```ts
    if (def.isCommutative) {
      patterns.push(new CommutativeConstantRight(def.name));
      if (def.isAssociative && def.fold) patterns.push(new AssociativeConstantReassoc(def.name, fastMath));
    }
  ```

  That is why `CanonicalizePass` takes a fast-math option and caches one pattern set per setting, the same shape `AlgebraicSimplificationPass` has. The general point: deriving patterns from declarations is the right architecture, and the price is that **a declaration's accuracy becomes load-bearing in a way a `switch` statement never made it** — and that a generated pattern needs a route by which context can reach it.

- **Neither termination nor confluence is checked.** Both are bounded or assumed. A pattern whose rewrite re-creates its own match will burn the safety budget on every compile — with one `INFO`-level line to say so — and a pair of patterns that disagree will produce whichever form the benefit ordering favours, silently. When you add a pattern, the question to ask is not "does it fire" but "can it fire on its own output".
- **Benefit ties are broken by insertion order.** `_ensureSorted` uses `Array.prototype.sort` on `b.benefit - a.benefit` ([`pattern.ts:47`](../../../src/compiler/ir/rewrite/pattern.ts)). The sort is stable in modern JavaScript engines, so equal-benefit patterns fire in registration order — which for canonicalize means op-registry iteration order. Deterministic, and not a designed guarantee.
- **`match` is called before every `rewrite` and both may walk the graph.** There is no shared state between them: `AddZero` tests `isConstantVal` in `match` and again in `rewrite`. For cheap predicates this is fine and it keeps `match` side-effect free, which is what makes trying patterns in benefit order safe. For an expensive match, it is a doubling.
- **The applicator does not recurse into regions during rewriting, only during collection.** The worklist is seeded from `opsRecursive()`, so operations inside a `fusion` or `scan` region *are* visited; but the re-enqueue walk after a rewrite uses `block._head`/`_next` within one block. A rewrite that changes something in a sibling region will not re-enqueue across the boundary.
- **`maxIterations` is passed to the applicator and is not an iteration count.** `applyPatterns(func, 10, trace)` uses the argument only to size the safety budget ([`pattern.ts:30`](../../../src/compiler/passes/rewrite/pattern.ts)). There is no outer loop over the worklist; the worklist *is* the loop. The parameter name is a leftover from an earlier design and the tests still describe rounds in its terms.
- **Which set a rule lives in is not always principled, and eight rules are in both.** §17.4 has the table. The clearest single case: `reshape` declares both of its rules — `FoldTrivialReshape` and `ReshapeReshape` — as canonicalization patterns ([`ops/shape.ts:158`](../../../src/compiler/ir/graph/ops/shape.ts)), while `transpose` declares only `FoldTrivialTranspose` ([`ops/shape.ts:184`](../../../src/compiler/ir/graph/ops/shape.ts)) and leaves the structurally identical `TransposeTranspose` to the algebraic pass. Both rules are unconditionally valid; there is no licence argument separating them. The observable consequence is in Chapter 15's lab: `transpose(transpose(x))` takes three rounds of the fixed-point group to disappear, and `reshape(reshape(x))` takes two — the transpose pair has to bounce between two passes, the reshape pair does not. The eight duplicated rules are the mirror-image cost: each is tried twice per round, in two different passes, and whichever runs first is the one that ever fires.
- **Half the combinator language has no users.** `isOp` and `wildcard` build the five two-deep matchers above; `capture`, `alt` and `hasAttr` are implemented and tested and called from nowhere in `src/`. And every pattern that needs to *read* what it matched — including `FoldTransposeIntoDot`, the most valuable one in the set ([`patterns.ts:413`](../../../src/compiler/ir/graph/patterns.ts)) — walks `definingOp` by hand instead, because a `match` that returns a boolean cannot hand its bindings to `rewrite`. That is the missing piece: the matcher language can bind, and the `Pattern` interface has nowhere to put a binding.

## 17.8 Read the tests

- [`tests/compiler/ir/rewrite/pattern.test.js:73`](../../../tests/compiler/ir/rewrite/pattern.test.js) — `PatternSet` ordering: descending benefit, and the specific/generic merge.
- [`tests/compiler/ir/rewrite/pattern.test.js:160`](../../../tests/compiler/ir/rewrite/pattern.test.js) — the applicator reaching a fixed point on `neg(neg(neg(neg(x))))`, the budget cap, benefit priority deciding between two matching patterns, and the case that matters most in practice: not crashing on an operation erased by an earlier pattern in the same run.
- [`tests/compiler/ir/rewrite/dfpattern.test.js`](../../../tests/compiler/ir/rewrite/dfpattern.test.js) — the combinators: nested structural match, captures, attribute match, alternation, wildcard.
- [`tests/compiler/passes/canonicalize/`](../../../tests/compiler/passes/canonicalize/) and [`tests/compiler/passes/simplify/`](../../../tests/compiler/passes/simplify/) — the patterns as passes, including the pairs where one enables another.

---

**Next:** [Chapter 18 — Watching the compiler work](../ch18-watching-the-compiler/README.md), which collects the trace stream this part has been quoting one field at a time, and asks what a compiler owes you when it fails.
