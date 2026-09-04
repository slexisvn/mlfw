# Part II — Representing programs

Part I argued that a compiler needs to see the whole computation at once, and showed you the thing it sees: a printed graph. This part is about what that graph *is* — the data structure, the rules it obeys, and why those rules are the ones that make the rest of the book possible.

Everything from here to the end of the book is a transformation of this object. It is worth knowing it well.

| Chapter | Title | The question it answers |
|---|---|---|
| [8](ch08-ssa-and-dataflow/README.md) | SSA and dataflow | Why does "every value is produced exactly once" make analysis cheap — and why does the order of the lines mean nothing? |
| [9](ch09-object-model/README.md) | Value, Operation, Block, Region, Function, Module | Six nouns. What does each own, and how does an operation come to contain a program? |
| [10](ch10-type-system/README.md) | The type system | What does a type know, when is one shape usable where another is expected, and what happens when a size is not known until run time? |
| [11](ch11-ops-as-a-dialect/README.md) | Ops as a dialect | How does a pass reason about an operation it has never heard of? |
| [12](ch12-valid-ir/README.md) | What "valid IR" means | Which invariants are enforced, by whom, and at what moment — and which are only intended? |
| [13](ch13-ir-as-text/README.md) | IR as text | Can you print the program, edit it in a text editor, and feed it back? |

## The argument in one paragraph

A graph IR is a set of **values**, each produced by exactly one **operation** — the SSA property (Chapter 8). That single restriction makes "who uses this?" a stored fact rather than a search, and it makes the program a directed acyclic graph in which the textual order of the lines carries no meaning at all; only the edges do. Around values and operations sit four containers — **blocks**, **regions**, **functions**, **modules** — and the interesting one is the region, because it lets an operation contain a program, which is how `fusion`, `scan` and `if` exist without being unrolled (Chapter 9). Every value carries a **type**: shape, element type, layout, with a dimension permitted to be unknown, and shape *compatibility* is a weaker and more useful relation than shape *equality* (Chapter 10). Operations are not hard-wired into passes; they are entries in a **registry** carrying traits, verification rules, type inference and folding rules, so a pass can ask "is this commutative?" instead of matching on a name (Chapter 11). What makes a graph valid is enforced in three places at three different moments, and knowing which is which is how you debug a compiler (Chapter 12). And all of it prints to text and parses back, losslessly, which is the difference between an IR you can work on and one you can only print (Chapter 13).

## What Part II establishes for later parts

- **SSA and the use-def graph** (Definition 8.1), which every analysis in Parts III–V walks.
- **Theorem 8.4** — textual order carries no semantics — which is the licence for every reordering pass in the book.
- **Regions** (Chapter 9), without which fusion has nowhere to put the operations it merges and `scan` has to be unrolled.
- **The specificity order on dimensions as a partial order, and the least upper bound in it as what type inference propagates** (Definitions 10.1–10.2). *Compatibility* — the relation `shapeCompatible` actually decides — is reflexive and symmetric and **not** transitive, so it is not a partial order at all; Theorem 10.3 is the counterexample.
- **Traits as queryable data** (Chapter 11), which is why the fusion engine in Part IV works on operations nobody had written when it was designed.
- **The invariant set** (Chapter 12), which the pipeline checks at four boundaries, Chapter 64 revisits as a verification strategy, and Chapter 67 turns into a debugging procedure.

## One pass, end to end

Six chapters build a data structure and never show you a complete consumer of it. That is the right order — you cannot read a pass before you can read the IR — but it leaves the loop open, so here it is closed. This is the whole of `DCEPass.run`, minus its tracing block ([`passes/simplify/dce.ts:20`](../../src/compiler/passes/simplify/dce.ts)):

```ts
  override run(target: PassTarget, analysisManager?: AnalysisManager): PassResultValue {
    const func = target as GraphFunction;
    let changed = false;
    const memEffects = this.getAnalysis(MemoryEffectAnalysis as never, func, analysisManager) as MemoryEffectResult;

    const worklist: Operation[] = [];
    for (const op of func.opsRecursive()) {
      if (this._isDead(op, memEffects)) worklist.push(op);
    }

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
      changed = true;

      for (const defOp of operandDefs) {
        if (defOp.parentBlock && this._isDead(defOp, memEffects)) worklist.push(defOp);
      }
    }

    return changed ? PassResult.CHANGED : PassResult.UNCHANGED;
  }
```

and the predicate the whole thing turns on ([`dce.ts:81`](../../src/compiler/passes/simplify/dce.ts)):

```ts
  _isDead(op: Operation, memEffects: MemoryEffectResult): boolean {
    if (isTerminatorOp(op.opName)) return false;
    for (let i = 0; i < op.numResults; i++) {
      if (op.getResult(i).hasUses) return false;
    }
    return !memEffects.hasSideEffect(op);
  }
```

**Every line of that is a chapter of this part.** Read it back with the chapter numbers attached:

| The code | The chapter it comes from |
|---|---|
| `op.getResult(i).hasUses` | Chapter 8. This is the intrusive use list, and it is why "is anybody reading this?" is a field test rather than a scan of the function. In a representation without it, DCE would be quadratic |
| `func.opsRecursive()` | Chapter 9. The generator that descends into regions. Written with `ops()` instead, this pass would silently never look inside a `fusion` or a `scan` |
| `_eraseRecursively`, `dropAllOperands` first | Chapter 9 again. Erasing an operation that owns a region means erasing its contents first, innermost last, dropping operands as you go so no use list is left pointing at a dead operation |
| `isTerminatorOp(op.opName)` | Chapters 11 and 12. A `return` has no results, so the loop above would call it dead every time. The trait is what rescues it, and the pass asks the registry rather than comparing against the string `'return'` |
| `memEffects.hasSideEffect(op)` | Chapter 11. "Has no users" is not grounds for deletion — Counterexample 8.5 — and the side-effect kind is declared on the `OpDef`, so a `scan` whose body writes memory is not deletable however unused its results look |
| the worklist, re-checking `_isDead` on pop | Chapter 8's Theorem 8.4. Deleting an operation orphans its producers, so the pass pushes them back and lets order sort itself out. Nothing here has to reason about *where in the block* anything sits |
| `CHANGED` / `UNCHANGED` | Chapter 14, the one thing in this listing Part II did not give you |

Forty lines, and the only thing it knows about any particular operation is what the registry told it. That ratio — a small pass over a well-chosen representation — is the argument of this whole part, and it is why the six chapters came before the transformations rather than after.

Two things this listing is *not*. It is not the full treatment of dead code elimination: Chapter 19 is, and it is mostly about the cases where `hasSideEffect` is the hard question. And it is not typical in its simplicity — the fusion engine of Part IV is two orders of magnitude larger. What is typical is the shape: ask the registry, walk the use lists, edit through the API, report whether anything moved.

## Labs

```bash
npm run build   # once, if you have not already

node docs/part2/ch08-ssa-and-dataflow/labs/01-use-def.mjs
node docs/part2/ch08-ssa-and-dataflow/labs/02-order-carries-nothing.mjs
node docs/part2/ch09-object-model/labs/01-the-six-nouns.mjs
node docs/part2/ch09-object-model/labs/02-what-a-region-sees.mjs
node docs/part2/ch10-type-system/labs/01-what-a-type-knows.mjs
node docs/part2/ch10-type-system/labs/02-static-dynamic-symbolic.mjs
node docs/part2/ch11-ops-as-a-dialect/labs/01-a-trait-is-data.mjs
node docs/part2/ch11-ops-as-a-dialect/labs/02-fold-and-canonicalize.mjs
node docs/part2/ch12-valid-ir/labs/01-break-it-eight-ways.mjs
node docs/part2/ch13-ir-as-text/labs/01-round-trip.mjs
node docs/part2/ch13-ir-as-text/labs/02-edit-by-hand.mjs
```

Unlike Part I's, none of these labs measure time. Every one of them is deterministic: the output printed in the chapters is the output you should get, character for character. If a lab prints something different, either you are reading a different version of the source or you have found a bug — both worth knowing.

A note on what the labs can reach. The package's public surface is deliberately small: `trace`, `compile`, `printModule`, `printFunction`, `parseModule`, `parseFunction`. That is enough, because `trace` and `parseModule` hand back the *real* IR objects, and from a module you can reach every function, block, operation, region, value and type by ordinary property access. Where a chapter discusses something the public surface does not expose — the op registry, the full verifier — it quotes the source and names the test that pins the behaviour, and says plainly that the lab cannot reach it.
