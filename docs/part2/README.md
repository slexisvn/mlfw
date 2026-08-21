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
- **The invariant set** (Chapter 12), which the pipeline checks at four boundaries and which Chapter 64 turns into a debugging procedure.

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
node docs/part2/ch12-valid-ir/labs/01-break-it-seven-ways.mjs
node docs/part2/ch13-ir-as-text/labs/01-round-trip.mjs
node docs/part2/ch13-ir-as-text/labs/02-edit-by-hand.mjs
```

Unlike Part I's, none of these labs measure time. Every one of them is deterministic: the output printed in the chapters is the output you should get, character for character. If a lab prints something different, either you are reading a different version of the source or you have found a bug — both worth knowing.

A note on what the labs can reach. The package's public surface is deliberately small: `trace`, `compile`, `printModule`, `printFunction`, `parseModule`, `parseFunction`. That is enough, because `trace` and `parseModule` hand back the *real* IR objects, and from a module you can reach every function, block, operation, region, value and type by ordinary property access. Where a chapter discusses something the public surface does not expose — the op registry, the full verifier — it quotes the source and names the test that pins the behaviour, and says plainly that the lab cannot reach it.
