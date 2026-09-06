# Chapter 13 — IR as text

Every chapter so far has shown you printed IR, and you have been reading it as documentation — a window onto a data structure that lives somewhere else. This chapter closes the loop. The text is not a view of the program. It *is* the program, in a second form, and the two forms are interchangeable.

That difference is larger than it sounds, and it is the last thing Part II owes you before the passes start.

## 13.1 The problem: a debug dump is not a format

Most compilers can print their IR. Far fewer can read it back.

The distinction decides what you can do when something goes wrong. With a print-only dump you can look at the IR, and that is all — to test a pass on a specific input you must find a program that traces to that input, which for a bug in the eleventh operation of a transformer means constructing a transformer. With a real textual format you paste the eleven operations into a file, run the pass on them, and iterate in seconds.

It also decides what a test can say. A test that asserts on a printed string is testing the printer. A test that parses IR, runs a pass, and compares against expected IR is testing the pass — and reads as a specification of the transformation rather than of its output formatting.

The property that makes the difference is round-tripping:

> **Definition 13.1 (Lossless round-trip).** **(stated here)** A printer *P* and parser *Q* round-trip losslessly on a class of programs when `P(Q(P(m))) = P(m)` for every module *m* in that class — printing, reading back, and printing again yields identical text.

Note the shape of the equation. It is not `Q(P(m)) = m`, which is too strong to be useful: the reconstructed module has different object identities, different internal `id` numbers, and a different history. Equality of the *printed form* is the right notion, because it says everything the text is meant to carry survived.

## 13.2 The format, read as a grammar

You have been reading it since Chapter 2; here is its structure, named. (Assembled to show every construct at once rather than lifted from one printout — but it is a real module: paste it into `parseModule` and it parses, and printing it back gives these exact lines.)

```
module @traced {
  func.func @traced(%0: tensor<2x2xf32>, %1: tensor<8x2xf32>, %2: tensor<8xf32>) -> (tensor<2x8xf32>) {
    %3 = tera.transpose %1 {permutation = array<i64: 1, 0>} : tensor<8x2xf32> -> tensor<2x8xf32>
    %4 = tera.dot %0, %3 {lhs_batch = array<i64>, lhs_contracting = array<i64: 1>, rhs_batch = array<i64>, rhs_contracting = array<i64: 0>} : (tensor<2x2xf32>, tensor<2x8xf32>) -> tensor<2x8xf32>
    %5 = "tera.fusion"(%4, %2) ({
      ^bb0(%6: tensor<2x8xf32>, %7: tensor<8xf32>):
        %8 = "tera.add"(%6, %7) : (tensor<2x8xf32>, tensor<8xf32>) -> tensor<2x8xf32>
        tera.yield %8 : tensor<2x8xf32>
    }) {fusion_kind = "kElementwise"} : (tensor<2x8xf32>, tensor<8xf32>) -> tensor<2x8xf32>
    return %5 : tensor<2x8xf32>
  }
}
```

| Piece | Syntax | Chapter |
|---|---|---|
| module | `module @name { ... }` | 9 |
| function | `func.func @name(args) -> (types) { ... }` | 9 |
| operation, custom form | `results = tera.name operands {attrs} : types` | 9 |
| operation, generic form | `results = "tera.name"(operands) {attrs} : (types) -> types` | 9 |
| value | `%n` | 8 |
| type | `tensor<2x8xf32>` | 10 |
| region | a `({ ... })` group inside the operation | 9 |
| block label | `^bbN(args):` | 9 |

This is MLIR's syntax, and not by coincidence: it is the tera dialect, the same
one the out-of-tree MLIR compiler in `terac/` defines. A module printed here can
be handed to `tera-opt` and it parses — which is the strongest statement the
format makes about itself, because a second implementation checks it.

Three design decisions are visible in that table.

**Nesting is by indentation and braces, not by a flat encoding.** A region prints as a `{ ... }` block below its operation, and the parser reconstructs depth from the indent width. That makes the format readable at the cost of making it whitespace-sensitive — `parseModule` takes an `indentWidth` option defaulting to 2 ([`parser.ts:851`](../../../src/compiler/ir/graph/parser.ts)) because it has to.

**Every operation has two spellings, and the printer picks the narrower one it can justify.** The *custom form* is the one the dialect defines for that operation — `tera.add %0, %1 : tensor<2x2xf32>` prints one type because `tera.add`'s operands and result must share it. The *generic form* spells the operation name as a string and lists every type — `"tera.add"(%0, %1) : (tensor<2x8xf32>, tensor<8xf32>) -> tensor<2x8xf32>`. That second line is not a formatting choice: mlfw's `add` broadcasts its operands and `tera.add` does not, so the custom form would be claiming the bias has a shape it does not have. The rule is [`mlir_format.ts:236`](../../../src/compiler/ir/graph/mlir_format.ts): print the custom form only when it carries enough to reconstruct the operation, and fall back to the generic form — which is always lossless — when it does not. Anything the dialect does not define at all goes generic too, which is why `"tera.fusion"` appears that way above.

**Types are on every result, not inferred on read.** `%3 = tera.transpose %1 {...} : tensor<8x2xf32> -> tensor<2x8xf32>` states the result type even though Chapter 11's `inferResultTypes` could compute it. That redundancy is deliberate: it means the text is self-describing, it means a hand-written module does not depend on inference being correct to be readable, and — as Chapter 12 showed — it gives the verifier two independent sources to compare.

## 13.3 In mlfw: printing

[`printer.ts:274`](../../../src/compiler/ir/graph/printer.ts) is where the choice of spelling is made, and it is two lines long:

```ts
    const form = this._formOf(op);
    if (form) this._printCustom(op, form, out);
    else this._printGeneric(op, out);
```

`_formOf` consults the shared table in `mlir_format.ts` — the one file the printer and the parser both read, so that one description of the dialect's syntax serves both directions. A format described twice is a format that drifts.

Three further details are what make Definition 13.1 hold, and each of them is a small decision that would silently break round-tripping if made the other way.

**Attributes print in sorted order** ([`printer.ts:223`](../../../src/compiler/ir/graph/printer.ts)):

```ts
    entries.sort((a, b) => (a[0] < b[0] ? -1 : a[0] > b[0] ? 1 : 0));
```

An operation's attributes live in a `Map`, which preserves insertion order. Two operations built by different routes can carry the same attributes inserted in different orders, and would print differently while being identical. Sorting makes the printed form a function of the operation's content alone — which is what lets a test compare two printouts, and what makes a diff between two compiler versions readable.

**Values are numbered by the printer, from zero, per function** ([`printer.ts:271`](../../../src/compiler/ir/graph/printer.ts)) — Chapter 8 §8.6 covered why. The consequence for round-tripping is that `%n` is not data to be preserved; it is a label regenerated on each print, and the second print of a re-parsed module reproduces it exactly because the walk order is the same.

**Non-finite floats print as their bit pattern** ([`mlir_format.ts:551`](../../../src/compiler/ir/graph/mlir_format.ts)):

```ts
export function formatFloatLiteral(value: number, dtype: ScalarDType): string {
  if (!Number.isFinite(value)) {
    const [total] = floatLayout(dtype);
    return `0x${nonFiniteBits(value, dtype).toString(16).toUpperCase().padStart(total / 4, '0')}`;
  }
```

`String(Infinity)` is `"Infinity"`, which is not a number literal any parser would accept, and MLIR has no `inf` keyword either — it spells a non-finite float as the hexadecimal contents of its bits. The width depends on the element type, which is why the function takes one: `-inf` is `0xFF800000` in `f32` and `0xFFF0000000000000` in `f64`. A masked softmax is full of `-inf`, so this is not an edge case — it is the second-most-common constant in an attention model.

Dense tensor attributes carry their type after the data, `dense<[1.0, 2.0]> : tensor<2xf32>` ([`printer.ts:319`](../../../src/compiler/ir/graph/printer.ts)), so a `Float32Array` and an `Int32Array` of the same numbers survive as different things.

## 13.4 In mlfw: parsing, and why it takes two passes

Here is where the IR's nature forces the design.

A parser for a normal language reads top to bottom and builds as it goes, because a use always follows its definition. This text has no such guarantee — Chapter 8's Lab 2 fed the parser a function printed in reverse, and it worked. More importantly, *nothing in the format promises otherwise*, so the parser cannot rely on it even for text a printer produced.

But an `Operation` cannot be constructed without its operands: look back at [`operation.ts:50`](../../../src/compiler/ir/graph/operation.ts) and every operand must be a live `Value` at construction time, because the constructor registers uses on it.

So the parser cannot build in reading order, and it splits in two.

**Phase one — read the text into records.** `RecordReader` ([`parser.ts:438`](../../../src/compiler/ir/graph/parser.ts)) turns each line into an `OpRecord`: the operation name, result names as *strings*, result types, attributes, operand names as *strings*, and nested region records. No `Operation` is constructed, and nothing is resolved. A record also carries its dependency set, gathered from its own operands and, recursively, from everything its regions reference ([`parser.ts:674`](../../../src/compiler/ir/graph/parser.ts)):

```ts
function collectRegionDeps(blocks: readonly BlockRecord[], deps: Set<string>): void {
  for (const block of blocks) {
    for (const op of block.ops) {
      for (const name of op.deps) deps.add(name);
    }
  }
}
```

That recursion matters: an operation with a region depends on everything the region's body reads, so a `fusion` must be built after the producers its body refers to — the same fact `capturedValues` computes for the topological sort in Chapter 9.

**Phase two — order by dependency, then materialize.** [`parser.ts:682`](../../../src/compiler/ir/graph/parser.ts):

```ts
function dependencyOrder(ops: readonly OpRecord[]): OpRecord[] {
  const producer = new Map<string, OpRecord>();
  for (const op of ops) {
    for (const name of op.resultNames) producer.set(name, op);
  }
  const ordered: OpRecord[] = [];
  const state = new Map<OpRecord, number>();
  const visit = (op: OpRecord): void => {
    const seen = state.get(op);
    if (seen === 2) return;
    if (seen === 1) throw new IRParseError(`'${op.opName}' participates in a value dependency cycle`, op.line.no);
```

A depth-first topological sort over records, with the grey/black colouring that detects a cycle — the same algorithm as the verifier's `detectCycles`, arriving at the same error message from the other side.

Then `Materializer.fillBlock` ([`parser.ts:444`](../../../src/compiler/ir/graph/parser.ts)) does the thing this chapter has been building toward:

```ts
  fillBlock(block: Block, record: BlockRecord): void {
    const built = new Map<OpRecord, Operation>();
    for (const record0 of dependencyOrder(record.ops)) {
      built.set(record0, this.buildOp(record0));
    }
    for (const op of record.ops) block.pushOp(built.get(op) as Operation);
  }
```

**Build in dependency order; insert in textual order.** Two loops, and the split between them is Theorem 8.4 written as four lines of code. Construction is constrained by dataflow; placement is not, so the block ends up holding the operations in the order the file listed them, whatever that order was.

If that looks familiar it is because you read it in Chapter 9, in `cloneRegion`, doing exactly the same thing for exactly the same reason. Two components, arrived at independently, forced into the same shape by the nature of the representation. That is usually a sign the representation is the real thing and the code is downstream of it.

## 13.5 Lab 1 — Does it actually round-trip?

```bash
node docs/part2/ch13-ir-as-text/labs/01-round-trip.mjs
```

The lab prints four real traced models, parses each printout, prints it again, and compares character by character:

```
two-layer MLP                 14 lines   round-trips: true
a scan region                 14 lines   round-trips: true
linear + relu + layernorm     12 lines   round-trips: true
dynamic batch dimension       14 lines   round-trips: true

all round-trip: true
```

Nested regions, multi-result operations, dynamic dimensions, float attributes printed at full precision — all survive. If a case ever fails, the lab prints the first differing line from each version side by side, which is the shape a round-trip test should have: the diff, not just the verdict.

Definition 13.1 is not a property you get for free. It is a property that holds because someone sorted the attributes, because `-inf` prints as a word, because dense arrays carry their dtype, and because the printer emits result types the parser then reads instead of re-inferring. Each of those is one line, and each of them is a round-trip failure if omitted. In this codebase the printer was made lossless *in order to* make the parser possible; they are one piece of work, not two.

### What "lossless" does not mean

Read Definition 13.1 literally, because it is stated as precisely as it is for a reason. It says `P(Q(P(m))) = P(m)` — *the text is stable*. Three stronger properties do **not** follow, and conflating them with round-tripping is how people get surprised.

**It is not object identity.** `Q(P(m))` is a fresh module. Nothing in it is the same object as anything in `m`, so a `WeakMap` keyed on operations, a cached analysis, or a held reference to a `Value` all point at the old graph after a round trip.

**It is not preservation of everything a module carries.** The printed form covers what the *text format* models: operations, operands, results, types, attributes, regions, signatures. A module in memory carries more, and the extra is silently dropped:

| Field | What it is | Survives a round trip? |
|---|---|---|
| `Value.id` | global allocation counter | no — and this is deliberate, §13.3 |
| `Value.symbolicShape` | the `SymInt` structure behind a `?` | **no** — the type prints as `?`, and the symbol's identity goes with it |
| `GraphFunction._partitionTarget` | which device a partitioned function is for | no |
| `_version` | the mutation counter of Chapter 9 | no — the fresh module starts over |

The second row is the one with teeth. Chapter 10 §10.3 made the case that `SymInt` exists precisely so that two dimensions both printed `?` can be *known equal* — that is what lets the compiler fuse them or prove an index in bounds. That knowledge lives in `Value.symbolicShape`, which the grammar has no syntax for. So a module round-tripped through text still round-trips by Definition 13.1, and has forgotten that its two dynamic dimensions were the same `n`. Printing is faithful to the *program*; it is not a serialization format for the compiler's working state, and Chapter 62's dynamic-shape machinery must not be moved across a round trip.

**It is not validity.** §13.6 is about that one.

## 13.6 Lab 2 — Editing a program in a text editor

The point of a real format is that you can work on the text.

```bash
node docs/part2/ch13-ir-as-text/labs/02-edit-by-hand.mjs
```

The lab takes the traced running example, changes one line with an ordinary string replacement — swapping the ReLU's `maximum(%7, %9)` for `tanh(%7)` — and parses the result:

```
=== after changing the activation, in a text editor ===
    %7 = "tera.add"(%6, %2) : (tensor<2x8xf32>, tensor<8xf32>) -> tensor<2x8xf32>
    %8 = tera.constant dense<0.0> : tensor<f32>
    %9 = tera.broadcast_in_dim %8 {broadcast_dimensions = array<i64>} : tensor<f32> -> tensor<2x8xf32>
    %10 = "tera.tanh"(%7) : (tensor<2x8xf32>) -> tensor<2x8xf32>
```

```
operations now: transpose, dot, add, constant, broadcast_in_dim, tanh, transpose, dot, add, return
the broadcast constant is still there, with 0 users
```

A different network, in one line of editing, with no model class and no tracing. Note what the parser did *not* do: it did not remove the now-unused broadcast. Parsing is not optimization — you get exactly the program you wrote, dead operations included, and DCE is a pass you choose to run.

The lab then writes a module from scratch, with no model behind it at all:

```
module @handwritten {
  func.func @handwritten(%0: tensor<3x4xf32>, %1: tensor<4x2xf32>) -> (tensor<3x2xf32>) {
    %2 = tera.dot %0, %1 {lhs_batch = array<i64>, lhs_contracting = array<i64: 1>, rhs_batch = array<i64>, rhs_contracting = array<i64: 0>} : (tensor<3x4xf32>, tensor<4x2xf32>) -> tensor<3x2xf32>
    %3 = "tera.tanh"(%2) : (tensor<3x2xf32>) -> tensor<3x2xf32>
    return %3 : tensor<3x2xf32>
  }
}
```

```
parsed: 3 operations, 2 inputs, 1 output
the dot contracts lhs dim 1 against rhs dim 0
round-trips: true
```

That is the capability this chapter is about. **You can now write IR.** Every pass in Parts III through V takes a module; you can hand it one that is four lines long and contains precisely the situation you are debugging, rather than a hundred lines of transformer you have to squint at.

And the edits that do not make sense are refused where you make them:

```
=== an edit that does not typecheck as text ===
IRParseError: line 8: use of undefined value '%99'
```

with the line number of the edit. That is Chapter 12's invariant 1, enforced at read time.

**Try this.** Take the handwritten module and change `tensor<3x2xf32>` on the `dot` result to `tensor<3x9xf32>`. The parser accepts it — the type is data, and the parser does not run type inference. Then recall Chapter 12's table: this is invariant 3, caught by `verifyModule` at a phase boundary and by nothing you can call yourself. Writing IR by hand is a sharp tool, and the parser is not the thing that keeps you honest.

## 13.7 Traps and limits

- **Indentation is significant, and the width is a parameter.** `toLines` divides leading spaces by `indentWidth` ([`parser.ts:794`](../../../src/compiler/ir/graph/parser.ts)). Text indented with tabs, or with three spaces, will not parse as the nesting you see. When hand-writing, copy a printout and edit it rather than typing from scratch.
- **The parser is more permissive than the verifier, and "it parsed" is not "it is valid".** The parser enforces the SSA core and the syntax: every name resolves, no name is bound twice, no dependency cycle. It does not check arity, unknown operations, types, traits, or **operation ordering**. §13.6's "Try this" is one demonstration; Chapter 8's Lab 2 is the sharper one, because it looks like a success. Reversing a printed function moves `return` to the top, the parser accepts it happily — it builds operations in dependency order rather than reading order, so a terminator appearing first is not a problem *for parsing* — and the resulting module then fails `verifyModule` with `a terminator must be the last operation in its block`. The parser will construct block structures the verifier rejects, which is the right division of labour (the parser's job is to read what the printer wrote) but a trap if you take acceptance as a verdict. Anything hand-written or hand-permuted should go through `verifyModule` before you trust it.
- **A function may have only one top-level block, and the entry block may not be labelled** ([`parser.ts:839`](../../../src/compiler/ir/graph/parser.ts) and [`parser.ts:842`](../../../src/compiler/ir/graph/parser.ts)). The data structure permits more; the format does not, because nothing generates it.
- **Round-tripping is a property of what the printer prints, not of the object.** `Q(P(m))` is a *different module* from `m` — new objects, new internal ids, and any field the printer does not emit is gone. Attributes the printer cannot represent fall through `formatAttrValue`'s final `String(val)` and come back as a string. If you attach an exotic attribute to an operation, check that it survives before relying on the text.
- **`%n` numbering is per function and per print.** Two printouts of the same function agree; a printout of a function and a printout of the module containing it also agree, because `printFunction` resets the counter. But value labels are not stable identifiers across edits — insert one operation near the top and every number below it shifts, which makes textual diffs of IR noisier than they look.

## 13.8 Read the tests

- [`tests/compiler/ir/graph/parser.test.js`](../../../tests/compiler/ir/graph/parser.test.js) — round-tripping, the two-phase ordering, every error the parser can raise, and the syntax corners: dense attributes, non-finite floats, nested regions, symbolic dimensions.
- [`tests/compiler/ir/graph/verifier.test.js`](../../../tests/compiler/ir/graph/verifier.test.js) — worth re-reading now, because many of its cases build their input by parsing text. That is §13.1's argument in practice.

---

**Part II ends here.** You have the representation: values and the single-assignment rule that makes them analysable (Chapter 8), the six nouns and the region that lets an operation hold a program (Chapter 9), the types every value carries and the two orders that govern them (Chapter 10), the registry that keeps operation knowledge out of passes (Chapter 11), the invariants that make a graph valid and the three places they are checked (Chapter 12), and a textual form you can read, write and test against (Chapter 13).

**Next:** [Part III — The transformation infrastructure](../../part3/README.md), which asks what a pass is, how a sequence of them is run, and how an analysis computed over one version of the IR is prevented from being used against another.
