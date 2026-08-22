# Chapter 8 — SSA and dataflow

Chapter 2 printed a graph and asked you to notice something in passing: every line names a value, and no value is ever named twice. This chapter is about why that restriction is there, what it buys, and the surprising thing it implies — that the order of the lines you were reading carries no meaning whatsoever.

## 8.1 The problem: what does a compiler need to know about a variable?

Take a fragment of ordinary code, in any language:

```js
let x = a + b;
x = x * 2;
const y = x + 1;
```

Now ask the question every optimization has to ask: **at the point where `y` is computed, what is `x`?**

You cannot answer it by looking at `x`. You have to look at the *program point*. There are two assignments to `x`, and which one is in force depends on where you are standing. So every analysis has to carry a notion of position, and every fact it computes is a fact about a (variable, position) pair rather than about a variable. Add a branch and it gets worse: after an `if`, `x` might be either of two things, and the analysis now needs to merge the possibilities.

That machinery — reaching definitions, dominance frontiers, φ-functions — is a large part of a classical compiler, and it exists to answer one question: *which assignment does this use refer to?*

The trick is to make the question unaskable.

## 8.2 Intuition: give every assignment its own name

Rewrite the fragment so that no name is ever reused:

```js
const x1 = a + b;
const x2 = x1 * 2;
const y  = x2 + 1;
```

Nothing about the computation changed. But the question "what is `x` here?" no longer arises, because there is no `x` — there is `x1` and there is `x2`, and each of them means one thing everywhere, forever. A use names its definition directly.

This is *static single assignment* form, and it is the single most consequential representational decision in modern compilers. In a tensor IR it is not even a transformation you apply: it is how the IR is built in the first place. When you write `a.add(b)` under a tracer, the result is a fresh value that never existed before and will never be assigned again.

> **Definition 8.1 (SSA form; after Cytron et al., 1991).** **(classical)** A program is in *static single assignment* form when every value is defined by exactly one instruction, and every use of a value refers unambiguously to that definition.

> **Definition 8.2 (Use-def and def-use graphs).** **(classical)** The *use-def graph* of a program in SSA form is the directed graph whose nodes are operations and whose edges run **from each operation to the operations producing its operands** — that is, consumer → producer, pointing backwards against the flow of data. The *def-use graph* is its reverse: an edge from each operation to the operations consuming its results, producer → consumer, pointing along the flow of data. The two carry the same information and are read in opposite directions, so which one a compiler stores is an engineering choice — and this one stores the second, since a `Value` holds a list of its users.

Keep the arrows straight, because the direction decides which topological order you mean. A topological order of the **def-use** graph lists producers before consumers: that is a valid *execution* order. A topological order of the **use-def** graph lists consumers before producers, which is the reverse of an execution order — useful when propagating information backwards, as reverse-mode differentiation does in Part V, and wrong if you were trying to schedule. Whenever this book says "topological order" without qualification it means the executable one, over the def-use graph.

## 8.3 In mlfw: a value is a node with a list of users

Here is the whole of what a value is — [`src/compiler/ir/graph/value.ts:26`](../../../src/compiler/ir/graph/value.ts):

```ts
export class Value {
  type: IRType;
  declare symbolicShape?: Shape;
  definingOp: Operation | null;
  resultIndex: number;
  id: number;
  private _useHead: UseLink | null;
  private _useTail: UseLink | null;
  private _useCount: number;
```

Four of those fields are the SSA property made concrete.

`definingOp` is the *single* producer — the "single assignment" of the name. It is a field, not a search: a value knows what made it, and there is exactly one answer because a value is created by an operation's constructor and handed out once.

`resultIndex` says which of that operation's results this is, because an operation may produce several (`scan` produces a carry and a stack; you saw it in Chapter 5).

`_useHead` / `_useTail` / `_useCount` are the *other* direction: the list of every place this value is consumed. Note what this is not — it is not recomputed by scanning the function. It is maintained incrementally, as an intrusive doubly-linked list of `UseLink` nodes, one per (consumer, operand position) pair ([`value.ts:12`](../../../src/compiler/ir/graph/value.ts)):

```ts
export class UseLink {
  user: Operation;
  operandIndex: number;
  prev: UseLink | null;
  next: UseLink | null;
}
```

The list is kept current by the operation constructor, which is where an edge in the use-def graph is actually born ([`operation.ts:50`](../../../src/compiler/ir/graph/operation.ts)):

```ts
    this.operands = new Array(operands.length);
    this._operandLinks = new Array(operands.length);
    for (let i = 0; i < operands.length; i++) {
      this.operands[i] = operands[i];
      const link = new UseLink(this, i);
      operands[i].addUse(link);
      this._operandLinks[i] = link;
    }
```

Constructing an operation registers it as a user of each of its operands. From that moment, `value.getUsers()` is a list traversal, not a graph search.

Both directions, for the single value `%6 = dot(%0, %5)` that `add` then consumes:

```
                      the "single assignment": one field, one answer
                      ..............................................
                      :                                            :
   Operation 'dot'    :                                            v
   +---------------+  :                                   +--------------------+
   | operands[0] --+--+---> %0                            |     Value  %6      |
   | operands[1] --+----> %5                              |--------------------|
   | results[0] ---+------------------------------------->| definingOp  -> dot |
   +---------------+                                      | resultIndex = 0    |
                                                          | _useHead ------+   |
                                                          | _useCount = 1  |   |
   Operation 'add'                                        +----------------|---+
   +---------------+                                                       |
   | operands[0] --+----> %6                                               v
   | operands[1] --+----> %2                            UseLink { user: add, operandIndex: 0 }
   | _operandLinks +--------------------------------------> ^     prev: null, next: null
   +---------------+       the same object, reachable       |
                           from both ends                   +-- the list `_useHead` walks
```

`definingOp` is one pointer, so "what produces `%6`?" is a field read. `_useHead` is a list exactly as long as the number of consumers, so "who consumes `%6`?" costs one walk of that list. Neither question touches the enclosing block, which is the content of the next lemma.

This is what SSA buys, stated as a cost:

> **Lemma 8.3 (Cost of the two questions).** **(invariant)** In this representation, "what produces this value?" is O(1), and "who consumes this value?" is O(k) in the number of consumers — neither depends on the size of the function.
>
> *Proof sketch.* The first is a field read. The second walks a list whose length is exactly the number of uses, maintained by `addUse` and `removeUse` at O(1) each. Neither traverses the enclosing block. ∎

In a non-SSA representation both questions require a walk over the program, repeated after every edit. That difference is why the passes in Part IV can afford to ask them constantly.

The one operation that exercises the whole structure is `replaceAllUsesWith` — the primitive every rewriting pass is built on ([`value.ts:96`](../../../src/compiler/ir/graph/value.ts)):

```ts
  replaceAllUsesWith(newValue: Value): void {
    if (this === newValue) return;
    const hadUses = this._useHead !== null;
    let cur = this._useHead;
    while (cur) {
      cur.user.operands[cur.operandIndex] = newValue;
      cur = cur.next;
    }
```

"Everywhere this value was used, use that one instead." Because the uses are a list, this is a walk over the consumers followed by a splice of two linked lists — not a scan of the function. When Chapter 11's canonicalizer folds a `transpose` into a `dot`, this is the call that does the rewiring.

### The one value with no producer

If every value has a defining operation, where does the first one come from? From a *block argument* — the exception that proves the rule ([`value.ts:130`](../../../src/compiler/ir/graph/value.ts)):

```ts
export class BlockArgument extends Value {
  ownerBlock: Block | null;
  argIndex: number;

  constructor(type: IRType, ownerBlock: Block | null, argIndex: number) {
    super(type, null, 0);
```

`super(type, null, 0)` — defining operation `null`. A block argument is a value whose producer is the act of entering the block. Function parameters are the entry block's arguments; the `^bb(...)` line inside a `fusion` or `scan` region declares that region's. Chapter 9 takes them properly.

## 8.4 Lab 1 — Reading the use-def graph

```bash
node docs/part2/ch08-ssa-and-dataflow/labs/01-use-def.mjs
```

The lab traces the running example and then asks every value the two questions from Lemma 8.3:

```
value      defined by            used by
%0         function argument     dot
%1         function argument     transpose
%2         function argument     add
%3         function argument     transpose
%4         function argument     add
%5         transpose             dot
%6         dot                   add
%7         add                   maximum
%8         constant              broadcast_in_dim
%9         broadcast_in_dim      maximum
%10        maximum               dot
%11        transpose             dot
%12        dot                   add
%13        add                   return

values claimed by more than one producer: 0
```

That table is both graphs at once, one per column: `defined by` is the use-def edge out of each value, `used by` is the def-use edge. Read the def-use direction and the program's shape appears — `%1 → transpose → %5 → dot`, and separately `%3 → transpose → %11 → dot`, the two `Linear` layers, structurally identical and independent. The second half of the lab reads the other direction, following `definingOp` back from the return.

That backwards walk, run on Chapter 5's dead-branch model:

```
the same walk on a model with a branch nobody reads:
  reachable from the return : %0 %3 %4 %12 %13 %14
  not reachable             : %1 %2 %5 %6 %7 %8 %9 %10 %11
  operations whose every result is unreachable: transpose, dot, add, constant, broadcast_in_dim, maximum, tanh
```

Nine of fifteen values cannot affect the output. That reachability walk is the whole of dead code elimination in a side-effect-free graph — twelve lines in the lab, and Chapter 19 explains what has to be added once side effects exist.

**Try this.** Change `getUsers()` to `useCount` in the first loop and print counts instead of names. Every value in this graph has exactly one user — there is not a single shared intermediate, which is why the whole program fuses so readily. Then trace a model that uses one activation twice — `const h = x.relu(); return h.add(h);` — and watch a value acquire two users. Chapter 24 is about what fusion does when it meets one.

## 8.5 The consequence nobody expects: order means nothing

Here is a claim that sounds wrong the first time you read it.

> **Theorem 8.4 (Textual order carries no semantics).** **(stated here)** Let *B* be a block of side-effect-free operations in SSA form whose use-def graph is acyclic, and let *t* be its terminator. Then any permutation of *B* that (a) is a topological order of the def-use graph — every producer before its consumers — and (b) leaves *t* last, denotes the same computation, and at least one such permutation exists.
>
> *Proof sketch.* Each operation's result is a function of its operands alone; no operation reads or writes state that another can observe. So the value of every result is determined by the def-use graph, not by position, and (a) guarantees each operand is defined before it is read. Existence is the standard fact that a finite DAG has a topological order, together with the observation that the terminator has no results and therefore no consumers, so it is free to be placed last in any such order. ∎

Three conditions are doing real work there, and it is worth being clear about what each one costs.

*Side-effect-free* is why the operations cannot communicate except through values — Chapter 19 returns to the operations for which this fails.

*Acyclic* is what makes a topological order exist at all, and it is checked: [`verifier.ts:136`](../../../src/compiler/ir/graph/verifier.ts) contains a `detectCycles` pass over every block, reporting `participates in a value dependency cycle`.

*Terminator last* is the condition that is easy to forget, and the reason it is stated explicitly is that the very next lab appears to violate it. A block's terminator is not just conventionally final: it is required to be, by a trait the verifier enforces. A permutation that satisfies (a) but not (b) — dataflow-consistent, terminator moved — is a structure the parser will happily build and the verifier will reject. §8.6 runs exactly that case.

The theorem is not a curiosity. It is the licence for everything later in the book. When Chapter 24's fusion engine reorders operations to bring a producer next to its consumer, when Chapter 52's memory scheduler moves independent work to shrink peak memory, when the lowering pass walks the graph in an order the user never wrote — none of them needs to prove that reordering is safe, because in this representation the order was never carrying information in the first place.

> **Counterexample 8.5.** The theorem's first condition is not decorative. Give two operations a shared mutable buffer — a `scatter` writing into a tensor another operation reads — and their relative order determines the answer. This is exactly why `SideEffectKind` exists in the op registry ([`op_registry.ts:7`](../../../src/compiler/ir/graph/op_registry.ts)) and why Chapter 19 finds that "has no users" is not sufficient grounds for deletion.

## 8.6 Lab 2 — Printing a program backwards

If Theorem 8.4 is true, then a printed function should survive having its lines shuffled. The lab does the most aggressive shuffle available: it reverses them.

```bash
node docs/part2/ch08-ssa-and-dataflow/labs/02-order-carries-nothing.mjs
```

```
=== the same function, printed bottom to top ===
module @traced {
  func @traced(%0: tensor<2x2xf32>, %1: tensor<8x2xf32>, %2: tensor<8xf32>, %3: tensor<1x8xf32>, %4: tensor<1xf32>) -> (tensor<2x1xf32>) {
    return(%13)
    %13 = add(%12, %4) : tensor<2x1xf32>
    %12 = dot(%10, %11) {lhs_batch = [], lhs_contracting = [1], rhs_batch = [], rhs_contracting = [0]} : tensor<2x1xf32>
    %11 = transpose(%3) {permutation = [1, 0]} : tensor<8x1xf32>
    %10 = maximum(%7, %9) : tensor<2x8xf32>
    %9 = broadcast_in_dim(%8) {broadcast_dimensions = [], result_shape = [2, 8]} : tensor<2x8xf32>
    %8 = constant() {tensor_type = tensor<xf32>, value = 0} : tensor<xf32>
    %7 = add(%6, %2) : tensor<2x8xf32>
    %6 = dot(%0, %5) {lhs_batch = [], lhs_contracting = [1], rhs_batch = [], rhs_contracting = [0]} : tensor<2x8xf32>
    %5 = transpose(%1) {permutation = [1, 0]} : tensor<2x8xf32>
  }
}

the parser accepted it.
```

Every single line now uses values that are defined *below* it. `return(%13)` comes first. Read as a sequence of instructions this is nonsense. Read as a set of edges it is the same graph, and the parser has no difficulty with it, because it builds operations in dependency order rather than in reading order — [`parser.ts:403`](../../../src/compiler/ir/graph/parser.ts), which Chapter 13 walks through.

> **This module is not valid IR, and that is Theorem 8.4's condition (b) showing its teeth.** Reversing the block moved the terminator to the top, and a terminator is required to be last. Run the result through the verifier and it says so:
>
> ```
> trait 'terminator': a terminator must be the last operation in its block
> Missing return op
> ```
>
> The same module before reversing verifies clean. So the lab demonstrates precisely as much as the theorem claims and no more: **the dataflow is order-independent, the block structure is not.** Parsing successfully is not the same as being valid — the parser enforces the SSA core and the syntax, the verifier enforces the rest, and Chapter 12 is about the gap between them. If you want a permutation that is both dataflow-consistent and *legal*, shuffle the non-terminator operations and leave the terminator where it is.

The lab then proves the two graphs are the same by traversing each from its return value:

```
dataflow order, original : transpose -> dot -> add -> constant -> broadcast_in_dim -> maximum -> transpose -> dot -> add
dataflow order, reversed : transpose -> dot -> add -> constant -> broadcast_in_dim -> maximum -> transpose -> dot -> add
identical                : true
```

Same program. And printing the reversed module back out shows what *did* change:

```
    return(%5)
    %5 = add(%6, %4) : tensor<2x1xf32>
    %6 = dot(%8, %7) {lhs_batch = [], lhs_contracting = [1], rhs_batch = [], rhs_contracting = [0]} : tensor<2x1xf32>
    ...
```

The numbers moved. `%13` became `%5`. That is because `%n` is not a property of a value at all — it is assigned by the printer, in the order it encounters values while walking the text ([`printer.ts:160`](../../../src/compiler/ir/graph/printer.ts)):

```ts
  _nameValue(value: Value): string {
    if (this.valueNames.has(value)) return this.valueNames.get(value) as string;
    const name = `%${this._nextValueId++}`;
```

Values do carry an `id` internally, allocated from a global counter at construction. But nothing the reader sees uses it. The `%5` in a printout is a label invented for the printout — which is the right design, because a stable global id in the text would make two structurally identical functions print differently depending on what had been built before them.

## 8.7 The one place order does matter

The lab ends with a wrinkle, and it is worth having:

```
one accessor does care about textual order:
  original  getReturnOp() -> found   findOp(name === 'return') -> found
  reversed  getReturnOp() -> null   findOp(name === 'return') -> found
```

[`function.ts:110`](../../../src/compiler/ir/graph/function.ts):

```ts
  getReturnOp(): Operation | null {
    const last = this.entryBlock.lastOp;
    if (last && last.opName === 'return') return last;
    return null;
  }
```

`getReturnOp` finds the return by looking at the *last* operation, not by looking for an operation named `return`. On the reversed module it returns `null`, and every caller that trusts it — `getReturnValues`, the function verifier's output check — silently sees a function with no return.

Is that a bug? Not quite. `return` is declared with the `TERMINATOR` trait, and terminators are required to be last: the trait verifier at [`trait_verifier.ts:151`](../../../src/compiler/ir/graph/trait_verifier.ts) enforces exactly that, and the block verifier enforces it again for region blocks. So in any IR that has passed verification, `lastOp` *is* the terminator and the accessor is correct.

What the reversed module shows is that the parser will build IR the verifier would reject. The two components disagree about how much they check, which is Chapter 12's subject. Keep the distinction: **terminator-last is a real invariant of valid IR; the order of everything else is not.**

## 8.8 Traps and limits

- **SSA is about values, not memory.** Every *value* is written once. The buffers those values eventually live in are written many times — Chapter 49 onward is entirely about reusing them. SSA at the graph level says nothing about storage, which is precisely why storage decisions can be postponed to Part IX.
- **There are no φ-functions here.** A classical SSA compiler needs them where control flow merges. This IR does not, because control flow is not represented as branching between blocks — it is an operation with regions (`if`, `while`, `scan`), and the merge is the operation's result. Chapter 9 explains the trade, and Chapter 31 explains what it costs when you differentiate one.
- **Acyclicity is a property of the graph, not of the model.** A recurrent network is a cycle in *time*, not in the use-def graph: `scan` carries the loop inside a region, and the graph containing it stays acyclic. If a cycle ever does appear in the graph, it is a compiler bug, and the verifier says so in those words.
- **Order is free within a block, not across scopes.** Theorem 8.4 is stated for one block. Operations inside a region cannot be interleaved with operations outside it, and Chapter 12's scope rule is what makes that precise.
- **A dead value is not always deletable.** §8.4's reachability walk finds what cannot affect the output. Removing it is only sound for operations without side effects, and Chapter 19 is about the gap between those two statements.

## 8.9 Read the tests

- [`tests/compiler/ir/graph/operation.test.js`](../../../tests/compiler/ir/graph/operation.test.js) — use lists, `replaceAllUsesWith`, `erase` refusing to drop an operation whose results still have users.
- [`tests/compiler/ir/graph/block-invariants.test.js`](../../../tests/compiler/ir/graph/block-invariants.test.js) — what the intrusive operation list guarantees under insertion and removal.
- [`tests/compiler/ir/graph/verifier.test.js`](../../../tests/compiler/ir/graph/verifier.test.js) — cycle detection, and the messages it produces.

---

**Next:** [Chapter 9 — Value, Operation, Block, Region, Function, Module](../ch09-object-model/README.md), which puts the four containers around the two nouns you now know.
