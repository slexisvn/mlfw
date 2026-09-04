# Chapter 9 — Value, Operation, Block, Region, Function, Module

Chapter 8 gave you two nouns: a value, and the operation that produces it. That is enough to describe a straight line of arithmetic and nothing else. This chapter adds the four containers that turn a list of operations into a program — and one of them, the region, is the reason this IR can represent a loop without unrolling it.

## 9.1 The problem: a flat list is not enough

Take the traced graph from Chapter 2. It is a flat sequence of operations, and for that program a flat sequence is exactly right.

Now ask for three things a real compiler needs.

**Somewhere to put a group.** Chapter 3 watched `add` and `maximum` disappear into a single `fusion` operation. Where did they go? They cannot be deleted — they are the fusion's definition. They cannot stay in the outer list — then nothing distinguishes "inside the fusion" from "next to it". They need a container that belongs to the `fusion` operation.

**A loop that is not unrolled.** Chapter 5's `scan` runs a body four times, and the body appears in the IR once. A flat list can only express that by writing the body out four times, which makes the graph grow with sequence length and makes differentiating it four times more work than it should be.

**More than one function.** A module holds a forward function and, after Part V, a backward one; after partitioning it may hold one function per device.

Four containers answer these: **Block**, **Region**, **Function**, **Module**. With `Value` and `Operation` from Chapter 8 they are the entire object model. There is nothing else.

## 9.2 Intuition: an alternating hierarchy

The containment rule is short enough to memorize, and it alternates:

```
Module
  └── Function
        └── Region           a function's body is a region
              └── Block      an ordered list of operations
                    └── Operation
                          └── Region      an operation may contain regions
                                └── Block
                                      └── Operation   ... and so on
```

An operation lives in a block; a block lives in a region; a region belongs either to a function or to an operation. That last clause is the recursive one, and it is what makes the structure interesting: **an operation can contain a program.**

This is MLIR's model, deliberately. If you have read MLIR's `Operation` / `Region` / `Block` documentation, the shape here will be familiar; the differences are that this IR has no branching between blocks, so a region almost always holds exactly one block, and that it has no operation names with dialect prefixes.

## 9.3 In mlfw: the four containers

### Block — an ordered list, with arguments

[`block.ts:6`](../../../src/compiler/ir/graph/block.ts):

```ts
export class Block {
  declare _parentFunction?: GraphFunction;
  parentRegion: Region | null;
  _head: Operation | null;
  _tail: Operation | null;
  _size: number;
  arguments: BlockArgument[];
```

Two things to notice.

The operations are an **intrusive doubly-linked list** — `_head`, `_tail`, and `_prev`/`_next` fields living on the operations themselves — not an array. That choice is made for the passes: a rewriting pass inserts and removes operations constantly, in the middle, while iterating. On an array each of those is O(n) and invalidates indices; here `insertBefore`, `insertAfter` and `removeOp` are all O(1) pointer surgery, and the `ops()` generator reads `cur._next` *before* yielding ([`block.ts:151`](../../../src/compiler/ir/graph/block.ts)) so that a pass may erase the operation it is currently looking at.

The `arguments` are the block's inputs: `BlockArgument` values, the producerless values of Chapter 8. For a function's entry block they are the parameters. For a region's block they are what the enclosing operation feeds in.

### Region — a list of blocks, owned by an operation or a function

[`block.ts:184`](../../../src/compiler/ir/graph/block.ts):

```ts
export class Region {
  parentOp: Operation | null;
  blocks: Block[];
```

A region is thin — it exists to be the thing an operation owns. `parentOp` is the back-pointer, and it is how the ownership chain is walked upward. Follow it far enough and you reach a block whose `_parentFunction` is set, which is what [`block.ts:31`](../../../src/compiler/ir/graph/block.ts) does:

```ts
  _owningFunction(): GraphFunction | null {
    let block: Block | null = this;
    while (block) {
      if (block._parentFunction) return block._parentFunction;
      const op: Operation | null = block.parentOp;
      block = op ? op.parentBlock : null;
    }
    return null;
  }
```

Any node, at any nesting depth, can find the function it belongs to. That matters for a reason that is not obvious, and §9.4 is about it: **mutation tracking**.

### Function — a signature and one region

[`function.ts:31`](../../../src/compiler/ir/graph/function.ts):

```ts
export class GraphFunction {
  declare _module?: GraphModule;
  declare _partitionTarget?: string;
  name: string;
  inputTypes: readonly IRType[];
  outputTypes: readonly IRType[];
  body: Region;
  _version: number;

  constructor(name: string, inputTypes: readonly IRType[], outputTypes: readonly IRType[]) {
    this.name = name;
    this.inputTypes = Object.freeze([...inputTypes]);
    this.outputTypes = Object.freeze([...outputTypes]);
    this.body = new Region();
    const entryBlock = new Block(inputTypes);
    entryBlock._parentFunction = this;
```

The signature is `Object.freeze`d. That is an invariant expressed in the type system rather than in a comment: a pass may rewrite everything inside a function, but it may not change what the function *is* to its callers. When Chapter 5's dead-branch model kept two unused parameters, this is why — the interface is not a pass's to edit.

Note also that the entry block is constructed *from* `inputTypes`, so the parameters and the block arguments are the same objects by construction. The verifier checks the correspondence anyway ([`verifier.ts:47`](../../../src/compiler/ir/graph/verifier.ts)), because a pass could add an argument later.

### Module — a map from name to function

[`module.ts:4`](../../../src/compiler/ir/graph/module.ts):

```ts
export class GraphModule {
  name: string;
  private _functions: Map<string, GraphFunction>;
  _version: number;
```

The module is the least interesting container and the shortest file, which is as it should be. Functions are keyed by name, which is what makes a `call` operation resolvable and what lets Chapter 26's partitioner split one function into several and keep them together.

## 9.4 Every edit through the API bumps a version number

[`block.ts:41`](../../../src/compiler/ir/graph/block.ts):

```ts
  _notifyMutation(): void {
    const fn = this._owningFunction();
    if (fn) fn.bumpVersion();
  }
```

`pushOp`, `insertBefore`, `insertAfter`, `removeOp`, `addArgument`, `removeArguments` and `replaceOperand` all call it. So does `replaceAllUsesWith`. The result is that a function carries a counter that changes whenever the *structure* inside it changes, at any depth — which operations exist, and which values they consume.

That counter is the foundation of Chapter 16's analysis caching: an analysis result computed at version 7 is known to be stale at version 8, without comparing anything.

**Attributes count as edits.** An attribute is not decoration: a `dot`'s `lhs_contracting` decides which axes are summed, and a comparison's `direction` decides whether the test is `<` or `>`. Changing one changes what the program computes, and passes do change them in place — [`patterns.ts:162`](../../../src/compiler/ir/graph/patterns.ts) inverts a comparison's `direction` during canonicalization, [`partition_pass.ts:59`](../../../src/compiler/passes/partition/partition_pass.ts) stamps `partition_id` onto existing operations. So the attribute mutators notify too ([`operation.ts:88`](../../../src/compiler/ir/graph/operation.ts)):

```ts
  setAttr(name: string, value: AttrValue): void {
    this.attributes.set(name, value);
    if (this.parentBlock) this.parentBlock._notifyMutation();
  }

  removeAttr(name: string): boolean {
    const removed = this.attributes.delete(name);
    if (removed && this.parentBlock) this.parentBlock._notifyMutation();
    return removed;
  }
```

Two details are worth copying. The notification is conditional on `parentBlock`, because an operation under construction is not yet in a function and has no version to bump. And `removeAttr` notifies only when it removed something — deleting an absent key is not a mutation, and reporting it as one would invalidate the cache on every miss.

So the mechanism is what this section's heading claims, with one word doing the work: every edit that goes *through the API* bumps the version. What it cannot see is an edit made *around* the API, which is §9.9's subject.

## 9.5 Lab 1 — The six nouns, printed

```bash
node docs/part2/ch09-object-model/labs/01-the-six-nouns.mjs
```

The lab traces the `scan` model from Chapter 5 and walks the containment tree, printing what each node owns:

```
=== the containment tree ===
Module 'traced'  1 function(s)
  Function 'traced'  2 in, 1 out
    Region  1 block(s)
      Block  2 argument(s), 2 operation(s)
        Operation 'scan'  2 operands, 2 results, 2 attributes, 1 regions
          Region  1 block(s)
            Block  2 argument(s), 5 operation(s)
              Operation 'constant'  0 operands, 1 results, 2 attributes, 0 regions
              Operation 'mul'  2 operands, 1 results, 0 attributes, 0 regions
              Operation 'add'  2 operands, 1 results, 0 attributes, 0 regions
              Operation 'tanh'  1 operands, 1 results, 0 attributes, 0 regions
              Operation 'yield'  2 operands, 0 results, 0 attributes, 0 regions
        Operation 'return'  1 operands, 0 results, 0 attributes, 0 regions
```

Read the indentation and the alternation from §9.2 is right there: Function → Region → Block → Operation → Region → Block → Operation. The `scan` operation *contains* a five-operation program, and the outer block contains exactly two operations.

That last fact is worth dwelling on, because it changes how you must count:

```
=== two ways to count ===
  ops() walks the top-level block only : 2
  opsRecursive() descends into regions : 7
  blocksRecursive()                    : 2
```

Two, or seven, depending on which question you asked. This distinction is a standing hazard: a pass that uses `ops()` when it meant `opsRecursive()` will silently skip everything inside every fusion and every loop. The two generators are [`function.ts:57`](../../../src/compiler/ir/graph/function.ts) and [`function.ts:63`](../../../src/compiler/ir/graph/function.ts):

```ts
  *opsRecursive(): Generator<Operation, void, undefined> {
    for (const op of this.ops()) {
      yield op;
      yield* opsInRegions(op);
    }
  }
```

The op counts you watched in Chapter 3's pass log — `7 ops -> 6 ops` — are top-level counts. Fusion does not delete two operations; it moves them one level down, and the counter cannot see them any more. Chapter 3 warned that op count measures size, not effect. Now you know exactly which size.

Finally, the lab walks *upward* from an operation buried inside the loop body:

```
=== who owns whom ===
  op 'tanh'  in  block(2 args)  in  op 'scan'  in  block(2 args)  in  function 'traced'
```

Every node knows its container, all the way to the top. That is `parentBlock` and `parentRegion.parentOp`, the same chain `_owningFunction` follows.

## 9.6 What a region is for

Regions do three jobs in this compiler, and it is worth naming them separately because they look unrelated until you see the mechanism is the same.

**They hold a fused group.** `fusion` carries a region containing the operations that were merged. Everything before Part VI reasons about the fusion as one operation; lowering reasons about its contents. Chapter 24.

**They hold a loop body once.** `scan` carries the body; the trip count is data, not structure. The graph does not grow with sequence length, the body is optimized once, and — the reason this matters most — it is *differentiated* once. Chapter 31.

**They hold the arms of a conditional.** `if` carries a region per branch, `while` carries condition and body. Control flow becomes an operation with operands and results rather than edges between blocks, which is why this IR needs no φ-functions: the merge is the operation's result.

The price is stated in the op registry. Region-carrying operations declare `RECURSIVE_MEMORY_EFFECTS` ([`op_registry.ts:39`](../../../src/compiler/ir/graph/op_registry.ts)), meaning "my effects are whatever my contents' effects are" — a pass that ignores this and treats a region-carrying operation as pure will delete a loop that writes to memory. That was a real bug in this codebase, and it is why the trait exists.

## 9.7 Lab 2 — What a region can see

A region contains a program. Which values from *outside* can that program refer to?

```bash
node docs/part2/ch09-object-model/labs/02-what-a-region-sees.mjs
```

The lab compiles the running example, grabs the post-fusion graph, and inspects the `fusion` operation:

```
    %7 = "tera.fusion"(%5, %2, %6) ({
      ^bb0(%8: tensor<2x8xf32>, %9: tensor<8xf32>, %10: tensor<2x8xf32>):
        %11 = "tera.add"(%8, %9) : (tensor<2x8xf32>, tensor<8xf32>) -> tensor<2x8xf32>
        %12 = tera.maximum %11, %10 : tensor<2x8xf32>
        tera.yield %12 : tensor<2x8xf32>
    }) {fusion_kind = "kElementwise"} : (tensor<2x8xf32>, tensor<8xf32>, tensor<2x8xf32>) -> tensor<2x8xf32>
```

```
the 'fusion' operation
  operands passed in       : 3
  block arguments inside   : 3
  values defined inside    : 5
  values captured from out : 0  (nothing crosses the boundary implicitly)

  operand i  ->  block argument i
    0: 2x8xf32  ->  2x8xf32   same type
    1: 8xf32  ->  8xf32   same type
    2: 2x8xf32  ->  2x8xf32   same type

  the region ends with 'yield', yielding 1 value(s) for the 1 result(s) of 'fusion'
```

Three operands go in, three block arguments come out, positionally matched and identically typed. The inner `add` reads `%8` and `%9` — the block's arguments — not `%5` and `%2` from the enclosing scope, even though those are the same tensors.

That is the **region scope contract**, and it is the design decision this chapter is really about:

> **Definition 9.1 (Region scope isolation).** **(stated here)** A region is *isolated* when every value used inside it is either a block argument of one of its blocks or defined by an operation within it — so that the only values crossing the boundary do so through the operation's explicit operands and results.

Isolation is a choice, not a necessity. MLIR supports both isolated and non-isolated regions, and a non-isolated region — one that can close over enclosing values — is more convenient to build. What isolation buys is that **the boundary is complete**: to know what a `fusion` reads you read its operand list, and to know what it produces you read its result list. No pass has to search inside it to discover a hidden dependency. Every fusion legality check in Part IV, every liveness computation in Part IX, and the buffer plumbing in every backend depend on that being true.

The lab computes the capture set the hard way — every value used inside, minus everything defined inside — and gets zero. The compiler has a function for this, `capturedValues` at [`graph_algorithms.ts:22`](../../../src/compiler/ir/graph/graph_algorithms.ts), used by the topological sort so that an operation with a region is ordered after everything its *contents* read. It exists precisely because isolation is enforced by convention and construction rather than by the verifier, which is the honest version of this story — see §9.9.

**Try this.** Run the lab against the `scan` model from Lab 1 instead. `scan` also takes 2 operands and its block also takes 2 arguments, and they still correspond position for position — operands are `(xs, h0)`, block arguments are `(element of xs, carry)`. What changes is that the correspondence is no longer type-preserving: operand 0 is the whole `tensor<4x3xf32>` stack while block argument 0 is one `tensor<3xf32>` slice of it, because the loop hands the body one timestep at a time. Only the carry passes through unchanged. Region operations get to define their own calling convention this way, and `num_carry` / `num_xs` are the attributes that record where the split falls; Chapter 5 §5.7 works through how to read it off a printout.

## 9.8 Cloning, and why it is harder than it looks

One operation on this structure is worth reading in full, because it is where all six nouns interact: cloning a region ([`operation.ts:300`](../../../src/compiler/ir/graph/operation.ts)):

```ts
export function cloneRegion(region: Region, valueMap: Map<Value, Value> = new Map()): Region {
  const newRegion = new Region();
  for (const block of region.blocks) {
    const argTypes = block.arguments.map(a => a.type);
    const BlockCtor = block.constructor as new (argTypes: readonly IRType[]) => Block;
    const newBlock = new BlockCtor(argTypes);
    for (let i = 0; i < block.arguments.length; i++) {
      valueMap.set(block.arguments[i], newBlock.arguments[i]);
    }
    const arr = block.opsArray();
    const inBlock = new Set(arr);
    const clonedByOrig = new Map<Operation, Operation>();
    for (const op of topoSortByOperands(arr, (o) => inBlock.has(o), 'ignore')) {
      clonedByOrig.set(op, op.clone(valueMap));
    }
    for (const op of arr) {
      newBlock.pushOp(clonedByOrig.get(op) as Operation);
    }
```

The `valueMap` threads through everything: it maps old values to new ones so that a cloned operation's operands point at the clones rather than at the originals. Block arguments are seeded into it first, because they have no producer to clone.

Then look at the two loops. Operations are **cloned** in `topoSortByOperands` order — an operation cannot be cloned before its operands' clones exist — but **pushed** in the original array order. Cloning order is a dataflow constraint; insertion order is cosmetic. That is Theorem 8.4 showing up as three lines of code, and Chapter 13 will show the parser making exactly the same split for exactly the same reason.

## 9.9 Traps and limits

- **`ops()` versus `opsRecursive()` is a real bug source.** They differ by everything inside every region. When you read a pass in Part IV, check which one it uses; when you write one, decide deliberately.
- **Region isolation is a contract, not a checked invariant.** The verifier in Chapter 12 checks that operands are defined *somewhere in the function's scope set*, which is deliberately permissive: it does not reject a region operation whose body reads an enclosing value. Isolation is upheld by the passes that build regions and is pinned by [`tests/compiler/ir/graph/region-scope-contract.test.js`](../../../tests/compiler/ir/graph/region-scope-contract.test.js) rather than by the verifier. Chapter 12 returns to why that boundary was drawn there.
- **A region here almost always has exactly one block.** The structure permits several, and `Region.blocks` is an array. But with no branch operation there is nothing to make a second block reachable, so multi-block regions are unused. The parser refuses more than one top-level block in a function outright ([`parser.ts:523`](../../../src/compiler/ir/graph/parser.ts)).
- **Encapsulation is a naming convention, and it does not cover the containers.** Every mutating *method* notifies (§9.4), but the underscore convention is the whole of the enforcement, and `Operation.attributes`, `.operands`, `.results` and `.regions` are all public, mutable and reachable. `op.attributes.set('direction', 'gt')` compiles, runs, changes what the program computes, and leaves the version untouched — by a route that is not even nominally private. Only `Block`'s intrusive list is underscore-protected. So "there is exactly one path by which the IR can be edited" describes an intention rather than a property of the code: the version counter is sound for every edit made through the API and cannot see one made around it. Freezing the containers, or hiding them behind accessors, is what would close the remaining gap.
- **`Object.freeze` on the signature is shallow.** `inputTypes` cannot be reassigned or resized; the `TensorType` objects inside it are immutable by their own construction rather than by the freeze. Chapter 10 makes that immutability explicit.

## 9.10 Read the tests

- [`tests/compiler/ir/graph/block-invariants.test.js`](../../../tests/compiler/ir/graph/block-invariants.test.js) — what the operation list guarantees under insertion, removal, and erasure during iteration.
- [`tests/compiler/ir/graph/region-scope-contract.test.js`](../../../tests/compiler/ir/graph/region-scope-contract.test.js) — the isolation property of §9.7, stated as a test because it is not stated as a verifier rule.
- [`tests/compiler/ir/deep-nesting.test.js`](../../../tests/compiler/ir/deep-nesting.test.js) — regions inside regions inside regions, and the traversals that must not blow the stack on them.

---

**Next:** [Chapter 10 — The type system](../ch10-type-system/README.md), which fills in the one field of `Value` this chapter kept saying "later" about.
