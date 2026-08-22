# Chapter 39 — The sref tree and block scopes

`schedule.parallelize('i1_7')` has to find a loop called `i1_7`, decide whether marking it parallel is legal, mark it, and leave every other query about the function still answerable. Doing that by walking the IR each time is correct and quadratic: a schedule of *k* primitives on a function of *n* nodes costs O(kn), and Part VIII will run schedules of a dozen primitives thousands of times.

This chapter is about the two indexes that reduce that cost, and about how much of the second one is currently unused.

**One correction to make before the chapter makes the claim, because the tempting figure is `O(k·depth)` and it is not what this implementation achieves.** Two costs sit on top of the tree lookup, and both are proportional to the work a primitive does rather than to the depth at which it does it:

- **`replaceNode` walks the replacement subtree.** Patching the shadow tree unregisters the srefs under the old node and *walks the new node* to build fresh ones (§39.4). That is `O(size of the replaced subtree)`, not `O(depth)`. For `split` the subtree is two loops and a body; for `rfactor` it is two complete nests. And when the old node is not found the fallback is `rebuildFrom`, which is `O(n)`.
- **Every primitive clears every memo.** `ScheduleState.invalidate` drops six cached analyses wholesale (§39.4), so the next legality query recomputes buffer accesses and dependences from scratch over the nest it asks about.

So the honest bound is: **the tree gives `O(depth)` *lookup*, and the rest of a primitive's cost is proportional to the subtree it rewrites plus whatever analysis the next query has to redo.** That is still a large improvement on re-walking the whole function for every query — which is the point of the chapter — and it is not a per-primitive constant. §39.4 is where the memo behaviour is set out and §39.7 is honest about what "incremental" covers.

## 39.1 The problem: a tree that is edited from the middle

The TIR of Chapter 32 is a tree of statements. Three facts about how a schedule uses it decide the data structure.

1. **Primitives address nodes by name.** `getLoops('matmul_1')` is the standard opening line of a rule. Names are stable; positions are not.
2. **Legality questions are about ancestors.** "Which loops enclose this block?" and "which blocks are under this loop?" are the two questions Chapter 42 asks, and neither is answerable from a node alone, because TIR nodes are not obliged to know their parents in the way graph IR values know their users (Chapter 8).
3. **Every primitive replaces a subtree.** `split` builds two new `ForNode`s and swaps them in for one; `rfactor` swaps a `SeqNode` of two nests in for one. Nothing is edited in place except a loop's `kind` field.

Point 3 is what makes point 2 hard. A parent pointer that is correct before a `split` is stale after it, in exactly the subtree that changed and nowhere else.

## 39.2 Intuition: a shadow tree that is cheap to patch

The answer is a second tree that mirrors the first at a coarser grain. It has a node — an **sref**, for *schedulable reference* — for each loop and each block, and for nothing else. Statement sequences, conditionals, allocations and `let`s are transparent: they are walked through, not represented. So a nest six statements deep might be three srefs deep, and every sref carries a parent pointer that the IR node does not have.

When a primitive replaces a subtree, the shadow tree is patched rather than rebuilt: the srefs under the replaced node are unregistered, the new node is walked, and the resulting srefs are spliced into the parent's child list at the position the old one occupied. Everything outside the replaced subtree keeps its sref, its parent pointer and its identity.

The second index answers a different question. Given the *siblings* inside one scope — the blocks that sit under a common parent block, or at the top of the function — which of them produce values the others consume? That is a graph on blocks, with an edge per (buffer, dependence kind) pair, and it is what a primitive needs before it can move one block past another.

## 39.3 Theory

> **Definition 39.1 (sref tree).** **(stated here)** For a `PrimFunc` `P`, the *sref tree* has one node per `ForNode` and per `BlockNode` in `P`'s body. The parent of an sref is the nearest enclosing loop or block; nodes of any other kind are transparent. An sref stores its IR node, its parent, and its ordered children.

> **Definition 39.2 (Block scope).** **(stated here)** The *scope root* of a block `B` is the nearest enclosing block, or `⊥` if there is none. The *block scope* of a root `R` is the set of blocks whose scope root is `R`, together with a *position* for each (its first access in program order) and a dependence graph over them.

> **Definition 39.3 (Scope dependence).** **(classical)** For blocks `X`, `Y` in one scope with `pos(X) ≤ pos(Y)`, and a buffer `b` both touch, there is an edge `X → Y` labelled RAW if `X` writes a region of `b` overlapping one `Y` reads, WAW if both write overlapping regions, and WAR if `X` reads a region `Y` writes.

The regions in Definition 39.3 are *hulls*, not exact sets, which makes the edge set conservative in the right direction:

> **Lemma 39.4 (Scope edges over-approximate).** **(invariant)** If two blocks in a scope access overlapping locations of a buffer, the scope contains an edge between them of the corresponding kind. The converse fails: an edge may exist between blocks that never touch the same location.

*Proof sketch.* `ScopeMember.hull` takes the bounding box of every access region a block makes to a buffer ([`block_scope.ts:67`](../../../src/compiler/schedule/block_scope.ts)), and `rangesOverlap` returns `true` whenever the boxes intersect and whenever either is unknown ([`dep_analysis.ts:5`](../../../src/compiler/schedule/dep_analysis.ts)). A real overlap implies overlapping hulls, so no edge is missed. Two blocks touching disjoint checkerboards of the same buffer have identical hulls and get an edge, so edges can be spurious. ∎

That is the correct direction for a legality check: an over-approximated dependence graph refuses transformations that were in fact safe and never permits one that was not.

The move a scope makes possible is stated once here and used by Chapter 41:

> **Proposition 39.5 (Relocation legality).** **(stated here)** Moving a block `X` from its position to a position `d` within the same scope preserves semantics if no block `Z` with `pos(Z)` strictly between `pos(X)` and `d` has an edge to or from `X`.

*Proof sketch.* Program order within a scope is a total order, and the scope's edges are exactly the pairs whose relative order is constrained (Lemma 39.4, in the safe direction). Moving `X` past only unconstrained blocks changes the order of no constrained pair, and the sequential composition of two commuting statements is order-independent. ∎

This is `_checkRelocationDependences` ([`schedule.ts:905`](../../../src/compiler/schedule/schedule.ts)) exactly, down to the interval:

```ts
    const lo = Math.min(self.position, destination);
    const hi = Math.max(self.position, destination);

    for (const dep of [...scope.depsBySrc(blockSRef), ...scope.depsByDst(blockSRef)]) {
      const other = dep.src === blockSRef ? dep.dst : dep.src;
      if (other === blockSRef) continue;
      const member = scope.memberOf(other);
      if (!member || member.position <= lo || member.position >= hi) continue;
      throw new Error(
        `${primitive}: moving '${(blockSRef.node as BlockNode).name}' across '${(other.node as BlockNode).name}' would violate a ` +
        `${dep.kind} dependence on buffer '${dep.buffer.name}'`
      );
    }
```

## 39.4 In mlfw

### `SRef` and `SRefTree`

[`schedule/sref.ts`](../../../src/compiler/schedule/sref.ts), 217 lines. `SRef` is three fields and eight accessors; the interesting part is `SRefTree`, which keeps four indexes ([`sref.ts:70`](../../../src/compiler/schedule/sref.ts)): node → sref, block name → sref, and the sets of all loop and all block srefs.

The builder is one switch and it is where "transparent" is defined ([`sref.ts:117`](../../../src/compiler/schedule/sref.ts)):

```ts
      switch (node.type) {
        case 'ForNode':
        case 'BlockNode': {
          const sref = new SRef(node, parentSRef);
          this._register(sref);
          if (isTop) top.push(sref);
          else (parentSRef as SRef).children.push(sref);
          stack.push({ node: (node as ForNode | BlockNode).body, parentSRef: sref, isTop: false });
          …
        case 'SeqNode': {
          const seq = node as SeqNode;
          for (let i = seq.stmts.length - 1; i >= 0; i--) stack.push({ node: seq.stmts[i], parentSRef, isTop });
        }
```

Note the second case passes `parentSRef` straight through: a `SeqNode`'s children become children of whatever encloses the `SeqNode`. `IfThenElseNode`, `AllocateNode` and `LetStmtNode` do the same. A guard introduced by `split` therefore does not deepen the sref tree, which matters because a schedule that splits a loop three times would otherwise bury the block under three conditionals.

`replaceNode` is the incremental patch ([`sref.ts:186`](../../../src/compiler/schedule/sref.ts)):

```ts
  replaceNode(oldNode: TirNode, newNode: TirNode): boolean {
    const oldSRef = this._nodeToSRef.get(oldNode);
    if (!oldSRef) return false;
    const parent = oldSRef.parent;
    const wasRoot = this.root === oldSRef;
    this._unregisterSubtree(oldSRef);
    const top = this._buildSubtree(newNode, parent);
    if (parent) {
      const idx = parent.children.indexOf(oldSRef);
      if (idx >= 0) parent.children.splice(idx, 1, ...top);
      else for (const s of top) parent.children.push(s);
    } else if (wasRoot) {
      this.root = this._nodeToSRef.get(newNode) || null;
    }
    return true;
  }
```

and the failure path is `ScheduleState.replaceNode` ([`schedule_state.ts:133`](../../../src/compiler/schedule/schedule_state.ts)): `if (!this.tree.replaceNode(oldNode, newNode)) this.tree.rebuildFrom(this.primFunc.body);`. If the node being replaced is not in the tree — because a previous primitive detached it — the whole tree is rebuilt rather than left inconsistent. Correct, and O(n) when it fires.

The one query with a filter in it is `loopsOf` ([`sref.ts:172`](../../../src/compiler/schedule/sref.ts)):

```ts
  loopsOf(blockName: string): SRef[] {
    const blockSRef = this._blockNameToSRef.get(blockName);
    if (!blockSRef) return [];
    return blockSRef.loopAncestors().filter(s => (s.node as ForNode).kind !== ForKind.RECURRENCE).reverse();
  }
```

A `@recurrence` loop — the sequential time axis a `scan` lowers to (Chapter 34) — is removed from the list a primitive sees. So `schedule.getLoops(b)[0]` is never the recurrence, and no rule, all of which start from `getLoops`, can accidentally parallelise it. That is a *representation-level* guarantee rather than a legality check, and it is the cheaper kind: the illegal argument is not offered. It is not airtight — `_resolveLoop` will still find a recurrence loop by name, and nothing downstream would object — but nothing in the compiler names loops that way.

### `ScheduleState`

[`schedule/schedule_state.ts`](../../../src/compiler/schedule/schedule_state.ts), 274 lines, is the sref tree plus six memoised analyses, all cleared by one method ([`schedule_state.ts:76`](../../../src/compiler/schedule/schedule_state.ts)):

```ts
  invalidate(): void {
    this._loopBindings = null;
    this._blockBindings = null;
    this._accessInfo = null;
    this._scopes = null;
    this._dependences = null;
    this._nestAnalyses = null;
  }
```

Every primitive calls it. This is Chapter 16's invalidation problem in its simplest possible form: one bit, cleared on every write, so nothing can be stale. The cost is that a rule applying five primitives to one block recomputes the buffer accesses five times, and Part VIII pays it thousands of times over.

**And it is worth being exact about which half of this chapter is incremental, because the two indexes behave differently.** The *sref tree* is genuinely maintained: a replacement patches the affected subtree and everything outside it keeps its identity, its parent pointer and its position. The *analyses over that tree* are not maintained at all — they are thrown away and recomputed on demand. So "the schedule edits a nest from the middle without invalidating everything else" is true of the structure and false of the derived facts, and a reader who takes the chapter's title as a claim about both will over-estimate what a primitive costs to apply.

The one place the second half is incremental is `nestAnalysis`, immediately below, which is keyed per loop node so that a legality question about one nest does not recompute the others.

`nestAnalysis` ([`schedule_state.ts:122`](../../../src/compiler/schedule/schedule_state.ts)) is the one memo with a key: it caches per loop node, because a legality question is about one nest and not the function.

```ts
  nestAnalysis(node: TirNode): NestAnalysis {
    if (!this._nestAnalyses) this._nestAnalyses = new Map();
    let analysis = this._nestAnalyses.get(node);
    if (!analysis) {
      const info = collectBufferAccesses(node, this._enclosingEnv(node));
      analysis = { info, deps: dependences(info.byBuffer) };
      this._nestAnalyses.set(node, analysis);
    }
    return analysis;
  }
```

`_enclosingEnv` is what makes analysing a *sub*-nest sound: it walks the sref chain from the node outward, collecting loop ranges and composing block bindings, so a dependence computed inside a nest still knows the extents of the loops above it. Without it, a subscript containing an outer loop variable would look unbounded.

### `BlockScope`

[`schedule/block_scope.ts`](../../../src/compiler/schedule/block_scope.ts), 257 lines. `buildBlockScopes` ([`block_scope.ts:189`](../../../src/compiler/schedule/block_scope.ts)) runs in four passes: assign each access to every block on its chain; sort each scope's members by position; take a hull per (member, buffer, direction); and pair the hulls up. The pairing is `linkAccessUnits` ([`block_scope.ts:158`](../../../src/compiler/schedule/block_scope.ts)), which is Definition 39.3 written out:

```ts
      if (src.write && dst.read && rangesOverlap(src.write, dst.read)) record(src, dst, DepKind.RAW, buffer);
      if (src.write && dst.write && rangesOverlap(src.write, dst.write)) record(src, dst, DepKind.WAW, buffer);
      if (src.read && dst.write && rangesOverlap(src.read, dst.write)) record(src, dst, DepKind.WAR, buffer);
```

Two flags are computed alongside and neither has a reader. `regionCover` ([`block_scope.ts:179`](../../../src/compiler/schedule/block_scope.ts)) asks whether everything a block reads of a buffer was written by producers in the same scope — the precondition for `computeAt` to be able to shrink the producer to just the region the consumer needs. `stagePipeline` asks whether a scope could be software-pipelined: no opaque accesses, every member's reads covered, and no WAR edge.

## 39.5 Lab — the sref tree, and what survives an edit

```bash
node docs/part7/ch39-sref-tree-and-block-scopes/labs/01-the-sref-tree.mjs
```

`x.mul(x).sum(1)` lowers to three blocks in three sibling statement chains. The tree over them:

```
=== every sref, with the chain that reaches it ===

  for i0_6 in 0..4  >  for i1_7 in 0..6  >  block mul_block_0
  for si0_10 in 0..4  >  block reduce_init_1
  for sa0_12 in 0..4  >  for r0_14 in 0..6  >  block reduce_acc_2

  loops registered: 5
  blocks registered: 3
  tree.root: null
```

Eight srefs for a function whose TIR has 47 nodes, and no root. `root` is `null` because `_build` finishes by looking the *body node* up in the node map ([`sref.ts:150`](../../../src/compiler/schedule/sref.ts)), and the body of any function with more than one statement is a `SeqNode`, which is transparent and therefore unregistered. The three chains hang off nothing — each top-level loop is its own sref with a `null` parent, in no one's child list. Nothing notices, because every query in the file goes through the name map or the loop/block sets instead; what is dead for the common case is `SRef.isRoot`, which has no caller at all, and the `wasRoot` branch of `replaceNode` and `removeNode`, which can only fire for a single-statement function.

Then the edit:

```
=== which srefs survive a split? ===

  mul_block_0      same sref object: true
  reduce_init_1    same sref object: true
  reduce_acc_2     same sref object: false

  loops before: 5   loops after: 6
```

`split` replaced `r0_14` with a two-deep nest, so the sref for `r0_14` and everything under it — including `reduce_acc_2` — was unregistered and rebuilt. The two blocks in the other chains were not touched. This is the property the design exists for, and it is also the reason **a primitive must never hold an `SRef` across another primitive**: after a `split`, the sref you were holding for a block inside the split subtree is no longer in the tree, and `getSRef` on its node returns `null`. Names survive; objects do not. `Schedule.tile` re-fetches `this.getLoops(blockName)` before every one of its splits, and matches the loop it wants *by name*, for exactly this reason ([`schedule.ts:549`](../../../src/compiler/schedule/schedule.ts)).

**Try this.** Split `i1_7` — a loop in the *first* chain — instead, and the surviving/​not-surviving column flips: `mul_block_0` is rebuilt and the two reduction blocks keep their srefs.

## 39.6 Lab — block scopes, and who reads them

```bash
node docs/part7/ch39-sref-tree-and-block-scopes/labs/02-block-scopes.mjs
```

The same function's one scope:

```
=== the scopes of one function ===

  scope rooted at the function body
    members        : mul_block_0, reduce_init_1, reduce_acc_2
    opaque accesses: 1
    stagePipeline  : false

=== the dependence edges between siblings ===

  mul_block_0    -> reduce_acc_2   RAW  on buf_5
  reduce_init_1  -> reduce_acc_2   RAW  on buf_3
  reduce_init_1  -> reduce_acc_2   WAW  on buf_3
```

Three edges, and each is a fact about the program. `mul_block_0 → reduce_acc_2` is the intermediate. The two `buf_3` edges are the same pair of blocks seen twice: the accumulation both reads the zero the init wrote (RAW) and overwrites it (WAW). Together they say the init cannot be moved after the accumulation, which is exactly right.

`stagePipeline` is `false`, and the reason is worth following because it is the kind of thing that only shows up when you print it. The function contains `buf_4[] = 0` — the scalar constant the reduction initialises from, lowered as a store at the top of the function body, outside every block. `buildBlockScopes` files any access whose innermost block is unknown under the null scope as an *opaque access* ([`block_scope.ts:205`](../../../src/compiler/schedule/block_scope.ts)), and `let pipeline = scope.opaqueAccesses.length === 0` ([`block_scope.ts:241`](../../../src/compiler/schedule/block_scope.ts)) then fails for the whole scope. Every function that lowers a scalar literal has one, so `stagePipeline` is `false` almost everywhere — which nothing observes, because:

```
=== who asks for any of this ===

  scope.deps / depsBySrc / depsByDst   _checkRelocationDependences (schedule.ts:905)
  scope.memberOf                       _checkRelocationDependences (schedule.ts:905)
  scope.producersOf                    — nothing in src/
  scope.consumersOf                    — nothing in src/
  scope.writersOf                      — nothing in src/
  scope.stagePipeline                  — nothing in src/
  BlockInfo.regionCover                — nothing in src/
  BlockInfo.affineBinding              — nothing in src/
```

and `_checkRelocationDependences` is reached only from `computeAt` and `reverseComputeAt`, neither of which has a caller in `src/`. `ScheduleState.scopes` is a lazy getter, so the honest summary is that **`buildBlockScopes` never runs during compilation** — the `BlockScope`, `BlockInfo` and `ScopeMember` classes, `computeRegionCover`, and the whole four-pass construction are reachable in principle and unreachable in fact.

One export of the file escapes that verdict, and it is the interesting one. `linkAccessUnits` ([`block_scope.ts:158`](../../../src/compiler/schedule/block_scope.ts)) — the thirteen lines that turn a set of hulls into RAW, WAW and WAR edges — is also imported by `MemorySchedulePass` ([`memory_scheduler.ts:6`](../../../src/compiler/passes/memory/memory_scheduler.ts)), which runs on every compilation because `memory.scheduleForPeak` defaults to `true` ([`compiler.ts:163`](../../../src/compiler/pipeline/compiler.ts)). Chapter 52 reorders whole statements to lower the peak, and it needs exactly Definition 39.3 and nothing else around it. So the one genuinely reusable piece of the file was factored out and reused by a different subsystem, and the 244 lines built around it are dead.

## 39.7 Traps and limits

- **The scope graph is built and never consumed.** §39.6. The producer/consumer view (`producersOf`, `consumersOf`, `writersOf`) has no caller anywhere; `stagePipeline` and `regionCover` are computed on every `scopes` access and read by nobody; and `scopes` itself is reached only from two primitives that nothing calls. This is Part V's "implemented and unreachable" pattern one level down, and unlike Part V's cases the code here is not merely unselected — its only callers are themselves unreachable. The exception is `linkAccessUnits`, which `MemorySchedulePass` imports directly and runs on every compilation.
- **`SRefTree.root` is `null` for every multi-statement function.** §39.5. Three members of `SRef`/`SRefTree` are defined against it (`isRoot`, the `wasRoot` branch of `replaceNode` and of `removeNode`) and are therefore dead in the common case; `rebuildFrom` sets it to `null` again on the next rebuild.
- **An sref is invalidated by any primitive that replaces its enclosing subtree, and nothing says so.** There is no version counter on `SRef` of the kind Chapter 9 puts on graph IR, and no check that a passed-in sref is still registered. `_checkRelocationDependences` will throw a clear "not part of any block scope" for a stale one; `_alignedLoopPairs` will silently produce a chain from a detached node.
- **`ScheduleState.invalidate` clears everything on every write.** The finest granularity available is "the whole state". A rule that splits, reorders and parallelises one block recomputes `collectBufferAccesses` over the nest three times.
- **`ScheduleState.dependences` — the function-wide dependence list — has no caller.** Every legality question goes through `nestAnalysis`, which computes its own. The getter and its memo slot are dead ([`schedule_state.ts:182`](../../../src/compiler/schedule/schedule_state.ts)).
- **Block names are the schedule's addressing scheme, and they are hints.** Chapter 33 noted that `ctx.blockName` appends a counter. Everything in this chapter and the next four addresses blocks by that string; a lowering rule that produced two blocks with the same name would make `_blockNameToSRef` keep only the second ([`sref.ts:91`](../../../src/compiler/schedule/sref.ts)), and the first would be unschedulable rather than mis-scheduled.

## 39.8 Read the tests

- [`tests/compiler/schedule/incremental-sref.test.js`](../../../tests/compiler/schedule/incremental-sref.test.js) — one case per primitive, each asserting that the patched tree equals a tree rebuilt from scratch. This is the specification of §39.5.
- [`tests/compiler/schedule/block-scope.test.js`](../../../tests/compiler/schedule/block-scope.test.js) — the edge kinds of Definition 39.3, including the WAR case no lowered program produces.
- [`tests/compiler/schedule/region-cover.test.js`](../../../tests/compiler/schedule/region-cover.test.js) — `regionCover` true and false, on hand-built nests, for the flag nothing reads.

---

**Next:** [Chapter 40 — Loop primitives](../ch40-loop-primitives/README.md), which is the first four primitives in the table of §38.4 and the predicate that shows up when an extent does not divide.
