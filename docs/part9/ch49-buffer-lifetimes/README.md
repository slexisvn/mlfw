# Chapter 49 — Buffer lifetimes

Chapter 32 lowered a graph to loops and two buffers appeared that nobody wrote: `buf_13` and `buf_24`, the intermediates the graph had been carrying as values. Chapter 2 §2.4 noticed the same thing from the other end — three temporaries declared at the top of a generated function — and deferred the obvious question. This chapter asks it.

## 49.1 The problem: a program does not need all its buffers at once

Take a chain of four elementwise operations on a 64 × 64 tensor. Lowering gives four loop nests and, between them, three intermediate buffers of 16,384 bytes each. Add the input, the output and the small constant buffers and the program names nine buffers in total.

The naive answer to "how much memory does this need" is to add them up. That answer is wrong, and it is wrong by a factor that grows with the depth of the model — which is to say, by the factor that decides whether a network fits on a card.

It is wrong because the third operation cannot start until the second has finished, and once the second has finished, the buffer feeding it is never read again. Its bytes are free. Nothing in the IR says so: a buffer is a name, and a name does not expire. The information is there — it is in *which statements mention the name* — but it has to be computed.

So the first job of memory planning is not allocation. It is establishing, for each buffer, the window of time during which its contents matter.

## 49.2 Intuition: a hotel register

A hotel with one room per guest needs as many rooms as it has ever had guests. A hotel that reads the register needs as many rooms as its busiest night.

The register is the useful object, and it holds one fact per guest: a check-in and a check-out. Two guests can share a room exactly when their stays do not overlap, and the number of rooms the hotel actually needs is the largest number of stays that cover any single night.

That is the whole of this chapter, with one substitution: guests are buffers, nights are statements, and the busiest night is peak memory. The two chapters after this one are about handing out the rooms (Chapter 50) and about moving the guests' dates around so the busiest night is less busy (Chapter 52).

The analogy has one seam, and it is worth naming now because it is where the real subtlety lives. A hotel guest's stay is an interval because time is a line. A program's statements are a line only if you *choose* a line — and a loop is not a line. Everything difficult in §49.3 comes from that.

## 49.3 Theory

Fix a program and a total order on its statements. Nothing yet says the order is the execution order; §49.4 says which order this compiler picks.

> **Definition 49.1 (Live interval).** Let statements be indexed `0, 1, …, n−1`. The *live interval* of a buffer `b` is `[first(b), last(b)]`, where `first(b)` is the least index of a statement that reads or writes `b` and `last(b)` is the greatest.

Note what the definition does **not** say: it does not say `b` is live at every index in between. A buffer written at 0, untouched at 1 through 8, and read at 9 has interval `[0, 9]` and is genuinely dead for eight of those ten statements. The interval is an over-approximation — a single range rather than a set of ranges — and the reason to accept it is in Theorem 49.4.

> **Definition 49.2 (Interference).** Buffers `a` and `b` *interfere* if their intervals overlap: `first(a) ≤ last(b)` and `first(b) ≤ last(a)`.

The overlap test is inclusive at both ends, so two buffers that share a single endpoint interfere. That is deliberate and it is not a rounding error: at the statement where one is last read and the other first written, both contents matter, because within a statement nothing orders the read against the write.

> **Theorem 49.3 (Disjoint intervals may share storage).** Let `a` and `b` be buffers that do not interfere, with `last(a) < first(b)`. If the total order on statements is a linearization of the execution order — every statement executes after all statements with smaller indices and before all statements with greater — then a program in which `a` and `b` name the same bytes computes the same results as one in which they do not.
>
> *Proof.* Every access to `a` occurs at a statement with index at most `last(a)`, and every access to `b` at a statement with index at least `first(b) > last(a)`. Under the hypothesis these sets of statements are executed in that order with none interleaved, so every read of `a` is executed before every write of `b`. A read of `a` therefore returns what the last write to `a` stored, whether or not `b` later overwrites those bytes; and a read of `b` returns what the last write to `b` stored, since no access to `a` follows it. Every load in the shared program returns the value it returned in the unshared one, so every store writes the same value. ∎

Two things about that proof are worth pausing on, because the rest of Part IX is built on them.

**The hypothesis is the whole content.** "The total order is a linearization of the execution order" is trivially true for a straight-line sequence of statements and *false* for a loop. In a loop, the statement at index 5 executes again after the statement at index 9, so "every read of `a` is executed before every write of `b`" does not follow from `last(a) < first(b)`. A liveness analysis that indexes statements textually and then applies Theorem 49.3 inside a loop body is unsound.

The repair is not to abandon the linearization but to widen the intervals until the theorem's hypothesis holds again:

> **Lemma 49.4 (Region extension).** Let `R` be a set of statements with indices `[s, e]` that may execute more than once, or in an order other than index order. Extend the interval of every buffer touched inside `R` so that it covers `[s, e]`. Then any two non-interfering buffers satisfy Theorem 49.3's hypothesis.
>
> *Proof.* After extension, two buffers both touched inside `R` have overlapping intervals — both contain `[s, e]` — so they interfere and the theorem is not applied to them. If only one of the pair is touched inside `R`, the other is touched only outside it, and outside `R` the index order is the execution order by construction. ∎

This is why an over-approximating single interval is the right shape after all. The analysis is not trying to describe liveness exactly; it is trying to describe a *conservative* liveness under which interval disjointness is a sound sharing test. Widening is always allowed; narrowing is what would be unsound.

> **Definition 49.5 (Peak, and the width of a program).** The *width* at index `i` is the number of buffers whose intervals contain `i`. The *peak* is the maximum over `i` of the total size of those buffers.

Peak is a lower bound on how many bytes any assignment needs, and Chapter 50 is about how close a real allocator gets to it.

## 49.4 In mlfw: 196 lines, one counter, and a touch log

[`buffer_liveness.ts:6`](../../../src/compiler/passes/memory/buffer_liveness.ts) is Definition 49.1 verbatim:

```ts
export class BufferInterval {
  buffer: Buffer;
  firstUse: number;
  lastUse: number;
  scope: string;
```

and [`buffer_liveness.ts:23`](../../../src/compiler/passes/memory/buffer_liveness.ts) is Definition 49.2 verbatim:

```ts
  overlaps(other: BufferInterval): boolean {
    return this.firstUse <= other.lastUse && other.firstUse <= this.lastUse;
  }
```

Both ends inclusive, as promised.

### What counts as a statement

The linearization is not "every TIR node". Exactly one node type advances the counter — [`buffer_liveness.ts:150`](../../../src/compiler/passes/memory/buffer_liveness.ts):

```ts
        case 'BlockNode': {
          const blk = node as BlockNode;
          stmtOrder.push({ idx: stmtIdx, node: blk });
          for (const r of blk.reads) touch(r.buffer);
          for (const w of blk.writes) touch(w.buffer);
          touchAll(blk.body, new Set());
          if (blk.initBody) touchAll(blk.initBody, new Set());
          stmtIdx++;
```

**The unit of time is a block**, Chapter 33's block: the indivisible piece of computation with a declared read set and write set. Loops do not advance the counter, and neither do `if`s or `let`s — they are containers, and a container that held no block would occupy no time at all. So the "program" this analysis reasons about is the sequence of blocks in the order the printer would print them, and a nest of six loops around one block is one statement.

That is a good choice for the same reason Chapter 33's block is a good unit: the read and write sets are already declared, so `touch` has something to consult without re-deriving anything.

### The two ways a buffer is touched

The declared sets are consulted first, then the body is walked. The walk exists because a block's declarations cover the buffers it reads and writes as *data*, and a body can mention a buffer in other positions — a bound, an index, a condition. `touchAll` ([`buffer_liveness.ts:96`](../../../src/compiler/passes/memory/buffer_liveness.ts)) walks arbitrary object graphs looking for anything buffer-shaped, guarded by a `seen` set and skipping the three parent-link keys that would otherwise walk back up the tree.

Recognising "buffer-shaped" is done structurally ([`buffer_liveness.ts:61`](../../../src/compiler/passes/memory/buffer_liveness.ts)):

```ts
function isBuffer(x: unknown): x is Buffer {
  const b = x as Buffer & { type?: unknown };
  return !!x && typeof x === 'object'
    && typeof b.name === 'string'
    && b.dtype !== undefined
    && b.shape !== undefined
    && b.type === undefined;
}
```

Three positive clauses and one negative one. The negative clause is the interesting one: every TIR *node* carries a `type` discriminant and a `Buffer` does not, so `b.type === undefined` is what separates a buffer from a node that happens to have a name, a dtype and a shape. §49.6 returns to what that costs.

### Region extension, which is Lemma 49.4

[`buffer_liveness.ts:114`](../../../src/compiler/passes/memory/buffer_liveness.ts):

```ts
    const extendRegion = (logStart: number, endIdx: number): void => {
      for (let i = logStart; i < touchLog.length; i++) {
        const interval = intervals.get(touchLog[i]);
        if (interval && endIdx > interval.lastUse) interval.lastUse = endIdx;
      }
    };
```

`touchLog` is an append-only list of every buffer touched, in touch order. A region records the log's length on the way in, and on the way out extends every buffer touched since then. That is a neat way to answer "which buffers were touched inside this subtree" without a second traversal or a per-region set.

Three node types call it — `ForNode` ([`:128`](../../../src/compiler/passes/memory/buffer_liveness.ts)), `WhileNode` ([`:139`](../../../src/compiler/passes/memory/buffer_liveness.ts)) and `IfThenElseNode` ([`:166`](../../../src/compiler/passes/memory/buffer_liveness.ts)) — and the first two are Lemma 49.4's "may execute more than once" while the third is its "or in an order other than index order". A branch executes at most once, but *which* of its two arms ran is not known, so treating the whole `if` as one region is the conservative reading.

### Parameters are not temporaries

[`buffer_liveness.ts:77`](../../../src/compiler/passes/memory/buffer_liveness.ts) seeds a set from `primFunc.bufferMap` — the buffers bound to the function's arguments and results — and `getTemporaries` ([`:43`](../../../src/compiler/passes/memory/buffer_liveness.ts)) returns everything else. A parameter's storage belongs to the caller, so the planner may not place it, reuse it, or free it. Chapter 51 is about the one circumstance under which the caller can hand that right over.

## 49.5 Lab — when is a buffer live

```bash
node docs/part9/ch49-buffer-lifetimes/labs/01-when-is-a-buffer-live.mjs
```

The lab compiles `t.mul(2).add(1).relu().mul(3)` on a 64 × 64 input with fusion switched off — fusion would delete the very intermediates this chapter is about, which is §49.5's last observation — and rebuilds the analysis from the printed IR: the blocks in order, then the interval of every buffer they touch.

```
=== the linearized program: one index per block ===
   0  mul_block_0        reads buf_1 buf_4          writes buf_8
   1  add_block_1        reads buf_8 buf_5          writes buf_13
   2  maximum_block_2    reads buf_13 buf_6         writes buf_18
   3  mul_block_3        reads buf_18 buf_7         writes buf_3
```

Four operations, four blocks, four indices. Now the register:

```
=== live intervals, one row per buffer ===
      buffer      [first,last]   0
      buf_1       [ 0, 0]      #...
      buf_4       [ 0, 0]      #...
      buf_8       [ 0, 1]      ##..
      buf_13      [ 1, 2]      .##.
      buf_5       [ 1, 1]      .#..
      buf_18      [ 2, 3]      ..##
      buf_6       [ 2, 2]      ..#.
      buf_3       [ 3, 3]      ...#
      buf_7       [ 3, 3]      ...#
```

**The three intermediates form a staircase.** `buf_8` is written at 0 and read at 1; `buf_13` is written at 1 and read at 2; `buf_18` is written at 2 and read at 3. Each overlaps only its immediate neighbour, and `buf_8` and `buf_18` — the first and last intermediate — do not overlap at all. Two 16,384-byte buffers that can share one 16,384-byte allocation, read straight off the picture.

That staircase is the shape of every elementwise chain, and it is why deep models are allocatable at all. It is also exactly the input Chapter 50's packer is designed for.

```
=== which pairs may share storage (disjoint intervals) ===
  24 disjoint pair(s), 12 interfering pair(s)
  widest point of the program: 3 buffers live at once, out of 9 total
  the planner reported peak: 16640 bytes over 7 temporaries
```

Nine buffers, but never more than three alive together. The planner's reported peak is 16,640 bytes — one full-size intermediate at 16,384 plus 256 bytes of constants — against 49,152 bytes for three unshared intermediates. The staircase was worth 3×, and it cost one traversal to find.

The last block of output is the one to think about longest:

```
=== the same program, fused ===
  1 block(s), 6 buffer(s) touched
  the planner reported peak: 256 bytes over 4 temporaries
```

**Fusion is the better memory optimization, and it is not in this part.** Four blocks became one, the three 16,384-byte intermediates became values inside a loop body, and the peak fell from 16,640 bytes to 256. Everything Part IX does is applied to what fusion leaves behind. Chapter 22 sold fusion on memory *traffic*; this is the same mechanism collecting a second dividend in memory *footprint*, and neither chapter's argument needs the other.

**Try this.** Lengthen the chain and watch the two numbers separate: the buffer count grows with the chain, the width does not. Then set the input to `[512, 512]` and compare the reported peak against `n × 16,384` for the `n` intermediates you counted.

## 49.6 Traps and limits

- **An interval is one range, so a buffer with a hole is live in the hole.** Definition 49.1. A buffer written at index 0 and read at index 40 interferes with everything alive in between, even though its bytes are untouched there. Live *ranges* — a set of intervals per buffer, split at the points where the value is dead — are the standard refinement and this compiler does not implement them. The cost is paid by any program that carries a value a long way, which is exactly the shape of a residual connection.
- **`isBuffer` is a structural test with a negative clause.** [`buffer_liveness.ts:61`](../../../src/compiler/passes/memory/buffer_liveness.ts) identifies a buffer as an object with `name`, `dtype` and `shape` and *without* `type`. Every TIR node carries `type`, so the test works — and it works by convention rather than by construction. A future node kind that omits its discriminant, or a `Buffer` that gains a `type` field, changes what liveness sees without changing anything in this file, and the failure would be a silently missing touch rather than an error.
- **Region extension moves the end of an interval and never the start.** [`buffer_liveness.ts:114`](../../../src/compiler/passes/memory/buffer_liveness.ts) sets `lastUse` to the region's end and leaves `firstUse` where the first touch put it. For a buffer first touched at the top of a loop body this is exactly Lemma 49.4; for one first touched at the *bottom* of a long loop body, the interval starts late, so it does not cover the earlier part of the loop that it is nevertheless live across on the second iteration. Nothing in the shipped lowering rules produces a loop-carried temporary of that shape — accumulators are read and written in the same block — so the asymmetry is latent rather than reachable, and it is the hypothesis of Lemma 49.4 that would have to be re-checked if that ever changed.
- **`AllocateNode` does not scope the interval it introduces.** [`buffer_liveness.ts:160`](../../../src/compiler/passes/memory/buffer_liveness.ts) touches the buffer and then walks the body. The buffer's interval is therefore where it is *used*, not the extent of the allocation that nominally owns it. That is the more useful of the two answers and it means an `AllocateNode` whose buffer is never used gets a degenerate one-index interval rather than none.
- **The analysis runs on one `PrimFunc` and knows nothing about the module.** Two functions' temporaries are planned independently against separate pools, so a program that calls three kernels in sequence has three peaks that the planner never adds up or overlaps. What a runtime does with those pools is Part XI's problem, and the number this analysis reports is not a whole-program figure.
- **Nothing here is a *decision*.** `BufferLiveness` computes intervals and stops; every claim in this chapter about two buffers sharing bytes is a claim about what the *next* chapter is permitted to do. It is worth keeping the two apart when reading a wrong answer: an interval that is too wide costs memory, and only an interval that is too *narrow* can cost correctness.

## 49.7 Read the tests

- [`tests/compiler/passes/memory/liveness.test.js:36`](../../../tests/compiler/passes/memory/liveness.test.js) — `BufferInterval` itself, including the case this chapter called deliberate: adjacent intervals sharing one endpoint *do* overlap.
- [`tests/compiler/passes/memory/liveness.test.js:70`](../../../tests/compiler/passes/memory/liveness.test.js) — the analysis: that sequential blocks get increasing indices, that an interval spans from first to last use across blocks, and that `isParam` separates the caller's buffers from the planner's.
- [`tests/compiler/passes/memory/planning.test.js`](../../../tests/compiler/passes/memory/planning.test.js) — liveness as the planner consumes it, which is the form Chapters 50 and 51 assume.

---

**Next:** [Chapter 50 — Arena allocation](../ch50-arena-allocation/README.md), which takes the intervals this chapter produced and turns them into addresses.
