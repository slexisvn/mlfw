# Chapter 50 — Arena allocation

Chapter 49 ended with a register: nine buffers, three of them alive at the widest point. That is a statement about *how much*. This chapter is about *where* — turning a set of intervals into a set of addresses, and doing it in one flat block of bytes rather than nine separate allocations.

## 50.1 The problem: a set of intervals is not a memory layout

Knowing that `buf_8` and `buf_18` do not interfere tells you they *may* share bytes. It does not tell you which bytes, and the difference matters for three reasons.

**An allocator has to answer for every buffer at once.** Sharing is not a pairwise decision. If `A` and `B` may share, and `B` and `C` may share, it does not follow that `A`, `B` and `C` may all share one region — and even when they may, they might have different sizes, so the region has to be as large as the largest and the leftover is wasted.

**A separate allocation per buffer is expensive in a way the interval picture hides.** Nine calls to the allocator have nine headers, nine chances to land on different cache lines, and on a GPU, nine round trips to a driver whose allocation call is measured in microseconds. What the backends want is one allocation and a set of offsets into it.

**The offsets have to obey alignment.** A `Float32Array` view cannot start at byte 3. Vector loads want more than that — 16 or 32 bytes — and cache behaviour wants a line, which is 64 on the machines this compiler targets. So the addresses are not free integers; they are multiples of something.

So the job is: given intervals and sizes, choose a byte offset for each buffer such that any two buffers that interfere get non-overlapping byte ranges, every offset is aligned, and the total is as small as you can make it.

## 50.2 Intuition: rectangles in a strip

Draw the program left to right as time and memory bottom to top. Each buffer is a rectangle: its width is its live interval, its height is its size. Placing a buffer means sliding its rectangle up or down — choosing an offset — but never left or right, because the interval is fixed.

A layout is valid when no two rectangles overlap. The arena is as tall as the tallest point of the stack. The job is to slide the rectangles so the whole thing is as short as possible.

This is a picture worth holding because it makes both the easy fact and the hard fact obvious. The easy fact: the arena can never be shorter than the tallest *column* — at the widest moment of the program, every live buffer needs its own bytes, and they are stacked. The hard fact: getting down to that height is not always possible, because rectangles that would fill a hole may be the wrong width for it, and choosing well for one buffer forecloses choices for another.

## 50.3 Theory

> **Definition 50.1 (Assignment).** An *assignment* maps each buffer `b` to an offset `off(b) ≥ 0`, occupying bytes `[off(b), off(b) + size(b))`. Its *height* is `max_b (off(b) + size(b))`.

> **Definition 50.2 (Valid assignment).** An assignment is *valid* if for every pair of interfering buffers `a` and `b` (Definition 49.2), their byte ranges are disjoint.

Validity is exactly the condition Theorem 49.3 needs, read backwards: buffers that do not interfere are *permitted* to overlap in bytes, and buffers that do interfere are *required* not to.

> **Theorem 50.3 (Width is a lower bound).** For any valid assignment, height ≥ `max_i Σ{ size(b) : i ∈ interval(b) }` — the peak of Definition 49.5.
>
> *Proof.* Fix an index `i` achieving the maximum, and let `L` be the buffers live at `i`. Any two members of `L` interfere, since both intervals contain `i`, so by Definition 50.2 their byte ranges are pairwise disjoint. A collection of pairwise disjoint intervals of the real line contained in `[0, height)` has total length at most `height`, so `Σ_{b ∈ L} size(b) ≤ height`. ∎

That bound is what the lab measures against. It is not always achievable.

> **Theorem 50.4 (The problem is NP-hard) — (classical)** *(Garey and Johnson, 1979, problem SR2)*. Deciding whether a set of intervals with sizes admits a valid assignment of height at most `k` is NP-complete. It is the **dynamic storage allocation** problem.

So a compiler does not solve it. It runs a heuristic and accepts a gap, and the interesting question becomes which heuristic and how large a gap. The standard answer, and the one used here, is to place the big rectangles first.

> **Proposition 50.5 (Greedy placement is valid, stated here).** Process buffers in any order. For each, compute the set of byte ranges occupied by already-placed buffers that interfere with it, and choose an aligned offset whose range meets none of them. The result is a valid assignment, and such an offset always exists.
>
> *Proof.* Validity is maintained as an invariant: each buffer is placed disjointly from every interfering buffer already placed, and interference is symmetric, so no later placement can violate an earlier pair. Existence: the occupied ranges are finite in number and bounded above, so the aligned offset at the first multiple of the alignment at or beyond the largest occupied end is always available. ∎

Note what Proposition 50.5 does *not* claim. It says the greedy is correct, not that it is good. The order is what decides quality, and the argument for size-descending is the same one that makes first-fit-decreasing a good bin packer: a large buffer placed late may find only holes too small for it and be forced to the top, while small buffers placed late fit into whatever remains.

There is one more choice once the order is fixed — which hole to use.

> **Definition 50.6 (First-fit and best-fit).** Given the interfering occupied ranges sorted by offset, *first-fit* places the buffer in the lowest gap large enough to hold it; *best-fit* places it in the smallest such gap.

Neither dominates the other. First-fit keeps allocations low and dense, which tends to leave one large hole at the top; best-fit preserves large holes for large buffers at the cost of leaving many small unusable ones. Which wins depends on the size distribution, and for the programs this compiler produces the lab finds they usually agree.

## 50.4 In mlfw: sort by size, sweep for a gap

[`buffer_assignment.ts:108`](../../../src/compiler/passes/memory/buffer_assignment.ts) is the whole allocator, and it has two halves: an order and a placement rule.

### The order

[`buffer_assignment.ts:150`](../../../src/compiler/passes/memory/buffer_assignment.ts):

```ts
    const sorted = [...intervals].sort((a, b) => {
      const aSize = a.size;
      const bSize = b.size;
      const aStatic = aSize > 0;
      const bStatic = bSize > 0;
      if (aStatic && bStatic) {
        const sizeDiff = bSize - aSize;
        if (sizeDiff !== 0) return sizeDiff;
      } else if (aStatic !== bStatic) {
        return aStatic ? -1 : 1;
      }
      return a.firstUse - b.firstUse;
    });
```

Size descending, ties broken by first use, and buffers whose size is not statically known sorted last. The tie-break matters more than it looks: on a program full of same-shaped tensors — which is what an elementwise chain is — *every* comparison is a tie, so the effective order is program order, and the allocator degenerates to a linear scan. That is the common case, and it is a good order.

### The placement rule

[`buffer_assignment.ts:224`](../../../src/compiler/passes/memory/buffer_assignment.ts) is Proposition 50.5 and Definition 50.6 together:

```ts
    const ranges: [number, number][] = [];
    for (const p of placed) {
      if (p.firstUse <= lastUseEff && firstUse <= p.lastUseEff) {
        ranges.push([p.offset, p.offset + p.size]);
      }
    }

    let cursor = 0;
    const sel = gapSelector(strategy);
    for (const [lo, hi] of ranges) {
      const start = Math.ceil(cursor / alignment) * alignment;
      const gap = lo - start;
      const hit = sel.consider(start, gap, size);
      if (hit !== null) return hit;
      if (hi > cursor) cursor = hi;
    }
    return sel.result(Math.ceil(cursor / alignment) * alignment);
```

Read it as a sweep. `placed` is kept sorted by offset — `insertByOffset` ([`buffer_assignment.ts:22`](../../../src/compiler/passes/memory/buffer_assignment.ts)) does a binary search and a splice — so walking it walks up the arena. The filter at the top is Definition 49.2 again: only buffers that *interfere* constrain this one, and a buffer whose interval is disjoint is not even looked at. `cursor` tracks the top of the occupied region so far, `gap` is the free space between the cursor and the next occupied range, and the strategy decides what to do with a gap big enough.

The strategy is a tiny object rather than a branch ([`buffer_assignment.ts:33`](../../../src/compiler/passes/memory/buffer_assignment.ts)):

```ts
    consider(start: number, gap: number, size: number): number | null {
      if (gap < size) return null;
      if (strategy === 'best-fit') {
        if (this.best === null || gap < this.best.gap) this.best = { offset: start, gap };
        return null;
      }
      return start;
    },
```

First-fit returns the moment it sees a gap that fits, ending the sweep. Best-fit never returns early — it records the tightest gap seen and the sweep runs to the end, after which `result` returns the recorded offset or, if no gap ever fit, the top of the occupied region. That is Proposition 50.5's existence argument, implemented.

### Alignment is applied to sizes, not just offsets

[`buffer_assignment.ts:90`](../../../src/compiler/passes/memory/buffer_assignment.ts) rounds a size *up* to a multiple of the alignment before placing it, so a 4-byte scalar occupies a 64-byte slot and the next buffer's offset is automatically aligned. Padding the size rather than only the offset is what keeps the sweep's arithmetic honest: `cursor` is always a multiple of the alignment, so `Math.ceil(cursor / alignment) * alignment` is a no-op in the common case and the alignment logic never has to look backwards.

The alignment itself comes from the target — [`memory_plan_pass.ts:21`](../../../src/compiler/passes/memory/memory_plan_pass.ts) reads `config.memory.alignment`, then the target's `cacheLineSizeBytes`, then falls back to 64.

### Scopes are separate arenas

Every buffer carries a `scope` — `global`, `shared`, `local` — and the assignment keeps one `MemoryPool` per scope ([`buffer_assignment.ts:198`](../../../src/compiler/passes/memory/buffer_assignment.ts)). This is not an optimization but a necessity: on a GPU those are physically different memories, and an offset in one means nothing in the other. `peakMemory()` with no argument sums the pools ([`buffer_assignment.ts:253`](../../../src/compiler/passes/memory/buffer_assignment.ts)), which is the right total for a CPU where there is one pool and a number to read carefully on a GPU where there are three.

## 50.5 Lab — one flat block of bytes

```bash
node docs/part9/ch50-arena-allocation/labs/01-one-flat-block-of-bytes.mjs
```

The lab compiles Chapter 49's chain on a 32 × 32 input with `poolAllocation` switched on, which is the setting that makes the allocator's answer visible in the generated code rather than only in a trace event.

```
=== one ArrayBuffer, every temporary at an offset inside it ===
  const _mem_pool = new ArrayBuffer(8324);

  offset     bytes   buffers placed there
       0      4096   buf_8, buf_18   <-- sharing one address
    4096      4096   buf_4, buf_13   <-- sharing one address
    8192         4   buf_5
    8256         4   buf_6
    8320         4   buf_7
```

One `ArrayBuffer` and five addresses, and the two lines marked are Chapter 49's staircase cashed in. `buf_8` is live over `[0, 1]` and `buf_18` over `[2, 3]`; they do not interfere, so they were given the same 4,096 bytes. The generated code says so literally — two `Float32Array` views constructed on the same buffer at the same offset.

The second sharing line is the more instructive one. `buf_4` is a *scalar constant* — the `2` in `mul(2)` — living at index 0 only, and `buf_13` is a full 4,096-byte intermediate living over `[1, 2]`. Disjoint intervals, so they share an address, and the region is as large as the larger of the two. A 4-byte constant and a 4-kilobyte tensor at one address is not a mistake; it is Definition 50.2 doing exactly what it says.

```
=== alignment ===
  distinct offsets: 0, 4096, 8192, 8256, 8320
  every offset is a multiple of 64: true
  the scalar run steps by 64, 64 — a 4-byte scalar still costs a 64-byte slot
```

The scalar run is the alignment made visible: three 4-byte constants at 8192, 8256 and 8320. Each uses four bytes and occupies sixty-four. That is 94% waste on those three buffers and 180 bytes on the program, and it is the right trade — the alternative is unaligned views, which the language will not construct and the hardware would not like.

```
=== what the packing achieved ===
  planner's reported peak      : 8384 bytes
  arena actually emitted       : 8324 bytes
  sum of every temporary       : 12544 bytes
  temporaries the planner saw  : 7
```

12,544 bytes of buffers in an 8,324-byte arena. The gap between the reported peak (8,384) and the emitted arena (8,324) is the alignment padding on the final scalar, which the pool does not need to round up because nothing follows it.

```
=== best-fit against first-fit, same program ===
  best-fit   peak=  8384  arena=  8324  offsets=[0 0 4096 4096 8192 8256 8320]
  first-fit  peak=  8384  arena=  8324  offsets=[0 0 4096 4096 8192 8256 8320]
```

**Identical, offset for offset.** That is worth expecting rather than being disappointed by: the two strategies differ only when the sweep meets a gap that is big enough but not the tightest, and on a program whose buffers come in two sizes and whose intervals form a staircase, the first gap that fits is the only gap that fits. The strategies are pinned apart at the unit level instead, in [`assignment.test.js:41`](../../../tests/compiler/passes/memory/assignment.test.js), on intervals constructed to have a choice.

```
=== the packing did not change the answer ===
  max difference against an unpooled compile: 0
```

Which is the claim Theorem 49.3 and Definition 50.2 exist to support, checked rather than assumed.

**Try this.** Set `alignment` to `4` in the `memory` options and watch the scalar run collapse from 64-byte steps to 4-byte ones, then confirm the answer is still bit-identical. Then set it to `256` and watch the arena grow.

## 50.6 Traps and limits

- **A zero-size buffer is skipped entirely and has no assignment.** [`buffer_assignment.ts:185`](../../../src/compiler/passes/memory/buffer_assignment.ts) `continue`s on `size === 0`, so `getOffset` later returns `-1` for it ([`buffer_assignment.ts:244`](../../../src/compiler/passes/memory/buffer_assignment.ts)) — the same sentinel `numel()` uses for "unknown" (Chapter 10 §10.7). A caller that does not distinguish "not placed" from "placed at −1" has two ways to be wrong and no type to stop it.
- **Every dynamically-sized buffer is assigned offset 0.** [`buffer_assignment.ts:186`](../../../src/compiler/passes/memory/buffer_assignment.ts) handles `size < 0` by recording `{ offset: 0, size: 0, isDynamic: true }` and moving on. Two dynamic buffers therefore carry the same offset with no interference check between them, because the interference sweep is skipped for both. Nothing downstream reads those offsets as addresses today — `_assignPoolOffsets` explicitly excludes `isDynamic` buffers from the pool ([`memory_planning.ts:162`](../../../src/compiler/passes/memory/memory_planning.ts)) — so the value is inert rather than wrong, and it is inert by the grace of one filter in a different file.
- **The interference sweep is linear in everything placed, not in what interferes.** [`buffer_assignment.ts:226`](../../../src/compiler/passes/memory/buffer_assignment.ts) walks all of `placed` for the scope and filters, so assigning `n` buffers costs `Θ(n²)` comparisons. For the buffer counts a lowered `PrimFunc` produces — tens, occasionally hundreds — that is nothing, and it is the reason no interval tree is warranted. It is worth knowing before pointing this allocator at a program with a hundred thousand temporaries.
- **The in-place fast path depends on the sort order and fails silently.** [`buffer_assignment.ts:169`](../../../src/compiler/passes/memory/buffer_assignment.ts) gives an in-place destination its source's offset — but only `if (srcAssignment)`, that is, only if the source has already been placed. The order is size-descending, and an in-place pair has equal sizes by construction (Chapter 51), so the tie-break on `firstUse` decides it, and a source used *later* than its destination is placed later. When that happens the `if` falls through and the destination is allocated its own bytes, with no diagnostic. The in-place is silently downgraded to an ordinary allocation.
- **`peakMemory()` sums pools that are never live simultaneously in the same address space.** [`buffer_assignment.ts:253`](../../../src/compiler/passes/memory/buffer_assignment.ts). On a GPU target, adding global, shared and local peaks produces a number that describes no physical resource. Read the per-scope figure instead; the summed one is meaningful only where there is one pool.
- **Height is minimized, fragmentation is not measured.** The allocator reports the top of the arena and nothing about the holes below it. A layout with 40% of its bytes stranded in unusable gaps and one with none report the same peak if their heights match, and nothing in the trace distinguishes them.

## 50.7 Read the tests

- [`tests/compiler/passes/memory/assignment.test.js:23`](../../../tests/compiler/passes/memory/assignment.test.js) — the gap strategies given an actual choice: first-fit takes the lowest hole, best-fit the tightest.
- [`tests/compiler/passes/memory/assignment.test.js:49`](../../../tests/compiler/passes/memory/assignment.test.js) — the properties this chapter argued: interfering intervals get disjoint offsets, non-overlapping ones reuse memory, and the order really is size-descending.
- [`tests/compiler/passes/memory/assignment.test.js:128`](../../../tests/compiler/passes/memory/assignment.test.js) — the regression worth reading, because it pins Definition 50.2 rather than an implementation detail: simultaneously-live buffers never share memory, checked against many randomly generated interval sets.
- [`tests/compiler/passes/memory/pool-allocation.test.js`](../../../tests/compiler/passes/memory/pool-allocation.test.js) — the arena as the backends see it, which is what §50.5 read out of the generated source.

---

**Next:** [Chapter 51 — In-place reuse and donation](../ch51-inplace-and-donation/README.md), where two buffers stop sharing an address after the fact and start being the same buffer.
