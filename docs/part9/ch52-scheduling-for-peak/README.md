# Chapter 52 — Scheduling to lower peak memory

Chapters 49 through 51 took the program's statement order as a fact and worked around it: intervals were computed from that order, the packer packed against those intervals, and in-place reuse asked what a single block did. This chapter removes the assumption. **The order of independent statements is not part of the program's meaning**, and choosing it differently changes peak memory by more than any of the previous three chapters could.

## 52.1 The problem: the widest point is a choice

Here is a computation with three independent halves:

```js
const wa = p.mul(p), wb = q.mul(q), wc = r.mul(r);
const ra = wa.sum(1), rb = wb.sum(1), rc = wc.sum(1);
return ra.add(rb).add(rc);
```

Each `mul` produces a 32 KiB intermediate. Each `sum` collapses one to 8 bytes. Run the statements in the order written and all three wide buffers are alive before the first `sum` runs: the widest point holds 96 KiB.

Now run them in a different order — `wa`, `ra`, `wb`, `rb`, `wc`, `rc` — and no two wide buffers are ever alive together. The widest point holds 32 KiB. Same operations, same dependences, same results, one third of the memory.

Nothing in Chapters 49–51 can find this. Liveness *reports* the intervals that the order produces; the packer packs whatever it is given; in-place reuse looks inside one block. The intervals themselves were the thing that needed changing, and they are a consequence of the order.

## 52.2 Intuition: finish what you start

The good order above has a name in every other setting it appears in: depth-first. Finish one subtree completely, releasing everything it allocated, before starting the next.

The bad order is breadth-first: open everything, then close everything. It is also, unfortunately, the order that falls out of writing code naturally, because a person computing three things writes the three computations next to each other.

The rule of thumb — **consume a value as soon as it exists** — is right often enough to be worth carrying, and it is not always right. A statement that frees 32 KiB is worth running early; a statement that allocates 32 KiB is worth running late; and a statement that does both is worth running early only if it frees more than it allocates. That last clause is what turns a rule of thumb into a heuristic with a score.

## 52.3 Theory

> **Definition 52.1 (Schedule).** **(classical)** Given statements with a dependence relation, a *schedule* is a topological order of them: every statement appears after all statements it depends on.

Any schedule computes the same results — that is what a dependence relation is for, and Chapter 36 is where the relation comes from. What differs is the memory.

> **Definition 52.2 (Peak of a schedule).** **(classical)** Walk a schedule in order, maintaining a live set. A buffer enters the live set at its first use and leaves after its last. The *peak* is the maximum total size of the live set over the walk.

> **Theorem 52.3 (Minimising peak is NP-hard; Sethi, 1975).** **(classical)** Deciding whether a directed acyclic graph of computations admits a schedule whose peak is at most `k` is NP-complete, already for unit-sized values, where it is the *register sufficiency* problem.

So again, heuristics. Two are worth knowing, and this compiler runs both.

> **Definition 52.4 (List scheduling by net release).** **(classical)** Maintain the set of statements whose predecessors have all run. Repeatedly pick the one maximizing `(bytes it frees) − (bytes it newly allocates)`, run it, and update.

> **Definition 52.5 (Depth-first by subgraph weight).** **(classical)** Give each statement a weight: the total bytes allocated by it and everything reachable from it. Traverse depth-first, visiting the *heaviest* ready statement first among those released, so the expensive subtree is finished while the cheap ones are still unstarted.

Neither dominates. List scheduling is greedy about the next step and can be led into a state where every remaining choice is bad; depth-first commits to a subtree and is blind to a cheap statement elsewhere that would have freed a large buffer. On the shape in §52.1 both find the good order, for different reasons — list scheduling because `ra` frees 32 KiB and allocates 8 bytes, depth-first because `wa`'s subtree is heavy and gets finished first.

The escape from having to choose is to run both:

> **Proposition 52.6 (Best-of-*k* is never worse).** **(stated here)** Let `H₁, …, H_k` be schedule heuristics and let `σ₀` be the program's original order. Simulating each and keeping the argmin of peak yields a schedule whose peak is at most that of `σ₀`.
>
> *Proof.* `σ₀` is in the candidate set, so the minimum over the set is at most its peak. ∎

That is trivial as mathematics and it is the entire safety argument for the pass. It is the same shape as the autotuner's fallback floor (Chapter 47): *measure the thing you already have, and only accept a replacement that beats it.* A transformation that can be evaluated exactly and cheaply before being applied does not need to be conservative — it needs to be checked.

## 52.4 In mlfw: units, edges, two heuristics, and a simulator

[`memory_scheduler.ts:248`](../../../src/compiler/passes/memory/memory_scheduler.ts) is the pass; everything above it is the machinery.

### What may move

A `ScheduleUnit` ([`memory_scheduler.ts:20`](../../../src/compiler/passes/memory/memory_scheduler.ts)) is one top-level statement of the function body, with the buffer regions it reads and writes. Only the top level is considered — `run` requires the body to be a `SeqNode` with at least two statements ([`memory_scheduler.ts:258`](../../../src/compiler/passes/memory/memory_scheduler.ts)) and moves whole statements within it. A loop nest is one unit and its interior is never rearranged, which is Part VII's job and needs Part VI's legality machinery.

Some statements may not move at all ([`memory_scheduler.ts:18`](../../../src/compiler/passes/memory/memory_scheduler.ts)):

```ts
export const UNSEQUENCED_EFFECT_NODES = new Set(['CallExternNode', 'SyncThreadsNode', 'WhileNode']);
```

and the response to meeting one is severe — [`memory_scheduler.ts:59`](../../../src/compiler/passes/memory/memory_scheduler.ts) returns `null` for the *whole function*, not just that unit. An external call may do anything, a barrier means something only where it stands, and a `while` has a trip count nobody knows. Rather than model any of that, the pass declines to run. Chapter 14's rule about reporting `CHANGED` conservatively has a partner here: when in doubt, do nothing at all.

### The edges

`buildDependenceEdges` ([`memory_scheduler.ts:67`](../../../src/compiler/passes/memory/memory_scheduler.ts)) groups accesses by buffer and hands each group to `linkAccessUnits`, the same routine Chapter 39's block scopes use. That is where read-after-write, write-after-read and write-after-write become edges. Reusing it matters: the legality of this pass is exactly the legality of Part VI's dependence relation, and a second implementation would be a second place for the answer to be wrong.

Note which dependences are being honoured. A *write-after-read* edge is the one that keeps this pass sound in the presence of Chapter 51's aliasing — two statements touching the same buffer are ordered even when the second only overwrites it.

### The objective, computed exactly

`simulatePeak` ([`memory_scheduler.ts:114`](../../../src/compiler/passes/memory/memory_scheduler.ts)) is Definition 52.2:

```ts
  for (const unit of order) {
    for (const buffer of unit.temporaries) {
      if (live.has(buffer)) continue;
      live.add(buffer);
      current += sizes.get(buffer) as number;
    }
    if (current > peak) peak = current;
    for (const buffer of unit.temporaries) {
      const left = (remaining.get(buffer) as number) - 1;
      remaining.set(buffer, left);
      if (left === 0 && live.delete(buffer)) current -= sizes.get(buffer) as number;
    }
  }
```

Allocate first, take the maximum, then free — in that order, because a statement's inputs and outputs are simultaneously live while it runs. `remaining` is a use count, decremented per statement, and a buffer is freed exactly when its count reaches zero. This is a *simulation*, not an estimate: for the order given, the number it returns is the peak that order will have.

Having an exact, cheap objective is what makes Proposition 52.6 usable, and it is worth contrasting with Chapter 46, where the cost model predicts a runtime it cannot compute and the search has to measure. Here the quantity of interest is a property of the schedule alone.

### The two heuristics

`listSchedule` ([`memory_scheduler.ts:145`](../../../src/compiler/passes/memory/memory_scheduler.ts)) is Definition 52.4, and its scoring function is three lines:

```ts
    for (const buffer of unit.temporaries) {
      const bytes = sizes.get(buffer) as number;
      if (remaining.get(buffer) === 1) score += bytes;
      if (!live.has(buffer)) score -= bytes;
    }
```

`remaining === 1` means this statement is the buffer's last use, so running it frees those bytes: credit. `!live.has(buffer)` means the buffer is not yet allocated, so running it costs those bytes: debit. A statement can score both for the same buffer — first and last use in one place — and then the two cancel, correctly.

`dfsSchedule` ([`memory_scheduler.ts:200`](../../../src/compiler/passes/memory/memory_scheduler.ts)) is Definition 52.5. `subgraphBytes` ([`memory_scheduler.ts:188`](../../../src/compiler/passes/memory/memory_scheduler.ts)) computes weights in one reverse pass, and the traversal uses an explicit stack with released successors pushed lightest-first so the heaviest is popped first.

### The floor

[`memory_scheduler.ts:221`](../../../src/compiler/passes/memory/memory_scheduler.ts) is Proposition 52.6:

```ts
  const original = [...units];
  let bestOrder = original;
  let bestPeak = simulatePeak(original, sizes);

  for (const candidate of [listSchedule(units, sizes), dfsSchedule(units, sizes)]) {
    if (!candidate) continue;
    const peak = simulatePeak(candidate, sizes);
    if (peak < bestPeak) {
      bestPeak = peak;
      bestOrder = candidate;
    }
  }
  return bestOrder === original ? null : { order: bestOrder, peak: bestPeak, originalPeak: simulatePeak(original, sizes) };
```

The original order is a candidate, seeded as the incumbent, and a heuristic replaces it only on a strict improvement. Returning `null` when nothing wins means the pass rewrites nothing and emits no event, so **silence in the trace means the program was already as good as the heuristics could make it** — not that the pass failed to run.

## 52.5 Lab — the order is not given

```bash
node docs/part9/ch52-scheduling-for-peak/labs/01-the-order-is-not-given.mjs
```

```
  scheduleForPeak=false   planned peak =  98432 bytes   pass reported: no change
  scheduleForPeak=true    planned peak =  32960 bytes   pass reported: 98316 -> 32788

  ratio: 2.99x
  the two orders compute the same numbers: max difference 0
  output: 12369.940, 12064.221
```

Three times the memory for the same arithmetic, and the pass finds it. Note that the two peaks the pass reports — 98,316 and 32,788 — are its own simulation over temporaries, while the planner's figures of 98,432 and 32,960 come from the arena of Chapter 50 after the reorder. The two agree on the ratio and differ by the constants and alignment the arena adds, which is the honest relationship between a heuristic's objective and the number the next pass produces.

The second block is where the chapter's real lesson is:

```
=== the same computation, already written in a good order ===

  scheduleForPeak=false   planned peak =  33024 bytes   pass reported: no change
  scheduleForPeak=true    planned peak =  32960 bytes   pass reported: 32796 -> 32788
```

Writing the same computation as three complete `p.mul(p).sum(1)` chains instead of three `mul`s followed by three `sum`s takes the *unscheduled* peak from 98,432 to 33,024. **How you write it matters enormously — right up until the pass runs, after which it does not matter at all.** Both versions land within 64 bytes of each other. That is the most useful thing this pass does: not the 3× on a badly-written program, but making the question stop being the user's problem.

```
=== a program with no independent work to move ===

  planned peak = 33024 bytes   pass reported: no change
  every statement depends on the one before it, so the dependence graph admits exactly one order.
```

An elementwise chain has one topological order, so its peak is not a choice and there is nothing to schedule. Chapter 51 is the only lever on that shape, which is why it is the chapter with the difficult legality condition.

**Try this.** Take the wide-then-narrow program to five branches and predict the ratio before running. Then insert a `while`-shaped construct anywhere in the function and watch the pass decline the whole function rather than the one statement.

## 52.6 Traps and limits

- **One unmovable statement disables the pass for the entire function.** [`memory_scheduler.ts:59`](../../../src/compiler/passes/memory/memory_scheduler.ts) returns `null` from `buildUnits` the moment any node in any statement is in `UNSEQUENCED_EFFECT_NODES`, and `run` then returns. A function with forty reorderable statements and one `CallExternNode` — which is what calling into cuBLAS looks like (Chapter 58) — gets no scheduling at all. Pinning the offending unit and scheduling around it is the obvious refinement and is not implemented.
- **Only the top level of the body is scheduled.** Statements inside a `ForNode` are one unit with the loop. So a program whose interesting structure is inside a loop nest — every scan, every RNN — presents the scheduler with one statement and nothing to do.
- **The pass optimizes a proxy for the number the planner reports.** `simulatePeak` sums buffer sizes with no alignment and no arena packing, while Chapter 50's assignment pads every size to 64 bytes and may leave holes. The two agree closely, and they are not the same function, so a reorder that improves the simulation by a few bytes can leave the planner's peak unchanged. The lab's second block is exactly this case: 32,796 → 32,788 in the simulation, 33,024 → 32,960 in the plan.
- **Buffers with non-finite or non-positive sizes are silently excluded from the objective.** [`memory_scheduler.ts:96`](../../../src/compiler/passes/memory/memory_scheduler.ts) skips them when building `sizes`, and `useCounts` only counts what is in `sizes`. A dynamically-shaped buffer therefore weighs nothing in the simulation, so under `dynamic_shapes` the pass optimizes the placement of the static buffers and treats the dynamic ones — usually the large ones — as free.
- **`buildScheduleUnits` is exported and has no caller, and it double-counts if used.** [`memory_scheduler.ts:240`](../../../src/compiler/passes/memory/memory_scheduler.ts) is reachable from neither `src/` nor `tests/`; the pass builds its units inline instead. It matters only because of what it would do: it calls `assignTemporaries` at [`:244`](../../../src/compiler/passes/memory/memory_scheduler.ts) and discards the result, and `scheduleForPeakMemory` calls it again at [`:222`](../../../src/compiler/passes/memory/memory_scheduler.ts) — and `assignTemporaries` *appends* to `unit.temporaries` rather than replacing it, so units passed through both would list every buffer twice. The peak survives that, since `simulatePeak`'s allocation loop guards on `live.has` and the doubled use counts are decremented twice per statement, cancelling. It is the sort of latent duplication that only stays harmless while the function has no users.
- **Peak is the only objective.** Reordering for memory can lengthen the distance between a value's definition and its use, which is bad for cache locality, and the pass has no term for that. It also cannot know that two statements it separated were about to be fused, because fusion has already run (Chapter 24). A change that lowers peak memory and slows the program down is not something this pass can detect.

## 52.7 Read the tests

- [`tests/compiler/passes/memory/memory-scheduler.test.js:41`](../../../tests/compiler/passes/memory/memory-scheduler.test.js) — the claim of §52.5, pinned: a graph of wide intermediates before their reductions has a strictly lower planned peak with the pass on.
- [`tests/compiler/passes/memory/memory-scheduler.test.js:55`](../../../tests/compiler/passes/memory/memory-scheduler.test.js) — the one that matters most, because it is the legality of the whole pass: the reordered program is bit-identical to the unscheduled one.
- [`tests/compiler/passes/memory/memory-scheduler.test.js:59`](../../../tests/compiler/passes/memory/memory-scheduler.test.js) — Proposition 52.6 as a test: when it cannot improve, it leaves the order alone.

---

Part IX ends here. The plan is complete: every temporary has a lifetime, an address, possibly a buffer it shares with the operation that produced it, and an order chosen to keep the widest moment narrow. [Part X](../../OUTLINE.md) takes that plan and emits code against it — which is where the two numbers this part has been careful to keep apart, what the plan says and what the machine allocates, finally have to agree.
