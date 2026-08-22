# Part IX — Memory

Part VIII searched for the fastest arrangement of a loop nest. This part asks a question with a single right answer instead of a best guess: **the program needs these buffers; how few bytes will hold them?**

It is the first part since Part II that is not about speed. It is about whether the program runs at all. A model that needs 40 GB of activations does not run slowly on a 24 GB card; it does not run. And unlike fusion or tiling, the decisions here are almost entirely *legality* decisions — two buffers may share bytes or they may not, and getting it wrong does not cost time, it costs correctness.

| Chapter | Title | The question it answers |
|---|---|---|
| [49](ch49-buffer-lifetimes/README.md) | Buffer lifetimes | When is a buffer live, and what does it mean for two of them to interfere? |
| [50](ch50-arena-allocation/README.md) | Arena allocation | Given the interferences, where in one flat block of bytes does each buffer go? |
| [51](ch51-inplace-and-donation/README.md) | In-place reuse and donation | When may an operation overwrite the very input it is reading? |
| [52](ch52-scheduling-for-peak/README.md) | Scheduling to lower peak memory | The order of independent work is free to change. Which order needs the least memory? |

## The argument in one paragraph

A temporary buffer is not needed for the whole program — it is needed from the statement that first touches it to the statement that last does, which over a linearized block order is an *interval*, and two buffers whose intervals are disjoint can occupy the same bytes (Chapter 49). Turning a set of intervals into addresses is a packing problem: buffers become rectangles whose width is a lifetime and whose height is a size, the arena is a strip, and minimizing the strip's height is NP-hard in general, so the compiler sorts by size descending and places each buffer in the first or tightest hole that no interfering buffer occupies (Chapter 50). Some pairs can do better than sharing an address after the fact: if an operation reads a buffer at exactly the index it is about to write, and nothing reads that buffer afterwards, the output can be computed *into the input*, which is where the deepest legality condition in this part lives — the index equality is not a convenience, it is the whole proof (Chapter 51). And because the order of independent statements is not fixed, peak memory is a property of the schedule rather than of the program: reordering to finish one subgraph before starting the next can shrink the high-water mark, minimizing it is NP-hard again, and the compiler runs two greedy heuristics and keeps whichever beats the original order (Chapter 52).

## What Part IX establishes for later parts

Part X generates code against the plan this part produces: a buffer's `poolByteOffset` and the `AllocateNode`s inserted here are what the backends turn into real allocations, and Chapter 53's flattened indices are computed against those offsets. Part XI's runtime owns the arena itself.

## What this part does not cover

Two memory mechanisms that ship in this compiler are deliberately elsewhere.

**Rematerialization** — dropping a value and recomputing it later instead of holding it — buys memory with time rather than packing what is already there, and the policy that decides it belongs to the backward pass that motivates it: [Chapter 30](../part5/ch30-memory-for-recomputation/README.md).

**The plan-level planner.** Everything here works inside one `PrimFunc`. When a graph compiles to a multi-step executor plan rather than a single kernel, a second planner runs over that plan's slots: it reuses slots whose lifetimes are disjoint and donates a dying elementwise input to its output, refusing any step that reindexes what it reads ([`plan_buffer_assignment.ts`](../../src/compiler/passes/memory/plan_buffer_assignment.ts), gated by `memory.planReuse` and `memory.planDonation`, both on by default). That is Chapters 49, 50 and 51 repeated at the granularity the runtime sees, and it is described with the runtime, in Part XI. Chapter 51 §51.6 marks the one place where this part's verdict would otherwise be read as the whole compiler's.

## Labs

```bash
npm run build   # once, if you have not already

node docs/part9/ch49-buffer-lifetimes/labs/01-when-is-a-buffer-live.mjs
node docs/part9/ch50-arena-allocation/labs/01-one-flat-block-of-bytes.mjs
node docs/part9/ch51-inplace-and-donation/labs/01-overwriting-your-own-input.mjs
node docs/part9/ch52-scheduling-for-peak/labs/01-the-order-is-not-given.mjs
```

One lab per chapter, and all four use only `compile` and the trace stream of Chapter 18 — nothing here reaches past the documented surface, which is a change from Parts VII and VIII and is possible because a memory plan is reported as an ordinary trace event. Each lab switches off the optimizations belonging to *later* chapters so that the number it reports is the one its own chapter is responsible for: Chapter 49's lab pins in-place reuse off, Chapter 50's turns pool allocation on, Chapter 51's sweeps both. Nothing in this part is timed, and every byte count, interval, offset and candidate count is deterministic and should reproduce exactly.

## A note on what this part found

Part VIII's findings were about seams between components. Part IX's are about the gap between **a plan and a program** — which is the same gap the preface below is about, and the reason this part reports two numbers where every other part reports one.

The one that matters is closed and is kept in the text because the chase is the content: a version of this pass recorded its in-place candidates, dropped the destination's allocation, materialized nothing, and removed the pair from the slot reuse that would otherwise have covered it — so the reported peak halved while the emitted program allocated 2.5× *more* (§51.6). Nothing failed, no test went red, and the trace reported an improvement. Every lab in this part now prints the plan's figure and the program's figure side by side for that reason.

The rest are smaller and named where they occur: donation is implemented, tested and unreachable from `compile()`; `layoutsMatch` compares two dynamic dimensions and calls them equal, producing candidates nothing acts on; one unmovable statement disables Chapter 52's pass for a whole function; a zero-size buffer's "no assignment" is the same `-1` sentinel as "unknown size"; and `buildScheduleUnits` is exported, uncalled, and would double-count if it were called. Each is carried into the outline's [Appendix E](../OUTLINE.md).

## A caution about this part's numbers

Every other part of this book can quote one number for a transformation's effect. This part has two: what the **plan** says the peak is, and what the **generated program** actually allocates. They are produced by different code and they can disagree — Chapter 51 §51.6 reconstructs a version of this compiler in which they moved in opposite directions, which is how the gap between them stopped being invisible. They agree in every configuration these chapters measure, and they are still two numbers: the plan pads to the alignment, counts constants the backend folds away, and on a GPU sums pools that are not one address space. Wherever a chapter here reports memory, it says which of the two it measured.

---

**Next:** [Chapter 49 — Buffer lifetimes](ch49-buffer-lifetimes/README.md).
