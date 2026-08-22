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

## A caution about this part's numbers

Every other part of this book can quote one number for a transformation's effect. This part has two, and they do not always agree: what the **plan** says the peak is, and what the **generated program** actually allocates. They are produced by different code, and Chapter 51 exhibits a configuration in which they move in opposite directions. Wherever a chapter here reports memory, it says which of the two it measured.

---

**Next:** [Chapter 49 — Buffer lifetimes](ch49-buffer-lifetimes/README.md).
