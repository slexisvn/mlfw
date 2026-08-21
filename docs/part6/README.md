# Part VI — Lowering to loops: TIR

Every program in Parts 0 to V is a graph of whole-tensor operations. This part is where one becomes loops.

It is the point of no return in the pipeline, and the reason the earlier parts are ordered the way they are. Differentiation needs dataflow, fusion needs values, layout needs a type — none of those survive the level change. After it, the program is buffers, subscripts and iteration order, which is what scheduling (Part VII), autotuning (Part VIII), memory planning (Part IX) and code generation (Part X) are all about. So Part VI is a hinge: two chapters describing the new representation, one describing the translation, and three describing the arithmetic that everything downstream reasons with.

| Chapter | Title | The question it answers |
|---|---|---|
| [32](ch32-tensor-algebra-to-loops/README.md) | From tensor algebra to loop nests | Why is there a second IR at all, and what does the translation into it cost you? |
| [33](ch33-buffers-blocks-itervars/README.md) | Buffers, blocks, iteration variables | After a scheduler has tiled and reordered a nest, what still says what the program computes? |
| [34](ch34-lowering-rules/README.md) | Lowering rules | 96 operations, 66 rules. How is one selected, and what happened to the other 30? |
| [35](ch35-index-arithmetic/README.md) | Index arithmetic | The subscripts are now the program. What are they made of, and when does a division disappear? |
| [36](ch36-dependence-analysis/README.md) | Dependence analysis | Can these two iterations touch the same element — and if so, how far apart are they? |
| [37](ch37-proving-things-about-indices/README.md) | Proving things about indices | Which guards are unnecessary, and in what precise sense is the analysis allowed to be wrong? |

## The argument in one paragraph

A graph says *what*; hardware runs *how*, so somewhere a step invents an iteration order and a set of addresses — and that step is a function, not a bijection, which is why every decision needing dataflow has already been made by the time it runs (Chapter 32). What comes out is written in a language of 21 node kinds, in which the loops carry the plan and a `block` carries the computation, each of its axes declared spatial or reducing; the declaration is what lets a scheduler decide legality without an analysis, and nothing verifies it (Chapter 33). The translation itself is 66 rules selected by a two-level priority scheme, most of them one shared eleven-line skeleton plus a callback, and the 30 operations with no rule are exactly the 21 the decomposition pass removes and the 9 the driver handles — an arithmetic that closes (Chapter 34). Every rule builds a subscript, and subscripts are affine forms: flattening a coordinate is writing a mixed-radix numeral, unflattening is reading one, and a division by `c` vanishes exactly when `c` falls on a digit boundary — which is why one reshape compiles to a bare index and another pays a division per element, measured here at 1.10× on a pure copy (Chapter 35). Given two such subscripts, deciding whether they can name the same element is a Diophantine equation with three tests of decreasing precision, and the answer is a direction vector that says which loops may be reordered — the reduction axis of every accumulation being the one that may not (Chapter 36). And all of it rests on interval arithmetic over the loop extents: sound in one direction, unable to relate two variables, and load-bearing not only for deleting `pad`'s two redundant comparisons but for the substitution of truncating division for floor division, which is legal exactly where the dividend is provably non-negative (Chapter 37).

## What Part VI establishes for later parts

- **The block and its iteration-variable kinds** (Definitions 33.2 and 33.3) as the object every scheduling primitive in Part VII moves without rewriting, and the contract every legality question consults first.
- **The affine form** (Definition 35.1) as the class every later analysis is confined to — and `split` staying inside it while a loop *fuse* leaves it, which is why Chapter 40's primitives are not symmetric.
- **Direction vectors** (Definition 36.2) as the input to Chapter 42's permutation legality and Chapter 43's cross-thread race detection.
- **`EVERYTHING`** (§37.4) as the honest answer, and the pattern of turning an unprovable precondition into a runtime obligation on the caller — which is what Chapter 62's guards do for shapes.
- **`//` and `%` are floor everywhere**, with `tdiv`/`tmod` introduced only where Theorem 37.6 makes them equivalent. Chapters 54 to 58 hold four backends to that one definition.

## Labs

```bash
npm run build   # once, if you have not already

node docs/part6/ch32-tensor-algebra-to-loops/labs/01-one-op-three-levels.mjs
node docs/part6/ch32-tensor-algebra-to-loops/labs/02-lowering-reads-the-users.mjs
node docs/part6/ch33-buffers-blocks-itervars/labs/01-anatomy-of-a-block.mjs
node docs/part6/ch33-buffers-blocks-itervars/labs/02-buffers-and-scopes.mjs
node docs/part6/ch34-lowering-rules/labs/01-the-catalogue.mjs
node docs/part6/ch34-lowering-rules/labs/02-one-rule-many-ops.mjs
node docs/part6/ch35-index-arithmetic/labs/01-flatten-and-unflatten.mjs
node docs/part6/ch35-index-arithmetic/labs/02-what-the-index-costs.mjs
node docs/part6/ch36-dependence-analysis/labs/01-which-loops-may-run-in-parallel.mjs
node docs/part6/ch36-dependence-analysis/labs/02-reading-the-dependence.mjs
node docs/part6/ch37-proving-things-about-indices/labs/01-guards-that-disappear.mjs
node docs/part6/ch37-proving-things-about-indices/labs/02-what-the-analyzer-cannot-see.mjs
```

Every lab in this part uses one public entry point, `compile`, and reads the TIR through the trace stream of Chapter 18. `irSnapshot: { afterLowering: true }` fires once per function immediately after the lowering phase; `afterScheduling: true` fires again after `SchedulePass`, which is what the two Chapter 36 labs compare. Most labs also pass `fusion: { enabled: false }` so that each operation produces its own nest and the output stays legible — the loop bodies are the same either way.

Nothing in this part reaches past the documented surface. `compile`, `TraceLevel`, `irSnapshot` and `compiled.source()` are the entire toolkit, which is a change from Parts III and V and is possible because the TIR text is a first-class trace artefact.

Only §35.6's timings are machine-specific. Every loop count, block name, subscript, comparison count and buffer count in this part is deterministic and should reproduce exactly.

## A note on what this part found

Part V ended on three mechanisms that were implemented and unreachable. Part VI's findings are a different species: everything here is reachable, runs on every compilation, and is either **less precise than its own proof allows** or **a declaration nothing checks**.

The largest is Chapter 35's. When a reshape fails the exact-split test, the compiler emits `(f tdiv c)·c + (f tmod c)` — an expression that is identically `f` for non-negative `f`. It chose truncating division *because* it proved `f ≥ 0`. So it holds a proof that the whole subscript is the identity and emits four arithmetic operations per element anyway, because the rewrite is not in the set. Measured at 1.10× on a 98,304-element copy.

The second is Chapter 33's. A block declares which buffers it reads, and every accumulation block in the compiler omits its own accumulator — because the read set is built from the operation's *operands* and the accumulator is a *result*. Nothing verifies it. Nothing breaks today, because every consumer that must be right walks the body instead; what consumes the declaration alone is Part VIII's workload key. In the same family: `BufferRegion` is exported and never constructed, so a block's declared regions are always "this buffer, unspecified"; `initBody` is implemented on both sides and set by no rule; and the iteration-variable kinds, which Chapter 36's dependence analysis is explicitly overruled by, are checked by nothing at all.

The rest are smaller and named where they occur: a printer that covers 17 of 21 node kinds and silently drops a unary minus; a lowering rule and an inline fusion builder both registered for an operation name the registry does not contain; a modulo bound that vanishes under symbolic shapes although the fact still holds; a divisor-split test that checks one condition more than its theorem needs; and a missing-rule error that reports one operation where it could report all of them.

As in Parts IV and V, each finding is named with its file and line, each is reproducible by a lab here, and each is carried into the outline's Appendix E.
