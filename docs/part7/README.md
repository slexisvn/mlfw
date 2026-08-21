# Part VII — Scheduling

Part VI ended with a loop nest: correct, complete, and written as if the machine had one core, no vector unit and infinite memory bandwidth. This part is the language for saying how it should actually run.

It is the first part in which the compiler makes decisions that cannot be justified from the program alone. Every transformation up to here — fusion, differentiation, decomposition, lowering — had an argument that appealed only to what the program computes. A schedule appeals to what the *hardware* is, and the same program gets four different answers on four targets. So Part VII is where the book stops asking "what is this program?" and starts asking "what should this machine do with it?", and where the compiler's answer stops being unique.

| Chapter | Title | The question it answers |
|---|---|---|
| [38](ch38-separating-what-from-how/README.md) | Separating what from how | One graph, four machines. What has to survive the difference, and what is allowed to change? |
| [39](ch39-sref-tree-and-block-scopes/README.md) | The sref tree and block scopes | How does a schedule edit a nest from the middle without invalidating everything else? |
| [40](ch40-loop-primitives/README.md) | Loop primitives | Split, fuse, reorder, tile. What appears when the tile size does not divide the extent? |
| [41](ch41-memory-and-reduction-primitives/README.md) | Memory and reduction primitives | Which primitives allocate, which reassociate, and what does each of them have to promise? |
| [42](ch42-legality/README.md) | Legality | Which reorderings preserve what a nest computes, and who in this compiler gets to decide? |
| [43](ch43-scheduling-for-gpus/README.md) | Scheduling for GPUs | When the loops become threads, what is a race — and which races can a schedule repair? |

## The argument in one paragraph

A loop nest answers two questions at once — what is computed and in what order — and only the first is a property of the program, so the compiler splits them: the block is the algorithm, the loops are the plan, and twenty-two primitives change the second while holding the first fixed (Chapter 38). Doing that efficiently needs an index into the IR that survives being edited from the middle, which is the sref tree: one node per loop and per block, transparent to everything else, patched per subtree rather than rebuilt, with a second index over sibling blocks for the primitives that move one nest inside another (Chapter 39). The four shape primitives are all the same move, division with remainder read forwards or backwards, and the whole of their difficulty is that a tile size need not divide an extent: `split` rounds up and guards, which is sound for every extent, and `fuse` then cannot undo it (Chapter 40). The primitives that allocate or reassociate need more than arithmetic, and each of them names a price: `rfactor` needs an associative operator, which floating-point addition is not; `cacheRead` and `cacheWrite` need a buffer nobody else holds; `computeInline` needs an invertible affine write index, and pays for the deleted buffer in recomputation (Chapter 41). Legality is where it gets interesting: the classical answer is direction vectors and lexicographic positivity, and this compiler's answer is to consult the block's declared axis kinds *instead* whenever they are available — a proof traded for a declaration, and Chapter 42 is where that trade is priced. And on a GPU the bound loops stop existing, so legality becomes a race question, with two repairs available and one case that has none (Chapter 43).

## What Part VII establishes for later parts

- **The twenty-two primitives** (§38.4) as the alphabet Part VIII searches over, and `ScheduleTrace` (§40.7) as the serialisable record of one point in that space.
- **The schedule as a *partial* function** (Definition 38.1): each primitive refuses rather than repairs, so a search composing sound primitives cannot produce a wrong program however badly it searches — with `rfactor` the one primitive that leaves that guarantee, and §38.3 the statement of what it costs.
- **`split`'s guard** (Theorem 40.2) as the reason Part VIII's tile sizes need not divide anything, and Chapter 62's dynamic shapes as the case where they must, since a non-constant extent has no `⌈n/c⌉`.
- **Direction vectors** (Definition 42.2) as the object Chapter 42's permutation test and Chapter 43's race detector both reduce to.
- **`IterVarPolicy`** (§42.4) as the exact point at which Chapter 33's declaration becomes load-bearing, with Counterexample 42.9 as the price of it being wrong.
- **Launch geometry** (Definition 43.2) as the output Part X's runtime reads to size a kernel launch, and the reason Chapter 62's symbolic extents disable GPU parallelism entirely.

## Labs

Part VII's labs drive the scheduling primitives by hand, and `Schedule` is not part of the package's public surface — nothing outside the compiler is meant to reshape a loop nest. Each lab therefore imports [`docs/part7/_internals.mjs`](_internals.mjs), which bundles the internal surface listed in [`docs/tools/internals-entry.ts`](../tools/internals-entry.ts) with esbuild, a devDependency the repository already has. The bundle lands in the OS temp directory and takes about a tenth of a second to build; nothing is written inside the repository, and there is still no build step to run.

```bash
npm install   # once, if you have not already — the labs need esbuild

node docs/part7/ch38-separating-what-from-how/labs/01-one-program-four-schedules.mjs
node docs/part7/ch38-separating-what-from-how/labs/02-what-the-annotation-is-worth.mjs
node docs/part7/ch39-sref-tree-and-block-scopes/labs/01-the-sref-tree.mjs
node docs/part7/ch39-sref-tree-and-block-scopes/labs/02-block-scopes.mjs
node docs/part7/ch40-loop-primitives/labs/01-split-fuse-reorder-tile.mjs
node docs/part7/ch40-loop-primitives/labs/02-the-guard.mjs
node docs/part7/ch41-memory-and-reduction-primitives/labs/01-rfactor.mjs
node docs/part7/ch41-memory-and-reduction-primitives/labs/02-cache-inline-decompose.mjs
node docs/part7/ch42-legality/labs/01-what-the-primitive-refuses.mjs
node docs/part7/ch42-legality/labs/02-direction-vectors.mjs
node docs/part7/ch43-scheduling-for-gpus/labs/01-thread-bindings.mjs
node docs/part7/ch43-scheduling-for-gpus/labs/02-races-and-templates.mjs
```

Three helpers appear in nearly every lab and are worth knowing before the first one. `lowerToTir(fn, inputs, target)` traces a function and lowers the traced graph, giving a real `PrimFunc` — the same object `SchedulePass` receives. `toKernel(primFunc, target)` takes one all the way to callable source, so a hand-written schedule can be *run* and its answer compared against a baseline; four labs do exactly that, and it is how Chapter 40's missing guard and Chapter 41's reassociation are demonstrated rather than asserted. `printTensorIR` is Chapter 32's printer.

Five labs also use the public `compile()` entry point, where the point is to show what the shipping compiler does rather than what the primitives can do: §38.5, §38.6's CUDA default, §40.6's surviving division, §42.6's WASM reduction and all of §43.6.

Every loop name, extent, subscript, block name, dependence direction and emitted kernel in this part is deterministic and should reproduce exactly. The labs call `resetVarCounter()` where the fresh-variable counter would otherwise leak between sections. Nothing here is timed.

## A note on what this part found

Part VI's findings were about *precision* — analyses less exact than their own proofs allowed. Part VII's are about *reach*. Nine of the twenty-two primitives have no caller in `src/`, and the pattern worth carrying is that these are the ones that turn out to be wrong: nothing runs them, so nothing has ever disagreed with them. The legality question of Chapter 42 has three answers in three layers of the same repository, all of which come out right, only one of which is a proof. And two shipped defaults cost real parallelism on a GPU. Each finding is named with its file and line in the chapter that meets it, each is reproducible by a lab here, and all eighteen are collected in the outline's [Appendix E](../OUTLINE.md) rather than restated in this page.
