# Part XI — From compiler to framework

Ten parts have built a compiler. It takes a graph and returns a `CompiledKernel`: a name, a source string, a target and a metadata record. Nobody can use that.

A framework is what stands between a person writing `model.forward(x)` and the object Part X produces — and it is four mechanisms, not one. Something has to *run* a compiled kernel on four incompatible targets. Something has to decide, for every operation a user calls, which of several implementations applies and in what order. Something has to turn a closure into a graph in the first place. And something has to decide what happens when the shape the graph was compiled for is not the shape the next call brings.

This part is those four mechanisms and then the program that exercises all of them at once.

| Chapter | Title | The question it answers |
|---|---|---|
| [59](ch59-the-runtime-module/README.md) | The runtime module | How does one caller launch a kernel on four targets that agree on nothing? |
| [60](ch60-a-pytorch-style-dispatcher/README.md) | A PyTorch-style dispatcher | How does one call compose four layers whose order matters? |
| [61](ch61-tracing/README.md) | Tracing | Where does the graph come from, and what does the recording lose? |
| [62](ch62-dynamic-shapes/README.md) | Dynamic shapes | When may a kernel compiled for one shape be reused for another? |
| [63](ch63-training-end-to-end/README.md) | Training end to end | Is the compiled path the same computation as the eager one? |

## The argument in one paragraph

A runtime module is a dictionary from kernel name to compiled kernel and a lazily-filled dictionary from name to live instance; which of four wildly different launch procedures runs is decided by one map lookup on a string in the metadata, and `runtime.ts` mentions no target by name (Chapter 59). Above it, a dispatcher gives every tensor a set of small integer keys, unions them per call, takes the highest, and calls the kernel registered there with the key removed — so autograd, tracing and four backends compose as *layers* rather than competing as branches, and the entire compiler becomes reachable as the fallback implementation of any eager operation nobody hand-wrote (Chapter 60). Installing one more such layer is what tracing is: a kernel on every operator that records instead of computing, entered through a guard so that everything inside the traced function is captured, with the closure's free variables becoming the tail of the function's parameter list (Chapter 61). The shapes that trace bakes in are not constants but *symbols with promises attached* — an equality per static dimension, a positivity per dynamic one, an equality for every place something asked a symbol for a number — and a compiled kernel may be reused exactly when every promise still holds, which makes static compilation the same mechanism at its maximal setting (Chapter 62). And a training step is forward, backward, apply, with the first two compiled from one trace and the third left in eager JavaScript, so that turning the compiler on is one boolean on the trainer and no second training loop (Chapter 63).

## What Part XI establishes for Part XII

Part XII is about being sure the thing is right, and this part hands it two things.

**A method.** Every number in Chapter 63 came from running two independent implementations of the same computation and subtracting. That method found a one-ulp agreement across forty optimizer steps — the strongest correctness statement in the book — and, on the same page, a structurally wrong parameter update that no convergence test would ever have reported.

**A rule for reading the result.** Theorem 63.5 divides what a differential test can find into two kinds: a *small* discrepancy, which is reassociation or precision or a different approximation and needs a tolerance derived from the arithmetic; and a *structured* one — a permutation, a transpose, a shifted index — for which no tolerance is the right answer. Getting that distinction wrong is how a test suite ends up with a tolerance wide enough to hide a bug.

## Labs

```bash
npm run build   # once, if you have not already

node docs/part11/ch59-the-runtime-module/labs/01-four-fields-and-a-registry.mjs
node docs/part11/ch59-the-runtime-module/labs/02-slots-and-the-plan.mjs
node docs/part11/ch60-a-pytorch-style-dispatcher/labs/01-one-call-many-answers.mjs
node docs/part11/ch60-a-pytorch-style-dispatcher/labs/02-boxing-schemas-and-a-new-layer.mjs
node docs/part11/ch61-tracing/labs/01-recording-a-program.mjs
node docs/part11/ch61-tracing/labs/02-around-the-compiler.mjs
node docs/part11/ch62-dynamic-shapes/labs/01-symbols-and-guards.mjs
node docs/part11/ch62-dynamic-shapes/labs/02-the-caches-and-the-gap.mjs
node docs/part11/ch63-training-end-to-end/labs/01-one-step-three-ways.mjs
node docs/part11/ch63-training-end-to-end/labs/02-the-whole-framework.mjs
```

Ten labs, none of which needs a GPU, and — unlike [Part X](../part10/README.md)'s — every one of them **runs** what it builds. That is the difference this part makes: a backend's output can be read as text, but a runtime, a dispatcher, a cache and a training loop can only be checked by executing them. Three of the ten register something into a live registry from outside `src/` — a runtime backend, an operator, a backend kernel, a dispatch layer — which is the sharpest available test of whether an interface is really an interface.

The labs reach past the public surface, so they read the internal modules listed in [`docs/tools/internals-entry.ts`](../tools/internals-entry.ts) through [`_internals.mjs`](_internals.mjs), the same way Parts VII, VIII and X do: `npm run build` emits them as `dist/internals.node.js` beside the public bundle, and the labs refuse to run against a build older than `src/`.

## A note on what this part found

Part X's findings were about the gap between what a mechanism was written to do and what the pipeline feeds it. Part XI's have a different shape, and it is one worth naming: **a mechanism that is complete, tested, and reachable only from the tests.**

The dispatcher implements a boxed calling convention with ten value tags, two adapters and a generic fallback table — and no code in `src/` registers a boxed kernel, sets a catch-all, or installs a fallback (Chapter 60 §60.7). The guard stack implements include and exclude, and `noGrad` — the one feature the exclude half exists for — uses a module-global boolean instead, so the autograd layer runs on every operation inside it (§60.5). `ShapeEnv` implements six comparison guards and a divisibility guard; two comparisons are ever recorded and the divisibility guard has no producer at all, which is the other half of why the scheduled path refuses symbolic shapes rather than guarding them (Chapter 62 §62.7). `Library` accepts a `DEF`/`IMPL` distinction from eleven call sites and drops it in the constructor.

Two findings are of a different and sharper kind, because they produce wrong answers rather than dead code.

**A guard set that is missing a guard is indistinguishable from one that never needed it.** Under `dynamic_shapes: [true]`, a matmul's contracting dimension is left symbolic with only `> 0` recorded against it, so an input whose inner dimension changed passes every guard, reuses the kernel and returns NaNs — where the default static path recompiles and the verifier reports the mismatch by name (Chapter 62 §62.5).

**And the four optimizers read a gradient's storage rather than the gradient.** Any VJP that returns a non-contiguous view — `transpose`, which is on the path of every `nn.Linear` — therefore feeds the optimizer a permuted update. Eager training converges anyway, which is why no test in the repository reports it; the compiled path does not have the problem, so comparing the two finds it in one screen (Chapter 63 §63.5).

Each is carried into the outline's [Appendix E](../OUTLINE.md).

## A caution about this part's numbers

Unlike [Part X](../part10/README.md), this part quotes wall-clock times, and they deserve the usual warning: one machine, one Node build, one date. The ratios are more durable than the absolutes, and the ratios that matter here are large — a compiled training step against an eager one, a cache hit against a cache miss — rather than the 5% differences [Chapter 61 §61.5](ch61-tracing/README.md) shows a measurement apparatus failing to resolve.

Everything else is a count, a structure or a difference: kernels launched, buffers allocated, guards recorded, operators with a kernel registered, and the elementwise difference between two implementations. Those should reproduce exactly, and where a lab's output varies between runs — the optimization gate's is the only one that does — the chapter says so and says why.

---

**Next:** [Chapter 59 — The runtime module](ch59-the-runtime-module/README.md).
