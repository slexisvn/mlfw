# Part III — The transformation infrastructure

Part II gave you the object every remaining chapter of this book transforms. This part is about the machinery that does the transforming — and it is deliberately placed *before* any actual optimization, because every optimization in Parts IV through X is written against these five ideas and none of them makes sense without them.

It is also the part that decides whether the compiler is debuggable. A wrong number produced by a fused kernel is a mystery; a wrong number produced by a fused kernel, in a pipeline that names the pass that broke the IR, streams the reason each fusion was chosen, and can be re-run with one pass switched off, is an afternoon's work.

| Chapter | Title | The question it answers |
|---|---|---|
| [14](ch14-what-a-pass-is/README.md) | What a pass is | What is the unit of transformation, and what does one owe the thing that runs it? |
| [15](ch15-the-pass-manager/README.md) | The pass manager | How do you run a sequence of them — including a set that has to run until it stops having an effect — and how do you find out which one broke the program? |
| [16](ch16-analyses-and-invalidation/README.md) | Analyses and the invalidation problem | Facts derived from the IR are expensive to compute and go stale the moment it changes. How is a cached fact prevented from outliving the program it describes? |
| [17](ch17-pattern-rewriting/README.md) | Pattern rewriting | What is a rewrite rule when it is an object rather than a branch in a `switch`, and what makes a set of them converge? |
| [18](ch18-watching-the-compiler/README.md) | Watching the compiler work | How do you see inside a compile without editing the compiler — and what does a compiler owe you when it fails? |

## The five objects, and who holds whom

Every chapter of this part is one box in this picture. It is worth having in front of you, because the chapters are written as if you already know where the thing being described sits.

```
   Compiler.compile                       drives a list of 14 named PHASES
        |                                 (ch15 15.8) -- graphPasses is one of them
        |  ctx.working : GraphModule      resilient mode works on a clone (ch18)
        v
   PassManager                            one entry at a time, in order
        |    +-----------------------------------------------------+
        |    |  entry = a Pass, or a FixedPointGroup of Passes      |  ch15
        |    +-----------------------------------------------------+
        |
        |  for each entry:
        |     shouldRun(pass)? ................ PassContext          ch14
        |     fetch pass.requiredAnalyses ..... AnalysisManager      ch16
        |     verdict = pass.run(target) ...... Pass                 ch14
        |     if CHANGED:  bumpVersion
        |                  invalidate(preservedAnalyses) .. ch16
        |                  verify -> names the pass ........ ch15
        |     emit a `pass` event ............. TraceLog            ch18
        v
   the same GraphModule, edited in place

   Two passes are not one transformation but a driver over many small rules:
        CanonicalizePass, AlgebraicSimplificationPass
             -> PatternApplicator over a PatternSet, worklist-driven   ch17
```

Read the middle block downward and you have the whole of Chapter 15: everything after the verdict is *keyed off* the verdict, which is why Chapter 14 spends its length on a three-valued return.

## The argument in one paragraph

A transformation is an object with a name, a target, and a three-valued verdict. The verdict is what everything else is keyed off — whether to iterate, whether to re-verify, whether to keep what you already know — which is why a pass that reports it wrongly gets a *later* pass blamed for the damage (Chapter 14).

A driver runs a sequence of those objects, rebuilding the sequence per compile from the config and the target. It iterates the cheap ones as a group until a whole round reports no change, which costs exactly one extra round to establish and terminates only because it is capped, not because anything decreases. And it verifies the IR after every pass that claims to have changed it, so an invalid graph arrives with the name of the pass that produced it (Chapter 15).

Passes want derived facts, so those are cached against a counter the IR increments on every structural edit — sound without anybody declaring anything, over exactly the edits it intercepts. A pass may additionally *declare* that it preserved a fact. That is faster, it is trusted, it is unchecked, and it must be applied transitively over the dependency graph or it buys nothing at all (Chapter 16).

Two of the five simplification passes are not single algorithms but collections of independent rewrite rules, driven by a worklist that re-queues whatever a rewrite disturbed. Rules cascade without knowing about each other, converging to a normal form that is bounded rather than proved — and one of those rules is generated from a trait that is false on floats (Chapter 17).

All of it emits a filtered stream of structured events: phases, verdicts, rewrite counts, IR snapshots, and the reasons behind individual decisions. A resilient mode turns a thrown exception into a recorded error, restores the module the failing pass abandoned, and leaves the caller's IR untouched (Chapter 18).

## What Part III establishes for later parts

- **The pass contract** (Definition 14.1) and the module/function granularity split (Definition 14.2) — the shape every transformation in Parts IV–X is written to.
- **The fixed-point group** (Definition 15.2) and its honest termination story (Theorem 15.4), which is why the simplification passes can be written as if they will be re-run, and why they are.
- **Per-pass verification** — more precisely, verification after every pass that *reports a change* — which turns Chapter 12's invariants into an attribution mechanism and is the foundation of Chapter 67's debugging procedure.
- **Preservation as a proof obligation** (Definition 16.2) and transitive invalidation (Theorem 16.3), which every pass that touches an analysis in Parts IV, V and IX has to reason about.
- **The pattern/applicator split** (Chapter 17), which is how Chapters 19, 20 and 21 add rewrite rules without adding passes.
- **The trace stream and `explain`** (Chapter 18), which is how every measurement and every "why did it choose that" in the rest of the book is obtained.

## Labs

```bash
npm run build   # once, if you have not already

node docs/part3/ch14-what-a-pass-is/labs/01-the-pass-ledger.mjs
node docs/part3/ch14-what-a-pass-is/labs/02-module-or-function.mjs
node docs/part3/ch14-what-a-pass-is/labs/03-turn-a-pass-off.mjs
node docs/part3/ch15-the-pass-manager/labs/01-the-fixed-point-group.mjs
node docs/part3/ch15-the-pass-manager/labs/02-verify-every-pass.mjs
node docs/part3/ch16-analyses-and-invalidation/labs/01-computed-once.mjs
node docs/part3/ch16-analyses-and-invalidation/labs/02-staleness-propagates.mjs
node docs/part3/ch17-pattern-rewriting/labs/01-a-rewrite-cascade.mjs
node docs/part3/ch17-pattern-rewriting/labs/02-canonical-form.mjs
node docs/part3/ch18-watching-the-compiler/labs/01-four-levels.mjs
node docs/part3/ch18-watching-the-compiler/labs/02-why-it-decided-that.mjs
node docs/part3/ch18-watching-the-compiler/labs/03-when-a-pass-fails.mjs
```

All of them are deterministic except the timings in Chapter 15's second lab and the `durationMs` fields in Chapter 18's first, which are machine-specific. Chapter 15 is explicit about which half of its measurement reproduces and which half sits inside the noise — that distinction is part of the lesson.

A note on how these labs reach the compiler, because it is different from Part II's. The public surface exposes no `PassManager` and no `AnalysisManager`. But `compile` accepts a `passContext`, and a `PassContext` is anything with a `shouldRun(pass)` method — which the pass manager calls, with the real `Pass` object, before every single pass. That one hook is enough to read what a pass declares, to switch it off, and — by replacing `pass.run` with a wrapper — to see what the manager hands it, to reach the live `AnalysisManager` it is given, and to inject a failure. Five labs do exactly that, and the chapter text says so wherever one does. It is a legitimate way to instrument a compile from outside and an illegitimate thing to do in production code: `shouldRun` is a predicate, and a predicate with side effects will surprise the next person to read it.
