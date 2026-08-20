# Chapter 14 — What a pass is

Part II left you with a data structure and the rules it obeys. This part is about changing it.

There are 31 transformations in this compiler. They run in an order, some of them repeatedly, some of them not at all depending on the target. Each one has to be written by somebody, tested by somebody, and debugged by somebody at three in the morning when the numbers come out wrong. The first question is not *what* they do — that is Parts IV through X. The first question is what a transformation **is**, as an object, so that thirty-one of them can be composed without every one of them knowing about the other thirty.

## 14.1 The problem: one program, thirty-one transformations

Here is the naive version. You have a graph. You want it simplified, then fused, then lowered. So you write:

```js
simplify(graph);
fuse(graph);
lower(graph);
```

Three problems appear immediately, and they appear in every compiler ever written.

**You cannot tell whether anything happened.** `simplify` might have removed forty operations or none. If it removed none, running `fuse` and then `simplify` again is wasted work; if it removed forty, running `simplify` again might remove forty more. Without a report there is no way to decide.

**You cannot turn one off.** A user reports a wrong answer. You suspect fusion. To check, you have to edit the source, rebuild, and re-run — instead of passing a flag.

**You cannot tell who broke it.** The program is wrong after `lower`. Was it wrong after `fuse`? After `simplify`? The three functions have no common shape, so there is nowhere to hang a check that runs between them.

All three are solved by the same move: stop writing transformations as functions you call, and start writing them as **objects that a driver runs**.

## 14.2 Intuition: a pass is a transformation that reports on itself

A pass is a unit of work with three properties:

- it has a **name**, so it can be named in a log, a flag, or an error message;
- it takes IR and edits it **in place**;
- it returns a **verdict** — did anything change?

That is nearly the whole idea. Everything else in this chapter is consequences.

The verdict is the part that does the most work, and it is worth being clear about what it is *not*. It is not "did the program get better". It is not "did the operation count go down". It is the answer to a much narrower question: *if I run you again, or run somebody else and then you, could the result differ from what it is now?* A pass that reports UNCHANGED is promising that the IR it was handed is the IR it is returning. That promise is what lets a driver stop iterating, skip re-verification, and keep cached analyses (Chapter 16).

## 14.3 Theory

> **Definition 14.1 (Pass).** A *pass* is a named partial function `P : IR → IR` applied by mutation, together with a *verdict function* `v : IR → {UNCHANGED, CHANGED, FAILED}` such that `v(m) = UNCHANGED` implies `P(m) = m`.

The implication runs one way only, and deliberately. Reporting CHANGED when nothing changed is legal and merely wasteful; reporting UNCHANGED when something changed is a bug that will show up somewhere else entirely, as a stale analysis or a skipped verification. Compiler infrastructure is full of contracts shaped like this: cheap to over-report, catastrophic to under-report.

Why three values rather than two? Because "I did not change anything" and "I could not do my job" are different facts and lead to different actions. UNCHANGED means the driver may proceed and may keep everything it knows. FAILED means the IR is in an unknown state: analyses computed over it must be thrown away, and the driver has to decide whether to abandon the compilation or quarantine what broke. Collapsing them loses the distinction exactly when you need it.

> **Definition 14.2 (Pass granularity).** A pass is *function-scoped* if its verdict and its edits depend only on one function of the module, and *module-scoped* otherwise.

This is not a stylistic choice. It is a claim about what the pass reads, and it buys two things. A function-scoped pass can be run over functions independently — in any order, and in principle in parallel — and a failure in one function does not implicate the others. A module-scoped pass cannot make either promise, because it may read a function it is not currently editing. Inlining is the standard example: to inline a call, you must read the callee, which is a different function.

So the driver needs to know which kind it is holding. In MLIR the same distinction appears as `OperationPass<ModuleOp>` versus `OperationPass<func::FuncOp>`; in LLVM as `ModulePass` versus `FunctionPass`. The names here are borrowed from LLVM's, and so is the meaning.

## 14.4 In mlfw: eighty-eight lines

The entire pass contract is [`passes/pass.ts`](../../../src/compiler/passes/pass.ts) — 88 lines, and it is worth reading all of it. Start with the verdict ([`pass.ts:17`](../../../src/compiler/passes/pass.ts)):

```ts
export const PassResult = Object.freeze({
  UNCHANGED: 0,
  CHANGED: 1,
  FAILED: 2
});
```

Then the base class ([`pass.ts:23`](../../../src/compiler/passes/pass.ts)):

```ts
export class Pass {
  name: string;
  preservedAnalyses: Set<AnalysisRef>;
  invalidatedAnalyses: Set<AnalysisRef>;
  requiredAnalyses: AnalysisCtor[];
  optLevel: number;
  trace: TraceLog | null;
  private _ownAnalyses: AnalysisManager | null;
```

Seven fields, and each one is a question the driver will ask before or after running it:

| Field | The question | Answered in |
|---|---|---|
| `name` | What do I call this in a log or a flag? | this chapter |
| `requiredAnalyses` | What must be computed before you run? | Chapter 16 |
| `preservedAnalyses` | What survives you? | Chapter 16 |
| `optLevel` | At which optimization level do you belong? | §14.4 |
| `trace` | Where do you write your explanation? | Chapter 18 |
| `_ownAnalyses` | What do you do if nobody gave you a manager? | Chapter 16 |

And the method that does the work ([`pass.ts:52`](../../../src/compiler/passes/pass.ts)):

```ts
  run(target: PassTarget, analysisManager?: AnalysisManager): PassResultValue {
    throw new Error('Not implemented');
  }
```

`PassTarget` is `GraphModule | GraphFunction` ([`pass.ts:8`](../../../src/compiler/passes/pass.ts)), which is the granularity distinction expressed in the type. The two subclasses at the bottom of the file carry no behaviour at all ([`pass.ts:78`](../../../src/compiler/passes/pass.ts)):

```ts
export class FunctionPass extends Pass {
  override run(func: PassTarget, analysisManager?: AnalysisManager): PassResultValue {
    throw new Error('Not implemented');
  }
}

export class ModulePass extends Pass {
  override run(module: PassTarget, analysisManager?: AnalysisManager): PassResultValue {
    throw new Error('Not implemented');
  }
}
```

They exist to be tested for. The pass manager branches on `pass instanceof ModulePass` versus `pass instanceof FunctionPass` ([`pass_manager.ts:116`](../../../src/compiler/passes/pass_manager.ts) and [`pass_manager.ts:148`](../../../src/compiler/passes/pass_manager.ts)) and hands each one a different argument. The subclass *is* the declaration.

### Who is a pass

31 classes in `src/compiler` extend one of these bases, spread over three IR levels:

| Level | Base class | Count | Where |
|---|---|---|---|
| Graph | `FunctionPass` / `ModulePass` | 21 | [`passes/`](../../../src/compiler/passes/) |
| TIR | `PrimFuncPass` / `TirModulePass` | 9 | [`passes/tir_pass.ts`](../../../src/compiler/passes/tir_pass.ts) |
| LIR | `LirFuncPass` | 1 | [`passes/lir_pass.ts`](../../../src/compiler/passes/lir_pass.ts) |

The TIR and LIR bases are separate class hierarchies with their own managers, not subclasses of `Pass`. That is a real design decision and Chapter 15 argues about it; the short version is that they transform a different IR, have a different notion of "unchanged" (a TIR pass may *return a new function* rather than mutating one — [`tir_pass.ts:27`](../../../src/compiler/passes/tir_pass.ts)), and sharing a base class would have meant a union type in every signature.

Only 9 of the 21 graph passes are instantiated for a default CPU compile. Which ones appear is decided by [`buildGraphPipeline`](../../../src/compiler/pipeline/graph_pipeline.ts) from the config and the target — quantization passes appear only when quantization is enabled, layout passes only when the target asks for them. That is Chapter 15's subject.

### The switch that turns one off

Pass selection is not a debugging afterthought bolted on later; it is a parameter to the run ([`pass.ts:57`](../../../src/compiler/passes/pass.ts)):

```ts
export class PassContext {
  optLevel: number;
  disabledPasses: ReadonlySet<string>;
  requiredPasses: ReadonlySet<string>;
  config: ReadonlyMap<string, unknown>;
```

with one method, consulted by the manager before every single pass ([`pass.ts:70`](../../../src/compiler/passes/pass.ts)):

```ts
  shouldRun(pass: Pass): boolean {
    if (this.disabledPasses.has(pass.name)) return false;
    if (this.requiredPasses.has(pass.name)) return true;
    if ((pass.optLevel || 0) > this.optLevel) return false;
    return true;
  }
```

Read the order of the three tests, because it encodes a policy. Disabling wins over requiring: if you say "never run X" and something else says "always run X", X does not run. Requiring wins over the optimization level: a pass in the require-set runs even at `-O0`, which is how a pass that is not an optimization at all — a legalization the target cannot do without — survives a low optimization level. And `optLevel` on the pass is a *threshold*, not a stage: the pass runs when the context's level is at least the pass's own.

Every pass in this compiler currently declares `optLevel: 0`, so the third test never fires today. The mechanism is there and tested ([`tests/compiler/passes/pass.test.js:11`](../../../tests/compiler/passes/pass.test.js)); it just has no users yet. That is worth flagging rather than hiding — a book that only describes the parts in active use gives you no way to tell a designed extension point from an unused one.

## 14.5 Lab 1 — The pass ledger

```bash
node docs/part3/ch14-what-a-pass-is/labs/01-the-pass-ledger.mjs
```

Every pass run reports itself into the trace stream at `VERBOSE` (Chapter 18 covers the stream itself; here we only read one field of it). The lab prints the verdict of every pass run in a compile of the two-layer MLP:

```
pass                    verdict     ops
CallInlinerPass         UNCHANGED   10 -> 10
DecompositionPass       UNCHANGED   10 -> 10
canonicalize            CHANGED     10 -> 10
algebraic_simplify      UNCHANGED   10 -> 10
constant_fold           CHANGED     10 -> 10
cse                     UNCHANGED   10 -> 10
dce                     CHANGED     10 -> 7
canonicalize            UNCHANGED   7 -> 7
algebraic_simplify      UNCHANGED   7 -> 7
constant_fold           UNCHANGED   7 -> 7
cse                     UNCHANGED   7 -> 7
dce                     UNCHANGED   7 -> 7
PriorityFusionPass      CHANGED     7 -> 6
MultiOutputFusionPass   UNCHANGED   6 -> 6
dce                     UNCHANGED   6 -> 6

15 pass runs, 4 changed something, 11 did not
```

Three things to take from this.

**Most passes do nothing, most of the time.** Eleven of fifteen runs report UNCHANGED. This is normal and it is the argument for the whole design: the price of a pass that does not apply to your program is one traversal and one `UNCHANGED`, so the pipeline can afford to contain transformations that are irrelevant to most programs.

**CHANGED does not mean fewer operations.** Look at `canonicalize: CHANGED 10 -> 10` and `constant_fold: CHANGED 10 -> 10`. The op count is identical before and after. Canonicalize rewrote the `dot`'s contracting-dimension attribute so the `transpose` feeding it became dead — the Chapter 11 rewrite — and constant-fold replaced a `broadcast_in_dim` of a scalar with a full-shape constant. Neither removed an operation; both changed the program. The count in the ledger is a diagnostic, not the verdict.

**The five middle passes appear twice.** That is a fixed-point group, and it is Chapter 15.

**Try this.** Change the model to `new Sequential(new Linear(2, 8))` — one layer, no activation — and re-run. Which passes still report CHANGED, and which fall silent?

## 14.6 Lab 2 — Module or function

```bash
node docs/part3/ch14-what-a-pass-is/labs/02-module-or-function.mjs
```

`PassContext.shouldRun` is handed the real `Pass` object before every run, which makes it the one place outside the compiler where a program can look at the pass objects themselves. The lab uses it to record what each pass declares, and to wrap `run` so it can see what the manager passes in:

```
=== what the pass manager hands each pass ===
CallInlinerPass         run(module 'Sequential') -> UNCHANGED
DecompositionPass       run(function 'Sequential') -> UNCHANGED
canonicalize            run(function 'Sequential') -> CHANGED
algebraic_simplify      run(function 'Sequential') -> UNCHANGED
constant_fold           run(function 'Sequential') -> CHANGED
cse                     run(function 'Sequential') -> UNCHANGED
dce                     run(function 'Sequential') -> CHANGED
...
```

`CallInlinerPass` receives the module; everything else receives a function. That is Definition 14.2 as a runtime observation rather than a claim: the inliner is the one graph pass here that has to look across function boundaries, and it is the one the manager calls once for the whole module.

The second half prints what each pass says about itself:

```
=== what each pass declares about itself ===
pass                    class                       opt  requires        preserves
CallInlinerPass         CallInlinerPass             0    -               -
DecompositionPass       DecompositionPass           0    -               -
canonicalize            CanonicalizePass            0    -               -
algebraic_simplify      AlgebraicSimplificationPass 0    -               -
constant_fold           ConstantFoldPass            0    -               -
cse                     CSEPass                     0    -               -
dce                     DCEPass                     0    memory_effect   memory_effect
PriorityFusionPass      PriorityFusionPass          0    use_def         -
MultiOutputFusionPass   MultiOutputFusionPass       0    -               -
```

Two passes name an analysis they need. Exactly one names an analysis it preserves. Hold that thought until Chapter 16; it is the entire subject of that chapter, and the sparseness of this column is the interesting part.

Note also that `name` and the class name are not the same string. `CanonicalizePass` calls itself `canonicalize`; `PriorityFusionPass` did not bother. The name is what appears in traces and in the disable-set, so it is the one users see — an inconsistency worth knowing about before you go looking for a pass called `CSEPass` in a log that says `cse`.

## 14.7 Lab 3 — Turn a pass off

```bash
node docs/part3/ch14-what-a-pass-is/labs/03-turn-a-pass-off.mjs
```

A `PassContext` is any object with a `shouldRun(pass)` method, so a disable-set is one line:

```js
passContext: { shouldRun: (pass) => !off.has(pass.name) }
```

The lab compiles the same MLP three times: everything on, dead-code elimination off, and all five simplification passes off. The output is the same in all three cases — `[[-0.0030276477336883545],[-0.18891724944114685]]` — and the program is not.

With everything on, the graph is six operations, one of them a fused elementwise region:

```
    %5 = dot(%0, %1) {lhs_batch = [], lhs_contracting = [1], rhs_batch = [], rhs_contracting = [1]} : tensor<2x8xf32>
    %6 = constant() {tensor_type = tensor<2x8xf32>, value = 0} : tensor<2x8xf32>
    %7 = fusion(%5, %2, %6) {fusion_kind = "kElementwise"} : tensor<2x8xf32>
    ...
```

With DCE off, two transposes nobody uses survive to the end:

```
    %5 = transpose(%1) {permutation = [1, 0]} : tensor<2x8xf32>
    %6 = dot(%0, %1) {lhs_batch = [], lhs_contracting = [1], rhs_batch = [], rhs_contracting = [1]} : tensor<2x8xf32>
    ...
    %15 = transpose(%3) {permutation = [1, 0]} : tensor<8x1xf32>
```

`%5` has no users. Neither does `%15`. They are the transposes canonicalization made redundant and DCE would have swept. And they are not free: the generated kernel goes from 38 lines to 48, because **nothing downstream of the graph passes decides not to emit an operation that is in the graph.** Lowering lowers what it is given. Dead code elimination is not tidiness; it is the only thing standing between a dead operation and a real pass over memory at run time.

With every simplification off, the transposes are not even dead — the `dot` still contracts dimension 0 of a transposed weight, so the transposes are load-bearing, and the fused region additionally contains a `broadcast_in_dim` that constant folding would have removed:

```
    %6 = dot(%0, %5) {lhs_batch = [], lhs_contracting = [1], rhs_batch = [], rhs_contracting = [0]} : tensor<2x8xf32>
    %8 = fusion(%6, %2, %7) {fusion_kind = "kBroadcast"} : tensor<2x8xf32>
    {
      ^bb(%9: tensor<2x8xf32>, %10: tensor<8xf32>, %11: tensor<xf32>):
      %12 = add(%9, %10) : tensor<2x8xf32>
      %13 = broadcast_in_dim(%11) {broadcast_dimensions = [], result_shape = [2, 8]} : tensor<2x8xf32>
      %14 = maximum(%12, %13) : tensor<2x8xf32>
      yield(%14)
    }
```

Three programs, one answer, three different amounts of work. This is what a pass pipeline buys, and being able to switch one element of it off from the outside is what lets you find out which element bought what.

**Try this.** Disable `PriorityFusionPass` and look at the graph. Then disable `canonicalize` alone and count the operations: which of the two costs more?

## 14.8 Traps and limits

- **No pass in this compiler ever returns FAILED.** Grep for it: the only writer of `PassResult.FAILED` is the pass manager itself ([`pass_manager.ts:126`](../../../src/compiler/passes/pass_manager.ts)), converting a thrown exception into a verdict in resilient mode. Passes signal failure by throwing. The three-valued contract is real and the manager handles all three, but in practice the third value is produced by the driver, not by a pass. If you write a pass, throwing is the idiom; returning FAILED is legal and nothing will call it.
- **`run` may lie about UNCHANGED and nothing checks.** Definition 14.1's implication is a contract, not an invariant. A pass that mutates and reports UNCHANGED will not be caught by verification — the IR is still valid, just different from what the driver believes. The one thing that saves you is that analyses are keyed by a mutation counter rather than by the verdict (Chapter 16), so a lying pass still invalidates the cache. That is a defence in depth that happens to work, not a designed check.
- **The `PassContext.config` map cannot be read by a pass at all.** [`pass.ts:61`](../../../src/compiler/passes/pass.ts) declares it and the constructor fills it from the caller's options ([`pass.ts:67`](../../../src/compiler/passes/pass.ts)) — but the flow of control only ever goes the other way: the manager hands a *pass* to `shouldRun`, and never hands the context to a pass. Per-pass configuration is done by constructor arguments in [`buildGraphPipeline`](../../../src/compiler/pipeline/graph_pipeline.ts) instead. Two mechanisms for one job, one of them unreachable.
- **A `FunctionPass` is run function-by-function but not in parallel.** [`pass_manager.ts:151`](../../../src/compiler/passes/pass_manager.ts) is an ordinary `for (const func of module)`. The granularity distinction licenses parallelism; nothing here takes it.
- **Mutating a pass object from `shouldRun` is not what the hook is for.** Lab 3 below uses it as intended — a predicate. Lab 2, and four labs in Chapters 15, 16 and 18, instead wrap `pass.run` from inside `shouldRun`, because it is the only way to reach a pass object from outside the package. It works because `shouldRun` receives the real object, and it is the right thing for a lab. It is the wrong thing in production code: `shouldRun` is a predicate, and a predicate with side effects is a trap for the next reader.

## 14.9 Read the tests

- [`tests/compiler/passes/pass.test.js`](../../../tests/compiler/passes/pass.test.js) — `PassContext` precedence: required beats optimization level, disabled beats required.
- [`tests/compiler/passes/pass-manager.test.js`](../../../tests/compiler/passes/pass-manager.test.js) — the FAILED half of the contract, including what the manager does to analyses when a `ModulePass` fails versus a `FunctionPass`. Note how the test builds `FakeFunc`/`FakeModule` objects with three methods each: the manager depends on very little.

---

**Next:** [Chapter 15 — The pass manager](../ch15-the-pass-manager/README.md), which takes these objects and asks what it means to run a *sequence* of them — including one that has to be run until it stops having an effect.
