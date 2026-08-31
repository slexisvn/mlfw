# Chapter 15 — The pass manager

A pass is a transformation that reports on itself. A pass manager is the thing that decides which ones run, in what order, how many times, and what happens when one of them is wrong.

It is 260 lines. It is also the single most useful piece of infrastructure in the compiler for the specific job of *finding out why the answer is wrong*, and this chapter is mostly about why.

## 15.1 The problem: three questions a list cannot answer

Suppose you have your 21 graph passes as objects, and you write down an order. Three questions remain, and none of them is answered by the list.

**When do I stop?** Canonicalization can enable constant folding; constant folding can enable canonicalization. Run each once and you leave work on the table. Run them alternately forever and you never finish. Something has to decide when the pair has settled.

**What do I do about a target that does not want a pass?** Quantization passes make no sense without a quantization config; layout passes make no sense for a target with one layout. Either the list is rebuilt per compile, or every pass begins by checking whether it should exist.

**Which pass broke the program?** This is the one that matters at three in the morning. Verification (Chapter 12) tells you the IR is invalid. A list of passes does not tell you which one made it so, and by the time you find out, twelve more passes have run over the wreckage.

## 15.2 Intuition: a driver, a loop, and a checkpoint

The answers are, in order: **a group that repeats until nothing happens**, **a pipeline that is built rather than written down**, and **a check between every pair of passes**.

The third is worth stating as a slogan, because it is the difference between a debuggable compiler and an undebuggable one: *if you verify only at the ends, you learn that something is broken; if you verify after each pass that changed anything, you learn who broke it.* Note the qualification — it is not decoration, and §15.6 is where it earns its keep. The cost is a full IR traversal per such pass, which is the kind of price you should expect to pay for that information — and §15.7 measures it.

## 15.3 Theory

> **Definition 15.1 (Pass pipeline).** **(stated here)** A *pipeline* is a finite sequence of entries, where each entry is either a pass or a *fixed-point group*: a finite list of passes together with an iteration bound.

> **Definition 15.2 (Fixed-point group).** **(stated here)** Running a group `G = (P₁ … Pₙ, k)` on IR `m` means: repeat up to `k` times the sequence `P₁ … Pₙ`, stopping early after any full round in which every `Pᵢ` reported UNCHANGED.

The stopping condition is the interesting half. "Every pass reported UNCHANGED for a whole round" is exactly the statement that the round was a no-op, so running the round again would also be a no-op — a fixed point of the composite transformation `Pₙ ∘ … ∘ P₁`. Hence the name.

> **Lemma 15.3 (The cost of knowing you are done).** **(stated here)** A group whose composite reaches a fixed point after `k` productive rounds performs `k + 1` rounds, provided `k + 1 ≤ maxIterations`.

*Proof.* Rounds 1 through `k` each contain at least one CHANGED report, so none of them triggers the early stop. Round `k + 1` reports UNCHANGED throughout and stops. ∎

That extra round is not waste; it is the evidence. There is no way to know a fixed point has been reached other than by trying and failing to move. It also means that the *observable* cost of a group is one round more than the work it does, which is why groups are made of cheap passes.

Now the question the implementation actually has to answer:

> **Theorem 15.4 (Termination).** **(invariant)** A fixed-point group terminates after at most `maxIterations` rounds, for every input, unconditionally.

*Proof.* The loop is bounded by a constant. ∎

That proof is deliberately anticlimactic, because the honest statement is that **the bound is the entire termination argument.** The classical alternative is available and is not used here: a rewrite system terminates if there is a well-founded order on terms that every rule strictly decreases *(classical, term rewriting)*. To apply it you would need a measure `μ` on IR such that any pass reporting CHANGED strictly decreases `μ`. No such measure exists in this compiler, and the ledger from Chapter 14 shows why:

> **Counterexample 15.5.** `canonicalize: CHANGED 10 -> 10`. Operation count is not decreased by a CHANGED report, so operation count is not the measure. Neither is any obvious refinement: canonicalization rewrote an attribute, fusion *adds* an operation (the `fusion` wrapper) while removing others, and rematerialization deliberately increases the operation count to lower peak memory.

So the cap is not a safety net around a proof. It is the proof — and it proves *termination*, which is a strictly weaker claim than *convergence*. Keep the two apart:

| Claim | Status |
|---|---|
| the group stops | proved, by the bound |
| the group stops **at a fixed point** | not proved, and not true in general |
| the group tells you which of the two happened | yes — §15.4's `<group>:max-iter` event |
| the group tells you **why** it hit the cap | no |

The last two rows are the practically useful ones. Falling out of the loop without a quiet round emits a synthetic trace event, so exhausting the budget is visible rather than silent; that is the one line of logging §15.4 points at. But the event carries no diagnosis, and there are two very different situations behind it. A group may be making genuine progress that simply needs more than eight rounds — raise the cap and it converges. Or two passes may be *oscillating*, each honestly reporting CHANGED as it undoes the other, in which case no cap is large enough and raising it only makes compilation slower. Nothing distinguishes them automatically; the way to tell is to re-run with a larger cap and see whether the event goes away.

Two further consequences follow, and both are visible in §15.5: a group can stop before it has converged, leaving a program that a further round would have improved; and a pass that reports CHANGED when nothing changed costs `maxIterations` rounds of everything else in its group.

Chapter 6 established the other half of the picture — Theorem 6.3, that no fixed pass order is optimal for all programs. The fixed-point group is one of the three standard answers to it: if you cannot pick the right order, run the cheap passes repeatedly until order stops mattering.

## 15.4 In mlfw: the manager

[`passes/pass_manager.ts`](../../../src/compiler/passes/pass_manager.ts). The group is a plain record ([`pass_manager.ts:51`](../../../src/compiler/passes/pass_manager.ts)):

```ts
export class FixedPointGroup {
  name: string;
  passes: Pass[];
  maxIterations: number;

  constructor(name: string, passes: Pass[], maxIterations = 8) {
```

and the loop is short enough to quote whole ([`pass_manager.ts:212`](../../../src/compiler/passes/pass_manager.ts)):

```ts
  _runGroup(group: FixedPointGroup, module: GraphModule, ctx: PassRunCtx, results: PassResultValue[]): boolean {
    const maxIter = group.maxIterations > 0 ? group.maxIterations : 1;
    for (let iter = 0; iter < maxIter; iter++) {
      let iterChanged = false;
      for (const pass of group.passes) {
        if (!ctx.passContext.shouldRun(pass)) continue;
        const { changed, fatal } = this._applyPass(pass, module, ctx, results);
        if (fatal) return true;
        if (changed) iterChanged = true;
      }
      if (!iterChanged) return false;
    }
    if (this.trace) this.trace.passRun(`${group.name}:max-iter`, PassResult.UNCHANGED, 0, -1, -1);
    return false;
  }
```

Definition 15.2, line for line. Note the last two lines before the return: falling out of the `for` without an early exit means the cap was reached without convergence, and the manager *says so* in the trace stream, under a synthetic pass name `<group>:max-iter`. A compiler that silently gives up is a compiler you will eventually mistrust; one line of logging is the difference.

### The pipeline is built, not written

[`buildGraphPipeline`](../../../src/compiler/pipeline/graph_pipeline.ts) is a function from `(config, target)` to a list of passes. Here is the middle of it ([`graph_pipeline.ts:39`](../../../src/compiler/pipeline/graph_pipeline.ts)):

```ts
  passes.push(new CallInlinerPass());
  passes.push(new DecompositionPass(target as unknown as null));
  passes.push(new FixedPointGroup('canonicalize', [
    new CanonicalizePass({ fastMath: config.optimization.fastMath }),
    new AlgebraicSimplificationPass({ fastMath: config.optimization.fastMath }),
    new ConstantFoldPass(),
    new CSEPass(),
    new DCEPass(),
  ], config.optimization.maxSimplifyIterations));

  if (config.optimization.layout && target) {
    passes.push(new LayoutTransformPass({ target }));
    passes.push(new DCEPass());
  }
```

Everything conditional is conditional here, in one file, rather than inside twenty passes each asking whether it should have been constructed. The five members of the `canonicalize` group are the five cheap simplifications, and the group's bound comes straight from user configuration — `optimization.maxSimplifyIterations`, default 8.

Two extension points are worth naming now because they are the reason this file has stayed short. `passesForPhase('pre')` and `passesForPhase('post')` bracket the whole pipeline with passes contributed by a registry ([`graph_pass_registry.ts:16`](../../../src/compiler/pipeline/graph_pass_registry.ts)), so a subsystem can inject a pass without editing the builder. And `activeExternalCodegenProviders` lets a backend that owns some operations — cuBLAS, for instance — contribute its own passes at the right point (Chapter 58).

### Applying one pass

`_applyPass` ([`pass_manager.ts:108`](../../../src/compiler/passes/pass_manager.ts)) is the heart of the file, and the module and function branches differ in more than the argument. The module branch, from just after the call ([`pass_manager.ts:130`](../../../src/compiler/passes/pass_manager.ts)):

```ts
      results.push(result);

      if (verbose) (this.trace as TraceLog).passRun(pass.name, result, performance.now() - t0, opsBefore, countOps(module));

      if (result === PassResult.CHANGED) {
        changed = true;
        ctx.anyChanged = true;
        this.analysisManager.invalidateFunctions(module, pass.preservedAnalyses);
        const verr = this._verifyAfter(pass, module, true);
```

and the function branch ([`pass_manager.ts:179`](../../../src/compiler/passes/pass_manager.ts)):

```ts
          for (const A of pass.requiredAnalyses) this.analysisManager.getAnalysis(A, func);
          const result = pass.run(func, this.analysisManager);
          if (verbose) (this.trace as TraceLog).passRun(pass.name, result, performance.now() - t0, opsBefore, countOps(func));
          if (result === PassResult.CHANGED) {
            passChanged = true;
            ctx.anyChanged = true;
            func.bumpVersion();
            this.analysisManager.invalidate(func, pass.preservedAnalyses);
```

Read the fixed sequence around the call: **required analyses in, pass runs, verdict recorded, version bumped, analyses invalidated, IR verified.** Everything after the verdict is keyed off it, which is what Chapter 14's contract was for. The `bumpVersion` and `invalidate` calls are Chapter 16; `_verifyAfter` is next.

## 15.5 Lab 1 — The fixed-point group

```bash
node docs/part3/ch15-the-pass-manager/labs/01-the-fixed-point-group.mjs
```

The program is chosen to need more than one round:

```js
const x = tensor([[1, 2], [3, 4]], { dtype: 'i32' });

class ThereAndBackAgain extends Module {
  forward(a) { return a.transpose(1, 0).transpose(1, 0).add(0); }
}
```

Two transposes that cancel, and an add of zero. The integer dtype is load-bearing and Chapter 20 is where it is explained: `x + 0` is an identity on integers and not on floats, where it maps `−0` to `+0`, so the rule that removes the add declines on an `f32` tensor. With the default bound of 8:

```
=== maxSimplifyIterations: 8 ===
  -- round 1 --
  canonicalize         CHANGED   5 -> 4
  algebraic_simplify   CHANGED   4 -> 4
  constant_fold        UNCHANGED 4 -> 4
  cse                  UNCHANGED 4 -> 4
  dce                  CHANGED   4 -> 2
  -- round 2 --
  canonicalize         CHANGED   2 -> 1
  algebraic_simplify   UNCHANGED 1 -> 1
  constant_fold        UNCHANGED 1 -> 1
  cse                  UNCHANGED 1 -> 1
  dce                  UNCHANGED 1 -> 1
  -- round 3 --
  canonicalize         UNCHANGED 1 -> 1
  algebraic_simplify   UNCHANGED 1 -> 1
  constant_fold        UNCHANGED 1 -> 1
  cse                  UNCHANGED 1 -> 1
  dce                  UNCHANGED 1 -> 1
module @ThereAndBackAgain {
  func @ThereAndBackAgain(%0: tensor<2x2xi32>) -> (tensor<2x2xi32>) {
    return(%0)
  }
}
```

Two productive rounds and a third that proves there is nothing left — Lemma 15.3 with `k = 2`. Follow what enabled what. In round 1, canonicalization removes the `add` of zero; algebraic simplification collapses `transpose(transpose(x))` into a single *identity* transpose (`CHANGED 4 -> 4` — a rewrite, not a deletion); DCE clears the orphans. Only in round 2 does canonicalization see an identity transpose it can fold away entirely, because the rule that *creates* an identity transpose lives in `algebraic_simplify` and the rule that *eliminates* one lives in `canonicalize`, which already ran.

Swapping those two passes would have finished this program in one round — and would have cost a second round on a program that needs the dependency the other way. That is Theorem 6.3 in miniature: there is no ordering that is right for every program, so the group stops trying to find one and iterates instead.

Now the same program with the bound set to 1:

```
=== maxSimplifyIterations: 1 ===
  -- round 1 --
  canonicalize         CHANGED   5 -> 4
  algebraic_simplify   CHANGED   4 -> 4
  constant_fold        UNCHANGED 4 -> 4
  cse                  UNCHANGED 4 -> 4
  dce                  CHANGED   4 -> 2
  canonicalize:max-iter UNCHANGED -1 -> -1
module @ThereAndBackAgain {
  func @ThereAndBackAgain(%0: tensor<2x2xi32>) -> (tensor<2x2xi32>) {
    %1 = transpose(%0) {permutation = [0, 1]} : tensor<2x2xi32>
    return(%1)
  }
}
```

The compiler stops one round early and ships `transpose {permutation = [0, 1]}` — the identity permutation, which will be lowered, code-generated, and executed as a full copy of the tensor at run time. The answer is still right. The program is worse, and the only trace of it is the `canonicalize:max-iter` line, which is exactly the line the manager emits to tell you the cap was hit rather than convergence reached.

That line is what Theorem 15.4 looks like from the outside. It is also the thing to grep for when a compile is slower than it should be.

**Try this.** Set the bound to 2 and confirm the identity transpose disappears but the confirming round does not run. Then predict what happens at 3.

## 15.6 Verifying after every pass

The manager holds one boolean and one method ([`pass_manager.ts:101`](../../../src/compiler/passes/pass_manager.ts)):

```ts
  _verifyAfter(pass: Pass, target: PassTarget, isModule: boolean): CompilationError | null {
    if (!this.checkEachPass) return null;
    const level = isModule ? IRLevel.GRAPH_MODULE : IRLevel.GRAPH_FUNC;
    const name = isModule ? (target.name || '<module>') : target.name;
    return checkIRInvariants(level, target, name, pass.name);
  }
```

The last argument is the whole point. `checkIRInvariants` ([`invariant_check.ts:23`](../../../src/compiler/support/invariant_check.ts)) uses it to build the message:

```ts
export function checkIRInvariants(irLevel: IRLevelValue, target: unknown, name: string, passName: string | null = null): CompilationError | null {
  const found = verifyIR(irLevel, target);
  if (found.length === 0) return null;
  const prefix = passName ? `pass '${passName}' produced invalid IR: ` : '';
  return new CompilationError('verification', name, prefix + found.join('; '), passName);
}
```

The same verifier, the same errors, and one difference: at a phase boundary nobody knows which pass to blame, so the prefix is empty. When the check runs immediately after a pass, the prefix names it.

How often it runs is a three-valued setting ([`invariant_check.ts:7`](../../../src/compiler/support/invariant_check.ts)):

```ts
export const VerifyLevel = Object.freeze({
  OFF: 'off',
  BOUNDARIES: 'boundaries',
  EACH_PASS: 'each-pass',
});
```

**The default is `EACH_PASS`** ([`invariant_check.ts:16`](../../../src/compiler/support/invariant_check.ts): `value ?? VerifyLevel.EACH_PASS`). That is an unusual choice and a deliberate one — most compilers make per-pass verification a debug build or a flag. Enabling it by default means the first report of a miscompile arrives with a pass name attached. The price is measured next.

> **Read the name as `EACH_CHANGING_PASS`.** **(invariant)** The setting is narrower than its name. It does not verify after every pass, and the guard is not in `_verifyAfter` at all — it is at the call site ([`pass_manager.ts:134`](../../../src/compiler/passes/pass_manager.ts)):
>
> ```ts
>   if (result === PassResult.CHANGED) {
>     changed = true;
>     ctx.anyChanged = true;
>     this.analysisManager.invalidateFunctions(module, pass.preservedAnalyses);
>     const verr = this._verifyAfter(pass, module, true);
> ```
>
> `_verifyAfter` is reached only from inside that branch. A pass reporting UNCHANGED is taken at its word and skipped entirely; look back at Chapter 14's ledger and count how many of the fifteen runs that is — eleven.

This is Chapter 14's contract being *spent* rather than merely declared, and it is the right optimization as long as the contract holds: verifying a module no pass touched is pure cost. But it means the two mechanisms that would catch a broken pass — verification and analysis invalidation — are keyed off the same self-report, and both fail together in exactly one case: **a pass that mutates the IR and reports UNCHANGED is never verified, so the corruption it introduced is first noticed after some later pass, and attributed to that one.** Chapter 14 §14.3 traces the sequence. If you are hunting a miscompile and `each-pass` blames a pass whose code cannot possibly have done it, the question to ask is which pass ran before it and what it reported.

## 15.7 Lab 2 — What each verification level buys

```bash
node docs/part3/ch15-the-pass-manager/labs/02-verify-every-pass.mjs
```

The lab sabotages one pass — it wraps `cse` so that after running normally it writes a result type that does not follow from its operands, exactly the failure a buggy rewrite produces — and compiles at each level:

```
=== one pass writes a result type that does not follow ===
  verify: each-pass
    threw:  Graph verification failed (after graph passes): [Sequential] op 'add' (id=2): trait 'elementwise': result 0 shape [2, 1] != broadcast of operand shapes [2, 32]
    blamed: cse -- pass 'cse' produced invalid IR: [Sequential] op 'add' (id=2): trait 'elementwise': result 0 shape [2, 1] != broadcast of operand shapes [2, 32]
  verify: boundaries
    threw:  Graph verification failed (after graph passes): [Sequential] op 'add' (id=23): trait 'elementwise': result 0 shape [2, 1] != broadcast of operand shapes [2, 32]
  verify: off
    compiled, no complaint
```

Three levels, three qualities of answer:

| Level | What you learn |
|---|---|
| `off` | Nothing. The compile succeeds and the kernel is wrong. |
| `boundaries` | The graph is invalid after the graph passes. Some pass among the fifteen runs did it. |
| `each-pass` | `cse` did it. |

The middle row is the one to sit with. "Invalid after graph passes" is a true statement that costs you an afternoon; "`cse` produced invalid IR" costs you ten minutes. And the bottom row is the reason the default is not `off`: an unverified compile of a broken pass does not fail, it *succeeds*, and hands you numbers.

Two details in the output are worth naming. The error surfaces twice at `each-pass` — once attributed to `cse` through the trace stream, once as the thrown exception from the boundary check — because in strict mode the pass manager records the error and stops running passes, and then the pipeline's own boundary check finds the same corruption still there. And the two levels report different operation ids for the same defect (`id=2` against `id=23`), because ids come from a process-global counter and the two compiles are consecutive; an id identifies an operation within one run, not across runs.

Then the price. The second half of the lab times a 49-operation graph (a 25-layer MLP) at each level, interleaving the three so drift cannot favour one, over forty rounds (Node 24.9, 2026-08-21):

```
=== compile time by verification level (49 graph ops, 40 interleaved rounds) ===
  level        median     ratio    IQR                min      max
  off           5.35 ms  1.00x   5.13-5.88      5.01     8.06
  boundaries    5.88 ms  1.10x   5.66-6.13      5.38     6.68   [inside the noise]
  each-pass     6.81 ms  1.27x   6.26-7.07      6.10     8.46
```

Per-pass verification costs 20–30% depending on the run, and that part reproduces. The boundary row does not, and **the table says so rather than leaving you to find out**: its interquartile range, 5.66–6.13 ms, overlaps `off`'s 5.13–5.88 ms, so the 1.10× is not distinguishable from noise on this workload. That is unsurprising once you count the work — three module traversals for the whole compile against fifteen for `each-pass` — but a bare ratio would have presented 1.10× and 1.27× as two facts of the same kind, and only one of them is.

### On choosing a statistic

The lab prints the minimum too, immediately below, and the two disagree in an instructive way:

```
for comparison, minimum of 40 rounds (the best case, not a central estimate):
  off           5.01 ms   1.00x
  boundaries    5.38 ms   1.07x
  each-pass     6.10 ms   1.22x
```

There is a real argument for the minimum here, and it is worth stating properly before setting it aside. Compile time is a *non-negative-interference* measurement: every sample is the true cost plus some amount of scheduler noise, cache disturbance and GC, and none of that can make the work go faster. Under that model the smallest sample is the one with the least contamination, and it is the best estimator of the underlying cost. An earlier version of this lab took the median of only fifteen rounds and produced orderings that reversed between runs — `boundaries` measuring *faster* than `off`, which is impossible — which is exactly the failure mode the minimum avoids.

What the minimum is not is a **robust** statistic, and this book does not call it one. Robustness is resistance to a few bad values, measured by breakdown point; the minimum has the worst breakdown point of any order statistic, because a *single* anomalously low sample — a mis-timed clock, a round that skipped work, a JIT that specialized one iteration — replaces the estimate entirely, with no other sample able to outvote it. It is also biased downward by construction, and the bias grows with the number of rounds: take more samples and the minimum drifts further from the typical cost, not closer. And it reports nothing at all about spread, which is the information that would have told you the `boundaries` row is noise.

So the rule this book follows, stated in Chapter 1 §1.8: **report the median with its spread as the headline, and the minimum only as a labelled best case.** The earlier version's problem was not that it used the median, it was that fifteen rounds were too few to have one; forty rounds and a printed IQR fix the reversal and keep the dispersion. Where the quantity you actually care about is the machine's best case, say so and quote the minimum — just do not call it robust, and never quote it alone.

## 15.8 Above the pass manager: phases

The pass manager runs graph passes. It is not the top of the pipeline. [`Compiler.compile`](../../../src/compiler/pipeline/compiler.ts) drives a list of *phases*, each a named record with an optional guard ([`compiler.ts:311`](../../../src/compiler/pipeline/compiler.ts)):

```ts
      {
        name: 'verify:pre',
        when: (ctx: CompileContext) => ctx.compiler.config.verifyEnabled,
        run: (ctx: CompileContext) => ctx.compiler._verifyGraph(ctx.working, 'before graph passes', ctx.trace, ctx.errors, ctx.failed, ctx.resilient),
      },
```

Fourteen phases, in order: `verify:pre`, `graphPasses`, `partition`, `split`, `verify:post`, `lowering`, `tirPasses`, `verify:tensor`, `lirLowering`, `lirPasses`, `verify:lir`, `codegen`, `relaunchOnSerialization`, `planBufferAssignment`. Four of them are verifications, at the four IR boundaries. Two of them — `tirPasses` and `lirPasses` — are entire other pass managers ([`TirPassManager`](../../../src/compiler/passes/tir_pass_manager.ts), Chapter 32) driving the passes of the next IR down.

The phase list is data, and one field explains why. `ctx.restartFrom` ([`compiler.ts:284`](../../../src/compiler/pipeline/compiler.ts)) lets a phase ask the driver to jump back to an earlier phase and re-run from there:

```ts
      const target = phases.findIndex(p => p.name === ctx.restartFrom);
      ctx.restartFrom = null;
      if (target < 0 || relaunches >= MAX_RELAUNCH_ATTEMPTS) continue;
      relaunches++;
      i = target - 1;
```

That is used by exactly one phase today: `relaunchOnSerialization`, which discovers during code generation that a fused kernel had to be serialized to be correct, splits the graph differently, and recompiles. A pipeline written as straight-line code cannot express "go back and try again"; a pipeline written as a list of named phases can, in six lines.

## 15.9 Traps and limits

- **The iteration cap is the only termination argument, and it is per group, not per pass.** A single pass that never settles internally is bounded by its own budget instead — the pattern applicator's `safetyBudget` ([`passes/rewrite/pattern.ts:30`](../../../src/compiler/passes/rewrite/pattern.ts)), Chapter 17. Two independent budgets, two independent silent-give-up paths, each with its own log line.
- **`maxIterations` counts rounds, not work.** A group of five passes with a bound of 8 can run 40 passes. On a large graph, most of them reporting UNCHANGED, that is 40 traversals; with `each-pass` verification it is 40 more.
- **A `FunctionPass` that fails stops the whole module in strict mode.** [`pass_manager.ts:192`](../../../src/compiler/passes/pass_manager.ts) breaks out of the loop over functions on the first failure. The resilient path ([`pass_manager.ts:157`](../../../src/compiler/passes/pass_manager.ts)) instead marks that function failed and carries on with the others — Chapter 18.
- **Instrumentation is a second observation mechanism, and it hands you live IR.** `addInstrument` and the `runBeforePass`/`runAfterPass` hooks ([`pass_manager_base.ts:43`](../../../src/compiler/passes/pass_manager_base.ts)) sit alongside the trace log; all three managers notify, tagging each call with the IR level it ran at. The difference that matters: the trace log emits plain data after the fact, while an instrument is handed the IR object itself, before and after — which is the only way to see what a pass did rather than what it reported. Nothing in `src/` registers one; the callers are outside the compiler.
- **Op counts in the trace are computed only at `VERBOSE` and above** ([`pass_manager.ts:117`](../../../src/compiler/passes/pass_manager.ts): `const opsBefore = verbose ? countOps(module) : -1`). `countOps` walks every function, so this is a real cost and it is correctly gated — but it also means the `-1` you see in a `max-iter` line is a sentinel, not a count.
- **The three pass managers share a base, not a loop.** `PassManagerBase` holds the pass list, the trace handle, per-pass verification and instrumentation; `PassManager`, `TirPassManager` and the LIR manager each still write their own run loop and their own error handling, and agree on the rest by convention. Only the graph manager has fixed-point groups and an analysis manager at all — which is why a graph pass reports `CHANGED`/`UNCHANGED` to an instrument and a TIR or LIR pass reports `null`: the lower managers never asked their passes to say.

## 15.10 Read the tests

- [`tests/compiler/passes/pass-manager.test.js:210`](../../../tests/compiler/passes/pass-manager.test.js) — `FixedPointGroup`: convergence, the cap when a pass never settles, and a one-shot change among otherwise stable passes.
- [`tests/compiler/passes/pass-manager.test.js:125`](../../../tests/compiler/passes/pass-manager.test.js) — attribution of invalid IR to the producing pass, in strict and resilient mode, and what happens with `checkEachPass` off.
- [`tests/compiler/pipeline/invariant-check.test.js`](../../../tests/compiler/pipeline/invariant-check.test.js) — the same three-level experiment as §15.7, run over graph, TIR and LIR, plus the assertion that every IR level the pipeline produces has a verifier registered for it.
- [`tests/compiler/passes/pass-instrument.test.js`](../../../tests/compiler/passes/pass-instrument.test.js) — one instrument watching a whole compile: every pass at all three levels, every `before` paired with its `after`, the repeats inside a fixed-point group, and the IR handed to `runAfterPass` already carrying the pass's edit.

---

**Next:** [Chapter 16 — Analyses and the invalidation problem](../ch16-analyses-and-invalidation/README.md), which is about the two lines of `_applyPass` this chapter skipped: `bumpVersion` and `invalidate`.
