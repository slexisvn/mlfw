# Chapter 16 — Analyses and the invalidation problem

Passes do not work on the IR directly. They work on facts *derived* from the IR — which operations use this value, which operations can be reached from that one, which buffers are live at this point. Computing those facts is expensive, several passes want the same ones, and the IR changes underneath them.

That is three requirements pulling against each other, and getting them wrong produces the worst class of compiler bug: one where the answer is wrong, intermittently, because a pass reasoned about a program that no longer exists. This chapter is about the mechanism that prevents it, and about the two ways it can still be defeated.

## 16.1 The problem: derived facts go stale

A concrete case. `PriorityFusionPass` needs a topological order of the graph to decide what to fuse (Chapter 24). Computing one is a traversal of every operation and every operand — cheap for ten operations, not cheap for ten thousand, and the fusion pass asks for it once per function.

Three passes want that same order. The obvious response is to compute it once and share it. The obvious problem with the obvious response is that fusion *changes the graph*, so the order the next pass gets is an order of a graph that no longer exists. Every value in it may still be a live object; every claim it makes may be false.

So the choice looks binary and both options are bad:

- **Recompute for every pass.** Correct, and quadratic in the worst case: 15 pass runs × a full traversal each, for facts that mostly did not change.
- **Compute once and share.** Fast, and wrong the moment anything moves.

The way out is a third option, and it needs a definition to state.

## 16.2 Intuition: a cache that knows when it expired

Keep the results in a cache, and attach to each entry the *state of the world when it was computed*. Serve a cached entry only when the world has not moved since. Now sharing is safe, and recomputation happens only when it must.

"The state of the world" can be measured two ways, and this compiler uses both:

- **Ask the IR.** Give every function a counter that increments on every structural edit. An entry computed at counter 22 is servable only while the counter reads 22. Nobody has to declare anything; the mechanism is driven by the mutation itself.
- **Ask the pass.** Let a pass declare "I did not disturb the liveness information", and keep that entry across it. This is faster — no recomputation at all across a pass that changed the graph — and it is a *claim*, which means it can be false.

The first is sound by construction — nobody can lie to a counter. The second is a promise, and a promise can be false. The rest of this chapter is about what the promise buys, what happens when it is broken, and — §16.7 — exactly which edits "by construction" covers.

## 16.3 Theory

> **Definition 16.1 (Analysis).** **(stated here)** An *analysis* `A` is a pure function from IR to a result, together with a finite list of analyses it depends on. `A`'s result is computed from the IR and from its dependencies' results.

Purity matters: an analysis may not edit the IR, and its result must be a function of the IR alone. If it were not, caching it would be meaningless, because "the world has not moved" would not imply "the answer has not changed".

> **Definition 16.2 (Preserved analysis).** **(stated here)** A pass `P` *preserves* analysis `A` if, for every IR `m` on which `P` is defined, the cached `A(m)` remains a correct answer to every query that can be made after `P` runs. Formally: for every query `q` that is well-formed against `P(m)`, `A(m).answer(q) = A(P(m)).answer(q)`.

**The observational form is the one that is usable, and the equality form is not.** It is tempting to write the condition as `A(P(m)) = A(m)` — the results are equal — and that version is both stronger than necessary and wrong about what the implementation does. Take the case argued in §16.4: DCE preserves the memory-effect analysis, whose result is a `Map` keyed by `Operation`. DCE erases operations. The cached map therefore still holds entries for operations that no longer exist, so it is *not equal* to the map a fresh run would produce — the fresh one has fewer keys. Under the equality reading, DCE does not preserve it, and the declaration in the source is a bug.

Under the observational reading the declaration is correct, and for the reason §16.4 gives: nothing can *ask* about an erased operation, because asking requires holding a reference to it, and it is gone from the function. The stale entries are unreachable, and an unreachable wrong answer is not a wrong answer. That is the property the compiler actually relies on, so it is the property the definition should state.

This is a genuine property of a pass, not a label. DCE preserving the memory-effect analysis is a theorem about DCE, argued in §16.4. A pass declaring preservation it does not have is not caught by anything — the declaration is trusted.

Now the result that shapes the implementation.

> **Theorem 16.3 (Transitive invalidation is required).** **(stated here)** Let `A` depend on `B`, and let pass `P` not preserve `B`. Then retaining a cached `A` across `P` while discarding `B` is unsound.

*Proof sketch.* `A`'s cached result was computed from `B`'s result at some earlier IR state. `P` not preserving `B` means `B`'s result may differ after `P`. Since `A` is a function of `B`'s result, `A`'s result may differ too. The manager has no way to tell whether this particular `A` reads the part of `B` that moved — the dependency is declared as a whole, not per field — so the only sound rule is to treat "a dependency was invalidated" as "the dependent is invalidated". Dependency is transitive, so the rule must be applied over the transitive closure of the dependency relation. ∎

The theorem is worth reading as an argument about *what a declaration can express*. A pass declares a set of preserved analyses. That set is a promise about the analyses named in it and says nothing about their dependencies. So a pass that preserves `liveness` but not `use_def` has, in effect, preserved nothing — and §16.6 measures exactly that, on the real pipeline.

> **Definition 16.4 (Mutation version).** **(invariant)** Each function carries a monotone counter, incremented by every structural edit to its body. A cache entry records the counter's value at computation time. An entry is *fresh* if the recorded value equals the current one.

Definition 16.4 is a mechanism, not a policy, and it is the sound half of the pair. It needs no declarations and cannot be lied to — **provided every edit path bumps it**. That proviso is doing all the work, it is a property of the IR data structure rather than of any pass, and §16.7 is where its scope is pinned down.

The two mechanisms interact in one specific way that is the trap of this chapter: **a preservation declaration overrides the version check.** Preserving an analysis does not merely skip its deletion; it re-stamps the cached entry with the *current* version, so that later freshness tests succeed. Preservation is therefore not a hint. It is an assertion that switches off the automatic mechanism for that entry.

## 16.4 In mlfw: 99 lines and one WeakMap

[`analysis/analysis_manager.ts`](../../../src/compiler/analysis/analysis_manager.ts). An analysis is not a class hierarchy; it is a shape ([`analysis_manager.ts:5`](../../../src/compiler/analysis/analysis_manager.ts)):

```ts
export type AnalysisCtor<TResult = unknown> = {
  readonly name: string;
  readonly depKey?: string;
  readonly dependencies?: readonly AnalysisCtor[];
  compute(func: GraphFunction, deps: AnalysisDeps): TResult;
};
```

Four static members. `UseDefAnalysis` implements them in four lines ([`use_def.ts:23`](../../../src/compiler/analysis/use_def.ts)):

```ts
  static get name(): string { return 'use_def'; }
  static get depKey(): string { return 'useDef'; }
  static get dependencies(): readonly AnalysisCtor[] { return []; }
```

and `LivenessAnalysis` declares its dependency the same way ([`liveness.ts:51`](../../../src/compiler/analysis/liveness.ts)):

```ts
  static get dependencies(): readonly AnalysisCtor[] { return [UseDefAnalysis as unknown as AnalysisCtor]; }
```

There are five analyses at the graph level, and the dependency graph is shallow:

| Analysis | Depends on | Read by |
|---|---|---|
| [`use_def`](../../../src/compiler/analysis/use_def.ts) | — | priority fusion, dominator fusion, layout, quantization |
| [`memory_effect`](../../../src/compiler/analysis/memory_effect.ts) | — | DCE |
| [`post_dominance`](../../../src/compiler/analysis/dominance.ts) | `use_def` | dominator fusion |
| [`liveness`](../../../src/compiler/analysis/liveness.ts) | `use_def` | rematerialization |
| [`layout`](../../../src/compiler/analysis/layout_analysis.ts) | `use_def` | layout transform |

### The cache

```ts
export class AnalysisManager {
  private _cache: WeakMap<GraphFunction, FuncCache>;
```

A `WeakMap` keyed by function, so a function that is discarded takes its cached analyses with it and no invalidation call is needed for the common case of a dead function. Inside, a `Map` from analysis class to `{ data, version }`.

The lookup is Definition 16.4 ([`analysis_manager.ts:22`](../../../src/compiler/analysis/analysis_manager.ts)):

```ts
    let result = funcCache.get(AnalysisClass as AnalysisCtor);
    if (!result || result.version !== func.version) {
      const deps = this._resolveDeps(AnalysisClass as AnalysisCtor, func, funcCache);
      const data = AnalysisClass.compute(func, deps);
      result = { data, version: func.version };
      funcCache.set(AnalysisClass as AnalysisCtor, result);
    }
```

Miss or stale version, recompute; otherwise serve. `_resolveDeps` recurses through `getAnalysis`, so a dependency shared by two analyses is computed once and the recursion is memoized by the same cache it is filling.

### The counter that nobody declares

`func.version` is not maintained by passes. It is maintained by the IR ([`block.ts:41`](../../../src/compiler/ir/graph/block.ts)):

```ts
  _notifyMutation(): void {
    const fn = this._owningFunction();
    if (fn) fn.bumpVersion();
  }
```

called from every structural edit on a block — insert, remove, replace — and from `Value.replaceAllUsesWith` ([`value.ts:117`](../../../src/compiler/ir/graph/value.ts)) and `Operation.replaceOperand` ([`operation.ts:101`](../../../src/compiler/ir/graph/operation.ts)). The pass manager adds one more bump per CHANGED verdict ([`pass_manager.ts:185`](../../../src/compiler/passes/pass_manager.ts)), so the counter over-counts rather than under-counts, which is the correct direction for a cache key.

This is why §16.5's first run is correct even though not a single pass declares a preservation: the cache is keyed to mutations, and a pass that edits the IR invalidates every entry as a side effect of editing.

### Invalidation, with the transitive rule

[`analysis_manager.ts:51`](../../../src/compiler/analysis/analysis_manager.ts). With no preserved set, everything for that function goes:

```ts
    if (!preservedSet) {
      this._cache.delete(func);
      return;
    }
```

With one, Theorem 16.3 is implemented directly ([`analysis_manager.ts:62`](../../../src/compiler/analysis/analysis_manager.ts)):

```ts
    const staleMemo = new Map<AnalysisCtor, boolean>();
    const isStale = (cls: AnalysisCtor): boolean => {
      const cached = staleMemo.get(cls);
      if (cached !== undefined) return cached;
      const deps = cls.dependencies;
      let stale = false;
      if (deps) {
        for (const dep of deps) {
          if (!isPreserved(dep) || isStale(dep)) { stale = true; break; }
        }
      }
      staleMemo.set(cls, stale);
      return stale;
    };
```

An analysis is stale if *any* dependency is not preserved, **or** if any dependency is itself stale. The second disjunct is the transitive closure; the memo makes the whole walk linear in the number of declared dependency edges rather than exponential in the depth.

And then the line that makes preservation an override rather than a hint ([`analysis_manager.ts:78`](../../../src/compiler/analysis/analysis_manager.ts)):

```ts
      if (!isPreserved(AnalysisClass) || isStale(AnalysisClass)) {
        toDelete.push(AnalysisClass);
      } else {
        const entry = funcCache.get(AnalysisClass);
        if (entry) entry.version = func.version;
      }
```

A surviving entry is re-stamped with the current version. Without that line the entry would be deleted on the next lookup anyway, by the version check, and preservation would buy nothing. With it, preservation is exactly as trustworthy as the pass that declared it.

### The one preservation in the compiler

Exactly one pass in `src/compiler` declares a non-empty preserved set ([`dce.ts:16`](../../../src/compiler/passes/simplify/dce.ts)):

```ts
    this.requiredAnalyses = [MemoryEffectAnalysis];
    this.preservedAnalyses = new Set([MemoryEffectAnalysis as never]);
```

It is worth checking the claim rather than accepting it, because Definition 16.2 is a proof obligation. `MemoryEffectResult` is a map keyed by `Operation` and by `Value`, recording each operation's side-effect kind ([`memory_effect.ts:21`](../../../src/compiler/analysis/memory_effect.ts)). DCE only *erases* operations; it never adds one, never rewires an operand, and never changes an operation's opcode or attributes. So for every operation that survives DCE, its entry is unchanged, and the entries left behind for erased operations are unreachable — nothing can ask about an operation that is no longer in the function. The claim holds.

That single declaration is also the honest summary of how much of this machinery is currently exercised: the infrastructure implements a general dependency-aware invalidation scheme, and the pipeline uses one edge of it. The labs below therefore drive the real manager with analyses of their own, because the compiler's own analyses do not yet exercise the interesting cases.

## 16.5 Lab 1 — Computed once, and the cost of a promise you cannot keep

```bash
node docs/part3/ch16-analyses-and-invalidation/labs/01-computed-once.mjs
```

The lab defines two analyses of its own — `op_count`, and `fanout` which depends on it — and registers them with the *real* `AnalysisManager` that the pass manager is using, by reaching the manager through a pass's `run` argument. Then it asks for `fanout` before every pass and reports whether it was served from cache, together with what the IR actually says at that moment.

First run: nothing preserved, which is the compiler's actual configuration.

```
=== no pass preserves anything (the default) ===
  DecompositionPass     v14   recomputed op_count+fanout fanout=1.40
  canonicalize          v22   from cache                 fanout=1.40
  algebraic_simplify    v23   recomputed op_count+fanout fanout=1.40
  constant_fold         v27   from cache                 fanout=1.40
  cse                   v28   recomputed op_count+fanout fanout=1.30
  dce                   v33   from cache                 fanout=1.30
  canonicalize          v34   recomputed op_count+fanout fanout=1.57
  algebraic_simplify    v34   from cache                 fanout=1.57
  constant_fold         v34   from cache                 fanout=1.57
  cse                   v34   from cache                 fanout=1.57
  dce                   v34   from cache                 fanout=1.57
  PriorityFusionPass    v40   from cache                 fanout=1.57
  MultiOutputFusionPass v41   recomputed op_count+fanout fanout=1.67
  dce                   v41   from cache                 fanout=1.67
  5 of 14 runs recomputed; 0 were served a stale answer
```

Three things are visible here and none of them was declared by anybody.

**The version counter counts mutations, not passes.** It jumps 14 → 22 across `canonicalize` because that pass performed eight structural edits, and stands still at 34 through five consecutive passes that changed nothing.

**Caching works without any declaration.** Nine of fourteen runs were served from cache, purely because the version had not moved since the last computation. This is Definition 16.4 doing the entire job.

**And it is never wrong.** Zero stale answers. Soundness here is a property of the mechanism, not of anyone's care.

Second run: every pass is made to declare that it preserves both analyses — a lie, and a plausible one, of exactly the kind a tired author writes when a profile says the analysis is hot.

```
=== every pass preserves both analyses ===
  DecompositionPass     v14   recomputed op_count+fanout fanout=1.40
  canonicalize          v22   from cache                 fanout=1.40
  algebraic_simplify    v23   from cache                 fanout=1.40
  constant_fold         v27   from cache                 fanout=1.40
  cse                   v28   from cache                 fanout=1.40 <- WRONG, the IR says 1.30
  dce                   v33   from cache                 fanout=1.40 <- WRONG, the IR says 1.30
  canonicalize          v34   from cache                 fanout=1.40 <- WRONG, the IR says 1.57
  ...
  MultiOutputFusionPass v41   from cache                 fanout=1.40 <- WRONG, the IR says 1.67
  dce                   v41   from cache                 fanout=1.40 <- WRONG, the IR says 1.67
  1 of 14 runs recomputed; 10 were served a stale answer
```

One recomputation instead of five — a real speedup — and ten passes reasoning about a graph that stopped existing twenty-seven mutations ago. Nothing fails. No verifier fires; the IR is perfectly valid, and it is the *belief about* the IR that is wrong. If those had been the fusion pass's topological order rather than a number, the compiler would have fused across an edge that no longer exists, and the failure would have surfaced as wrong numbers in a kernel, six phases downstream.

This is the trap in Definition 16.2 made concrete: **preservation is the only place in this infrastructure where a pass can be wrong without being caught.** Verification checks the IR. Nothing checks a claim about what a pass did *not* do.

**Try this.** Preserve both analyses for `dce` alone — the one pass that would be entitled to make such a claim about the memory-effect analysis — and count the stale answers.

## 16.6 Lab 2 — Staleness propagates

```bash
node docs/part3/ch16-analyses-and-invalidation/labs/02-staleness-propagates.mjs
```

Theorem 16.3, measured. The lab builds a three-deep chain — `shape` depends on `fanout` depends on `op_count` — and runs the same compile four times, varying only which analyses the passes claim to preserve:

```
dependency chain: shape -> fanout -> op_count

all three declared preserved           preserves {op_count, fanout, shape}
  over 14 pass runs: op_count recomputed 1x, fanout recomputed 1x, shape recomputed 1x

the two dependents, not the root       preserves {fanout, shape}
  over 14 pass runs: op_count recomputed 5x, fanout recomputed 5x, shape recomputed 5x

the root only                          preserves {op_count}
  over 14 pass runs: op_count recomputed 1x, fanout recomputed 5x, shape recomputed 5x

nothing                                preserves {nothing}
  over 14 pass runs: op_count recomputed 5x, fanout recomputed 5x, shape recomputed 5x
```

Compare rows two and four. **They are identical.** Declaring that you preserve `fanout` and `shape` while saying nothing about `op_count` buys precisely nothing — not for `op_count`, which was never claimed, and not for the two analyses that were, because `isStale` walked from each of them down to an unpreserved root and deleted them anyway. That is the theorem: staleness propagates *up* the dependency chain from the thing that was not preserved to everything built on it, and it propagates through `shape` even though `shape` never mentions `op_count`.

Row three is the useful contrast. Preserving only the root saves the root's recomputations — 1× instead of 5× — while the dependents still recompute, because they were not claimed. Preservation is per-analysis and it is not inherited in either direction.

The practical rule, which is worth memorising before writing a `preservedAnalyses` set: **preserve from the bottom up, or do not bother.** An analysis is worth naming only if everything it depends on is named too.

## 16.7 Traps and limits

Everything in this chapter rests on Definition 16.4's proviso — *every edit path bumps the counter* — which §16.2 sold as the mechanism nobody can lie to. The claim is only as wide as the set of edits the counter intercepts, and the rule to carry is that **a mechanism is "sound by construction" only over the operations it intercepts**.

> **Counterexample 16.5.** Let `A` be an analysis whose result depends on an attribute — a FLOP estimate reading `getFlops`, a layout analysis reading `layoutSensitivity`, a fusion cost model reading a reduction's axes. Compute `A`, then rewrite an attribute *without going through `setAttr`*: `op.attributes.set('direction', 'gt')` is one line and `op.attributes` is a public `Map`. The version does not move, the freshness test passes, and the manager serves a result that disagrees with what recomputing `A` would produce.

Chapter 9 §9.4 covers the API side: every mutating method, attributes included, notifies. §9.9 covers the rest — the containers hanging off an `Operation` are public, so the counter is sound for edits made *through* the object and blind to edits made *to its fields*. Two things keep the exposure small: most cached analyses are *structural* — liveness, use-def, dominance, the cycle check and the op-count analyses read which operations exist and how they are wired, and every edit to that shape goes through a method — and no analysis this compiler caches today reads an attribute at all. The gap is real, narrow, and entirely in front of whoever caches the first attribute-dependent analysis.

- **A false preservation is undetectable.** §16.5 is the demonstration. There is no verification of Definition 16.2, no "recompute and compare" debug mode, and no assertion. The infrastructure trusts the declaration completely. If you add a preservation, the argument for it belongs in a comment or a test, because nothing else will carry it.
- **Analyses are keyed by function, and module passes invalidate by iterating.** `invalidateFunctions` ([`analysis_manager.ts:92`](../../../src/compiler/analysis/analysis_manager.ts)) loops over the module applying the per-function rule. There is no module-level analysis cache at all; a whole-module fact — a call graph, say — has nowhere to live.
- **Two call paths route around the manager entirely.** `LayoutAnalysis.compute(graphFunc, { useDef }, this._policy)` is called directly by the layout pass ([`layout_transform.ts:47`](../../../src/compiler/passes/layout/layout_transform.ts)), and the AD builders call `UseDefAnalysis.compute` directly ([`backward_builder.ts:169`](../../../src/compiler/ad/backward_builder.ts)). Neither result is ever cached. The layout case is forced: the analysis takes a third *policy* argument, and `AnalysisCtor.compute` has room for exactly two — the protocol has no way to express a parameterized analysis, so the parameterized one opted out.
- **`invalidatedAnalyses` is declared and never read.** [`pass.ts:26`](../../../src/compiler/passes/pass.ts) gives every pass the field; nothing in `src/` consults it. Invalidation is driven entirely by the complement of `preservedAnalyses`. A pass that wants to say "I definitely broke liveness" has a field to say it in and no listener.
- **A pass without a manager gets its own.** [`pass.ts:42`](../../../src/compiler/passes/pass.ts) lazily creates a private `AnalysisManager` when `run` is called without one. That keeps a pass usable in isolation — a test can construct one and call `run(func)` — at the cost of a cache that shares nothing. It is the right default and it means "the analysis was computed twice" can have a boring explanation.
- **`WeakMap` keying means a cloned function shares nothing.** Resilient mode clones the module before running passes (Chapter 18), and `cloneGraphFunction` copies `_version` across ([`function.ts:164`](../../../src/compiler/ir/graph/function.ts)) — but the clone is a different object, so it starts with an empty cache. Correct, and worth knowing when a resilient compile looks slower than a strict one.

## 16.8 Read the tests

- [`tests/compiler/analysis/analysis-manager-wiring.test.js:59`](../../../tests/compiler/analysis/analysis-manager-wiring.test.js) — computed-once sharing, forced recomputation after a CHANGED pass, a dependency shared by two analyses computed once, and the private-manager fallback.
- [`tests/compiler/analysis/analysis-manager-wiring.test.js:104`](../../../tests/compiler/analysis/analysis-manager-wiring.test.js) — the declarations themselves: that DCE requires and preserves the memory-effect analysis, that preserving carries an entry across a CHANGED pass, and that not preserving drops it. These are the assertions that pin §16.5 to the source rather than to a lab.

---

**Next:** [Chapter 17 — Pattern rewriting](../ch17-pattern-rewriting/README.md), which zooms in on the passes that do most of the work in the fixed-point group, and asks what a rewrite rule is when it is an object rather than a branch in a `switch`.
