# Chapter 48 — Reproducibility

You have spent thirty seconds. Somewhere in the process there is a `PrimFunc` that is 8% faster than the one the rules would have produced, and in a moment the process will exit. What do you write down?

The obvious answer — write down the schedule — is the wrong one, because a scheduled `PrimFunc` is a graph of mutable nodes with buffer identities and parent pointers, and serialising it means inventing a text format for TIR, a parser for it, and a promise that the two agree. Chapter 32 already noted that TIR has no parser. The autotuner needs something smaller.

The answer this part has been building towards is that a schedule is not a *thing* but a *derivation*: a sequence of primitive applications, each of which is a name and a few numbers. Record the derivation and you can replay it. That object is the `ScheduleTrace`, it is seventy lines, and this chapter is about the exact conditions under which replaying it gives you back what you had.

## 48.1 The problem: what does it take to get the same kernel tomorrow?

Concretely, here is a tuning result:

```
  split        ["ls0_6",4]
  reorder      [["ls0_6_o_0","rs0_7","ls0_6_i_1"]]
  parallelize  ["ls0_6_o_0"]
  vectorize    ["ls0_6_i_1"]
```

Four steps, whose arguments are strings, a number and one array of strings, so the whole thing JSON round-trips exactly. Write it to a file, load it tomorrow, apply it to a freshly lowered matmul, and you have your kernel back.

Except that three of those four steps name a loop that did not exist when the trace started. `ls0_6_o_0` was created *by* step one. Whether step two can find it depends on step one producing a loop with that name — and the name came from a counter that is not in the trace, not in the `PrimFunc`, and not in the tuning record. §48.5 replays the trace above successfully and then, after two unrelated variables have been allocated elsewhere in the process, watches it fail on step two.

That is the chapter's technical content. The wider question is which properties an autotuner's persistence layer actually needs, since the compiler has two mechanisms with different ones — the trace, which is complete and fragile, and the tuning record's `(sketchName, params)` pair, which is incomplete and robust — and it uses the second.

## 48.2 Intuition: a recipe, not a photograph

A photograph of a cake tells you what it looks like. A recipe tells you how to make one, and it works in a different kitchen, but only if the kitchen has the same ingredients and you read the steps in order. Between the two there is a real trade.

A photograph — a serialised IR — carries the result rather than the derivation, so it does not depend on the schedule primitives still meaning what they meant. It is not version-free: it depends instead on the parser, the node schema and the backends' reading of them, which move too. It is also enormous and pins down decisions you would rather leave open.

A recipe — a trace — is tiny and readable, and it composes: you can look at it, edit a factor, replay it. But every step is *interpreted* by the schedule primitives at replay time, so a change to what `split` does changes what the trace means. And a step that refers to something an earlier step created is referring to the interpreter's state, not to anything in the file.

The second intuition is that these are different kinds of promise, and a system should know which one it is making. "Replaying this file gives the same program" is a claim about the interpreter. "Compiling this graph with these tuning parameters gives the same program" is a weaker claim about a smaller interface, and it is the one this compiler actually relies on.

## 48.3 Theory

> **Definition 48.1 (Schedule trace).** **(stated here)** A *schedule trace* is a finite sequence of steps `(π₁, a₁), …, (π_n, a_n)` where each `π_i` is the name of a schedule primitive and each `a_i` is a list of JSON values. It is *recorded* by a `Schedule` when each mutating primitive appends its own name and arguments, and *replayed* on a `Schedule` by dispatching `π_i` as a method with arguments `a_i`, in order.

> **Definition 48.2 (Replayable).** **(stated here)** A trace `T` is *replayable on a `PrimFunc` `P`* if applying every step of `T` in order to `Schedule(P)` completes without throwing. It is *faithful* if the resulting `PrimFunc` is *structurally identical* to the one recorded: the same tree of nodes with the same field values, including the same loop-variable names. Replay always starts from a fresh clone, so node identity is never in question and is not what is being claimed.

The distinction between replayable and faithful is the chapter's. A trace can complete and produce a different nest; a trace can produce the right nest with different variable names, which is neither identity nor a difference anyone would care about; and a trace can fail halfway, leaving a program that is neither the old one nor the new one.

> **Theorem 48.3 (Conditions for faithful replay).** **(stated here)** Let `T` be recorded on `P` with the fresh-variable counter at value `v₀`, producing `P'`. Then replaying `T` on a structurally identical copy of `P` yields exactly `P'` provided that (i) every primitive named in `T` still exists and has the same behaviour, (ii) the counter is again at `v₀` when replay begins, and (iii) no step's arguments depend on state outside `P` and the counter. If (ii) fails and some step names a loop introduced by an earlier step, replay is not faithful and in general not replayable.

*Proof.* For sufficiency: `Schedule` is deterministic given its `PrimFunc` and the counter, since the only non-argument input to any primitive is `freshVar` ([`schedule.ts:194`](../../../src/compiler/schedule/schedule.ts)), which is a pure function of the module-global `_varId`. Replaying the same steps against a structurally identical input therefore produces a structurally identical output, name for name — which is what §48.5 checks, by comparing the printed IR.

For the converse: `split` records `[loop.loopVar.name, factor]` ([`schedule.ts:314`](../../../src/compiler/schedule/schedule.ts)) and introduces two loops named `` `${old}_o_${_varId++}` `` and `` `${old}_i_${_varId++}` ``. A later step referring to either records the name as it was at record time. If the counter differs, the loop created at replay carries a different name, `_resolveLoop` fails to find the recorded one, returns its string argument unchanged ([`schedule.ts:254`](../../../src/compiler/schedule/schedule.ts)), and the primitive receives a `string` where it expects a `ForNode`. ∎

Condition (ii) is the one that has no representation anywhere. There is exactly one way to set the counter — `resetVarCounter()` ([`schedule.ts:198`](../../../src/compiler/schedule/schedule.ts)) — and `TuningRecord` ([`tuning_db.ts:29`](../../../src/compiler/autotune/tuning_db.ts)) stores a workload key, a sketch name, parameters, a score, a trace, a version and two timings. It does not store the counter, and could not usefully: what would have to be recorded is not a value but the guarantee that the process reaches the same value, which is a property of everything the process did beforehand.

> **Proposition 48.4 (Faithful replay requires complete recording).** **(stated here)** Let a primitive change the `PrimFunc` and append no step, and let that change **persist** — that is, no later recorded step overwrites or removes the part of the function it touched. Then for any schedule using it, the recorded trace — *if it replays at all* — is not faithful.

*Proof.* Replay applies the recorded steps and nothing else, so the unrecorded change is absent from the result. Persistence is what makes that absence observable: the recorded steps produce the same function as the original run everywhere except in the part the unrecorded change touched, and by hypothesis nothing later rewrote it. The qualification about replaying at all is necessary in both directions: the omission itself raises no error, but a later step may still throw, and may throw *because* the omitted change is missing. What cannot happen is a faithful result. ∎

**The persistence hypothesis is not a technicality; without it the proposition is false.** An unrecorded mutation that a later recorded step *destroys* leaves no trace in the final function, and replay produces exactly the right answer despite the gap. Two shapes of this are ordinary: an unrecorded annotation on a loop that a later `split` replaces with two fresh loops, and an unrecorded field set on a subtree that a later `rfactor` swaps out wholesale. Chapter 39 §39.2 makes this likely rather than exotic — *every* primitive replaces a subtree, so an unrecorded edit inside one is discarded the moment anything above it is rewritten.

The practical reading is therefore uncomfortable rather than reassuring: **an incomplete trace is not reliably broken.** It may replay faithfully on the schedules you tested and unfaithfully on a schedule that happens to order its primitives differently — which is precisely the situation a search creates, since it explores orderings nobody wrote by hand. A test that replays one recorded schedule and finds it faithful has not established that the recording is complete.

> **Counterexample 48.5.** `tensorize` ([`schedule.ts:1093`](../../../src/compiler/schedule/schedule.ts)) sets `FuncAttr.TENSOR_INTRIN` on the function and records nothing. A schedule that splits and then tensorises records one step; replaying it reproduces the split and drops the intrinsic, and the CUDA backend reads that attribute, so the two programs compile to different kernels. `createMatmulRegisterBlockGPUSketch` is the extreme case: it assigns `schedule.func.body` directly ([`gpu_matmul_sketch.ts:20`](../../../src/compiler/autotune/gpu_matmul_sketch.ts)) and its recorded trace is empty, so replay produces the *unscheduled* function.

> **Corollary 48.6 (A trace is a recipe, not a certificate).** **(invariant)** `ScheduleTrace.replay` performs exactly one check — that the named primitive exists ([`trace.ts:51`](../../../src/compiler/schedule/trace.ts)) — and applies the steps in order with no transaction. A step that throws leaves every earlier step applied. Nothing validates the result: `ScheduleValidator` is run by the tuning session ([`session.ts:186`](../../../src/compiler/autotune/session.ts)) and never by `replay`.

**And "the named primitive exists" is a weaker check than it reads as, because of how the lookup is done:**

```ts
      const method = schedule[step.primitive] as ((...args: unknown[]) => unknown) | undefined;
      if (typeof method !== 'function') {
        throw new Error(`Unknown schedule primitive: ${step.primitive}`);
      }
```

That is an ordinary property access, so it walks the prototype chain — through `Schedule.prototype` and on to `Object.prototype`. **The names a trace can invoke are therefore not the 22 primitives, nor even the ~28 public members of `Schedule`, but every callable property reachable from a `Schedule` instance**, which includes `toString`, `valueOf`, `hasOwnProperty`, `isPrototypeOf`, `propertyIsEnumerable` and `constructor`, plus every private `_`-prefixed method the class defines. A step naming one of those passes the check and is called.

Nothing in this compiler *writes* such a step — traces are produced by the primitives recording themselves — so this is a property of the surface rather than an observed failure. It matters because a trace is a **serialized, persisted, externally-editable artifact**: `TuningRecord.traceData` is written to a database file, and Definition 48.7 exists precisely so that data can come back from disk. Treating a deserialized trace as a list of primitive invocations, when the mechanism will dispatch any inherited method name, is the same category of assumption as trusting a file's contents because your own program wrote it.

The fix is one line — look the name up in an explicit allow-list of the 22 primitives rather than on the object — and it would also make Corollary 48.6's check mean what the chapter says it means.

Finally, the alternative the compiler actually uses.

> **Definition 48.7 (Provenance of a tuning result).** **(stated here)** To reproduce a tuned kernel one needs: the workload it was tuned for, the sketch it came from, the parameters, and the interpreter that turns those into a program. A *trace-based* record fixes the third and fourth together and must therefore pin the interpreter's state. A *parameter-based* record fixes only the second and third and re-derives the rest, and must therefore assume the sketch still exists under the same name and still accepts the same parameters.

Neither is stronger; they assume different things. §48.6 shows that the compiler stores both and reads only the second.

## 48.4 In mlfw

### `ScheduleTrace`

[`schedule/trace.ts`](../../../src/compiler/schedule/trace.ts), 70 lines. A step is a name and an argument array; the trace is a list of them; `serialize` maps over it and `deserialize` maps back:

```ts
export class ScheduleStep {
  primitive: string;
  args: ScheduleArgs;
  …
  serialize(): SerializedStep {
    return { primitive: this.primitive, args: this.args };
  }
```

`args` is typed `readonly unknown[]` ([`trace.ts:1`](../../../src/compiler/schedule/trace.ts)) and is stored by reference and serialised unchanged, so the design rests on the primitives passing only JSON-safe values — a convention, not a constraint the format enforces. Twenty-one of the twenty-two keep it: every `trace.record` call in `schedule.ts` passes loop *names* rather than `ForNode`s, block names rather than blocks, and numbers:

```ts
      this.trace.record('split', [loop.loopVar.name, factor]);
      this.trace.record('reorder', [newOrder.map(l => l.loopVar.name)]);
      this.trace.record('rfactor', [blockName, reductionVarName, factor]);
      this.trace.record('storageAlign', [blockName, bufferName, axis, factor, offset || 0]);
```

That is why `_resolveLoop` ([`schedule.ts:239`](../../../src/compiler/schedule/schedule.ts)) exists at all: every primitive accepts either a `ForNode` or a name, and the name form is what makes a step serialisable. Chapter 40's trap — that `_resolveLoop` returns its argument unchanged when the name is not found — is the same design decision seen from its failure side.

`replay` ([`trace.ts:48`](../../../src/compiler/schedule/trace.ts)) is fourteen lines:

```ts
  replay(schedule: ReplayTarget): void {
    for (const step of this.steps) {
      const method = schedule[step.primitive] as ((...args: unknown[]) => unknown) | undefined;
      if (typeof method !== 'function') {
        throw new Error(`Unknown schedule primitive: ${step.primitive}`);
      }
      schedule._replaying = true;
      try {
        method.call(schedule, ...step.args);
      } finally {
        schedule._replaying = false;
      }
    }
  }
```

`schedule[step.primitive]` is a dynamic property lookup on the `Schedule` instance, so the trace's vocabulary is exactly the object's method names — which means a trace can also name `getLoops` or `verify`, and calling one is harmless but not a schedule step. The `_replaying` flag is the reason replaying does not double the trace: every `trace.record` in `schedule.ts` is guarded by `if (!this._replaying)`. The `finally` restores it even when the step throws, which matters because the caller may keep using the schedule afterwards.

### Where the counter lives

Three lines, at [`schedule.ts:193`](../../../src/compiler/schedule/schedule.ts):

```ts
let _varId = 0;
function freshVar(hint: string, dtype = 'int32'): VariableNode {
  return new VariableNode(`${hint}_${_varId++}`, dtype);
}
```

Module-global and monotone. Every fresh loop variable in the compiler comes from here — `split`'s two, `fuseLoops`' one, `rfactor`'s three, `cacheRead`'s array — and so does every name a later trace step can refer to. `resetVarCounter` is exported and has **no caller in `src/`**: the counter is never reset during a compilation, so it grows for the life of the process and its value when a given block is scheduled depends on everything scheduled before it, in this compilation and in every earlier one.

### The record

`TuningRecord` ([`tuning_db.ts:29`](../../../src/compiler/autotune/tuning_db.ts)) has nine fields, and `bestTrace` ([`session.ts:151`](../../../src/compiler/autotune/session.ts)) fills the trace one:

```ts
  bestTrace(): SerializedStep[] | null {
    if (!this._best) return null;
    const sketch = this.sketchByName.get(this._best.sketchName);
    if (!sketch) return null;
    try {
      const sch = new Schedule(clonePrimFunc(this.primFunc));
      sketch.instantiate(this._best.params)(sch, this.blockName, this.target);
      return sch.trace.serialize();
    } catch (e) {
      this._warn('best-trace', this.blockName, e);
      return null;
    }
  }
```

Note what this is: the winning parameters are re-applied to a fresh clone *purely in order to record what they do*. The schedule it builds is thrown away and only the trace survives. It is the one place in the compiler where a trace is produced deliberately rather than as a side effect.

### The cache hit

`Autotuner.tune` on a cache hit ([`autotuner.ts:228`](../../../src/compiler/autotune/autotuner.ts)):

```ts
      if (task.kind === 'cache') {
        const cached = task.cached as TuningRecord;
        results.set(name, { blockName: name, sketchName: cached.sketchName, params: cached.params, score: cached.score, fromCache: true });
        continue;
      }
```

Three fields out of nine. `_buildTunedSchedule` then re-derives the sketches for the block and looks the name up ([`autotuner.ts:323`](../../../src/compiler/autotune/autotuner.ts)):

```ts
          const sketches = getSketchesForBlock(work, blockName, this.target, blockMap, { … });
          const sketch = sketches.find(s => s.name === result.sketchName);
          if (sketch && this._fitsThreadBlock(work, blockName, sketch, result.params)) {
            const apply = sketch.instantiate(result.params);
            apply(sch, blockName, this.target);
          }
```

This is Definition 48.7's parameter-based path, and it is the whole of the cache. `traceData` is never consulted. The assumptions it makes instead — that `getSketchesForBlock` still returns a sketch of that name for this block, and that the sketch still accepts those parameters — are checked by the `find` returning `undefined` and by the surrounding `try`.

> **"A stale record degrades to no tuning, not to a wrong kernel" needs those two checks to be doing more than they look.** They establish that a sketch of that *name* exists and that it *accepts* those parameters. Neither establishes that the sketch still **means** what it meant when the record was written, and those are different properties:
>
> | What the pair verifies | What it does not |
> |---|---|
> | a sketch named `mlt_cpu` is derived for this block | that it is the *same* `mlt_cpu` — same skeleton, same primitive sequence |
> | it accepts the recorded parameter vector | that the parameters mean the same thing — same order, same semantics per position |
>
> A sketch whose parameter list is reordered, whose skeleton gains a `reorder`, or whose `rfactor` moves to a different axis still parses an old record, still accepts the old parameters, and applies a *different transformation*. The record is a `(name, params)` pair, and the name is a string.

### Two version guards and a key

The database therefore carries two version stamps, and rejects wholesale on either mismatch. `CODEGEN_VERSION` covers the emitted-code format; `SCHEDULE_SEMANTICS_VERSION` covers what the primitives *mean*, and is bumped by hand whenever one of them changes — the `split` lower-bound and `rfactor` identity rules of Chapters 40 and 41 are exactly the kind of change that has to bump it. A database written before the stamp existed is rejected too, since its absence is indistinguishable from an older semantics.

Wholesale rejection is coarse, and it is the right shape for a *format or meaning* change: every record in the file was written under the old meaning, so none of them is salvageable.

The **numerical mode** is different, because records from the two modes are simultaneously valid — they just answer different questions. A schedule tuned with fast-math off must not be served to a fast-math compile, and vice versa, since the set of legal schedules differs. So the mode goes into the *workload key* rather than into a version stamp, and the two sets of records coexist in one database.

That leaves what Proposition 47.6 already told you the key does not contain, and it is worth reading as a list of things a cache hit does not promise:

- **the primitive implementations.** Covered by the semantics version, and only because somebody remembers to bump it.
- **the target's behaviour beyond `name` and `kind`.** Two CPU targets with different cache sizes or vector widths share a key — which is the point on one machine and wrong across two.
- **the enclosing loops.** A block's key describes the block, not its context.

So the accurate statement of the cache contract is: **a hit means a block that looks like this was tuned once, against a target with this name, in this numerical mode, under this schedule semantics, and the parameters still fit.** A `(sketchName, params)` pair is robust in that it survives irrelevant changes; the cost of that robustness is that it survives some relevant ones too, and the version stamps are what keep that set small.

## 48.5 Lab — record and replay

```bash
node docs/part8/ch48-reproducibility/labs/01-record-and-replay.mjs
```

Recording four steps on an 8×8 matmul gives the trace from §48.1, and:

```
  4 steps, and every argument is a string, a number or an
  array of them: JSON round-trips exactly (true).

  the nest it produced:
   for di0_12 in 0..8 {
   for di1_14 in 0..8 {
   for ls0_6_o_0 in 0..2 @parallel {
   for rs0_7 in 0..8 {
   for ls0_6_i_1 in 0..4 @vectorized {
   for c0_8 in 0..8 {
```

Replayed into a fresh copy of the same program:

```
  identical IR: true
  the replay recorded nothing of its own: 0 steps
```

Theorem 48.3's sufficient direction, and the `_replaying` flag doing its job. Condition (ii) holds here because `lowerToTir` — a lab helper, not compiler code — calls `resetVarCounter` before lowering, so both programs are built from the same counter value. The compiler never resets it (§48.4), so condition (ii) is not something the compiler arranges; it is something nothing in the compiler needs, because nothing in the compiler replays a trace.

> **What this lab is evidence for, stated narrowly.** One schedule, four primitives — `split`, `parallelize`, `vectorize`, `reorder` — on one 8×8 matmul, replayed once, compared by printed IR. That is a real check and it is worth running. It is **not** evidence that record/replay is faithful in general, and three of the chapter's own results say why: Counterexample 48.5 names two constructs (`tensorize`, and the register-block GPU sketch) whose traces are incomplete and which this schedule does not use; Proposition 48.4 shows an incomplete trace can replay *correctly* when a later step happens to overwrite the gap, so a passing replay does not certify completeness; and Corollary 48.6 shows nothing validates a replayed result at all. A one-case demonstration cannot distinguish "the mechanism is faithful" from "this schedule avoids the constructs where it is not". The mechanism is faithful *for traces made only of primitives that record themselves completely*, and the set of primitives that do is smaller than the set that exist.

Now break condition (ii) by allocating two variables elsewhere first:

```
  replay threw: reorder expects ForNode arguments

  what changed:

  0 unrelated split(s) beforehand:  split('ls0_6', 4) produces ls0_6_o_0 / ls0_6_i_1
  1 unrelated split(s) beforehand:  split('ls0_6', 4) produces ls0_6_o_2 / ls0_6_i_3
  2 unrelated split(s) beforehand:  split('ls0_6', 4) produces ls0_6_o_4 / ls0_6_i_5
  3 unrelated split(s) beforehand:  split('ls0_6', 4) produces ls0_6_o_6 / ls0_6_i_7
```

The names a `split` introduces are a function of how much scheduling has happened *in the process*, not of the program being scheduled. Step one still succeeds — `ls0_6` is a name the lowering rule chose, so it is stable — and produces `ls0_6_o_2`. Step two asks for `ls0_6_o_0`, `_resolveLoop` does not find it, returns the string, and `reorder` reports a type error.

Two things about that error are worth saying. It is Chapter 40's unresolved loop name arriving where it does the most damage: the message names a type rather than the missing name, so the failure mode of a replayed trace is maximally uninformative. And it is a *loud* failure, which is the good case. A trace whose last step is a `parallelize` on a stale name would fail the same way; a trace consisting only of steps that name original loops would replay silently and correctly. Whether a given trace is fragile depends on whether any step names something an earlier step created — which for every tiling sketch is all of them.

Last, what replay checks:

```
  a trace naming a primitive that does not exist: Unknown schedule primitive: thisIsNotAPrimitive
  and the split before it has already been applied: true
```

Corollary 48.6. One check, no transaction. The `PrimFunc` handed to a failed replay is left in whatever state the last successful step produced — for the search that is harmless, since the session clones before every attempt, and for a hypothetical "restore my tuned kernel from disk" path it would not be.

The section closes with the reassurance that matters:

```
  max |difference| against the unscheduled nest: 0.00e+0
```

A replayed schedule computes the same values as no schedule at all, to the bit. Sound primitives again.

## 48.6 Lab — what the trace omits

```bash
node docs/part8/ch48-reproducibility/labs/02-what-the-trace-omits.mjs
```

Which primitives record, derived by asking each method for its own source:

```
  27 methods = 5 queries + 22 primitives

  primitives that record a step of their own: 18
  primitives that do not:                     4   tile, computeInline, computeInlineBlock, tensorize
```

Twenty-two primitives, matching Chapter 38's count — by a different route, since the outline's repo baseline reaches 22 as "28 public members less six queries" from a hand tally and this reaches it as 27 prototype methods less five. The 22 is the figure both agree on and the one Part VII uses. Three of the four silent ones are composites: `tile` is a sequence of `split`s and a `reorder`, each of which records itself, and `computeInline`/`computeInlineBlock` both route through `_applyInline`, which records under whichever name it was called with ([`schedule.ts:890`](../../../src/compiler/schedule/schedule.ts)). Their traces are complete; they are just written in terms of parts.

And the one argument type the format never constrained:

```
=== `annotate` records whatever it is given ===

  annotate(loop, 'pragma', 4             ) -> ["ls0_6","pragma",4]
  annotate(loop, 'pragma', 'unroll'      ) -> ["ls0_6","pragma","unroll"]
  annotate(loop, 'pragma', [2, 4]        ) -> ["ls0_6","pragma",[2,4]]
  annotate(loop, 'pragma', 1n  (BigInt)  ) -> TypeError: Do not know how to serialize a BigInt
  annotate(loop, 'pragma', () => 1       ) -> ["ls0_6","pragma",null]
  annotate(loop, 'pragma', undefined     ) -> ["ls0_6","pragma",null]
```

`annotate(loop, key, value: unknown)` records `value` verbatim ([`schedule.ts:1089`](../../../src/compiler/schedule/schedule.ts)), and `ScheduleArgs` is `readonly unknown[]`, so nothing between the call and the file says what a step's arguments may be. Three of the six values above survive the round trip. The BigInt throws — and it throws out of `JSON.stringify`, which `saveToFile` runs over the whole database at once ([`tuning_db.ts:146`](../../../src/compiler/autotune/tuning_db.ts)), so one such step costs the file rather than the step. The other two are worse than a throw: a function and `undefined` both serialise to `null`, and replaying `annotate(loop, 'pragma', null)` sets an annotation the recorded schedule did not have.

Nothing in the compiler can reach this, because `annotate` is one of Chapter 38's nine primitives with no caller in `src/`. It is worth naming anyway, because Definition 48.1 says "a list of JSON values" and the code says `unknown` — and the gap between those two is the kind of thing a format only discovers when someone uses it.

`tensorize` is not a composite:

```
  trace steps before tensorize: 1   after: 1
  the function did change:      TENSOR_INTRIN = {"name":"wmma_16x16x16","info":{"M":16,"N":16,"K":16,"a":"A","b":"B","c":"C"}}
```

Counterexample 48.5, executed. One step recorded, two changes made.

And the sketch with no trace at all:

```
  sketch: matmul_register_block_gpu, 32 configurations
  after applying one: 0 trace steps, body replaced: true
  the TuningRecord it would produce: traceData = []
```

A record whose `traceData` is `[]` is not distinguishable from a record whose schedule was a no-op. Both replay to the unscheduled function.

Then the fact that makes all of this latent rather than live:

```
  records written by the first compile:
    key 99cbef04  sketch elementwise_cpu  params {"vector_width":4}
                 traceData: 3 steps  parallelize split vectorize
    key 1d1320f0  sketch mlt_cpu          params {"s0":[3,1,1,2],"s1":[2,3,1,1],"r0":[4]}
                 traceData: 9 steps  split split split split split split reorder parallelize vectorize
    key de0c3ce8  sketch elementwise_cpu  params {"vector_width":4}
                 traceData: 3 steps  parallelize split vectorize

  every stored trace replaced with a nonsense step, then recompiled:
    the compile succeeded:            true
    same kernel up to variable names: true
```

Every stored trace was overwritten with a step naming a primitive that does not exist, and the cached compilation produced the same kernel. Nothing read them. `traceData` is computed by `bestTrace()`, stored, ranked alongside, serialised to JSON, written to disk by `saveToFile`, read back by `deserialize` — and never used. `ScheduleTrace.replay` has no caller in `src/` either.

That is worth being precise about rather than dismissive. The trace is not dead code in the ordinary sense: it is a correctly implemented mechanism whose consumer was never written, and the consumer that *was* written — the parameter-based cache hit — makes a different and weaker set of assumptions. It is also why Theorem 48.3's condition (ii) has never bitten anyone: the compiler reproduces tuned kernels by re-deriving them, and re-derivation regenerates the names rather than referring to them.

Finally, what a trace is bound to and what it is not:

```
  a trace recorded on a 16x16 matmul: [{"primitive":"split","args":["ls0_6",4]}]
  replayed onto a 14x14 matmul: succeeded, outer extent 4
  and a guard appeared, because 4 does not divide 14: true
```

The trace transfers to a different problem size, which is not obvious and is a direct consequence of Chapter 40's Theorem 40.2: `split` rounds up and guards, so a factor recorded against one extent is meaningful against any extent. What a trace does not carry is the target, the block name, or the sketch and parameters it came from — those are arguments to `sketch.instantiate(params)(schedule, blockName, target)` and live in the record beside the trace rather than inside it. A trace alone does not say what problem it solves.

## 48.7 Traps and limits

- **`TuningRecord.traceData` is written, ranked, serialised, persisted and never read.** [`session.ts:151`](../../../src/compiler/autotune/session.ts) computes it by re-applying the winning parameters to a throwaway clone; [`tuning_db.ts:116`](../../../src/compiler/autotune/tuning_db.ts) serialises it; nothing in `src/` consumes it, and `ScheduleTrace.replay` has no caller in `src/` at all. Replacing every stored trace with garbage does not change a cached compilation's output.
- **Faithful replay requires the fresh-variable counter to be at its record-time value.** Theorem 48.3 and §48.5. The counter is a module global ([`schedule.ts:193`](../../../src/compiler/schedule/schedule.ts)) that nothing in `src/` ever resets, so it never returns to a previous value within a process; nothing in the trace, the record or the serialised database mentions it; and the failure mode is `_resolveLoop` returning a string, so the error names a type rather than a missing loop.
- **`tensorize` records no step.** [`schedule.ts:1093`](../../../src/compiler/schedule/schedule.ts) sets a function attribute the CUDA backend reads and appends nothing to the trace, so a tensorised schedule's trace is incomplete by Proposition 48.4.
- **The register-blocked GPU sketch records nothing.** It replaces `schedule.func.body` wholesale ([`gpu_matmul_sketch.ts:20`](../../../src/compiler/autotune/gpu_matmul_sketch.ts)), so its trace is `[]` and is indistinguishable from a schedule that did nothing. Since it is the compiler's fastest GPU matmul, the one kernel most worth reproducing is the one the trace mechanism describes least.
- **A trace step's arguments are `unknown`, and one primitive takes advantage.** `annotate` records its `value` verbatim ([`schedule.ts:1089`](../../../src/compiler/schedule/schedule.ts)) and `ScheduleArgs` is `readonly unknown[]` ([`trace.ts:1`](../../../src/compiler/schedule/trace.ts)). A BigInt makes `JSON.stringify` throw, which in `saveToFile` ([`tuning_db.ts:146`](../../../src/compiler/autotune/tuning_db.ts)) loses the whole database; a function or `undefined` serialises silently to `null` and replays as an annotation that was never set. Unreachable today only because `annotate` has no caller in `src/`.
- **`replay` has no transaction.** [`trace.ts:48`](../../../src/compiler/schedule/trace.ts). A step that throws leaves the earlier steps applied, and the only pre-check is that the primitive name resolves to a function — which it also does for `getLoops`, `verify` and `getTrace`.
- **A trace step can name a primitive that mutates without recording, and replay cannot tell.** The dispatch is `schedule[step.primitive]`, so the trace's vocabulary is the whole public surface of `Schedule` rather than a declared set of primitives.
- **The counter also makes cached compilations non-identical.** §47.6 measured it: two compilations of the same graph with the same cache produce the same kernel up to `_\d+` numbering and not byte for byte, and the cross-session test in `autotuner.test.js` normalises the digits away before comparing. Any downstream consumer keyed on source text — a kernel cache, a build-artefact hash — sees two different kernels.
- **The version stamps cover the primitives and the code generator, and nothing covers the sketches.** `SCHEDULE_SEMANTICS_VERSION` ([`tuning_db.ts:28`](../../../src/compiler/autotune/tuning_db.ts)) is bumped by hand when a primitive's meaning changes, so a change to what `split` does can invalidate every stored record — provided somebody remembers. What has *no* stamp is the thing the cache actually reads: a record is a `(sketchName, params)` pair, and a sketch whose parameter order or skeleton changes still parses an old record, still accepts the old parameters, and applies a different transformation (§48.4). The path that is versioned is the one nothing consumes; the path that is consumed is the one that is not versioned.

## 48.8 Read the tests

- [`tests/compiler/schedule/trace.test.js`](../../../tests/compiler/schedule/trace.test.js) — six tests, and reading them beside Theorem 48.3 is instructive. Three pin the `_replaying` flag: cleared after a step, cleared when a step throws, and an unknown primitive rejected without leaving it set. Two replay `setScope` and `storageAlign`, whose arguments are block and buffer names that no primitive generates. The sixth replays a *single* `split` and asserts that the replayed nest has the same number of loop variables and no longer contains the original one — deliberately not that the names match, which is what lets it pass while the counter differs between the two schedules. No test in the file replays a trace whose later steps name a loop an earlier step created, which is the only shape condition (ii) can break.
- [`tests/compiler/autotune/autotuner.test.js`](../../../tests/compiler/autotune/autotuner.test.js) — `a cache-hit compile regenerates the same tuned source and stays correct` is the closest thing in the suite to a reproducibility test, and it is a test of the *parameter* path: it compares `getSource` after replacing `_\d+` with `_N`. That normalisation is the counter, acknowledged and stepped around.
- [`tests/compiler/autotune/autotuner.test.js`](../../../tests/compiler/autotune/autotuner.test.js) again, the versioning block — a database round-trips, a database with a foreign `codegenVersion` loads as empty. The case Chapter 47 found, a database with no `codegenVersion` at all, is the one shape not tested.

---

Part VIII ends here. [Part IX](../../part9/README.md) leaves the search behind and returns to a question with a single right answer: given the loop nests this part has been permuting, when may two buffers share the same bytes, and what is the smallest arena that holds a program?
