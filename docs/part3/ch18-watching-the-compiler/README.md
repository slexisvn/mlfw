# Chapter 18 — Watching the compiler work

Every lab in this book so far has reached into the same place. The pass ledger in Chapter 14, the round-by-round breakdown in Chapter 15, the rewrite counts in Chapter 17, the IR snapshots going back to Chapter 3 — all of them are one mechanism, read through different filters.

This chapter is about that mechanism, and about the harder question underneath it: what does a compiler owe you when it fails?

## 18.1 The problem: a black box that returns a number

You compile a model. The kernel runs. The number is wrong by 0.3%.

Everything in Parts IV through X is a candidate. A fusion decision, a schedule choice, a memory alias, a code generator's index arithmetic. The pipeline ran fifteen passes and fifteen phases and produced 1,296 characters of generated source, and the only thing you know is that the last digit is wrong.

The instinct is to add print statements, and the instinct is wrong for three reasons. Prints you add are prints somebody has to remove. Prints inside a pass cost something on every compile, even when nobody is reading them. And a print is a *string*: by the time it reaches you, the structure has been flattened into prose you have to parse back.

There is a second, sharper version of the problem. A pass throws. What should happen? The obvious answer — propagate the exception — destroys all the information the compiler had accumulated, including whether the *other* functions in the module were fine and how far the pipeline got. And it leaves the IR in whatever half-rewritten state the failing pass abandoned it in, which is a real problem if the caller intends to look at it.

## 18.2 Intuition: emit structured events, filter at the source

Replace prints with **events**: objects with a type and fields, handed to a sink the caller supplies. The compiler decides what is worth reporting; the caller decides what to do with it — count it, print it, index it, ignore it.

Attach a **level** to every event and a threshold to the log. An event above the threshold is dropped, and — this is the part that makes it cheap — a *caller* that is about to construct an expensive event can ask whether the threshold would drop it and skip the construction.

Be precise about what "cheap" costs, though, because the mechanism is opt-in per call site. Dropping happens inside `emit`, which means reaching it has already cost a method call and the construction of every argument. The one-integer-comparison price is only paid where the caller *guards itself* with `explainsEnabled` first — and some do:

```ts
    if (!this.trace || !this.trace.explainsEnabled) return;
    this.trace.explain('fusion', ops.join('+'), ...);
```

while others do not ([`fusion_pass.ts:94`](../../../src/compiler/passes/fusion/fusion_pass.ts) calls `explain` straight out, so `ops.join('+')` runs and an object literal is allocated before anything checks the level). So the honest statement is: **a disabled diagnostic costs one integer comparison at a guarded call site, and a string join plus an allocation plus a call at an unguarded one.** Neither is expensive next to a graph traversal, which is why the inconsistency has survived; the reason to know about it is that "tracing is free when off" is the kind of claim that stops being true the moment somebody puts an unguarded `explain` inside a loop over operands.

For failure, the intuition is borrowed from databases: make the compile a **transaction**. Work on a copy. If a pass fails, record the failure as data, restore that function from the original, mark it as failed, and keep going with the rest. The caller gets a result object carrying both what succeeded and a list of what did not — rather than an exception carrying one message and no context.

## 18.3 Theory

> **Definition 18.1 (Trace level).** A *trace level* is a totally ordered severity attached to each event and a threshold on the log, such that an event is delivered exactly when `event.level ≤ log.level`.

The ordering must be monotone in a specific sense: raising the threshold may only *add* events, never change or remove ones already delivered. That sounds obvious and it is a real constraint on where you attach levels. Put a level on a *class* of information rather than on an individual message, and never make a low-level event summarize what a high-level event would have said in detail — otherwise turning up the verbosity silently rewrites history, and two runs at two levels can no longer be compared.

> **Definition 18.2 (Explanation, stated here).** An *explanation* is an event carrying four fields: the *category* of decision, the *subject* it was made about, the *decision* taken, and the *reason* — where the reason is expressed in the terms the decision procedure actually used.

The last clause is what separates an explanation from a log line. "fused add+maximum" is a log line: it tells you what happened, which you could have seen in the IR anyway. "fused add+maximum because it saves 256 bytes of traffic against a 5µs launch cost" is an explanation: it names the quantities the cost model compared, so you can disagree with it. A compiler that logs its decisions tells you what it did; a compiler that explains them tells you what it would take to make it decide otherwise.

> **Definition 18.3 (Transactional compilation, stated here).** A compilation is *transactional* if, when a pass fails, the IR the caller handed in is left exactly as it was, the working module is left as the failing pass found it, and every function other than the failing one still produces output.

**The middle clause has to be earned per pass rather than per compilation.** A database transaction rolls back the *failed unit of work*; rolling back only at the outermost edge would protect the caller and leave the compiler itself working from a half-edited module. §18.7 has the code and the counterexample it rules out.

Two clauses, and both are needed. The first is about the caller's data: a pass that throws in the middle of a rewrite leaves IR that is not merely unoptimized but *invalid*, and handing that back is worse than handing back nothing. The second is about salvage: in a module of forty functions, one failure should cost you one function.

The cost of the first clause is a copy of the module before any pass runs, and it is paid only in resilient mode.

## 18.4 In mlfw: one log, eleven event types

[`pipeline/trace.ts`](../../../src/compiler/pipeline/trace.ts) — 127 lines. The levels ([`trace.ts:8`](../../../src/compiler/pipeline/trace.ts)):

```ts
export const TraceLevel = Object.freeze({
  SILENT: 0,
  INFO: 1,
  VERBOSE: 2,
  DEBUG: 3,
});
```

and the filter, which is the entire delivery mechanism ([`trace.ts:57`](../../../src/compiler/pipeline/trace.ts)):

```ts
  emit(event: TraceEvent): void {
    if (event.level > this.level) return;
    event.timestamp = performance.now();
    this.sink(event);
  }
```

Everything else in the file is a typed constructor on top of it. `phaseStart`, `phaseEnd`, `passRun`, `functionEvent`, `irDump`, `memoryStats`, `autotuneStats`, `codegenStats`, `errorEvent`, `warn`, `explain` — eleven methods, each producing a fixed event type at a fixed level, plus `pass_detail`, which passes emit through `emit` directly with whatever fields they have. The default sink is `() => {}` ([`trace.ts:37`](../../../src/compiler/pipeline/trace.ts)), so a compile with no trace configured pays one comparison per event and nothing else.

The level assignment is the design decision worth arguing about, and it lines up with what a reader at each level is trying to learn:

| Level | Event types | The question being answered |
|---|---|---|
| `INFO` | `phase`, `function`, `error`, `warning` | Where did the time go, and did anything go wrong? |
| `VERBOSE` | `pass`, `memory`, `autotune`, `codegen` | What did each pass and each subsystem do? |
| `DEBUG` | `ir_snapshot`, `explain`, `pass_detail` | Why did it decide that, and what did the IR look like? |

### Two gates, not one

IR snapshots are the expensive case: printing a module is a full traversal producing kilobytes of text, and a compile has four points where it could do it. So they are gated twice ([`trace.ts:123`](../../../src/compiler/pipeline/trace.ts)):

```ts
  shouldSnapshot(point: keyof IRSnapshotFlags): boolean {
    return this.level >= TraceLevel.DEBUG && !!this.irSnapshot[point];
  }
```

DEBUG *and* an explicit opt-in per point. Turning on DEBUG to see explanations does not flood you with three copies of the IR; you ask for `afterGraphPasses` and get exactly that. Every lab in this book that prints IR does so through this flag.

Explanations get the cheaper version of the same idea ([`trace.ts:121`](../../../src/compiler/pipeline/trace.ts)):

```ts
  get explainsEnabled(): boolean { return this.level >= TraceLevel.DEBUG; }
```

and the call sites are expected to ask *before* building the message ([`priority_fusion.ts:243`](../../../src/compiler/passes/fusion/priority_fusion.ts)):

```ts
  _explain(group: FusionGroup): void {
    if (!this.trace || !this.trace.explainsEnabled) return;
    const ops = group.ops.map(o => o.opName);
    this.trace.explain('fusion', ops.join('+'), 'fused', null, { groupSize: ops.length, strategy: 'priority' });
  }
```

The guard is not saving the `emit` call — `emit` would have dropped it anyway. It is saving the two lines under it: a `map` over the group and a `join`, allocating two objects per fusion decision on the hot path of the fusion engine. That is the pattern: **the level check belongs where the message is constructed, not where it is delivered.**

### The error record

A failure is a value ([`trace.ts:15`](../../../src/compiler/pipeline/trace.ts)):

```ts
export class CompilationError {
  phase: string;
  funcName: string;
  message: string;
  passName: string | null;
```

Four fields, which is three more than an exception carries. `toString` assembles them into `[graphPasses] Sequential (cse): deliberate failure inside cse` — phase, function, pass, message. Chapter 15's per-pass verification is what fills in the third field; without it the parenthesis is empty and you are back to knowing only the phase.

### Resilient mode

One line makes the compile transactional. `Compiler.compile` reads the mode ([`compiler.ts:264`](../../../src/compiler/pipeline/compiler.ts): `const resilient = this.config.errorMode === 'resilient';`) and then builds the context every phase will work through ([`compiler.ts:270`](../../../src/compiler/pipeline/compiler.ts)):

```ts
    const ctx: CompileContext = {
      compiler: this,
      trace,
      errors,
      failed,
      resilient,
      original: graphModule,
      working: resilient ? cloneGraphModule(graphModule) : graphModule,
```

In strict mode the compiler edits the caller's module in place, which is fast and which is why a strict-mode failure can leave that module wrecked. In resilient mode it works on a clone, and the caller's module is untouched no matter what happens — Definition 18.3's first clause, bought with one copy.

The second clause is the pass manager's ([`pass_manager.ts:157`](../../../src/compiler/passes/pass_manager.ts)), which wraps each function's pass run in a `try` and, on a throw, records a `CompilationError` and adds the function to `ctx.failedFunctions`. Every later pass skips it ([`pass_manager.ts:152`](../../../src/compiler/passes/pass_manager.ts): `if (ctx.failedFunctions.has(func.name)) continue;`), and so does every later *phase* ([`compiler.ts:454`](../../../src/compiler/pipeline/compiler.ts)). And the half-rewritten function is replaced with a fresh copy of the original ([`compiler.ts:424`](../../../src/compiler/pipeline/compiler.ts)):

```ts
          if (resilient && original && original !== graphModule) {
            const orig = original.getFunction(name);
            if (orig) graphModule.addFunction(cloneGraphFunction(orig));
          }
```

so the module the compiler goes on working with is valid IR throughout, even though one of its functions is now flagged as un-compilable.

## 18.5 Lab 1 — Four levels of the same compile

```bash
node docs/part3/ch18-watching-the-compiler/labs/01-four-levels.mjs
```

The same MLP, compiled four times, counting events by type:

```
event type       SILENT    INFO VERBOSE   DEBUG
phase                 0      22      22      22
function              0       4       4       4
pass                  0       0      15      15
memory                0       0       1       1
codegen               0       0       1       1
pass_detail           0       0       0       4
explain               0       0       0       1
ir_snapshot           0       0       0       2
TOTAL                 0      26      43      50
```

Definition 18.1 in a table: each column is a superset of the one to its left. Nothing is reworded, nothing disappears, and 26 events is enough to see the whole shape of a compile.

The twenty-two phase events are eleven start/end pairs, and it is worth resisting the obvious guess about which eleven. They are:

```
compile  graphPasses  lowering  scheduling  scheduling  simplify
memoryScheduling  memoryPlanning  lirLowering  lirSimplify  codegen
```

Only four of those — `graphPasses`, `lowering`, `lirLowering`, `codegen` — are entries in Chapter 15's fourteen-phase list. `compile` is the outer bracket around the whole run and is not in that list at all, and the remaining six are TIR and LIR *passes* announcing a phase of their own, because both of those managers call `phaseStart(pass.phase)` per pass ([`tir_pass_manager.ts:51`](../../../src/compiler/passes/tir_pass_manager.ts), [`lir_pass_manager.ts:48`](../../../src/compiler/passes/lir_pass_manager.ts)). `scheduling` appears twice because two different TIR passes declare that same phase name — `InlineReindexPass` ([`inline_reindex_pass.ts:58`](../../../src/compiler/passes/schedule/inline_reindex_pass.ts)) and `SchedulePass` ([`schedule_pass.ts:18`](../../../src/compiler/passes/schedule/schedule_pass.ts)).

Going the other way, ten of Chapter 15's fourteen phases emit nothing: `verify:pre`, `verify:post`, `verify:tensor`, `verify:lir`, `split`, `tirPasses`, `lirPasses`, `relaunchOnSerialization` and `planBufferAssignment` never call `phaseStart` at all, and `partition` does ([`compiler.ts:446`](../../../src/compiler/pipeline/compiler.ts)) but is guarded off in a default compile. So the phase *stream* and the phase *list* are two different enumerations that happen to share four names — which is worth knowing before you try to reconstruct the pipeline from a trace.

The second half prints one event of each type, which is the fastest way to learn what is available:

```
  {"type":"phase","action":"start","phase":"compile"}
  {"type":"pass","passName":"CallInlinerPass","changed":false,"durationMs":0.0164,"opCountBefore":10,"opCountAfter":10}
  {"type":"pass_detail","passName":"PatternApplicator","totalRewrites":2,"patternCount":30}
  {"type":"explain","category":"fusion","subject":"add+maximum","decision":"fused","reason":null,"groupSize":2,"strategy":"priority"}
  {"type":"ir_snapshot","label":"afterGraphPasses"}  (+ 16 lines of IR)
  {"type":"function","phase":"lowering","funcName":"Sequential","durationMs":0.1926}
  {"type":"memory","funcName":"Sequential","durationMs":0.7004,"peakMemory":128,"totalTemporaries":4,"totalInplace":1}
  {"type":"codegen","funcName":"Sequential","durationMs":0.1376,"sourceSize":1296,"targetName":"cpu_generic"}
```

These are objects, not lines. A sink that wants "total time in codegen across every function" adds two numbers; one that wants "every pass that changed something" filters on a boolean. That is the whole argument for structured events over prints, and it is why every lab in this book is fifteen lines instead of a log parser.

(`durationMs` values are machine-specific and are rounded here; everything else in this output is deterministic.)

**Try this.** Write a sink that prints only the three slowest phases. Then one that reconstructs the pass ledger from Chapter 14 — it is a four-line filter.

## 18.6 Lab 2 — Why it decided that

```bash
node docs/part3/ch18-watching-the-compiler/labs/02-why-it-decided-that.mjs
```

The same five-layer MLP compiled under two fusion strategies, with scheduling on:

```
=== fusion strategy: priority ===
  add+maximum    fused      because: (no reason recorded)
  add+maximum    fused      because: (no reason recorded)
  and 6 block(s) scheduled by rule 'fallback', 3 block(s) scheduled by rule 'reduction_cpu'

=== fusion strategy: dominator ===
  add+maximum    fused      because: saves 256 bytes, 5us launch
  add+maximum    fused      because: saves 256 bytes, 5us launch
  and 6 block(s) scheduled by rule 'fallback', 3 block(s) scheduled by rule 'reduction_cpu'

=== the same compile with explains off (INFO) ===
  explain events: 0
```

The `dominator` line is Definition 18.2 satisfied: category `fusion`, subject `add+maximum`, decision `fused`, reason *saves 256 bytes, 5µs launch*. Those two numbers are the two sides of the cost model's comparison — the memory traffic a fusion removes against the kernel launch it saves — and seeing them means you can check the arithmetic. Two 2×32 f32 tensors is 256 bytes; if that number looked like 4 bytes you would know the cost model was reading the wrong shape.

The `priority` line is the same decision without the explanation, and this is a real gap rather than a teaching simplification. The priority engine picks fusions from a benefit heap (Chapter 24); every one of those benefits is a number it computed and then discarded. It is the default strategy, and it is the one that cannot tell you why.

The schedule explanations show the other common shape: not a cost comparison but *which rule matched*. `reduction_cpu` and `fallback` are named rules in the target's schedule table (Chapter 43), and knowing which one claimed a block is usually enough to know why the loop nest looks the way it does.

**Try this.** Compile with `fusion: { enabled: false }` and confirm the fusion explanations vanish while the schedule ones do not. Then look at what happens to the `reduction_cpu` count.

## 18.7 Lab 3 — When a pass fails

```bash
node docs/part3/ch18-watching-the-compiler/labs/03-when-a-pass-fails.mjs
```

The lab breaks two passes — one module pass and one function pass — and compiles under both error modes.

```
=== errorMode: strict ===
  compile threw: deliberate failure inside CallInlinerPass
  error events seen before the throw: 0

=== errorMode: resilient ===
  compile returned; succeeded: false
  errors recorded: 2
    [graphPasses] Sequential (CallInlinerPass): deliberate failure inside CallInlinerPass
    [graphPasses] Sequential (cse): deliberate failure inside cse
  functions marked failed: Sequential
  kernels emitted: 0
```

Strict mode gives you the first exception and nothing else — not the phase, not the function, not whether anything else would also have failed. Resilient mode gives you a result object: `succeeded` is false, two `CompilationError`s carry phase, function and pass, and the compile ran to the end.

The second error is the interesting one. In strict mode you would never have learned that `cse` was also broken, because the compile stopped at the first failure. Resilient mode kept going and found it — the module pass failure did not poison the function pass that followed, because a `ModulePass` throwing in resilient mode records an error and returns control without marking any function failed ([`pass_manager.ts:123`](../../../src/compiler/passes/pass_manager.ts)). This is what makes resilient mode the right setting for a bisection: one run enumerates every failure instead of the earliest.

`kernels emitted: 0` is the honest limit of this demonstration. Definition 18.3's second clause is about *other functions* surviving, and a traced model is one function, so there is nothing to salvage. The property is real and the test that pins it builds a two-function module where one function fails and the other compiles and runs correctly — [`tests/compiler/pipeline/resilient-transaction.test.js`](../../../tests/compiler/pipeline/resilient-transaction.test.js), which also asserts the first clause directly: a pass that stamps an attribute on an operation and *then* throws leaves no trace of that attribute on the caller's IR.

### What "transactional" does not cover

Read that last assertion carefully — *on the caller's IR*. It holds because of the clone, and the clone is the only rollback there is. Follow what happens to the module the compiler is actually working on ([`pass_manager.ts:120`](../../../src/compiler/passes/pass_manager.ts)):

```ts
      let result;
      try {
        result = pass.run(module, this.analysisManager);
      } catch (e) {
        if (!resilient) throw e;
        this.analysisManager.invalidateAll();
        results.push(PassResult.FAILED);
        ctx.errors.push(new CompilationError('graphPasses', module.name || '<module>', (e as Error).message, pass.name));
        return { changed, fatal: false };
      }
```

The handler records the error, drops the analysis cache, and restores the module. That last step is the one worth dwelling on, because without it the phases that follow would carry on from a half-edited module:

> **Counterexample 18.6.** A `ModulePass` that erases three operations and then throws on the fourth leaves the module three operations lighter. In resilient mode the pipeline records one error and continues, so subsequent passes optimize, lower and generate code from IR that no complete pass ever produced. If the partial edit happens to leave the module *valid*, nothing downstream notices and the compile succeeds — emitting kernels for a program that is neither the original nor the transformed one.

The snapshot is taken per module pass, and only when the mode asks for it:

```ts
      const snapshot = resilient ? cloneGraphModule(module) : null;

      let result;
      try {
        result = pass.run(module, this.analysisManager);
      } catch (e) {
        if (!resilient) throw e;
        module.restoreFrom(snapshot as GraphModule);
```

`restoreFrom` refills the module's function table *in place* rather than swapping the object, because the pipeline and the caller both hold a reference to it — the same reason `Compiler.compile` clones into `ctx.working` instead of reassigning. Strict mode pays nothing: no clone is taken and the exception propagates.

The cost is a module clone per module pass in resilient mode, which is why it is gated on the mode rather than always on. Two things still hold and are worth knowing rather than assuming. Per-pass verification (Chapter 15) runs only on `CHANGED`, and a throwing pass reports `FAILED`, so it does not fire here — the *phase* boundary check is what would catch a structurally invalid module, attributed to the phase rather than the pass. And a `FunctionPass` failure marks its function failed, so that function is dropped rather than compiled from a mangled state.

So Definition 18.3 holds at all three levels: **the caller's IR is safe, the working module is left as the failing pass found it, and other functions still produce output.** [`resilient-transaction.test.js`](../../../tests/compiler/pipeline/resilient-transaction.test.js) pins the middle clause with a module pass that deletes a function and then throws — the deleted function is restored, and its kernel still computes the right numbers.

**Try this.** Sabotage `dce` instead of `cse` and watch the error count. Then sabotage a pass name that does not exist and confirm the compile succeeds — `shouldRun` is called with every pass, so a typo in a disable-set fails silently.

## 18.8 Traps and limits

- **The default sink discards everything, and the default level is SILENT.** A compile with no `trace` option emits nothing. That is the right default and it means "I added a trace and saw nothing" almost always means the level was left at the default, not that the event is missing.
- **`explain` has seven call sites.** Fusion (three strategies), scheduling (two), tensorization, and the serialization relaunch. Lowering, memory planning, layout and code generation make decisions and explain none of them. The mechanism is good; the coverage is a fraction of the decisions a wrong number could come from.
- **The priority fusion engine explains only its successes.** `explain('fusion', ops, 'fused', null, …)` — the reason is always `null`, and a pair the engine *declined* to fuse produces no event at all. The dominator strategy reports both outcomes with reasons ([`dominator_fusion.ts:96`](../../../src/compiler/passes/fusion/dominator_fusion.ts)). Since priority is the default, the default configuration is the least explicable one.
- **A `pass` event's `durationMs` measures more than the pass, and less than the pass costs.** The clock starts before the manager fetches the pass's required analyses ([`pass_manager.ts:155`](../../../src/compiler/passes/pass_manager.ts)) and stops as soon as `run` returns ([`pass_manager.ts:181`](../../../src/compiler/passes/pass_manager.ts)), so the figure includes any analysis computed on this pass's behalf — which may be a cache hit for one pass and a full traversal for the next — and excludes the invalidation and the per-pass verification that follow. Two passes' durations are not comparable quantities.
- **Warnings are `INFO`-level and easy to miss.** `TraceLog.warn` ([`trace.ts:113`](../../../src/compiler/pipeline/trace.ts)) has five call sites, and two of them report something you would very much want to know: *kernel serialized to a single thread* ([`compiler.ts:661`](../../../src/compiler/pipeline/compiler.ts)) and *the graph cannot be split any further* ([`compiler.ts:545`](../../../src/compiler/pipeline/compiler.ts)). Both are performance cliffs of an order of magnitude, both are delivered as events into a sink that is `() => {}` by default, and neither is surfaced anywhere else. If you compile with no trace configured, a compiler that has decided to run your GPU kernel on one thread will not mention it.
- **Resilient mode costs a full module clone even when nothing fails.** `cloneGraphModule` runs before the first pass, unconditionally, whenever `errorMode` is `resilient`. It also means analyses cached against the original module's functions are useless, because the clone's functions are different objects (Chapter 16).
- **A failed function is skipped, not compiled unoptimized.** The restore in `_runGraphPasses` puts the original IR back into the module so the module stays valid, but `failedFunctions` still contains the name, and lowering skips it. There is no "fall back to the unoptimized version" path — a failure removes the function from the output, it does not degrade it.

## 18.9 Read the tests

- [`tests/compiler/pipeline/resilient-transaction.test.js`](../../../tests/compiler/pipeline/resilient-transaction.test.js) — both clauses of Definition 18.3, with a pass that mutates and then throws, and the assertion that the surviving function still computes the right numbers.
- [`tests/compiler/pipeline/invariant-check.test.js:130`](../../../tests/compiler/pipeline/invariant-check.test.js) — how a corruption is attributed to a pass across the three verification levels, which is the same machinery seen from the error side.
- [`tests/compiler/pipeline/relaunch-on-serialization.test.js`](../../../tests/compiler/pipeline/relaunch-on-serialization.test.js) — the one phase that uses `restartFrom`, and the `explain('relaunch', …)` event that records it happened.

---

**Part III ends here.** You have the infrastructure every later part is written against: a pass as a named transformation with a verdict (Chapter 14), a manager that sequences them, iterates the cheap ones to a fixed point, and verifies between each pair so a corruption arrives with a culprit attached (Chapter 15), an analysis cache keyed to mutations and to declarations, where one of the two can be lied to (Chapter 16), rewrite rules as independent objects driven by a worklist to a normal form (Chapter 17), and an event stream that lets you watch all of it without editing any of it (Chapter 18).

Everything from here is a pass. Part IV starts writing them.

**Next:** [Part IV — Graph-level optimization](../../part4/README.md), which opens with the three passes you have watched run fifteen times without being told what they do: constant folding, common subexpression elimination, and the one that is much harder than it looks — dead code elimination.
