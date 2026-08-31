# Chapter 12 — What "valid IR" means

Chapter 3 made a claim in passing and moved on: interleaving verification between phases "converts *the model produced garbage* into *pass X produced invalid IR*, which is a debuggable statement." This chapter makes good on it.

The question is narrower than it sounds. Not *is this program correct?* — no compiler can answer that. The question is: **what must be true of the data structure for the rest of the compiler's assumptions to hold?** That set is small, it is written down as code, and knowing it is most of what you need to debug a compiler.

## 12.1 The problem: what would you even check?

A verifier that checks too little is useless; a pass corrupts the IR, three phases run happily on nonsense, and the failure surfaces as a wrong number at the end.

A verifier that checks too much is worse. Every intermediate state a pass passes through must satisfy it, and passes legitimately pass through inconsistent states — a rewrite that replaces one operation with another has a moment where the old one has no users and the new one is not yet inserted. A verifier that rejects those states forces every pass to be written defensively around it.

So the design question is: which invariants are *load-bearing* — relied on by some later component that will misbehave without them?

Four answers here, and it is worth naming them before reading any code:

1. **Every use has a definition, in scope.** Otherwise there is nothing to read.
2. **No dependency cycles.** Otherwise no execution order exists (Theorem 8.4).
3. **Operations match their declarations.** Arity, required attributes, declared traits, inferred types.
4. **The function's boundary matches its signature.** Otherwise callers are lied to.

Notice what is *not* on the list, and this is the interesting part.

## 12.2 The invariant that is deliberately absent: dominance

A classical SSA compiler enforces **dominance**: a definition must dominate every use, meaning every path from the function's entry to the use passes through the definition. It is the invariant that makes SSA sound in the presence of branches, and its machinery — dominator trees, dominance frontiers — is a substantial part of LLVM.

This IR does not check dominance, and it is right not to.

Dominance is a statement about *paths through a control-flow graph*. It exists because in a branching program a definition may be reached on one path and not another. Here there is no control-flow graph: a function's body is a single block, control flow lives inside operations as regions (Chapter 9), and a region's block is entered exactly once per execution of its parent operation.

In a structure with no branching, every path from entry to a use is the same path, and "definition dominates use" collapses into "definition is in scope and the graph is acyclic" — which is invariants 1 and 2.

> **Theorem 12.1 (Scope plus acyclicity suffices).** **(stated here)** In a dataflow IR whose functions have a single block and whose control flow is expressed as operations carrying regions, a use-def graph in which every operand is defined within the enclosing scope and no dependency cycle exists admits an execution order in which every definition precedes every use. No dominance relation need be computed.
>
> *Proof sketch.* Acyclicity gives a topological order of the operations under the def-use edges (Theorem 8.4) — producers first, which is the executable direction. In that order every operand's producer precedes its consumer. Because a function body has one block and a region's block is entered unconditionally when its parent operation executes, there is no alternative path along which a definition could be skipped. So the topological order is a valid execution order. ∎

This is why Chapter 8 could spend its length on use-def lists and never mention a dominator tree. It is a genuine simplification bought by the representational choice in Chapter 9 — and it is one worth carrying to other compilers as a question: *does this IR actually branch, or has it merely inherited the machinery of one that did?*

## 12.3 In mlfw: three checkers, at three moments

The surprise on first reading this codebase is that there is not one verifier. There are three, they check different things, and they run at different times. Learning which is which is the debugging skill this chapter is for.

### The parser — structure, at read time

Chapter 13 covers it properly, but note now that reading IR text enforces invariants 1 and 2 by construction. The parser resolves operand names against a symbol table ([`parser.ts:438`](../../../src/compiler/ir/graph/parser.ts)):

```ts
  resolve(name: string, line: number): Value {
    const value = this.values.get(name);
    if (!value) throw new IRParseError(`use of undefined value '${name}'`, line);
    return value;
  }
```

and refuses a name bound twice ([`parser.ts:433`](../../../src/compiler/ir/graph/parser.ts)), and detects cycles while ordering ([`parser.ts:413`](../../../src/compiler/ir/graph/parser.ts)). Text that violates the SSA core never becomes a module at all.

### `verify()` — the signature, cheaply, on demand

`GraphModule.verify()` and `GraphFunction.verify()` are instance methods, available to anyone holding a module ([`function.ts:121`](../../../src/compiler/ir/graph/function.ts)):

```ts
  verify(): string[] {
    const errors: string[] = [];
    if (!this.entryBlock) {
      errors.push('Function has no entry block');
      return errors;
    }
    if (this.entryBlock.arguments.length !== this.inputTypes.length) {
      errors.push(`Entry block has ${this.entryBlock.arguments.length} args but function expects ${this.inputTypes.length}`);
    }
    const ret = this.getReturnOp();
    if (!ret) {
      errors.push('Function body has no return op');
    } else if (ret.numOperands !== this.outputTypes.length) {
```

That is the whole of it: entry block arity, a return exists, return arity. Invariant 4, and nothing else. It does not look at a single operation's operands.

### `verifyModule` — everything, at phase boundaries

The real verifier is a free function in [`verifier.ts:43`](../../../src/compiler/ir/graph/verifier.ts), and it is what the pipeline runs at `verify:pre`, `verify:post`, and — in its TIR and LIR equivalents — `verify:tensor` and `verify:lir`. It descends: module → function → block → operation.

At the function level it establishes the **scope set** ([`verifier.ts:81`](../../../src/compiler/ir/graph/verifier.ts)):

```ts
  const scope: Scope = { defs: new Set<Value>(), parent: null };
  for (const arg of func.entryBlock.arguments) {
    scope.defs.add(arg);
  }
  for (const block of func.body) {
    collectScopeDefs(block, scope.defs);
  }
```

A `Scope` is a set plus a link to its enclosing scope ([`verifier.ts:32`](../../../src/compiler/ir/graph/verifier.ts)), and membership walks the chain ([`verifier.ts:36`](../../../src/compiler/ir/graph/verifier.ts)). A region opens a child scope holding only its own definitions, so the outer set is never copied — a function with `r` region operations and `v` values costs `O(v)` rather than `O(r·v)`.

Read that carefully, because the order matters: **the entire scope set is collected before any operand is checked.** So "used before definition" here means "used without any definition anywhere in this scope", not "used before its definition textually". That is exactly right for a DAG — Theorem 8.4 says textual position carries no meaning among the non-terminator operations, so a verifier that insisted on textual precedence would reject a module whose only sin was being printed in an unusual order. Note the limit of that licence, though: Chapter 8's Lab 2 reverses the *whole* block, terminator included, and the verifier does reject that — not for operand order, which it ignores, but for the terminator rule checked one level up at the block. Dataflow order is free; block structure is not.

At the block level it checks acyclicity, with an explicit iterative depth-first search rather than recursion ([`verifier.ts:149`](../../../src/compiler/ir/graph/verifier.ts)), reporting `participates in a value dependency cycle` on the offending operation — and it checks that a region's block ends in a terminator ([`verifier.ts:203`](../../../src/compiler/ir/graph/verifier.ts)).

At the operation level ([`verifier.ts:212`](../../../src/compiler/ir/graph/verifier.ts)) it walks the `OpDef` from Chapter 11 field by field: operands in scope, results whose `definingOp` points back, every result shape free of a negative extent that is not `DYNAMIC`, the operation is registered, arity matches `numOperands` / `numResults`, required attributes present, `numRegions` matches — and then two delegations that are the interesting part:

```ts
  for (const message of verifyTraits(op)) {
    errors.push(new VerificationError(message, op, func));
  }

  if (opDef.verify) {
    const opErrors = opDef.verify(op);
```

The verifier does not know what `ELEMENTWISE` means or what makes a `dot` well-formed. It asks. Chapter 11's registry is the answer, and the verifier is a driver over it.

Finally it re-runs type inference and compares ([`verifier.ts:298`](../../../src/compiler/ir/graph/verifier.ts)):

```ts
    const inferred = opDef.inferResultTypes(operandTypes, op.attributes, op.results.map((r: Value) => r.type));
    if (inferred) {
      for (let i = 0; i < Math.min(inferred.length, op.numResults); i++) {
        const actual = op.getResult(i).type;
        const expected = inferred[i];
```

This is the strongest check in the file. It says: *the type this value claims must be the type this operation's own rule would produce from these operands.* A pass that rewires an operand and forgets to update the result type is caught here, one phase after the mistake — and the comparison uses `shapeCompatible`, Chapter 10's actual-against-expected relation, not `equals`, so that a pass narrowing `?` to `4` is not punished for making the program more specific.

### Traits verify themselves

`verifyTraits` ([`trait_verifier.ts:29`](../../../src/compiler/ir/graph/trait_verifier.ts)) is a second small registry:

```ts
export function verifyTraits(op: Operation): string[] {
  const def = registry.get(op.opName);
  if (def === null) return [];
  const errors: string[] = [];
  for (const trait of def.traits) {
    const verify = _traitVerifiers.get(trait);
    if (!verify) continue;
    for (const message of verify(op)) errors.push(`trait '${trait}': ${message}`);
  }
  return errors;
}
```

Eight traits have verifiers, and reading them is the fastest way to learn what each trait actually promises. `TERMINATOR` ([`trait_verifier.ts:151`](../../../src/compiler/ir/graph/trait_verifier.ts)):

```ts
registerTraitVerifier(OpTrait.TERMINATOR, (op) => {
  const block = op.parentBlock;
  if (block === null) return [];
  if (block.lastOp !== op) return ['a terminator must be the last operation in its block'];
  return [];
});
```

That is the invariant Chapter 8 discovered by accident when `getReturnOp()` returned `null` on a reversed module. It is real, it is enforced, and it is the one place textual order carries meaning.

`VIEW` is the one to read for the flavour of the rest ([`trait_verifier.ts:158`](../../../src/compiler/ir/graph/trait_verifier.ts)):

```ts
registerTraitVerifier(OpTrait.VIEW, (op) => {
  const errors: string[] = [];
  if (op.numOperands !== 1) errors.push(`a view op reads exactly 1 operand, got ${op.numOperands}`);
  if (op.numResults !== 1) errors.push(`a view op produces exactly 1 result, got ${op.numResults}`);
  if (errors.length > 0) return errors;
  const operand = op.getOperand(0).type;
  const result = op.getResult(0).type;
  if (operand instanceof TensorType && result instanceof TensorType && operand.dtype !== result.dtype) {
    errors.push(`a view op cannot change dtype: ${typeToString(operand)} -> ${typeToString(result)}`);
  }
  return errors;
});
```

A view reshapes; it does not reinterpret bytes. Declaring `VIEW` on an operation that changes dtype is now an error rather than a miscompile in whichever pass trusted the trait.

And `IDEMPOTENT` ([`trait_verifier.ts:132`](../../../src/compiler/ir/graph/trait_verifier.ts)) carries its own justification in the message text — `so folding f(x, x) -> x would not preserve types` — which is exactly the right way to write a verifier message: it names the optimization that would break.

## 12.4 Lab — Break it seven ways

```bash
node docs/part2/ch12-valid-ir/labs/01-break-it-seven-ways.mjs
```

The lab takes a valid two-operation module, damages it seven different ways, and hands each to the two checkers a user can reach.

```
the valid module round-trips: true

a value nobody defines
  parser        : rejected -- line 3: use of undefined value '%9'

one name, two definitions
  parser        : rejected -- line 4: value '%2' is defined twice

a dependency cycle
  parser        : rejected -- line 3: 'add' participates in a value dependency cycle

an operation nobody registered
  parser        : accepted
  module.verify(): no complaints

the wrong number of operands
  parser        : accepted
  module.verify(): no complaints

a result type that does not follow
  parser        : accepted
  module.verify(): no complaints

a return that does not match the signature
  parser        : accepted
  module.verify(): m: Return has 2 operands but function declares 1 outputs
```

Line those up against §12.1's four invariants and the division of labour is exact:

| Broken invariant | Caught by |
|---|---|
| 1. Every use has a definition | the parser |
| 1. Each name defined once | the parser |
| 2. No cycles | the parser |
| 3. Operation matches its declaration | **`verifyModule` only** — neither checker here |
| 4. Boundary matches signature | `module.verify()` |

The three that fall through are exactly invariant 3, and they are exactly what `verifyModule` checks: `Unknown op 'frobnicate'`, `'add' expects 2 operands, got 1`, and a result-type mismatch. They are not unchecked by the compiler — they are checked four times per compilation, at the phase boundaries from Chapter 3 — but they are not checked by anything a user can call from the package's public surface.

That is worth stating plainly rather than glossing: **if you build a module by hand and want it fully checked, you must run it through the compiler.** `module.verify()` is a signature check with a name that promises more than it delivers.

**Try this.** Take the "wrong number of operands" case and pass the resulting module through `printModule`. It prints happily — `%2 = add(%0) : tensor<2x2xf32>` — and re-parses. The printer and parser are faithful to the structure, not to the semantics, and nothing between them notices that `add` takes two operands. Then look at the message `verifyModule` produces for the same input in [`tests/compiler/ir/graph/verifier.test.js`](../../../tests/compiler/ir/graph/verifier.test.js), and note it names both the expected and the actual count. Chapter 64 is about why that phrasing matters at 2 a.m.

## 12.5 Errors carry their location

One small thing separates a verifier you can use from one you cannot ([`verifier.ts:12`](../../../src/compiler/ir/graph/verifier.ts)):

```ts
export class VerificationError {
  message: string;
  op: Operation | null;
  func: GraphFunction | null;

  ...

  toString(): string {
    let loc = '';
    if (this.func) loc += `[${this.func.name}] `;
    if (this.op) loc += `op '${this.op.opName}' (id=${this.op.id}): `;
    return loc + this.message;
  }
```

An error is an object holding the offending operation, not a string. So a caller can print it, or navigate to it, or count how many errors one pass introduced. And `verifyModule` **collects rather than throws** — it returns every error it finds, so one run tells you whether a pass broke one thing or forty. A verifier that throws on the first problem makes you fix them one recompile at a time.

Note `id=${this.op.id}` — the global operation counter from Chapter 8. This is the one place it surfaces to a human, and it is the right place: within a single compilation it uniquely identifies an operation, which the printer's `%n` labels do not.

## 12.6 Traps and limits

- **Region isolation is not verified.** Chapter 9's Definition 9.1 says a region only sees values through its block arguments. The scope set at [`verifier.ts:68`](../../../src/compiler/ir/graph/verifier.ts) is collected across the whole function, so an operation *inside* a region that reads a value from *outside* it passes verification. That is a real gap between a documented contract and an enforced one, held instead by [`tests/compiler/ir/graph/region-scope-contract.test.js`](../../../tests/compiler/ir/graph/region-scope-contract.test.js). It is also not an accident: making the scope set per-region would require deciding what to do about the values a fusion region legitimately closes over during construction, and the current answer is to enforce it in the builders.
- **Seven of fifteen traits have no verifier.** Declaring `ASSOCIATIVE` on an operation that is not associative produces no error, and the first symptom is a wrong number after a reassociating rewrite. Chapter 11 §11.8 lists which.
- **Verification is per-level, and the levels do not check each other.** `verify:post` proves the graph is well-formed and `verify:tensor` proves the TIR is; neither proves the TIR *computes the same thing* as the graph. Chapter 6 §6.6's integer-division story is what that gap looks like when it bites. Differential testing (Chapter 65) is the only thing that closes it.
- **Nothing verifies the absence of a cycle across a region boundary.** `detectCycles` runs per block, over operations in that block. A cycle threading out of a region and back in is not something the builders can produce, but it is not something the verifier would catch either.
- **`verify()` and `verifyModule` sharing a name-root is a trap.** They are unrelated functions with a fifteen-fold difference in coverage. When a bug report says "it verified fine", find out which one was called.
- **A module can be invalid between two verifications, and nothing notices.** Verification is a *phase boundary* activity, not an invariant maintained continuously. Every public mutation — `pushOp`, `replaceOperand`, `setAttr`, or reaching into `op.operands` directly (Chapter 9 §9.9) — can leave the module in a state that `verifyModule` would reject, and it stays that way until a verification path happens to run. Three consequences follow. Inside a pass, transient invalidity is *expected* and is the reason §12.1 argues against checking too much. Between passes, invalidity is caught only if the pass manager runs the verifier there — and Chapter 15 §15.4 shows it runs only when a pass reports `CHANGED`, so a pass that mutates and reports `UNCHANGED` is not checked at all. And outside the pipeline entirely — a script that builds IR by hand, a test that patches a module — nothing runs unless you call it. If you have mutated a module and want to know whether it is still valid, the answer is always to call `verifyModule` yourself.

### "Valid" means *structurally well-formed*, and the word will mislead you if you let it

This is worth restating at the end because the chapter's own vocabulary invites the slide. The four invariants of §12.1 are all statements about the data structure: something is defined, something is acyclic, arities agree, types agree. None of them is a statement about what the program *computes*. Concretely, `verifyModule` returning `[]` does not establish any of the following, and each has bitten somewhere in this book:

| Not established by validity | Where it matters |
|---|---|
| that a declared trait is true of the operation | the traps above; Chapter 11 §11.8 — seven traits have no verifier, and `ASSOCIATIVE` is *false* on the operations declaring it |
| that this level computes what the level above computed | the traps above; only differential testing closes it |
| that a numeric result equals the eager one, or is within any tolerance | Chapter 19's `f32` folding, Chapter 20's reassociation |
| that the version counter reflects the edits made | Chapter 9 §9.9 — every mutating *method* notifies, but `op.attributes` and `op.operands` are public, so an edit made around the API is invisible to the counter |
| that a region reads nothing from outside itself | the first trap above |

So a valid module can be a wrong program, and this compiler contains valid modules that are wrong programs. Validity is the property that lets the *rest of the compiler run without tripping over its own data structures* — which is exactly what §12.1 set out to define, and exactly why it is worth checking four times per compilation. Read "invalid IR" as "a pass broke the machine", never as "a pass broke the mathematics".

## 12.7 Read the tests

- [`tests/compiler/ir/graph/verifier.test.js`](../../../tests/compiler/ir/graph/verifier.test.js) — the full checker: scope, cycles, arity, attributes, type inference mismatch. The clearest statement anywhere of what valid IR is.
- [`tests/compiler/ir/graph/trait-verifier.test.js`](../../../tests/compiler/ir/graph/trait-verifier.test.js) — one violating operation per trait verifier, spelled out, plus a check that the traits the registry declares are the traits something verifies.
- [`tests/compiler/ir/graph/region-scope-contract.test.js`](../../../tests/compiler/ir/graph/region-scope-contract.test.js) — the invariant §12.6 says the verifier does not enforce, pinned where it actually is enforced.
- [`tests/compiler/pipeline/`](../../../tests/compiler/pipeline/) — verification as a phase: which boundaries run it, and what happens when it fails.

---

**Next:** [Chapter 13 — IR as text](../ch13-ir-as-text/README.md), the last chapter of Part II, which closes the loop: everything you have been reading in printed form goes back in.
