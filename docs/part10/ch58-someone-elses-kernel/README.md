# Chapter 58 — Calling someone else's kernel

Five chapters of this part have been about generating code. This one is about deciding not to.

There are operations a compiler will not beat by generating loops. A single-precision GEMM on an NVIDIA device has had a decade of hand-tuned assembly poured into it, with per-architecture tile shapes, software pipelining, register-level double buffering and instruction schedules chosen by measurement. Chapter 56's register-blocked matmul reaches a good fraction of that, and cuBLAS reaches more. The interesting engineering question is not how to close the gap; it is **what a compiler has to give up to hand a piece of its program to a library**, and how to build that hand-off so it is one mechanism rather than a special case in five places.

## 58.1 The problem: the fastest kernel is one you did not write

Suppose the compiler decides that a particular matmul should be a cuBLAS call. Four things immediately stop being true.

**There is no source.** Every other kernel this compiler produces is a string the runtime compiles or assembles. This one is a *call*, and what codegen must produce is not text but a description: which library function, at what dimensions, with which of the kernel's arguments in which role.

**It cannot be fused.** [Chapter 22](../../part4/ch22-fusion-why/README.md)'s whole argument is that an elementwise operation following a matmul should be computed in the matmul's epilogue, while the product is still in registers. A library kernel's epilogue belongs to the library. So the `relu` after the matmul must be a separate kernel, and the product must be written to global memory and read back — the exact round trip fusion exists to avoid.

**The decision has to be taken early.** Because of the previous point, the *graph* has to be shaped so the matmul is alone in its kernel. That is a graph pass, running long before codegen, and it has to know at that point that the library will be used.

**The compiler stops being able to reason about it.** The library kernel's numerics, its memory traffic, its launch geometry are all outside the compiler's model. Any cost model that ranks it is guessing.

So the mechanism has to span the whole pipeline — a predicate at configuration time, a pass at graph time, an annotation at TIR time, a special entry at codegen time — and it has to do so without any of the four levels growing a `if (name === 'cublas')` branch.

## 58.2 Intuition: a claim, carried down the pipeline

The design is one idea repeated: **a provider makes a claim, and the claim travels as an attribute.**

At configuration time a *provider* — a small record with a name and a predicate — says whether it is active for this compilation. If it is, it contributes graph passes, which reshape the graph so its precondition holds. After lowering, its *annotate* step walks the TIR module and attaches an attribute to every function it recognises: "this function is mine, and here is what it is". At codegen, the backend pipeline looks for that attribute first, and if it finds one whose name resolves to a registered emitter for this target, it calls that emitter instead of the ordinary backend.

Nothing in the graph passes, the lowering, or the four backends knows the word "cuBLAS". What they know is `FuncAttr.EXTERNAL_CODEGEN`, a name, and a registry.

## 58.3 Theory

> **Definition 58.1 (External kernel).** **(stated here)** An *external kernel* is a `PrimFunc` whose body the compiler does not emit. It is described by an *interface*: a provider name, and enough information to make the call — the problem dimensions, and the position of each operand in the function's argument list.

The second half is the part that is easy to underrate. A compiled kernel's arguments are a list of buffers in `bufferMap` order; a library call needs to know which of them is *A*, which is *B*, and which is *C*. `aIdx`, `bIdx`, `cIdx` are that mapping, and they are why detection has to run on the *lowered* function rather than on the graph — the argument order is a property of lowering.

> **Definition 58.2 (External codegen provider).** **(invariant)** A provider is `{ name, enabled(config, target), graphPasses?(config, target), annotate?(tirModule, split), suppressesEpilogueFusion? }`. Providers live in one registry; `activeExternalCodegenProviders` filters it by `enabled`.

> **Definition 58.3 (External codegen entry).** **(invariant)** An entry is `{ targetKind, runtimeKind, compile(primFunc, target, info) }`, in a second registry keyed by provider name. `BackendPipeline.compile` consults it before the ordinary codegen registry, and uses it only when `entry.targetKind` equals the target's kind.

The split into two registries is deliberate: *when* to use a library is a property of the compilation, and *how* to emit the call is a property of the target. One provider can have entries for several target kinds.

> **Proposition 58.4 (An external kernel is a fusion barrier).** **(stated here)** If a `PrimFunc` is external, no operation may be fused into it, because fusion works by extending the body the compiler emits and there is no such body.
>
> *Consequence.* Any consumer that would have been fused must be a separate kernel, and the external kernel's output must be materialized in memory. The extra traffic is 2 × |output| bytes — one write and one read — that fusion would have avoided.

Proposition 58.4 is why `suppressesEpilogueFusion` exists as a flag on the *provider* rather than a decision at codegen: the graph must be split before the fusion pass runs, not after.

> **Definition 58.5 (Detection).** **(invariant)** `detectPureMatmul` accepts a `PrimFunc` when every `BlockNode` in its body has a name containing `matmul`, one of them has at least two reads and one write, all three buffers are `f32` of rank 2, the three extents it reads off them — *M* and *N* from the output, *K* from the first input — are numbers rather than symbols, and all three buffers appear in the buffer map.

That is a pattern match on a **block name**. Part IV's recurring argument — that asking "what operation is this?" by matching a name is the compiler's oldest and weakest habit, and that the answer belongs in an op-attribute registry — applies here in full, and §58.7 says so.

## 58.4 In mlfw: 87 lines and two registries

[`src/compiler/pipeline/external_codegen.ts`](../../../src/compiler/pipeline/external_codegen.ts) is the whole provider mechanism, and it is short enough to read in one sitting.

### The registry and the filter

```ts
const _providers = new Map<string, ExternalCodegenProvider>();

export function activeExternalCodegenProviders(config: CompilerConfig, target: CompileTarget): ExternalCodegenProvider[] {
  const active: ExternalCodegenProvider[] = [];
  for (const provider of _providers.values()) {
    if (provider.enabled(config, target)) active.push(provider);
  }
  return active;
}
```

([`external_codegen.ts:19`](../../../src/compiler/pipeline/external_codegen.ts) and [`:29`](../../../src/compiler/pipeline/external_codegen.ts).) The compiler calls this in two places: once when building the graph pipeline, to collect `graphPasses`, and once immediately after lowering, to run `annotate` ([`compiler.ts:341`](../../../src/compiler/pipeline/compiler.ts)):

```ts
        name: 'lowering',
        run: (ctx: CompileContext) => {
          const cfg = ctx.compiler.config;
          ctx.tirModule = ctx.compiler._lowerAll(ctx.working, ctx.trace, ctx.errors, ctx.failed, ctx.resilient);
          for (const provider of activeExternalCodegenProviders(cfg, cfg.target)) {
            if (provider.annotate) provider.annotate(ctx.tirModule, ctx.split);
          }
        },
```

### The one provider

```ts
registerExternalCodegenProvider({
  name: CUBLAS_PROVIDER,
  suppressesEpilogueFusion: true,
  enabled: (config) => config.matmulBackend === CUBLAS_PROVIDER,
  graphPasses: () => [new CublasRewritePass()],
  annotate: (tirModule, split) => {
    const fromSplit = split && split.cublasInfos ? split.cublasInfos : null;
    for (const primFunc of tirModule) {
      const info = fromSplit ? fromSplit.get(primFunc.name) : detectPureMatmul(primFunc);
      if (info) primFunc.setAttr(FuncAttr.EXTERNAL_CODEGEN, { name: CUBLAS_PROVIDER, info });
    }
  },
});
```

([`external_codegen.ts:75`](../../../src/compiler/pipeline/external_codegen.ts).) Note the `fromSplit` branch: when the graph was partitioned into a multi-kernel plan, the partitioner already recorded which sub-functions are matmuls and with what dimensions, so the annotate step trusts that rather than re-detecting. When there was no split, it detects.

### The emitter

The cuBLAS entry is the shortest codegen in the compiler ([`codegen_registry.ts:47`](../../../src/backend/codegen_registry.ts)):

```ts
registerExternalCodegen(CUBLAS_PROVIDER, {
  targetKind: TargetKind.CUDA,
  runtimeKind: 'cuda',
  compile(primFunc: PrimFunc, target: TargetFeatures, info: ExternalKernelInfo): CodegenOutput {
    return { source: '', metadata: { kind: 'cuda', cublas: info, outputIndices: [info.cIdx] } };
  },
});
```

An empty source and a descriptor. Everything a normal backend spends hundreds of lines on — declarations, loops, addresses, launch geometry — is the library's problem.

### The hand-off

`BackendPipeline.compile` ([`pipeline.ts:36`](../../../src/backend/pipeline.ts)) is where the attribute is read:

```ts
    const external = primFunc.getAttr<ExternalCodegenAttr>(FuncAttr.EXTERNAL_CODEGEN);
    if (external) {
      const entry = getExternalCodegen(external.name);
      if (entry && entry.targetKind === this.target.kind) {
        const { source, metadata } = entry.compile(primFunc, this.target, external.info);
        return new CompiledKernel(primFunc.name, source, this.target, metadata);
      }
    }
    const entry = ((this.context && this.context.getCodegenEntry(this.target.kind)) || getCodegenEntry(this.target.kind)) as CodegenEntry | null;
```

A handful of lines, and a fall-through. The fall-through is what makes the mechanism composable — a provider that is not registered for this target simply does not apply — and it is also the last of §58.7's three findings, because a fall-through and a mistake look the same from outside.

## 58.5 Lab — a kernel with no source

```bash
node docs/part10/ch58-someone-elses-kernel/labs/01-a-kernel-with-no-source.mjs
```

The same graph, two values of one option:

```
  matmulBackend  kernel     source chars  metadata
  native         mm                 2243  {"kind":"cuda","blockDim":[8,16,1],"gridDim":[2,1,1],"sharedMemBytes":3072,"params":["buf_1","buf_3","buf_5"],"outputInd
  cublas         mm                    0  {"kind":"cuda","cublas":{"M":64,"N":48,"K":32,"aIdx":0,"bIdx":1,"cIdx":2},"outputIndices":[2]}
```

**Zero characters.** The compiled kernel is a descriptor, and it is an otherwise entirely ordinary `CompiledKernel` — same class, same registry, same runtime module — so nothing downstream needs to know that this one is different until the moment it is launched.

Definition 58.5, exercised:

```
  a plain matmul         M=8 N=6 K=4  operands 0,1 -> 2
  matmul then relu       not a pure matmul
  a batched matmul       not a pure matmul
  an elementwise chain   not a pure matmul
```

The `relu` case is Proposition 58.4 arriving as a *detection* failure: the function has a block whose name does not contain `matmul`, so `detectPureMatmul` refuses it — which is correct, because there is no way to hand that whole function to a GEMM. What turns it into a case that *can* be handed over is upstream: the provider contributes a graph rewrite, and the partitioner ([`cublas_split.ts:291`](../../../src/compiler/passes/partition/cublas_split.ts)) splits the graph so the matmul is alone in its own sub-function, recording the descriptor as it goes.

The batched case is refused on rank. That is a real limit rather than an oversight, and the reason is that a batched GEMM is a different library entry point with a different descriptor.

Then the two registries:

```
  default                active providers: (none)   enabled(cublas)=false
  matmulBackend: cublas  active providers: cublas   enabled(cublas)=true
```

And then the fall-through, made visible by re-registering the emitter against the wrong target kind:

```
  the graph still compiles: mm, 633 characters of cuda
  the attribute is still on the function, and it was ignored.
```

The attribute was set, the lookup succeeded, the target check failed, and the pipeline generated an ordinary CUDA kernel — correct, and not what was asked for, with no warning anywhere.

The lab then registers a provider of its own, in about twenty lines, and routes every pure matmul in a compilation through it. That is the mechanism working as designed: no core file changed.

Finally, the cost:

```
  native   1 kernel(s): mmr(2257 chars)
  cublas   2 kernel(s): mmr_p0(0 chars), mmr_p1(327 chars)
```

`matmul` then `relu`. Native: one kernel, the `relu` in the epilogue, the product never leaving registers. cuBLAS: two kernels, and a 64×48 `f32` product — 12,288 bytes — written to global memory and read straight back. **That round trip is Proposition 58.4's consequence, measured in kernels rather than in nanoseconds**, and it is the reason a library call is a graph-level decision: at codegen time the split can no longer be made.

**Try this.** Put the `relu` *before* the matmul instead of after and count the kernels again. A prologue is not an epilogue: the input can be materialized by an earlier kernel without any loss the fusion pass would have recovered, so the boundary costs nothing there. The asymmetry is why `suppressesEpilogueFusion` names the epilogue specifically.

## 58.6 What this part established

Six chapters, one pipeline stage. It is worth naming what the rest of the book can now assume.

**[Definition 53.7](../ch53-lir-the-third-ir/README.md)'s backend contract now has five implementations**, and this chapter's is the one that tests where the contract was drawn: an external kernel is an ordinary `CompiledKernel` whose source happens to be empty, and no consumer downstream needed a special case for it. Everything Part XI's runtime does is against those four fields — it compiles or assembles the source, allocates against the metadata's buffer offsets or bindings, and launches with the metadata's geometry.

**The metadata table in §53.4 is the row-by-row answer** to what each backend could not say in its text: a page count and an import list for WASM, a launch geometry and scratch for CUDA, a binding table for WebGPU, a library descriptor here.

**Numerical equivalence is a per-backend property, and this part named six places where it is not N0.** The CPU backend accumulates reductions in `f64` (Chapter 54 §54.8). The WASM backend reassociates a vectorised reduction (Chapter 55 §55.6). The CUDA backend's block reduction would reassociate too, if it were reachable (Chapter 56 §56.6). The WebGPU backend clamps infinite literals (Chapter 57 §57.5). `remainder` is floor-modulo on CPU and truncating on the other three (Chapter 53 §53.6). And `erf`, `erfc`, `lgamma` and `gamma` come from a shared approximation on three backends and from the device math library on CUDA (Chapter 54 §54.3).

They are not the same kind of difference, and a differential test has to know which is which. Two are **reassociations**, and their error grows with the length of the reduction, so no fixed tolerance covers them — one of the two is currently unreachable. One is a **different approximation**, a fixed error of order 10⁻⁷ that a tolerance can absorb. One is a **clamped value**, exact on finite inputs and wrong on infinite ones, which a tolerance cannot see at all. And one is a **sign convention**, where the two answers differ by the divisor and calling it a tolerance question would be a category error. Part XII's tolerances have to be derived from that list rather than guessed at, and two of the six need a targeted test rather than a tolerance.

## 58.7 Traps and limits

### Three ways a generic interface turns out not to be generic

**The generic interface has one operation's signature in its type.** `ExternalKernelInfo` is `{ M, N, K, transB?, aIdx, bIdx, cIdx }` ([`external_codegen.ts:7`](../../../src/compiler/pipeline/external_codegen.ts)). Every provider, present and future, must describe its kernel in GEMM's vocabulary. A convolution has no `M`, `N`, `K`; an attention kernel has a mask and a scale; an RNN has a sequence length and a layer count. The mechanism around the type is genuinely generic — two registries, a predicate, an attribute — and the payload is not, which means the second provider will either widen the type for everyone or smuggle its description through `info` as an untyped bag. Making `info` a provider-owned opaque type, with the provider's own emitter the only thing that reads it, is a one-line change to the type and no change to any call site.

**Detection is a pattern match on a block name.** `detectPureMatmul` opens with `if (!b.name.includes('matmul')) return null` ([`external_codegen.ts:57`](../../../src/compiler/pipeline/external_codegen.ts)). Block names are set by lowering rules for readability; nothing declares them part of an interface. A rule that renamed its block `gemm_block` would silently disable cuBLAS for every model. The op-attribute registry this compiler already has — the one `launchBoundaryClass` and `hasLibraryOp` read — already answers that question for the fusion and scheduling layers, and routing detection through it would make `TargetFeatures.libraryClasses` (which already lists `matmul` and `conv`) the single source of what a target can offload.

**A failed hand-off is indistinguishable from no hand-off.** `BackendPipeline.compile` falls through when the entry is missing or its `targetKind` does not match ([`pipeline.ts:40`](../../../src/backend/pipeline.ts)), producing a correct kernel and no signal. §58.5 executes it. A user who asks for `matmulBackend: 'cublas'` on a target with no cuBLAS entry gets native kernels and is never told. One `trace.warn` naming the provider and the target kind would close it.

### The rest

- **`unregisterCodegen` does not exist.** `registerExternalCodegen`/`unregisterExternalCodegen` and `registerExternalCodegenProvider`/`unregisterExternalCodegenProvider` are symmetric pairs; `registerCodegen` ([`codegen_registry.ts:27`](../../../src/backend/codegen_registry.ts)) has no inverse, so an ordinary backend can be replaced but never removed. Both registries are module-global mutable maps, which makes them convenient to extend and impossible to scope to one compilation — a `CompilerContext` override exists for the ordinary registry ([`pipeline.ts:45`](../../../src/backend/pipeline.ts)) and not for the external one.
- **`suppressesEpilogueFusion` is declared and read elsewhere.** The flag is part of Definition 58.2 and is consumed by the fusion configuration rather than by anything in this file, so the coupling between "this provider is active" and "do not fuse epilogues" is by convention across two modules. A provider that forgets the flag gets a graph whose fused matmuls are then not detected — a silent loss, not an error.
- **The annotate step trusts the partitioner without checking it.** When `split.cublasInfos` is present, `annotate` uses the recorded info for a function name and does not re-run detection ([`external_codegen.ts:83`](../../../src/compiler/pipeline/external_codegen.ts)). That is the right performance choice and it means the two sources of `ExternalKernelInfo` — the partitioner's and the detector's — are never checked against each other.
- **Nothing decides whether the library is actually faster.** The provider's `enabled` predicate is a user configuration option, not a measurement. Chapter 47's optimization gate compiles candidate configurations and keeps the winner; `matmulBackend` is not among the knobs it sweeps, so "cuBLAS or native" is a decision the user makes blind, on a per-compilation rather than per-kernel basis — and at small *M*, *N*, *K* the launch overhead of a library call and the lost epilogue fusion can easily lose.
- **cuDNN is not reachable through this mechanism.** The runtime binds cuDNN for RNNs and convolutions on the eager path, and no provider routes a compiled graph to it. That is a scope statement rather than a defect, and it is the clearest evidence for the first finding: the second provider anyone writes is a convolution, and `{ M, N, K }` cannot describe one.

## 58.8 Read the tests

- [`tests/compiler/pipeline/external-codegen.test.js`](../../../tests/compiler/pipeline/external-codegen.test.js) — the provider registry, `enabled` filtering, the annotate step attaching the attribute, and `detectPureMatmul`'s accept and reject cases including the rank, dtype and static-extent clauses.
- [`tests/backend/pipeline.test.js`](../../../tests/backend/pipeline.test.js) — `BackendPipeline.compile`'s dispatch: the external path, the target-kind check, and the fall-through.
- [`tests/backend/cuda/compile.test.js`](../../../tests/backend/cuda/compile.test.js) — a whole graph compiled with `matmulBackend: 'cublas'`, and the split that the provider's graph pass produces.

---

**Part X ends here.** [Part XI](../../part11/README.md) takes the `CompiledKernel` this part produces and makes it a framework: a runtime that loads and launches it, a dispatcher that routes an eager call to it, a tracer that produced the graph in the first place, and a training loop that runs all of it end to end.
