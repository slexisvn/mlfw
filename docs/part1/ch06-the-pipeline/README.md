# Chapter 6 — The pipeline in one picture

Chapter 3 showed you the pipeline as a map: five representations, a directory for each. This chapter asks the question that map does not answer.

Three of those five are intermediate representations the compiler owns — Graph IR, TIR, LIR. A compiler's job is to turn a graph into machine code. Why not do that in one step?

## 6.1 The problem: one translation, or several?

The direct approach is tempting. You have `dot(%0, %1)` and you want a loop nest that multiplies matrices. Write a function from the first to the second and you are done — no intermediate representations, no verifiers, no printers, no pass infrastructure in between.

For one operation on one target, that works. It starts to strain at the second target, and it fails outright at the first optimization that has to see more than one operation at a time.

To see why, look at the same program at successive levels.

## 6.2 Lab — One program, three levels

```bash
node docs/part1/ch06-the-pipeline/labs/01-one-program-three-levels.mjs
```

The two-layer network, after graph optimization:

```
    %5 = dot(%0, %1) {lhs_contracting = [1], rhs_contracting = [1]} : tensor<2x8xf32>
    %7 = fusion(%5, %2, %6) {fusion_kind = "kElementwise"} : tensor<2x8xf32>
    {
      ^bb(%8: tensor<2x8xf32>, %9: tensor<8xf32>, %10: tensor<2x8xf32>):
      %11 = add(%8, %9) : tensor<2x8xf32>
      %12 = maximum(%11, %10) : tensor<2x8xf32>
      yield(%12)
    }
```

and the same program one level down, in TIR:

```
prim_func Sequential(arg_0, arg_2, arg_4, arg_6, arg_8, ret_10) {
  buf_1 = buffer_map(arg_0, shape=[2,2], dtype=f32)
  buf_3 = buffer_map(arg_2, shape=[8,2], dtype=f32)
  ...
  for ls0_14 in 0..2 {
    for rs0_15 in 0..8 {
      for c0_16 in 0..2 {
        block matmul_1 {
          bind vls0_17 = ls0_14
          bind vrs0_18 = rs0_15
          bind vc0_19 = c0_16
          reads([buf_1[...], buf_3[...]])
          writes([buf_13[...]])
          buf_13[vls0_17, vrs0_18] = (buf_13[vls0_17, vrs0_18] + (buf_1[vls0_17, vc0_19] * buf_3[vrs0_18, vc0_19]))
        }
      }
    }
  }
  for i0_25 in 0..2 {
    for i1_26 in 0..8 {
      block fusion_block_2 {
        bind v0_27 = i0_25
        bind v1_28 = i1_26
        reads([buf_13[...], buf_5[...], buf_12[...]])
        writes([buf_24[...]])
        buf_24[v0_27, v1_28] = max((buf_13[v0_27, v1_28] + buf_5[v1_28]), buf_12[0, 0])
      }
    }
  }
  ...
```

Compare them as *languages*, not as programs.

The graph knows there is a `dot`. It does not know, and cannot say, in what order the multiply-accumulates happen — that is not expressible in it. TIR knows the loop order exactly (`ls0`, `rs0`, `c0`), which memory locations each iteration reads and writes, and that `matmul_1` accumulates into `buf_13`. What TIR no longer knows is that any of this was a matrix multiply. The name `matmul_1` is a label; the *concept* is gone, replaced by three loops and an accumulation.

The lab prints a third version, the generated JavaScript you already met in Chapter 2, and it continues the same slide. There, `buf_13[vls0_17, vrs0_18]` has become `buf_13[(ls0_14 * 8) + rs0_15]`: the two-dimensional index is gone, replaced by arithmetic on a flat array. TIR knew the buffer had a shape; the emitted code knows only offsets.

That trade — gaining detail by losing meaning, at every step — is what a level of a compiler is.

> **Definition 6.1 (Lowering; stated here).** *Lowering* is a translation from a representation to a more detailed one that (i) preserves the program's semantics and (ii) is not invertible: information present in the source representation is not recoverable from the result.

Irreversibility is the point, not a defect. It is why the order of the levels is fixed, and it is why every design decision about *where* a transformation belongs has consequences that cannot be undone later.

## 6.3 Every optimization has a natural level

An optimization is expressible at a given level only if that level can *say* the thing the optimization is about.

| Optimization | Needs to talk about | Natural level | Why not elsewhere |
|---|---|---|---|
| Fold transpose into dot | Operations and their attributes | Graph | TIR has no `transpose` to fold; only index expressions |
| Fuse `add` into `maximum` | Producer/consumer between operations | Graph | After lowering they are two loop nests; merging them means loop fusion, which is harder and needs dependence proofs |
| Choose NCHW vs NHWC | Tensor layout as a property of a value | Graph | Below this, layout has already been baked into index arithmetic |
| Tile a loop for cache | Loop bounds and iteration order | TIR | The graph cannot express "iterate in blocks of 32"; there are no loops in it |
| Bind an axis to GPU threads | Loop nests and hardware axes | TIR | Same reason |
| Reuse one buffer for two tensors | Addresses and lifetimes | TIR/LIR | The graph has values, not addresses |
| Strength-reduce an index expression | Flat integer arithmetic | LIR | Multi-dimensional indices have not been flattened yet |

Read the middle column downward and you have the shape of the pipeline. Each level exists because some family of optimizations needs a vocabulary that neither its neighbour above nor its neighbour below provides.

This also explains why the multi-level structure is not unique to this compiler. TVM has Relax → TensorIR → target code. XLA lowers HLO through backend-specific representations down to LLVM IR or GPU assembly. MLIR generalizes the idea into *progressive lowering* through arbitrarily many dialects. Halide made the earliest version of the argument by separating *algorithm* from *schedule* — the same split you see between the graph and TIR here.

## 6.4 The phase-ordering problem

Once you have many transformations, you must run them in some order — and the order changes the outcome.

You already have the evidence, from Chapter 3's pass log:

```
    pass canonicalize: 10 ops -> 10 ops
    pass constant_fold: 10 ops -> 10 ops
    pass dce: 10 ops -> 7 ops
```

`canonicalize` folded the transposes into the `dot`, leaving them unused. `constant_fold` replaced a broadcast with a constant, leaving that unused. Only then could `dce` remove three operations.

Run `dce` first and it removes nothing: at that moment every operation still has a user. The deletions exist only because two other passes ran and left orphans behind.

That is the small version of a general problem.

> **Definition 6.2 (Phase-ordering problem; classical).** Given a set of program transformations, the *phase-ordering problem* is to choose a sequence of applications that yields the best final program. Transformations are not commutative: applying A then B may enable, disable, or undo the effect of applying B then A.

> **Theorem 6.3 (folklore).** No fixed order of transformations is optimal for all programs.
>
> *Proof sketch.* It suffices to exhibit two programs whose optimal orders differ. Take A = constant folding, B = loop unrolling. For a loop whose trip count becomes constant only after folding, A-then-B unrolls and B-then-A does not. For a loop whose body contains a constant expression exposed only by unrolling, B-then-A folds it and A-then-B does not. No single order handles both. ∎

Compilers respond in one of three ways, and this one uses all three.

**Fix a good order by hand.** The pipeline is a list, written down and defended by tests. That is [`src/compiler/pipeline/graph_pipeline.ts`](../../../src/compiler/pipeline/graph_pipeline.ts).

**Iterate to a fixed point.** Where transformations enable each other, run them repeatedly until nothing changes — [`graph_pipeline.ts:39`](../../../src/compiler/pipeline/graph_pipeline.ts):

```ts
  passes.push(new FixedPointGroup('canonicalize', [
    new CanonicalizePass(),
    new AlgebraicSimplificationPass({ fastMath: config.optimization.fastMath }),
    new ConstantFoldPass(),
    new CSEPass(),
    new DCEPass(),
  ], config.optimization.maxSimplifyIterations));
```

This dissolves ordering *within* the group: any enabling relationship among those five is eventually exploited, whatever order it needs. The three passes from the example above are all members of this group, which is precisely why their relative order is not something anyone has to get right — run `dce` too early and the next iteration cleans up anyway. What the group does not dissolve is ordering *between* groups, and it is bounded by an iteration cap, because a fixed point is not guaranteed to be reached quickly — Chapter 15 proves what can be guaranteed.

**Measure instead of guessing.** For a few expensive, program-dependent choices, the compiler compiles more than one version and times them ([`src/compiler/pipeline/opt_gate.ts`](../../../src/compiler/pipeline/opt_gate.ts)). This is the honest response to Theorem 6.3: when analysis cannot tell you which order wins, run both and look. Part VIII scales this idea into full autotuning.

## 6.5 What each level guarantees

Levels are not only about expressiveness; they are also where correctness is pinned down. Each representation has an invariant set and a verifier that enforces it, and the compiler checks at every boundary — the `verify:pre`, `verify:post`, `verify:tensor`, `verify:lir` phases from Chapter 3.

| Level | What the verifier actually checks | Where |
|---|---|---|
| Graph | Every operand is defined in scope; no dependency cycles; operand and result counts match the operation's declaration; required attributes present; declared traits hold; result types match what the operation's inference rules produce | [`ir/graph/verifier.ts`](../../../src/compiler/ir/graph/verifier.ts) |
| TIR | Root is a `PrimFunc`; no loop or block variable is bound twice; every variable used is bound; buffer loads and stores have a buffer and an index count matching its rank | [`ir/tensor/verifier.ts`](../../../src/compiler/ir/tensor/verifier.ts) |
| LIR | Root is an `LIRFunc`; every loop has a bound variable and an extent; flat loads and stores name a buffer; accumulators declare a name and dtype; every variable used is bound | [`ir/lir/verifier.ts`](../../../src/compiler/ir/lir/verifier.ts) |

That table is deliberately a list of what *is* checked rather than what *should* be. Notice how much thinner the lower two rows are than the top one. The graph verifier knows about types, traits and inference; the TIR verifier checks scoping and ranks but not, for example, that a block's declared read and write regions actually cover what its body touches; the LIR verifier checks structural well-formedness and little else. Index bounds are checked, but by a separate validator that runs during scheduling ([`schedule/validator.ts`](../../../src/compiler/schedule/validator.ts)), not by the TIR verifier.

Even so, the arrangement earns its keep. If the graph verifier passes and the TIR verifier fails, the fault is in lowering — you have bisected the whole compiler in one step. Chapter 64 develops this into a debugging procedure, and is candid about which properties are enforced and which are merely intended.

## 6.6 The cost of levels

Three representations mean three sets of node types, three verifiers, three printers, three pass managers, and three places where a bug can hide. That is real, and the code shows it: [`src/compiler/ir/`](../../../src/compiler/ir/) is 9,034 lines, and much of it is the same idea repeated at three levels of detail.

The alternative is worse in a specific way. A single IR expressive enough for everything — operations *and* loops *and* addresses — forces every pass to handle every level of detail. A fusion pass would have to know what to do when its operands are already loop nests. In practice that is where compilers go to die, and the discipline of "each level says less than the last, and says it more precisely" is what keeps each pass small enough to be correct.

There is one honest caveat, and this codebase supplied a textbook instance of it. Each lowering is a place where the compiler can silently disagree with itself: TIR assumes something about an operation, and a backend implements it differently. Integer division was exactly that. The symbolic layer folded `//` as floor division, TIR's constant folder used truncation, the CPU backend emitted truncation for `//` but floor for `%`, and CUDA, WebAssembly and WGSL emitted truncation for both — four levels, three incompatible definitions, no test failing anywhere, because every index this compiler generates happens to be non-negative and the definitions only diverge on negative operands. Chapter 35 tells the full story, including the fix: one scalar definition in [`src/util/divmod.ts`](../../../src/util/divmod.ts), which `SymInt` and TIR constant folding call directly and which the four backends now match expression for expression — every one of them emits `((a % b) + b) % b` for `%` and reserves truncation for the separate `tdiv`/`tmod` pair.

Levels buy modularity, and they charge for it in interface agreements that nothing checks automatically. The agreement here was never written down, so nothing noticed when three of the four parties disagreed about it.

## 6.7 Traps and limits

- **Level boundaries are design decisions, not laws.** Layout selection sits at the graph level here; some compilers place it lower. Where you put a transformation determines what information it can use, and the choice is rarely obvious in advance.
- **A fixed-point group is not a fixed point.** It is bounded by `maxSimplifyIterations` (default 8). Programs that would keep improving stop improving when the budget runs out, silently.
- **Verification is only as good as the invariants written down.** As §6.5 shows, the lower two verifiers check structure and scoping, not semantics. A pass can produce TIR that verifies cleanly and computes the wrong thing.

## 6.8 Read the tests

- [`tests/compiler/pipeline/`](../../../tests/compiler/pipeline/) — the pipeline itself: which passes run under which configuration, and in what order.
- [`tests/compiler/passes/simplify/`](../../../tests/compiler/passes/simplify/) — the passes in the fixed-point group, including the cases where one enables another.

---

**Next:** [Chapter 7 — Vocabulary](../ch07-vocabulary/README.md), which collects every term used so far, defines it precisely, and points at the code where it lives.
