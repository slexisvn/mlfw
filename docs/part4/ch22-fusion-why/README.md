# Chapter 22 — Fusion I: why it is the single most valuable optimization

Chapter 4 measured an eager operation and found `T(n) = α + βn`: a fixed cost plus a per-element cost. Chapter 21 just multiplied the number of operations in the graph by five, and by nine for a `layer_norm`. This chapter is where that becomes affordable, and where the largest number in this book gets explained.

The claim is narrow and checkable. For the elementwise operations that make up most of a decomposed network, `β` is almost entirely **memory traffic**, and traffic is the one cost a compiler can remove without doing anything clever about the arithmetic.

## 22.1 The problem: every operation is a round trip

Consider four elementwise operations on tensors of `n` elements:

```
t1 = x + y
t2 = t1 * x
t3 = t2 - y
r  = t3 + x
```

Executed one at a time, each is a loop that reads its operands from memory and writes its result to memory. Four loops, each touching three tensors: **twelve tensor-sized transfers**.

Now look at what the program actually needs. It consumes `x` and `y` and produces `r`; three transfers. The other nine are avoidable, and they are avoidable for two different reasons. Six of them are `t1`, `t2` and `t3` — values that exist only to be handed to the next operation, so the compiler stores each one and immediately loads it back. The remaining three are `x` and `y` themselves: `x` is read by three of the four loops and `y` by two, and each of those loops fetches it from memory again. Keep that split in mind; §22.3 is the theorem that counts both kinds, and stating only the first is the easy way to get it wrong.

If instead you write one loop:

```
for i in 0..n:
    r[i] = ((x[i] + y[i]) * x[i] - y[i]) + x[i]
```

the intermediates never reach memory at all. They live in registers for the three instructions between their production and their last use. **Three transfers instead of twelve**, for exactly the same arithmetic.

That is fusion, and the reason it dominates is arithmetic intensity. Chapter 4 defined it as FLOPs per byte moved. The unfused chain does 4 FLOPs per element and moves 48 bytes per element (12 transfers × 4 bytes); intensity `1/12`. The fused loop does 4 FLOPs per element and moves 12 bytes; intensity `1/3`. Both are far below the machine's balance point, so both are memory-bound, and by Chapter 4's roofline theorem the runtime of a memory-bound kernel is proportional to the bytes it moves. Cut the bytes by four and you cut the time by up to four.

## 22.2 Intuition: the loops were always the same loop

Here is the geometric way to see it. An elementwise operation over an `n`-element tensor is a loop over `0..n`. Two elementwise operations of the same shape are two loops over `0..n`. Two loops over the same range, where the second reads only what the first wrote at the same index, can be merged into one loop — this is *loop fusion*, and it is one of the oldest transformations in compilers.

What makes it easy here and hard in C is that the graph IR has already told you the loop bounds and the dependencies. In C you have to prove that `b[i]` in the second loop is the same memory the first loop wrote, and that no aliasing pointer sits between them. In a tensor graph the edge from producer to consumer *is* the proof: SSA (Chapter 8) says the value has one producer, and the type says its shape.

So fusion at the graph level is not "prove these loops can merge". It is "decide which merges are worth making", which is a cost question, plus one legality question that turns out to be about cycles rather than about memory. Chapter 23 is the legality; Chapter 24 is the deciding; this chapter is the cost model they both use.

## 22.3 Theory

> **Definition 22.1 (Memory traffic of a kernel).** The *traffic* of a kernel is the total bytes of its input tensors plus the total bytes of its output tensors, counting each distinct tensor once.

This is a model, and it is worth being explicit about what it assumes: that every input is read exactly once from memory and every output written exactly once, with no reuse across kernels. For elementwise kernels over tensors larger than cache, both assumptions hold well. For small tensors they do not — the second kernel finds its input still in cache — and §22.5 measures exactly where the model starts to be right.

> **Definition 22.2 (Fusion of a group).** Let `G` be a set of operations. Its *inputs* are the values used by an operation in `G` and produced outside it; its *outputs* are the values produced in `G` and used outside it, or returned. Fusing `G` replaces its operations with one kernel whose traffic is the bytes of its inputs plus the bytes of its outputs.

The accounting is easier to get right if you stop thinking about edges and think about *touches*. Before fusion, a value costs a transfer every time an operation in the group touches it — once for the operation that produces it, once for each operation that reads it. After fusion there is one kernel, so it costs a transfer only if it crosses the boundary. Write `writes_G(w) ∈ {0,1}` for whether an operation in `G` produces `w`, and `reads_G(w)` for how many operations in `G` take `w` as an operand.

> **Theorem 22.3 (What fusion removes, stated here).** Let `G` be fused, and let `∂G` be its inputs and outputs in the sense of Definition 22.2. For every value `w` touched by an operation in `G`,
>
> `traffic_unfused(G) − traffic_fused(G) = Σ_w (writes_G(w) + reads_G(w) − [w ∈ ∂G]) · bytes(w)`
>
> where `[w ∈ ∂G]` is 1 if `w` crosses the boundary and 0 otherwise. Every term is non-negative.

*Proof sketch.* Before fusion, every operation contributes the bytes of its operands plus the bytes of its results, so `w` is charged exactly `writes_G(w) + reads_G(w)` times. After fusion, Definition 22.2 charges the group once for each distinct input and once for each distinct output, so `w` is charged `[w ∈ ∂G]` times. Subtracting gives the sum. Non-negativity: a boundary value is either an input, hence read by at least one operation in `G`, or an output, hence produced by one — either way its bracket is at most `writes_G(w) + reads_G(w)`. ∎

Two special cases carry almost all the intuition, and it is worth naming them because it is easy to state only the first and think you are done.

- **An internalized value** — produced in `G`, read once in `G`, used nowhere else — contributes `1 + 1 − 0 = 2`: the write and the read that fusion deletes. This is the round trip everyone means when they say fusion removes an intermediate.
- **A repeated external input** — read by `k` operations in `G`, produced outside — contributes `0 + k − 1 = k − 1`: the fused kernel loads it once where `k` loops each loaded it separately.

The second case is the one that gets dropped, and dropping it understates the saving badly on exactly the graphs fusion is for. §22.5's chain reads `x` in three of its four operations and `y` in two, so three of the nine tensor round trips it removes — a third of the total — are of the second kind. The theorem also credits a value read *twice* inside the group with `1 + 2 = 3` transfers saved, which is right for traffic and is precisely where the implementation disagrees: it charges that value as shared-memory pressure instead and may refuse the fusion outright. That is the distinction Chapter 24 will pay 2.4× for.

> **Corollary 22.4 (Chains are the ideal case).** For a chain of `k` elementwise binary operations over `n`-element tensors of `b` bytes each, where every intermediate is used once, unfused traffic is `3knb` — two reads and one write per operation. Fused traffic is `(i + 1)nb`, where `i` is the number of *distinct* external inputs, so the ratio is `3k / (i + 1)`. In the case §22.5 measures, where the whole chain is written over the same two tensors, `i = 2` and the ratio is exactly `k`.

Which gives the headline: **fusing a chain of `k` operations over a fixed set of inputs reduces memory traffic by a factor of `k`**, and for memory-bound kernels the runtime follows. A chain that pulls in a fresh tensor at every step keeps those inputs and saves less.

The cost model in the compiler computes Theorem 22.3's difference exactly, because it charges every operation for its own operands and its own results ([`fusion_cost.ts:235`](../../../src/compiler/passes/fusion/fusion_cost.ts)):

```ts
      memorySaved: totalBytes - fusedBytes,
      launchSaved: (group.size - 1) * this.launchOverheadUs,
```

`totalBytes` is the sum over the group's operations of their own operand and result bytes; `fusedBytes` is Definition 22.2's inputs plus outputs. The second line is the `α` from Chapter 4: fusing `k` operations into one removes `k − 1` kernel launches.

## 22.4 In mlfw: what the cost model actually counts

[`passes/fusion/fusion_cost.ts`](../../../src/compiler/passes/fusion/fusion_cost.ts). The per-operation traffic is a straight sum over the type system ([`fusion_cost.ts:106`](../../../src/compiler/passes/fusion/fusion_cost.ts)):

```ts
  estimateBytes(op: Operation): number {
    let total = 0;
    for (let i = 0; i < op.numOperands; i++) {
      const t = op.getOperand(i).type;
      if (t instanceof TensorType) {
        const bytes = t.sizeInBytes();
        if (bytes !== DYNAMIC) total += bytes as number;
      }
    }
```

Note `if (bytes !== DYNAMIC)`. A tensor with a symbolic dimension (Chapter 10) contributes **zero** to the traffic estimate, because the compiler does not know how big it is. A fully dynamic graph therefore looks free to this model, and every fusion decision on it is made on the launch-saving term alone. That is a real limitation with a real cause and no warning attached to it.

The group-level estimate ([`fusion_cost.ts:125`](../../../src/compiler/passes/fusion/fusion_cost.ts)) computes four more quantities beyond traffic: recomputation cost for values used more than once internally, peak live values as a proxy for register pressure, shared-memory bytes for internally reused values, and a parallelism-loss term when a reduction and an elementwise operation of different sizes end up together. The decision procedure ([`fusion_cost.ts:244`](../../../src/compiler/passes/fusion/fusion_cost.ts)) is a series of vetoes, the last of which is the only one that mentions benefit at all ([`fusion_cost.ts:270`](../../../src/compiler/passes/fusion/fusion_cost.ts)):

```ts
    if (cost.memorySaved <= 0 && cost.launchSaved <= 0) {
      return { fuse: false, reason: 'no memory or launch benefit', cost };
    }
```

and when it passes, the reason string is the theorem's two terms, which is why the `explain` events in Chapter 18 read *saves 37748736 bytes, 15us launch*.

## 22.5 Lab — The model, and then the stopwatch

```bash
node docs/part4/ch22-fusion-why/labs/01-the-traffic-model.mjs
```

The program is Corollary 22.4's ideal case: four binary elementwise operations, every intermediate used once.

```js
class Chain extends Module {
  forward(x, y) { return x.add(y).mul(x).sub(y).add(x); }
}
```

The compiler fuses all four into one region:

```
module @Chain {
  func @Chain(%0: tensor<1048576xf32>, %1: tensor<1048576xf32>) -> (tensor<1048576xf32>) {
    %2 = fusion(%0, %1) {fusion_kind = "kElementwise"} : tensor<1048576xf32>
    {
      ^bb(%3: tensor<1048576xf32>, %4: tensor<1048576xf32>):
      %5 = add(%3, %4) : tensor<1048576xf32>
      %6 = mul(%5, %3) : tensor<1048576xf32>
      %7 = sub(%6, %4) : tensor<1048576xf32>
      %8 = add(%7, %3) : tensor<1048576xf32>
      yield(%8)
    }
    return(%2)
  }
}
```

Two operands in, one result out, four operations inside, three intermediates that never leave the region. Definition 22.2, in IR.

Counting by hand: twelve round trips become three, so the model predicts nine tensors of traffic removed per call, independent of size. Theorem 22.3 gets there in two kinds of term — six transfers from the three internalized intermediates, three more from the repeated reads of `x` and `y` — and the `traffic saved` column below is the compiler's own arithmetic agreeing to the byte. Count only the intermediates and you would predict six, which is what makes the second kind worth stating.

Now the measurement, sweeping the tensor from 4 KiB to 16 MiB:

```
elements   tensor    unfused   fused    speedup   traffic saved
    1024       4 KiB    0.013    0.007     1.78x       0.04 MiB
   16384      64 KiB    0.125    0.051     2.45x       0.56 MiB
   65536     256 KiB    0.434    0.171     2.54x       2.25 MiB
  262144    1024 KiB    1.348    0.529     2.55x       9.00 MiB
 1048576    4096 KiB    5.315    2.080     2.56x      36.00 MiB
 4194304   16384 KiB   21.079    8.385     2.51x     144.00 MiB
```

Three things to take from this table, and the third is the most important.

**The speedup is real and large.** Two and a half times, for a transformation that changed no arithmetic whatsoever. Nothing else in Part IV comes close: Chapter 19's three passes removed operations that were doing nothing, and this one removed nothing at all — it only changed where the numbers live between operations.

**The speedup is flat once the tensors leave cache.** From 64 KiB to 16 MiB — a 256-fold span — the ratio sits between 2.45 and 2.56 on the machine that produced this table, and stays in a similarly narrow band on others. That flatness is the model being right: traffic saved scales linearly with `n`, traffic total scales linearly with `n`, so the ratio is a constant. A speedup that climbed or fell steadily with size would mean the model had missed the dominant term.

**And it is 2.55, not the 4.0 the traffic ratio predicts.** The model says twelve transfers become three. The clock says 2.55. The gap is the model's assumption: `T ∝ bytes` holds for a kernel that is purely memory-bound, and these kernels also execute four arithmetic operations and a loop per element, which the fused version still does. Traffic went down 4×; the part of the runtime that was traffic went down 4×; the rest did not. Chapter 4's roofline says the same thing in one sentence: you can only remove the memory-bound part of a memory-bound kernel.

The small end is the other boundary. At 4 KiB the two tensors fit in L1, so the "unfused" version is not really doing twelve trips to memory — it is doing twelve trips to cache — and the remaining 1.78× is mostly the four kernel launches becoming one, which is Chapter 4's `α`. The model's assumption fails exactly where you would expect it to, and the failure is in the direction of over-predicting the benefit.

**Try this.** Change the chain to `x.add(y).exp().log().add(x)` — same length, two transcendental operations. Predict what happens to the speedup before running, using Chapter 4's Theorem 4.4.

## 22.6 What this buys the rest of the book

Three consequences worth carrying forward.

**Decomposition becomes free.** Chapter 21 turned one `softmax` into nine operations and the pipeline turned it back into one kernel. The reason that is not a disaster is this chapter: the nine operations were elementwise and reduction operations over the same shape, so their intermediates were internalizable, so fusing them cost nothing that the original `softmax` kernel would not also have paid.

**The graph's operation count stops being a proxy for cost.** After fusion, a graph of six operations may be one kernel or six; the number that matters is the number of *regions* and what is inside them. Every measurement from here on counts kernels, not operations.

**And the cost model becomes load-bearing.** Everything above assumed the compiler fuses when fusing helps. It fuses when its cost model says fusing helps, and Chapters 23 and 24 are about the two ways that goes wrong: a merge that is illegal, and a merge the model declines for a reason that does not apply to your target.

## 22.7 Traps and limits

- **Dynamic shapes contribute zero traffic.** `sizeInBytes()` returns `DYNAMIC` for a symbolic dimension and the estimator skips it ([`fusion_cost.ts:112`](../../../src/compiler/passes/fusion/fusion_cost.ts)). A graph compiled with `dynamicShapes` therefore makes every fusion decision on `launchSaved` alone. It still fuses — the launch term is positive for any group of two — but the memory argument that justifies fusion is absent from the decision that performs it.
- **The traffic model assumes no cache reuse between kernels.** §22.5's small end shows where that breaks. The model has no cache-size parameter, so it cannot distinguish "these two kernels would have hit in L2" from "these two kernels each stream from DRAM".
- **A value used twice inside a group is charged as if it must be stored.** Theorem 22.3 counts it as a saving; the implementation counts it as shared-memory pressure ([`fusion_cost.ts:203`](../../../src/compiler/passes/fusion/fusion_cost.ts)) and may refuse the fusion outright. Chapter 24 measures what that costs on a CPU — where there is no shared memory at all.
- **`estimateFLOPs` is an element count, not a FLOP count.** Without a `getFlops` in the registry, the estimate is the number of output elements ([`fusion_cost.ts:85`](../../../src/compiler/passes/fusion/fusion_cost.ts)) — one FLOP per element, whether the operation is an add or an `exp`. Chapter 4 measured transcendentals at roughly an order of magnitude more, so the recomputation term is systematically wrong for exactly the operations where recomputation is expensive.
- **`memoryBandwidthGBs` and `computeTFLOPs` are configured and unused in the decision.** They are constructor parameters on the cost model ([`fusion_cost.ts:57`](../../../src/compiler/passes/fusion/fusion_cost.ts)) and no branch of `shouldFuse` reads them. The decision is made on raw byte counts and a launch-overhead constant, so it is a *relative* judgement that cannot be calibrated against a real machine's balance point.

## 22.8 Read the tests

- [`tests/compiler/passes/fusion/`](../../../tests/compiler/passes/fusion/) — the cost model's individual terms, including the byte accounting Theorem 22.3 describes.
- [`tests/e2e/`](../../../tests/e2e/) — fused and unfused compilations of the same model compared numerically, which is what makes "fusion changed no arithmetic" a checked claim rather than a stated one.

---

**Next:** [Chapter 23 — Fusion II: legality](../ch23-fusion-legality/README.md), which asks the one question this chapter assumed away: when is merging two operations into one kernel not merely unprofitable, but impossible?
