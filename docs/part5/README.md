# Part V — Automatic differentiation

Parts III and IV rewrote programs into faster programs computing the same thing. This part writes a program the user did not: given a forward graph, it constructs the graph that computes the gradient.

It sits here, between graph optimization and lowering, for one reason. Differentiation is a graph-to-graph transformation and the only level at which the chain rule is expressible as a rewrite — once a program is loops and buffers (Part VI) the dataflow the chain rule needs has been flattened away. So the backward graph is built at the top, and then falls through every pass you have just read about: verified by Chapter 15, canonicalized by Chapter 17, folded by Chapter 19, fused by Chapter 24. That is why the rules in Chapter 28 can afford to be wasteful, and §31.6 measures exactly how much waste the pipeline absorbs.

| Chapter | Title | The question it answers |
|---|---|---|
| [27](ch27-differentiating-programs/README.md) | Differentiating programs | Why does a gradient of a million parameters cost the same as one forward pass, and what does that cost instead? |
| [28](ch28-writing-a-vjp-rule/README.md) | Writing a VJP rule | What is the derivative of one operation when it has to be an object, and what must it be handed to be writable at all? |
| [29](ch29-building-the-backward-graph/README.md) | Building the backward graph | What does the driver around 67 independent rules have to do that no rule can? |
| [30](ch30-memory-for-recomputation/README.md) | Trading memory for recomputation | The backward pass needs the forward pass's values. Which ones do you keep, and what does keeping fewer cost? |
| [31](ch31-differentiating-control-flow/README.md) | Differentiating control flow | What is the derivative of an operation whose body is a program — and of one that has no derivative? |

## The argument in one paragraph

The chain rule is a product of Jacobians, and matrix products associate two ways. That is the entire difference between the two modes of automatic differentiation: forward mode costs one sweep per input, reverse mode one sweep per output, and a loss has one output and a million inputs (Chapter 27).

Reverse mode's sweep applies one linear map per operation, so the derivative of each operation is a registered function that emits a subgraph rather than computing a number. There are 67 of them. Some read the operation's operands and some its results — a distinction that costs nothing to write and decides everything about memory (Chapter 28).

Around those rules sits a driver doing the three things no rule can: prune the operations that cannot reach the output, sum the contributions arriving at a value the forward pass forked, and answer every request for a forward value either from a saved argument or by rebuilding it. The result is packaged either as two functions or as one. Both produce identical numbers, and the joint form was 15% *slower* until it stopped running its kernel twice (Chapter 29).

Which values are saved is a policy, consulted once per operation. The shipped one recomputes elementwise work and keeps contractions, and it cannot help with depth: there the saved set grows linearly, and the √n checkpointing that would fix it is implemented, tested, and unreachable from the public API (Chapter 30).

Two operations are not leaves at all. A `scan`'s backward is emitted by unrolling the loop at compile time, into a graph 76 operations per timestep long; an `if`'s backward runs both branches. And three other operations have no derivative, distinguished by whether the compiler returns a zero, declares a barrier, or refuses (Chapter 31).

## What Part V establishes for later parts

- **The backward graph is an ordinary graph.** Everything Parts III, IV and VI do applies to it unchanged, and §31.6 shows the simplification passes removing a third of what the rules emit.
- **The saved set** (Definition 29.4) as the thing Chapter 49's liveness analysis and Chapter 50's allocator are really planning for: activations held across the forward-backward boundary are the dominant term in training memory.
- **Rematerialization** (Definition 30.1) as the graph-level version of the same trade Chapter 26's rematerialization pass makes at the buffer level — and the two are separate implementations of one idea.
- **The joint graph** (Definition 29.5) as the shape Chapter 63's end-to-end training loop actually wants, and the two-call API that currently prevents it.
- **The gradient barrier** (Definition 31.3) as the pattern for every later "declare it or fail" decision: a claim the compiler cannot verify, made explicit rather than inferred from silence.

## Labs

```bash
npm run build   # once, if you have not already

node docs/part5/ch27-differentiating-programs/labs/01-one-pass-for-every-input.mjs
node docs/part5/ch27-differentiating-programs/labs/02-the-jacobian-both-ways.mjs
node docs/part5/ch28-writing-a-vjp-rule/labs/01-a-rule-is-a-subgraph.mjs
node docs/part5/ch28-writing-a-vjp-rule/labs/02-the-shape-fix-up.mjs
node docs/part5/ch29-building-the-backward-graph/labs/01-the-reverse-sweep.mjs
node docs/part5/ch29-building-the-backward-graph/labs/02-separate-or-joint.mjs
node docs/part5/ch30-memory-for-recomputation/labs/01-what-to-save.mjs
node docs/part5/ch31-differentiating-control-flow/labs/01-the-loop-unrolled.mjs
node docs/part5/ch31-differentiating-control-flow/labs/02-three-ways-to-lose-a-gradient.mjs
```

Every lab in this part uses one public entry point, `compileWithBackward`, and reads the graphs it builds through the trace stream of Chapter 18 — `irSnapshot: { afterGraphPasses: true }` fires once for the forward module and once for the backward one, which is why the labs can show both. Several switch fusion off so the emitted subgraphs stay legible; the numbers they report are the same either way.

Two labs reach past the documented surface in the way Part III established. Chapter 30's passes a `rematPolicy` that is not a `RematPolicy` — the builder calls exactly one method on it, so any object with `shouldRematerialize(op)` drives the saved set from outside. Chapter 31's second lab passes a `passContext` to switch the simplification passes off. Both are legitimate for a lab and neither is a supported API.

Only the timings are machine-specific: §27.5's scaling ratios, §29.6's mode comparison. Every graph size, argument count, operation count and gradient in this part is deterministic and should reproduce exactly.

## A note on what this part found

Three of the five chapters end on a mechanism that is implemented, tested, and unreachable from the API the labs use: `CheckpointPolicy` and its four segmenters, `scanCheckpoint`, and `findUnsupportedGradOps`. A fourth was reachable and slower than the alternative for a reason that had nothing to do with the transformation itself — joint mode recomputed the forward pass because the two-call API appeared to give it nowhere else to put the result, and §29.6 is both the measurement and the fix. And the largest limit in the part is not a bug at all but a decision with a number attached: differentiating a `scan` unrolls it, so a sequence model's backward graph is proportional to its sequence length, and 512 steps of the lab's LSTM would reach the lowering pipeline as roughly 39,000 operations.

As in Part IV, the chapters keep the measurement on both sides of any fix rather than only after it, because the chase is the content. Each finding is named with its file and line, each is reproducible by a lab here, and each is carried into the outline's Appendix E — the three unreachable mechanisms and the `scan` unroll still on the open list, joint mode's double forward moved to the closed one.
