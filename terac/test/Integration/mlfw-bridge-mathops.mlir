// RUN: %if mlfw-oracle %{ %node %tera_harness/mlfw_oracle.mjs mathops --emit-mlir --out %t.mlir --json %t.json %}
// RUN: %if mlfw-oracle %{ tera-opt %t.mlir -o %t.rt.mlir %}
// RUN: %if mlfw-oracle %{ FileCheck %s --check-prefix=MLIR --input-file=%t.mlir --implicit-check-not='"tera.' %}
// RUN: %if mlfw-oracle %{ tera-runner %t.mlir --shared-libs=%mlir_c_runner_utils --data=%t.json --check %}
// RUN: %if mlfw-oracle %{ %if cuda %{ tera-runner %t.mlir --target=cuda --shared-libs=%mlir_cuda_runtime --shared-libs=%mlir_c_runner_utils --data=%t.json --check %} %}
// RUN: %if mlfw-oracle %{ %node %tera_harness/mlfw_reader.mjs %t.rt.mlir --data %t.json %}

// RUN: %if mlfw-oracle %{ %node %tera_harness/mlfw_oracle.mjs mathops --grad --emit-mlir --out %t.g.mlir --json %t.g.json %}
// RUN: %if mlfw-oracle %{ tera-opt %t.g.mlir --tera-autodiff -o %t.vjp.mlir %}
// RUN: %if mlfw-oracle %{ FileCheck %s --check-prefix=VJP --input-file=%t.vjp.mlir %}
// RUN: %if mlfw-oracle %{ tera-runner %t.vjp.mlir --entry=mathops_vjp --shared-libs=%mlir_c_runner_utils --data=%t.g.json --check %}
// RUN: %if mlfw-oracle %{ %if cuda %{ tera-runner %t.vjp.mlir --entry=mathops_vjp --target=cuda --shared-libs=%mlir_cuda_runtime --shared-libs=%mlir_c_runner_utils --data=%t.g.json --check %} %}

// What was missing from the dialect, measured rather than guessed.
//
// The other bridge programs were written to exercise a piece of machinery. This
// one was written from a survey: mlfw's tracer was run over an MLP, a softmax, a
// layer norm, a GELU, a sigmoid, a convolution, a max pool, an embedding and a
// batch norm, and every op left after `DecompositionPass` was checked against
// the dialect. Almost all of them were already there -- a GELU is `erf` written
// out, a softmax is a max, an exp and a sum -- and what was not came to four
// things, three of which are here.
//
// `rsqrt` is the one every normalisation divides by, and its own op rather than
// a `div` of a `sqrt` because the hardware has the instruction. `sqrt` and
// `tanh` are the same kind of op and came with it.
//
// `reduce mean` is the interesting one. It is the only reduction here that is
// not a monoid: it sums and then divides by how many it summed, so it has no
// identity element and stablehlo leaves it out, spelling a mean as a sum and a
// divide. The dialect keeps it, because that is the shape mlfw's IR holds it in
// and because splitting it here would only make the reduction fusion downstream
// put it back together. The lowering sums into the accumulator and divides once
// at the end -- one rounding rather than `n` of them -- and the reverse pass
// hands every element the same `1/n` share of the adjoint.
//
// The fourth was `gather`, which is what an embedding is, and it is not here:
// an index is not a shape and a lookup is not an elementwise map, so it needed
// an op of its own, and `scatter` with it, because the derivative of a gather
// is a scatter. Both are in `mlfw-bridge-embedding.mlir`, and `conv` and
// `pool2d` -- the two the survey called larger -- are in
// `mlfw-bridge-convnet.mlir`. Nothing the survey named is missing now.
//
// The answer is not exact -- a reciprocal square root is not a dyadic rational
// -- so this is the one bridge program `--check` compares through its tolerance
// rather than bit for bit.

// MLIR-LABEL: func.func @mathops
// MLIR: tera.reduce mean
// MLIR: tera.rsqrt
// MLIR: tera.tanh
// MLIR: tera.sqrt
// MLIR: return

// The derivative of each, against mlfw's own: `1/n` of the adjoint spread back
// over the axis a mean reduced, `1 - tanh^2`, and `1 / 2 sqrt(x)`.
// VJP: func.func @mathops_vjp
// VJP: tera.tanh
// VJP: tera.sqrt
