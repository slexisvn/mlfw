// RUN: %if mlfw-oracle %{ %node %tera_harness/mlfw_oracle.mjs convnet --emit-mlir --out %t.mlir --json %t.json %}
// RUN: %if mlfw-oracle %{ tera-opt %t.mlir -o %t.rt.mlir %}
// RUN: %if mlfw-oracle %{ FileCheck %s --check-prefix=MLIR --input-file=%t.mlir --implicit-check-not='"tera.' %}
// RUN: %if mlfw-oracle %{ tera-runner %t.mlir --shared-libs=%mlir_c_runner_utils --data=%t.json --check %}
// RUN: %if mlfw-oracle %{ %if cuda %{ tera-runner %t.mlir --target=cuda --shared-libs=%mlir_cuda_runtime --shared-libs=%mlir_c_runner_utils --data=%t.json --check %} %}
// RUN: %if mlfw-oracle %{ %node %tera_harness/mlfw_reader.mjs %t.rt.mlir --data %t.json %}

// RUN: %if mlfw-oracle %{ %node %tera_harness/mlfw_oracle.mjs convnet --grad --emit-mlir --out %t.g.mlir --json %t.g.json %}
// RUN: %if mlfw-oracle %{ tera-opt %t.g.mlir --tera-autodiff -o %t.vjp.mlir %}
// RUN: %if mlfw-oracle %{ FileCheck %s --check-prefix=VJP --input-file=%t.vjp.mlir %}
// RUN: %if mlfw-oracle %{ tera-runner %t.vjp.mlir --entry=convnet_vjp --shared-libs=%mlir_c_runner_utils --data=%t.g.json --check %}
// RUN: %if mlfw-oracle %{ %if cuda %{ tera-runner %t.vjp.mlir --entry=convnet_vjp --target=cuda --shared-libs=%mlir_cuda_runtime --shared-libs=%mlir_c_runner_utils --data=%t.g.json --check %} %}

// The end of the op-coverage survey.
//
// `mlfw-bridge-mathops.mlir` records what that survey found missing, and this
// is the last of it: a strided, padded convolution and the two ways to pool
// its output. With `mlfw-bridge-embedding.mlir` alongside, every op the tracer
// still emits for an MLP, a softmax, a layer norm, a GELU, a sigmoid, a
// convolution, a max pool, an embedding and a batch norm now prints in the
// custom form, which is to say it crosses.
//
// Two of those ops are new here and neither is `conv` or `pool2d`. A
// convolution's derivative runs the adjoint back through the kernel read
// backwards, and spaces it out first when the forward pass strided over
// positions; `tera.reverse` and `tera.pad` are those two, and both are ops
// mlfw already had and the dialect did not. `tera.pad` also replaced the
// reshape-and-concatenate that `tera.slice` was undoing itself with, so the
// dialect came out of this shorter in one place as well as longer in two.
//
// The numbers are multiples of 1/8, every window sums at most eighteen
// products, and both pools divide by four, so the whole program is exact in
// f32 and `--check` compares bit for bit -- in both directions, on both
// targets.

// MLIR-LABEL: func.func @convnet
// MLIR: tera.conv %{{.*}} {dilation = array<i64: 1, 1>, groups = 1 : i64, padding = array<i64: 1, 1, 1, 1>, strides = array<i64: 2, 2>}
// The two pools read the same features, so the adjoint of that tensor is a sum
// of what each of them sends back.
// MLIR: tera.pool2d max
// MLIR: tera.pool2d average
// MLIR: tera.add
// MLIR: return

// The derivative, against mlfw's own. Both convolutions are here: the one that
// runs the spaced-out adjoint back through the flipped kernel, and the one
// that contracts the input against the adjoint over the batch axis.
// VJP: func.func @convnet_vjp
// VJP: tera.pad
// VJP: tera.reverse
// VJP: tera.conv
// VJP: tera.conv
// VJP: tera.slice
