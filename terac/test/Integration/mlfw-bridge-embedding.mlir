// RUN: %if mlfw-oracle %{ %node %tera_harness/mlfw_oracle.mjs embedding --emit-mlir --out %t.mlir --json %t.json %}
// RUN: %if mlfw-oracle %{ tera-opt %t.mlir -o %t.rt.mlir %}
// RUN: %if mlfw-oracle %{ FileCheck %s --check-prefix=MLIR --input-file=%t.mlir --implicit-check-not='"tera.' %}
// RUN: %if mlfw-oracle %{ tera-runner %t.mlir --shared-libs=%mlir_c_runner_utils --data=%t.json --check %}
// RUN: %if mlfw-oracle %{ %if cuda %{ tera-runner %t.mlir --target=cuda --shared-libs=%mlir_cuda_runtime --shared-libs=%mlir_c_runner_utils --data=%t.json --check %} %}
// RUN: %if mlfw-oracle %{ %node %tera_harness/mlfw_reader.mjs %t.rt.mlir --data %t.json %}

// RUN: %if mlfw-oracle %{ %node %tera_harness/mlfw_oracle.mjs embedding --grad --emit-mlir --out %t.g.mlir --json %t.g.json %}
// RUN: %if mlfw-oracle %{ tera-opt %t.g.mlir --tera-autodiff -o %t.vjp.mlir %}
// RUN: %if mlfw-oracle %{ FileCheck %s --check-prefix=VJP --input-file=%t.vjp.mlir %}
// RUN: %if mlfw-oracle %{ tera-runner %t.vjp.mlir --entry=embedding_vjp --shared-libs=%mlir_c_runner_utils --data=%t.g.json --check %}
// RUN: %if mlfw-oracle %{ %if cuda %{ tera-runner %t.vjp.mlir --entry=embedding_vjp --target=cuda --shared-libs=%mlir_cuda_runtime --shared-libs=%mlir_c_runner_utils --data=%t.g.json --check %} %}

// The last three ops the op-coverage survey found missing, two of them.
//
// `mlfw-bridge-mathops.mlir` records that survey: an MLP, a softmax, a layer
// norm, a GELU, a convolution, a max pool and an embedding were traced, and
// what the dialect had no op for came to four things. Three were unary maths
// and a reduction and are in that file. The fourth was `gather`, which is what
// an embedding is, and it could not join them: an index is not a shape and a
// lookup is not an elementwise map, so it needed its own op, its own lowering
// and its own rule -- and `scatter` with it, because the derivative of a gather
// is a scatter and there is no way to check one without the other.
//
// `conv` and `pool2d` are what is left, and are still generic here.
//
// Two shapes of the same op cross. An embedding reads whole rows, so the row
// axis collapses, the width axis survives as a window, and the index tensor has
// no axis for coordinates at all -- one row is one number, which mlfw spells as
// an `index_vector_dim` equal to the rank. Indexing along an axis reads single
// elements, so every operand axis collapses, nothing survives, and the
// coordinates are a real axis built by an `iota` and a `concat`. Neither shape
// reaches the other's branch of the lowering.
//
// Three of the six rows looked up are repeats, which is the only thing that can
// tell a scatter that adds from one that overwrites: both have the right shape
// and both agree everywhere a position is named once.

// MLIR-LABEL: func.func @embedding
// The embedding: a window survives and the index vector is implicit.
// MLIR: tera.gather %{{.*}} {collapsed_slice_dims = array<i64: 0>, index_vector_dim = 1 : i64, offset_dims = array<i64: 1>, slice_sizes = array<i64: 1, 4>, start_index_map = array<i64: 0>}
// Indexing along an axis: the coordinates are an axis of their own, and mlfw
// builds the axis it does not have out of an `iota` and a `concat`.
// MLIR: tera.iota
// MLIR: tera.concat
// MLIR: tera.gather %{{.*}} {collapsed_slice_dims = array<i64: 0, 1>, index_vector_dim = 2 : i64, offset_dims = array<i64>, slice_sizes = array<i64: 1, 1>, start_index_map = array<i64: 0, 1>}
// The scatter mlfw writes with an empty combiner region, which the custom form
// leaves out because addition is the only thing that region ever means.
// MLIR: tera.scatter %{{.*}} {index_vector_dim = 2 : i64, inserted_window_dims = array<i64: 0, 1>, scatter_dims_to_operand_dims = array<i64: 0, 1>, update_window_dims = array<i64>}
// MLIR: return

// Each op differentiates into the other, and the bridge runs both reverse
// passes on the same numbers: mlfw's `gather` rule builds a scatter-add into
// zeros and its `scatter` rule gathers the adjoint back, and these are the
// same two ops with the same attributes read across.
// VJP: func.func @embedding_vjp
// VJP: tera.gather
// VJP: tera.scatter
// VJP: tera.constant dense<0.000000e+00>
// VJP: tera.scatter
