// RUN: %if mlfw-oracle %{ %node %tera_harness/mlfw_oracle.mjs integers --emit-mlir --out %t.mlir --json %t.json %}
// RUN: %if mlfw-oracle %{ tera-opt %t.mlir %}
// RUN: %if mlfw-oracle %{ FileCheck %s --check-prefix=MLIR --input-file=%t.mlir --implicit-check-not='"tera.' %}
// RUN: %if mlfw-oracle %{ tera-runner %t.mlir --shared-libs=%mlir_c_runner_utils --data=%t.json --check %}
// RUN: %if mlfw-oracle %{ %if cuda %{ tera-runner %t.mlir --target=cuda --shared-libs=%mlir_cuda_runtime --shared-libs=%mlir_c_runner_utils --data=%t.json --check %} %}

// Differentiating it reaches the cast's rule from the float side, which is the
// only argument a gradient flows to.
// RUN: %if mlfw-oracle %{ %node %tera_harness/mlfw_oracle.mjs integers --grad --emit-mlir --out %t.grad.mlir --json %t.grad.json %}
// RUN: %if mlfw-oracle %{ tera-opt %t.grad.mlir --tera-autodiff -o %t.vjp.mlir %}
// RUN: %if mlfw-oracle %{ FileCheck %s --check-prefix=VJP --input-file=%t.vjp.mlir %}
// RUN: %if mlfw-oracle %{ tera-runner %t.vjp.mlir --shared-libs=%mlir_c_runner_utils --data=%t.grad.json --check %}
// RUN: %if mlfw-oracle %{ %if cuda %{ tera-runner %t.vjp.mlir --target=cuda --shared-libs=%mlir_cuda_runtime --shared-libs=%mlir_c_runner_utils --data=%t.grad.json --check %} %}

// Every other bridge program is f32 from end to end, so the integer half of the
// dialect crossed only in tests written by hand. elementwise.mlir has the same
// ops; this one is traced.
//
// What the integer path costs is not in the printer. `tera.div` truncates
// towards zero and `tera.neg` has to be synthesised from a subtraction, and
// running this found mlfw's CPU backend getting the first of those wrong:
// integer arithmetic there is JavaScript arithmetic, which has no integer
// division, and a quotient stored straight into an integer buffer only came out
// right because the store truncated it. Fused into a longer chain there is no
// such store, and the float leaked into the result -- so mlfw's own CPU and
// WebAssembly backends disagreed with each other, and both with terac.
// tests/backend/cpu/integer-arithmetic.test.js is the regression test.
//
// `tera.convert` arrives without being asked for. mlfw has no user-facing cast,
// so nothing traced could reach the op -- but mlfw's elementwise ops unify the
// dtypes of their operands, and multiplying an i32 by an f32 is what inserts
// one. That is the path a real program takes to a cast, and it is why the
// output is f32 while everything above it is not.

// MLIR-LABEL: func.func @integers
// MLIR-SAME:    (%{{.*}}: tensor<2x3xi32>, %{{.*}}: tensor<2x3xi32>, %{{.*}}: tensor<2x3xf32>) -> (tensor<2x3xf32>)
// MLIR: tera.div %{{.*}}, %{{.*}} : tensor<2x3xi32>
// MLIR: tera.maximum %{{.*}}, %{{.*}} : tensor<2x3xi32>
// MLIR: tera.compare lt, %{{.*}}, %{{.*}} : tensor<2x3xi32> -> tensor<2x3xi1>
// MLIR: tera.neg %{{.*}} : tensor<2x3xi32>
// MLIR: tera.select %{{.*}}, %{{.*}}, %{{.*}} : tensor<2x3xi1>, tensor<2x3xi32>
// MLIR: tera.convert %{{.*}} : tensor<2x3xi32> -> tensor<2x3xf32>
// MLIR: tera.mul %{{.*}}, %{{.*}} : tensor<2x3xf32>
// MLIR: return

// Only the float argument carries a gradient: the two integer ones are indices
// as far as a derivative is concerned, and `tera.diff_args` says so.
// VJP: func.func @integers_vjp({{.*}}) -> tensor<2x3xf32> attributes {tera.diff_args = array<i64: 2>}
