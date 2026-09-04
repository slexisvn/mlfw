// RUN: tera-opt %s | tera-opt | FileCheck %s
// RUN: tera-opt %s --canonicalize | FileCheck %s --check-prefix=CANON
// RUN: tera-runner %s --shared-libs=%mlir_c_runner_utils --data=%S/demo.json --check | FileCheck %s --check-prefix=VALUE
// The same program on the device, checked against the same frozen answer: the
// two targets share nothing below bufferized linalg, so agreeing on 34 is the
// only evidence that the GPU pipeline lowers the program and not some other.
// RUN: %if cuda %{ tera-runner %s --target=cuda --shared-libs=%mlir_cuda_runtime --shared-libs=%mlir_c_runner_utils --data=%S/demo.json --check | FileCheck %s --check-prefix=VALUE %}

// RUN: %if mlfw-oracle %{ %node %tera_harness/mlfw_oracle.mjs demo --out %t.json %}
// RUN: %if mlfw-oracle %{ tera-runner %s --shared-libs=%mlir_c_runner_utils --data=%t.json --check %}
// RUN: %if mlfw-oracle %{ diff -u %S/demo.json %t.json %}

// The program the mlfw tracer emits for `relu(matmul(x, w)).sum()`, written in
// the tera dialect. This is the Phase 1 gate: the first real program the
// dialect has to hold.

// CHECK-LABEL: func @demo
// CHECK-SAME:    (%[[X:.*]]: tensor<2x4xf32>, %[[W:.*]]: tensor<4x2xf32>) -> tensor<f32>
func.func @demo(%x: tensor<2x4xf32>, %w: tensor<4x2xf32>) -> tensor<f32> {
  // CHECK: %[[DOT:.*]] = tera.dot %[[X]], %[[W]]
  // CHECK-SAME: lhs_contracting = array<i64: 1>
  // CHECK-SAME: rhs_contracting = array<i64: 0>
  %0 = tera.dot %x, %w {lhs_batch = array<i64>,
                        lhs_contracting = array<i64: 1>,
                        rhs_batch = array<i64>,
                        rhs_contracting = array<i64: 0>}
      : (tensor<2x4xf32>, tensor<4x2xf32>) -> tensor<2x2xf32>

  // CHECK: %[[ZERO:.*]] = tera.constant dense<0.000000e+00> : tensor<f32>
  %1 = tera.constant dense<0.0> : tensor<f32>

  // CHECK: %[[B:.*]] = tera.broadcast_in_dim %[[ZERO]]
  %2 = tera.broadcast_in_dim %1 {broadcast_dimensions = array<i64>}
      : tensor<f32> -> tensor<2x2xf32>

  // CHECK: %[[RELU:.*]] = tera.maximum %[[DOT]], %[[B]] : tensor<2x2xf32>
  %3 = tera.maximum %0, %2 : tensor<2x2xf32>

  // CHECK: tera.reduce sum, %[[RELU]] {dimensions = array<i64: 0, 1>}
  %4 = tera.reduce sum, %3 {dimensions = array<i64: 0, 1>}
      : tensor<2x2xf32> -> tensor<f32>

  return %4 : tensor<f32>
}

// Canonicalization must not disturb a program that is already minimal. Every op
// survives; the only movement is the constant, which the folder hoists to the
// top of the block as it does for any ConstantLike op.
// CANON-LABEL: func @demo
// CANON: tera.constant
// CANON: tera.dot
// CANON: tera.broadcast_in_dim
// CANON: tera.maximum
// CANON: tera.reduce

// The Phase 2 gate: lowered to LLVM and run, this program answers 34, the
// value the mlfw compiler gives for the same inputs.
//
// demo.json is that answer, frozen. Where mlfw can be run the last three RUN
// lines regenerate it, check tera-runner against the fresh copy, and diff the
// two, so the frozen file cannot drift away from the oracle it came from.
// VALUE: "data"
// VALUE-NEXT: 34
