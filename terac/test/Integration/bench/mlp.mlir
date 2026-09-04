// RUN: tera-runner %s --entry=mlp --benchmark=1 --warmup=1 --shared-libs=%mlir_c_runner_utils | FileCheck %s
// RUN: tera-opt %s --tera-autodiff | tera-runner - --entry=mlp_vjp --benchmark=1 --warmup=1 --shared-libs=%mlir_c_runner_utils | FileCheck %s --check-prefix=VJP

// A three-layer perceptron at a size where a measurement means something:
// 64 x 256 in, two hidden layers of 512, ten classes out. Roughly 25 million
// multiply-accumulates forward, which is enough that the timing describes the
// kernels rather than the cost of making the call.
//
// It is not a gradcheck model. The models in test/gradcheck are chosen for
// awkward shape, and finite differences over 400,000 weights would take longer
// than the whole suite; this one is marked differentiable so that the reverse
// pass can be timed as well, and the rules it uses are the ones mlp.mlir there
// already checks.

// CHECK: mlp: 1 runs
// VJP: mlp_vjp: 1 runs

func.func @mlp(%x: tensor<64x256xf32>, %w1: tensor<256x512xf32>,
               %b1: tensor<512xf32>, %w2: tensor<512x512xf32>,
               %b2: tensor<512xf32>, %w3: tensor<512x10xf32>,
               %b3: tensor<10xf32>) -> tensor<64x10xf32>
    attributes {tera.differentiable} {
  %zero1 = tera.constant dense<0.000000e+00> : tensor<64x512xf32>

  %0 = tera.dot %x, %w1 {lhs_batch = array<i64>, lhs_contracting = array<i64: 1>,
                         rhs_batch = array<i64>, rhs_contracting = array<i64: 0>}
      : (tensor<64x256xf32>, tensor<256x512xf32>) -> tensor<64x512xf32>
  %1 = tera.broadcast_in_dim %b1 {broadcast_dimensions = array<i64: 1>}
      : tensor<512xf32> -> tensor<64x512xf32>
  %2 = tera.add %0, %1 : tensor<64x512xf32>
  %3 = tera.maximum %2, %zero1 : tensor<64x512xf32>

  %4 = tera.dot %3, %w2 {lhs_batch = array<i64>, lhs_contracting = array<i64: 1>,
                         rhs_batch = array<i64>, rhs_contracting = array<i64: 0>}
      : (tensor<64x512xf32>, tensor<512x512xf32>) -> tensor<64x512xf32>
  %5 = tera.broadcast_in_dim %b2 {broadcast_dimensions = array<i64: 1>}
      : tensor<512xf32> -> tensor<64x512xf32>
  %6 = tera.add %4, %5 : tensor<64x512xf32>
  %7 = tera.maximum %6, %zero1 : tensor<64x512xf32>

  %8 = tera.dot %7, %w3 {lhs_batch = array<i64>, lhs_contracting = array<i64: 1>,
                         rhs_batch = array<i64>, rhs_contracting = array<i64: 0>}
      : (tensor<64x512xf32>, tensor<512x10xf32>) -> tensor<64x10xf32>
  %9 = tera.broadcast_in_dim %b3 {broadcast_dimensions = array<i64: 1>}
      : tensor<10xf32> -> tensor<64x10xf32>
  %10 = tera.add %8, %9 : tensor<64x10xf32>
  return %10 : tensor<64x10xf32>
}
