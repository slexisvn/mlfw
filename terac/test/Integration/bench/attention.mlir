// RUN: tera-runner %s --entry=attention --benchmark=1 --warmup=1 --shared-libs=%mlir_c_runner_utils | FileCheck %s
// RUN: tera-opt %s --tera-autodiff | tera-runner - --entry=attention_vjp --benchmark=1 --warmup=1 --shared-libs=%mlir_c_runner_utils | FileCheck %s --check-prefix=VJP

// The causal attention block of test/gradcheck/attention.mlir at a size that
// can be measured: four batches, 128 positions, 64 channels. The two
// contractions are 4.2 million multiply-accumulates each, and between them
// sits a softmax over 65,536 elements, so the shape of the number says which
// of the two the pipeline is failing to optimise.

// CHECK: attention: 1 runs
// VJP: attention_vjp: 1 runs

func.func @attention(%q: tensor<4x128x64xf32>, %k: tensor<4x128x64xf32>,
                     %v: tensor<4x128x64xf32>) -> tensor<4x128x64xf32>
    attributes {tera.differentiable} {
  %0 = tera.dot %q, %k {lhs_batch = array<i64: 0>, lhs_contracting = array<i64: 2>,
                        rhs_batch = array<i64: 0>, rhs_contracting = array<i64: 2>}
      : (tensor<4x128x64xf32>, tensor<4x128x64xf32>) -> tensor<4x128x128xf32>
  %1 = tera.constant dense<0.125000e+00> : tensor<4x128x128xf32>
  %2 = tera.mul %0, %1 : tensor<4x128x128xf32>

  %3 = tera.iota {iota_dimension = 1} : tensor<4x128x128xi64>
  %4 = tera.iota {iota_dimension = 2} : tensor<4x128x128xi64>
  %5 = tera.compare ge, %3, %4 : tensor<4x128x128xi64> -> tensor<4x128x128xi1>
  %6 = tera.constant dense<-1.000000e+30> : tensor<4x128x128xf32>
  %7 = tera.select %5, %2, %6 : tensor<4x128x128xi1>, tensor<4x128x128xf32>

  %8 = tera.reduce maximum, %7 {dimensions = array<i64: 2>}
      : tensor<4x128x128xf32> -> tensor<4x128xf32>
  %9 = tera.stop_gradient %8 : tensor<4x128xf32>
  %10 = tera.broadcast_in_dim %9 {broadcast_dimensions = array<i64: 0, 1>}
      : tensor<4x128xf32> -> tensor<4x128x128xf32>
  %11 = tera.sub %7, %10 : tensor<4x128x128xf32>
  %12 = tera.exp %11 : tensor<4x128x128xf32>
  %13 = tera.reduce sum, %12 {dimensions = array<i64: 2>}
      : tensor<4x128x128xf32> -> tensor<4x128xf32>
  %14 = tera.broadcast_in_dim %13 {broadcast_dimensions = array<i64: 0, 1>}
      : tensor<4x128xf32> -> tensor<4x128x128xf32>
  %15 = tera.div %12, %14 : tensor<4x128x128xf32>

  %16 = tera.dot %15, %v {lhs_batch = array<i64: 0>, lhs_contracting = array<i64: 2>,
                          rhs_batch = array<i64: 0>, rhs_contracting = array<i64: 1>}
      : (tensor<4x128x128xf32>, tensor<4x128x64xf32>) -> tensor<4x128x64xf32>
  return %16 : tensor<4x128x64xf32>
}
