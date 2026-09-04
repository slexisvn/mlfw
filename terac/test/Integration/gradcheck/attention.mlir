// RUN: tera-gradcheck %s --shared-libs=%mlir_c_runner_utils

// A batched causal attention block: scaled dot-product scores, a mask built
// from two iotas, a softmax stabilised by a detached row maximum, and a second
// contraction against the values.
//
// The batch axes are what make this worth running: the dot rule has to carry
// them through both halves of its own derivative and permute the result back,
// and nothing smaller catches that. The `stop_gradient` on the row maximum is
// the other half of the point — a softmax is invariant to the shift, so
// blocking that gradient has to leave the answer unchanged.

func.func @attention(%q: tensor<2x4x3xf64>, %k: tensor<2x4x3xf64>,
                     %v: tensor<2x4x3xf64>) -> tensor<2x4x3xf64>
    attributes {tera.differentiable} {
  %0 = tera.dot %q, %k {lhs_batch = array<i64: 0>, lhs_contracting = array<i64: 2>,
                        rhs_batch = array<i64: 0>, rhs_contracting = array<i64: 2>}
      : (tensor<2x4x3xf64>, tensor<2x4x3xf64>) -> tensor<2x4x4xf64>
  %1 = tera.constant dense<0.57735026918962584> : tensor<2x4x4xf64>
  %2 = tera.mul %0, %1 : tensor<2x4x4xf64>

  %3 = tera.iota {iota_dimension = 1} : tensor<2x4x4xi64>
  %4 = tera.iota {iota_dimension = 2} : tensor<2x4x4xi64>
  %5 = tera.compare ge, %3, %4 : tensor<2x4x4xi64> -> tensor<2x4x4xi1>
  %6 = tera.constant dense<-1.000000e+30> : tensor<2x4x4xf64>
  %7 = tera.select %5, %2, %6 : tensor<2x4x4xi1>, tensor<2x4x4xf64>

  %8 = tera.reduce maximum, %7 {dimensions = array<i64: 2>}
      : tensor<2x4x4xf64> -> tensor<2x4xf64>
  %9 = tera.stop_gradient %8 : tensor<2x4xf64>
  %10 = tera.broadcast_in_dim %9 {broadcast_dimensions = array<i64: 0, 1>}
      : tensor<2x4xf64> -> tensor<2x4x4xf64>
  %11 = tera.sub %7, %10 : tensor<2x4x4xf64>
  %12 = tera.exp %11 : tensor<2x4x4xf64>
  %13 = tera.reduce sum, %12 {dimensions = array<i64: 2>}
      : tensor<2x4x4xf64> -> tensor<2x4xf64>
  %14 = tera.broadcast_in_dim %13 {broadcast_dimensions = array<i64: 0, 1>}
      : tensor<2x4xf64> -> tensor<2x4x4xf64>
  %15 = tera.div %12, %14 : tensor<2x4x4xf64>

  %16 = tera.dot %15, %v {lhs_batch = array<i64: 0>, lhs_contracting = array<i64: 2>,
                          rhs_batch = array<i64: 0>, rhs_contracting = array<i64: 1>}
      : (tensor<2x4x4xf64>, tensor<2x4x3xf64>) -> tensor<2x4x3xf64>
  return %16 : tensor<2x4x3xf64>
}
