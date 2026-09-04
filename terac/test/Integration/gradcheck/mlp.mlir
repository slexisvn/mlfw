// RUN: tera-gradcheck %s --shared-libs=%mlir_c_runner_utils

// A two-layer perceptron: matmul, broadcast bias, relu, matmul, broadcast
// bias. Between them these exercise dot, broadcast_in_dim, add, maximum and
// constant, and the relu makes the check bite on a rule that is only piecewise
// linear.

func.func @mlp(%x: tensor<4x3xf64>, %w1: tensor<3x5xf64>, %b1: tensor<5xf64>,
               %w2: tensor<5x2xf64>, %b2: tensor<2xf64>) -> tensor<4x2xf64>
    attributes {tera.differentiable} {
  %0 = tera.dot %x, %w1 {lhs_batch = array<i64>, lhs_contracting = array<i64: 1>,
                         rhs_batch = array<i64>, rhs_contracting = array<i64: 0>}
      : (tensor<4x3xf64>, tensor<3x5xf64>) -> tensor<4x5xf64>
  %1 = tera.broadcast_in_dim %b1 {broadcast_dimensions = array<i64: 1>}
      : tensor<5xf64> -> tensor<4x5xf64>
  %2 = tera.add %0, %1 : tensor<4x5xf64>
  %3 = tera.constant dense<0.000000e+00> : tensor<4x5xf64>
  %4 = tera.maximum %2, %3 : tensor<4x5xf64>
  %5 = tera.dot %4, %w2 {lhs_batch = array<i64>, lhs_contracting = array<i64: 1>,
                         rhs_batch = array<i64>, rhs_contracting = array<i64: 0>}
      : (tensor<4x5xf64>, tensor<5x2xf64>) -> tensor<4x2xf64>
  %6 = tera.broadcast_in_dim %b2 {broadcast_dimensions = array<i64: 1>}
      : tensor<2xf64> -> tensor<4x2xf64>
  %7 = tera.add %5, %6 : tensor<4x2xf64>
  return %7 : tensor<4x2xf64>
}
