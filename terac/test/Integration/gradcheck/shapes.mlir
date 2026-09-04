// RUN: tera-gradcheck %s --shared-libs=%mlir_c_runner_utils

// The rules the two model gates do not reach. `slice` carries a stride of two,
// which is the branch of its rule that dilates the adjoint rather than only
// padding it, and the sliced value is used twice, so the tree accumulation in
// the pass has something to sum.
//
// `exp` sits in front of both reductions on purpose. The product rule divides
// by the operand, so it needs one that cannot be zero, and the minimum rule
// needs a row whose smallest element is unique.

func.func @shapes(%a: tensor<2x6xf64>, %b: tensor<3x2xf64>) -> tensor<3xf64>
    attributes {tera.differentiable} {
  %0 = tera.transpose %a {permutation = array<i64: 1, 0>}
      : tensor<2x6xf64> -> tensor<6x2xf64>
  %1 = tera.reshape %0 : tensor<6x2xf64> -> tensor<3x4xf64>
  %2 = tera.slice %1 {start_indices = array<i64: 0, 1>,
                      limit_indices = array<i64: 3, 4>,
                      strides = array<i64: 1, 2>}
      : tensor<3x4xf64> -> tensor<3x2xf64>
  %3 = tera.neg %b : tensor<3x2xf64>
  %4 = tera.add %2, %3 : tensor<3x2xf64>
  %5 = tera.concat %4, %2 {dimension = 1 : i64}
      : tensor<3x2xf64>, tensor<3x2xf64> -> tensor<3x4xf64>
  %6 = tera.exp %5 : tensor<3x4xf64>
  %7 = tera.reduce product, %6 {dimensions = array<i64: 1>}
      : tensor<3x4xf64> -> tensor<3xf64>
  %8 = tera.reduce minimum, %6 {dimensions = array<i64: 1>}
      : tensor<3x4xf64> -> tensor<3xf64>
  %9 = tera.mul %7, %8 : tensor<3xf64>
  return %9 : tensor<3xf64>
}

// A broadcast that both adds an axis and stretches one of extent 1. Its rule
// has to sum over both, and only the stretched axis distinguishes it from the
// simpler rule that sums the axes the operand never named.
func.func @stretch(%a: tensor<3x1xf64>, %b: tensor<2x3x4xf64>) -> tensor<3xf64>
    attributes {tera.differentiable} {
  %0 = tera.broadcast_in_dim %a {broadcast_dimensions = array<i64: 1, 2>}
      : tensor<3x1xf64> -> tensor<2x3x4xf64>
  %1 = tera.mul %0, %b : tensor<2x3x4xf64>
  %2 = tera.reduce sum, %1 {dimensions = array<i64: 0, 2>}
      : tensor<2x3x4xf64> -> tensor<3xf64>
  return %2 : tensor<3xf64>
}
