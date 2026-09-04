// RUN: tera-gradcheck %s --shared-libs=%mlir_c_runner_utils

// The one line of the dot rule that is not obvious. The derivative contracts
// the adjoint against the other operand, and what survives on that side is its
// contracting axes — but a dot result carries surviving axes in increasing
// order, not in the order the attribute listed them. So contraction `k` has to
// be found again by where its axis sorts, not by `k`.
//
// Both operands list their contracting axes out of order here, so both halves
// of the rule go through that remapping. With the axes in order the mistake is
// invisible, which is why a matmul-shaped test would not catch it.
//
// lhs [2, 3, 4, 5] contracting (3, 1) against rhs [2, 3, 5, 6] contracting
// (2, 1): axis 3 of the lhs pairs with axis 2 of the rhs, axis 1 with axis 1.
// One batch axis, one free axis each, so the result is [2, 4, 6].

func.func @tangled(%a: tensor<2x3x4x5xf64>, %b: tensor<2x3x5x6xf64>)
    -> tensor<2x4x6xf64> attributes {tera.differentiable} {
  %0 = tera.dot %a, %b {lhs_batch = array<i64: 0>,
                        lhs_contracting = array<i64: 3, 1>,
                        rhs_batch = array<i64: 0>,
                        rhs_contracting = array<i64: 2, 1>}
      : (tensor<2x3x4x5xf64>, tensor<2x3x5x6xf64>) -> tensor<2x4x6xf64>
  return %0 : tensor<2x4x6xf64>
}
