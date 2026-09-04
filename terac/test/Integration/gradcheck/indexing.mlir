// RUN: tera-gradcheck %s --shared-libs=%mlir_c_runner_utils

// The numbers behind the gather and scatter rules. FileCheck can see that the
// derivative of a gather is a scatter with the attributes read across; it
// cannot see whether the attributes were read across correctly, and reversing
// two of them leaves a rule that still builds and still type-checks.
//
// Every index list below names a position twice. That is what these cases are
// for: a gather that reads one row twice sends two contributions back to it,
// and a scatter that overwrote instead of adding would agree with finite
// differences everywhere except there. The positions are constants rather than
// arguments because an index is not a differentiable quantity and a random one
// would not even be in range.

// An embedding: whole rows, so the row axis collapses and the width axis
// survives as a window, and one position is a single number -- the implicit
// index vector an `index_vector_dim` equal to the rank stands for. Row 3 is
// read twice.
func.func @gather_rows(%table: tensor<5x3xf64>, %weight: tensor<4x3xf64>)
    -> tensor<4x3xf64> attributes {tera.differentiable} {
  %ids = tera.constant dense<[3, 0, 3, 1]> : tensor<4xi32>
  %0 = tera.gather %table, %ids {offset_dims = array<i64: 1>,
                                 collapsed_slice_dims = array<i64: 0>,
                                 start_index_map = array<i64: 0>,
                                 slice_sizes = array<i64: 1, 3>,
                                 index_vector_dim = 1 : i64}
      : (tensor<5x3xf64>, tensor<4xi32>) -> tensor<4x3xf64>
  %1 = tera.mul %0, %weight : tensor<4x3xf64>
  return %1 : tensor<4x3xf64>
}

// The other shape: indexing along an axis reads one element, so both operand
// axes collapse, nothing survives as a window, and the coordinates are a real
// axis of the index tensor. Element (0, 3) is read twice.
func.func @gather_elements(%x: tensor<3x4xf64>) -> tensor<3x2xf64>
    attributes {tera.differentiable} {
  %at = tera.constant dense<[[[0, 3], [2, 1]],
                             [[1, 0], [0, 3]],
                             [[2, 2], [1, 3]]]> : tensor<3x2x2xi32>
  %0 = tera.gather %x, %at {offset_dims = array<i64>,
                            collapsed_slice_dims = array<i64: 0, 1>,
                            start_index_map = array<i64: 0, 1>,
                            slice_sizes = array<i64: 1, 1>,
                            index_vector_dim = 2 : i64}
      : (tensor<3x4xf64>, tensor<3x2x2xi32>) -> tensor<3x2xf64>
  %1 = tera.exp %0 : tensor<3x2xf64>
  return %1 : tensor<3x2xf64>
}

// A scatter differentiated directly, so the rule under test is the one that
// hands the operand its adjoint unchanged and gathers the rest back for the
// updates. Both arguments take a gradient, and the two writes to (0, 3) mean
// the updates gradient reads the same element of the adjoint twice.
func.func @scatter_elements(%x: tensor<3x4xf64>, %updates: tensor<3x2xf64>)
    -> tensor<3x4xf64> attributes {tera.differentiable} {
  %at = tera.constant dense<[[[0, 3], [2, 1]],
                             [[1, 0], [0, 3]],
                             [[2, 2], [1, 3]]]> : tensor<3x2x2xi32>
  %0 = tera.scatter %x, %at, %updates
      {update_window_dims = array<i64>,
       inserted_window_dims = array<i64: 0, 1>,
       scatter_dims_to_operand_dims = array<i64: 0, 1>,
       index_vector_dim = 2 : i64}
      : (tensor<3x4xf64>, tensor<3x2x2xi32>, tensor<3x2xf64>) -> tensor<3x4xf64>
  return %0 : tensor<3x4xf64>
}

// A windowed scatter, which is the shape an embedding is differentiated into.
// Its own derivative gathers a window back, so `slice_sizes` has to be the
// width the window covered rather than one.
func.func @scatter_rows(%table: tensor<5x3xf64>, %updates: tensor<4x3xf64>)
    -> tensor<5x3xf64> attributes {tera.differentiable} {
  %ids = tera.constant dense<[3, 0, 3, 1]> : tensor<4xi32>
  %0 = tera.scatter %table, %ids, %updates
      {update_window_dims = array<i64: 1>,
       inserted_window_dims = array<i64: 0>,
       scatter_dims_to_operand_dims = array<i64: 0>,
       index_vector_dim = 1 : i64}
      : (tensor<5x3xf64>, tensor<4xi32>, tensor<4x3xf64>) -> tensor<5x3xf64>
  return %0 : tensor<5x3xf64>
}

// The two composed, with the gathered value read by both the scatter and the
// multiply so its adjoint is a sum of two rather than one. This is the case a
// rule that is right in isolation and wrong about which operand it belongs to
// fails on.
func.func @round_trip(%table: tensor<5x3xf64>, %weight: tensor<4x3xf64>)
    -> tensor<5x3xf64> attributes {tera.differentiable} {
  %ids = tera.constant dense<[3, 0, 3, 1]> : tensor<4xi32>
  %0 = tera.gather %table, %ids {offset_dims = array<i64: 1>,
                                 collapsed_slice_dims = array<i64: 0>,
                                 start_index_map = array<i64: 0>,
                                 slice_sizes = array<i64: 1, 3>,
                                 index_vector_dim = 1 : i64}
      : (tensor<5x3xf64>, tensor<4xi32>) -> tensor<4x3xf64>
  %1 = tera.mul %0, %weight : tensor<4x3xf64>
  %2 = tera.scatter %table, %ids, %1
      {update_window_dims = array<i64: 1>,
       inserted_window_dims = array<i64: 0>,
       scatter_dims_to_operand_dims = array<i64: 0>,
       index_vector_dim = 1 : i64}
      : (tensor<5x3xf64>, tensor<4xi32>, tensor<4x3xf64>) -> tensor<5x3xf64>
  return %2 : tensor<5x3xf64>
}
