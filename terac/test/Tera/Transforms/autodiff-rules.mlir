// RUN: tera-opt %s --tera-autodiff --split-input-file | FileCheck %s

// One case per op, checking the shape of the derivative it builds. The
// gradcheck under test/gradcheck proves the numbers; these prove which ops the
// rules reach for, which is what a later refactor breaks first.

// d(a/b) = da/b, and db = -(da/b) * (a/b). max routes the adjoint by a
// comparison rather than by recomputing the branch.
// CHECK-LABEL: func @arithmetic_vjp
// CHECK: %[[FWD:.*]] = tera.div %arg0, %arg1
// CHECK: %[[ZERO:.*]] = tera.constant dense<0.000000e+00>
// CHECK: %[[PICK:.*]] = tera.compare ge, %[[FWD]], %arg1
// CHECK: %[[TOMAX:.*]] = tera.select %[[PICK]], %arg2, %[[ZERO]]
// CHECK: %[[TORHS:.*]] = tera.select %[[PICK]], %[[ZERO]], %arg2
// CHECK: %[[DA:.*]] = tera.div %[[TOMAX]], %arg1
// CHECK: %[[SCALED:.*]] = tera.mul %[[DA]], %[[FWD]]
// CHECK: %[[DB:.*]] = tera.neg %[[SCALED]]
// CHECK: %[[TOTAL:.*]] = tera.add %[[TORHS]], %[[DB]]
// CHECK: return %[[DA]], %[[TOTAL]]
func.func @arithmetic(%a: tensor<4xf32>, %b: tensor<4xf32>) -> tensor<4xf32>
    attributes {tera.differentiable} {
  %0 = tera.div %a, %b : tensor<4xf32>
  %1 = tera.maximum %0, %b : tensor<4xf32>
  return %1 : tensor<4xf32>
}

// -----

// sub and neg, and a select whose predicate takes no gradient at all.
// CHECK-LABEL: func @choices_vjp
// CHECK-SAME: {tera.diff_args = array<i64: 1, 2>}
// CHECK: tera.select %arg0, %arg1
// CHECK: %[[ZERO:.*]] = tera.constant dense<0.000000e+00>
// CHECK: %[[TRUE:.*]] = tera.select %arg0, %arg3, %[[ZERO]]
// CHECK: %[[FALSE:.*]] = tera.select %arg0, %[[ZERO]], %arg3
// CHECK: %[[NEG:.*]] = tera.neg %[[FALSE]]
// CHECK: return %[[TRUE]], %[[NEG]]
func.func @choices(%p: tensor<4xi1>, %a: tensor<4xf32>, %b: tensor<4xf32>)
    -> tensor<4xf32> attributes {tera.differentiable} {
  %0 = tera.constant dense<0.000000e+00> : tensor<4xf32>
  %1 = tera.sub %0, %b : tensor<4xf32>
  %2 = tera.select %p, %a, %1 : tensor<4xi1>, tensor<4xf32>
  return %2 : tensor<4xf32>
}

// -----

// A conversion between float types runs the other way in the derivative. The
// finite-difference gate cannot check this one: a lossy conversion is a step
// function, so only the shape of the rule is testable.
// CHECK-LABEL: func @widen_vjp
// CHECK: %[[NARROW:.*]] = tera.convert %arg1 : tensor<4xf64> -> tensor<4xf32>
// CHECK: return %[[NARROW]]
func.func @widen(%a: tensor<4xf32>) -> tensor<4xf64>
    attributes {tera.differentiable} {
  %0 = tera.convert %a : tensor<4xf32> -> tensor<4xf64>
  return %0 : tensor<4xf64>
}

// -----

// A broadcast is undone by summing the axes it copied along, and the reshape
// puts back the rank the sum dropped. Axis 0 was added and axis 2 was
// stretched from extent 1, so both are summed.
// CHECK-LABEL: func @broadcast_vjp
// CHECK: %[[SUM:.*]] = tera.reduce sum, %arg1 {dimensions = array<i64: 0, 2>}
// CHECK-SAME: tensor<2x3x4xf32> -> tensor<3xf32>
// CHECK: %[[BACK:.*]] = tera.reshape %[[SUM]] : tensor<3xf32> -> tensor<3x1xf32>
// CHECK: return %[[BACK]]
func.func @broadcast(%a: tensor<3x1xf32>) -> tensor<2x3x4xf32>
    attributes {tera.differentiable} {
  %0 = tera.broadcast_in_dim %a {broadcast_dimensions = array<i64: 1, 2>}
      : tensor<3x1xf32> -> tensor<2x3x4xf32>
  return %0 : tensor<2x3x4xf32>
}

// -----

// A transpose is undone by the inverse permutation, and a reshape by a reshape
// back to the operand type.
// CHECK-LABEL: func @layout_vjp
// CHECK: %[[UNSHAPE:.*]] = tera.reshape %arg1 : tensor<60xf32> -> tensor<5x3x4xf32>
// CHECK: %[[UNPERM:.*]] = tera.transpose %[[UNSHAPE]] {permutation = array<i64: 1, 2, 0>}
// CHECK-SAME: tensor<5x3x4xf32> -> tensor<3x4x5xf32>
// CHECK: return %[[UNPERM]]
func.func @layout(%a: tensor<3x4x5xf32>) -> tensor<60xf32>
    attributes {tera.differentiable} {
  %0 = tera.transpose %a {permutation = array<i64: 2, 0, 1>}
      : tensor<3x4x5xf32> -> tensor<5x3x4xf32>
  %1 = tera.reshape %0 : tensor<5x3x4xf32> -> tensor<60xf32>
  return %1 : tensor<60xf32>
}

// -----

// A strided window is undone by writing the adjoint back where it was read
// from: `low` and `high` put the offset and the tail back, and `interior`
// spaces the elements out by the stride. The window read elements 1, 3 and 5
// of six, so one goes before, none after, and one gap sits between each pair.
// CHECK-LABEL: func @strided_window_vjp
// CHECK: %[[ZERO:.*]] = tera.constant dense<0.000000e+00> : tensor<f32>
// CHECK: %[[GRAD:.*]] = tera.pad %arg1, %[[ZERO]]
// CHECK-SAME: high = array<i64: 0>
// CHECK-SAME: interior = array<i64: 1>
// CHECK-SAME: low = array<i64: 1>
// CHECK-SAME: (tensor<3xf32>, tensor<f32>) -> tensor<6xf32>
// CHECK: return %[[GRAD]]
func.func @strided_window(%a: tensor<6xf32>) -> tensor<3xf32>
    attributes {tera.differentiable} {
  %0 = tera.slice %a {start_indices = array<i64: 1>,
                      limit_indices = array<i64: 6>,
                      strides = array<i64: 2>}
      : tensor<6xf32> -> tensor<3xf32>
  return %0 : tensor<3xf32>
}

// -----

// Each input of a concat takes back the window of the adjoint it wrote.
// CHECK-LABEL: func @join_vjp
// CHECK: %[[FIRST:.*]] = tera.slice %arg2
// CHECK-SAME: limit_indices = array<i64: 2, 3>
// CHECK-SAME: start_indices = array<i64: 0, 0>
// CHECK: %[[SECOND:.*]] = tera.slice %arg2
// CHECK-SAME: limit_indices = array<i64: 6, 3>
// CHECK-SAME: start_indices = array<i64: 2, 0>
// CHECK: return %[[FIRST]], %[[SECOND]]
func.func @join(%a: tensor<2x3xf32>, %b: tensor<4x3xf32>) -> tensor<6x3xf32>
    attributes {tera.differentiable} {
  %0 = tera.concat %a, %b {dimension = 0 : i64}
      : tensor<2x3xf32>, tensor<4x3xf32> -> tensor<6x3xf32>
  return %0 : tensor<6x3xf32>
}

// -----

// A sum reduction is undone by a broadcast back along the axes it collapsed.
// CHECK-LABEL: func @total_vjp
// CHECK: %[[SPREAD:.*]] = tera.broadcast_in_dim %arg1
// CHECK-SAME: broadcast_dimensions = array<i64: 0>
// CHECK-SAME: tensor<2xf32> -> tensor<2x3xf32>
// CHECK: return %[[SPREAD]]
func.func @total(%a: tensor<2x3xf32>) -> tensor<2xf32>
    attributes {tera.differentiable} {
  %0 = tera.reduce sum, %a {dimensions = array<i64: 1>}
      : tensor<2x3xf32> -> tensor<2xf32>
  return %0 : tensor<2xf32>
}

// -----

// A product reduction reweights the adjoint by the result and divides by the
// operand, rather than recomputing the reduction once per element.
// CHECK-LABEL: func @running_product_vjp
// CHECK: %[[FWD:.*]] = tera.reduce product, %arg0
// CHECK: %[[SCALED:.*]] = tera.mul %arg1, %[[FWD]]
// CHECK: %[[SPREAD:.*]] = tera.broadcast_in_dim %[[SCALED]]
// CHECK: %[[GRAD:.*]] = tera.div %[[SPREAD]], %arg0
// CHECK: return %[[GRAD]]
func.func @running_product(%a: tensor<2x3xf32>) -> tensor<2xf32>
    attributes {tera.differentiable} {
  %0 = tera.reduce product, %a {dimensions = array<i64: 1>}
      : tensor<2x3xf32> -> tensor<2xf32>
  return %0 : tensor<2xf32>
}

// -----

// An extremum reduction routes the adjoint to every element that ties for it,
// which keeps the rule independent of the order the reduction ran in.
// CHECK-LABEL: func @smallest_vjp
// CHECK: %[[FWD:.*]] = tera.reduce minimum, %arg0
// CHECK: %[[SPREADFWD:.*]] = tera.broadcast_in_dim %[[FWD]]
// CHECK: %[[TIED:.*]] = tera.compare eq, %arg0, %[[SPREADFWD]]
// CHECK: %[[ZERO:.*]] = tera.constant dense<0.000000e+00>
// CHECK: %[[SPREAD:.*]] = tera.broadcast_in_dim %arg1
// CHECK: %[[GRAD:.*]] = tera.select %[[TIED]], %[[SPREAD]], %[[ZERO]]
// CHECK: return %[[GRAD]]
func.func @smallest(%a: tensor<2x3xf32>) -> tensor<2xf32>
    attributes {tera.differentiable} {
  %0 = tera.reduce minimum, %a {dimensions = array<i64: 1>}
      : tensor<2x3xf32> -> tensor<2xf32>
  return %0 : tensor<2xf32>
}

// -----

// A contraction differentiates into two more contractions, each against the
// other operand over the axes the forward pass left free there. The batch axis
// stays a batch axis in both, and the transpose puts each result back into the
// layout of the operand it belongs to.
// CHECK-LABEL: func @batched_vjp
// CHECK: %[[DLHS:.*]] = tera.dot %arg2, %arg1
// CHECK-SAME: lhs_batch = array<i64: 0>
// CHECK-SAME: lhs_contracting = array<i64: 2>
// CHECK-SAME: rhs_batch = array<i64: 0>
// CHECK-SAME: rhs_contracting = array<i64: 2>
// CHECK-SAME: (tensor<2x3x5xf32>, tensor<2x4x5xf32>) -> tensor<2x3x4xf32>
// CHECK: %[[LHS:.*]] = tera.transpose %[[DLHS]] {permutation = array<i64: 0, 1, 2>}
// CHECK: %[[DRHS:.*]] = tera.dot %arg2, %arg0
// CHECK-SAME: lhs_batch = array<i64: 0>
// CHECK-SAME: lhs_contracting = array<i64: 1>
// CHECK-SAME: rhs_batch = array<i64: 0>
// CHECK-SAME: rhs_contracting = array<i64: 1>
// CHECK-SAME: (tensor<2x3x5xf32>, tensor<2x3x4xf32>) -> tensor<2x5x4xf32>
// CHECK: %[[RHS:.*]] = tera.transpose %[[DRHS]] {permutation = array<i64: 0, 2, 1>}
// CHECK: return %[[LHS]], %[[RHS]]
func.func @batched(%a: tensor<2x3x4xf32>, %b: tensor<2x4x5xf32>)
    -> tensor<2x3x5xf32> attributes {tera.differentiable} {
  %0 = tera.dot %a, %b {lhs_batch = array<i64: 0>,
                        lhs_contracting = array<i64: 2>,
                        rhs_batch = array<i64: 0>,
                        rhs_contracting = array<i64: 1>}
      : (tensor<2x3x4xf32>, tensor<2x4x5xf32>) -> tensor<2x3x5xf32>
  return %0 : tensor<2x3x5xf32>
}

// -----

// A gather copies elements, so its derivative writes the adjoint back where it
// was read from and adds where a position was read twice -- which is a scatter,
// with every attribute crossing over unchanged. The indices take no gradient,
// so the vjp differentiates one argument out of two.
// CHECK-LABEL: func @lookup_vjp
// CHECK-SAME: attributes {tera.diff_args = array<i64: 0>}
// CHECK: %[[ZERO:.*]] = tera.constant dense<0.000000e+00> : tensor<5x3xf32>
// CHECK: %[[GRAD:.*]] = tera.scatter %[[ZERO]], %arg1, %arg2
// CHECK-SAME: index_vector_dim = 1 : i64
// CHECK-SAME: inserted_window_dims = array<i64: 0>
// CHECK-SAME: scatter_dims_to_operand_dims = array<i64: 0>
// CHECK-SAME: update_window_dims = array<i64: 1>
// CHECK: return %[[GRAD]]
func.func @lookup(%table: tensor<5x3xf32>, %ids: tensor<4xi32>) -> tensor<4x3xf32>
    attributes {tera.differentiable} {
  %0 = tera.gather %table, %ids {offset_dims = array<i64: 1>,
                                 collapsed_slice_dims = array<i64: 0>,
                                 start_index_map = array<i64: 0>,
                                 slice_sizes = array<i64: 1, 3>,
                                 index_vector_dim = 1 : i64}
      : (tensor<5x3xf32>, tensor<4xi32>) -> tensor<4x3xf32>
  return %0 : tensor<4x3xf32>
}

// -----

// The other direction. A scatter adds to what it was given and never scales it,
// so the operand takes the adjoint unchanged; the updates take back the window
// of the adjoint each was added into, which is a gather with the attributes
// read the other way round. `slice_sizes` is the width the window covered, so
// it is 1 on the inserted axis and the operand extent on the one it spans.
// CHECK-LABEL: func @write_back_vjp
// CHECK-SAME: attributes {tera.diff_args = array<i64: 0, 2>}
// CHECK: %[[DUPDATES:.*]] = tera.gather %arg3, %arg1
// CHECK-SAME: collapsed_slice_dims = array<i64: 0>
// CHECK-SAME: index_vector_dim = 1 : i64
// CHECK-SAME: offset_dims = array<i64: 1>
// CHECK-SAME: slice_sizes = array<i64: 1, 3>
// CHECK-SAME: start_index_map = array<i64: 0>
// CHECK: return %arg3, %[[DUPDATES]]
func.func @write_back(%table: tensor<5x3xf32>, %ids: tensor<4xi32>,
                      %updates: tensor<4x3xf32>) -> tensor<5x3xf32>
    attributes {tera.differentiable} {
  %0 = tera.scatter %table, %ids, %updates
      {update_window_dims = array<i64: 1>,
       inserted_window_dims = array<i64: 0>,
       scatter_dims_to_operand_dims = array<i64: 0>,
       index_vector_dim = 1 : i64}
      : (tensor<5x3xf32>, tensor<4xi32>, tensor<4x3xf32>) -> tensor<5x3xf32>
  return %0 : tensor<5x3xf32>
}

// -----

// A convolution differentiates into two more convolutions. The input's runs
// the adjoint back through the kernel read backwards -- a correlation undone
// is a convolution, which is what the flip is for -- and the kernel's swaps
// the two leading axes of both operands so the batch is what gets contracted.
//
// The stride is what the interior padding puts back: the forward pass skipped
// a position between windows, so the adjoint gets a hole between its elements
// before it is run backwards. In the second convolution the strides and the
// dilation change places instead.
// CHECK-LABEL: func @sliding_vjp
// CHECK: %[[ZERO:.*]] = tera.constant dense<0.000000e+00> : tensor<f32>
// CHECK: %[[SPACED:.*]] = tera.pad %arg2, %[[ZERO]]
// CHECK-SAME: interior = array<i64: 0, 0, 1, 1>
// CHECK: %[[SWAPPED:.*]] = tera.transpose %arg1 {permutation = array<i64: 1, 0, 2, 3>}
// CHECK: %[[FLIPPED:.*]] = tera.reverse %[[SWAPPED]] {dimensions = array<i64: 2, 3>}
// CHECK: %[[DINPUT:.*]] = tera.conv %[[SPACED]], %[[FLIPPED]]
// CHECK-SAME: padding = array<i64: 1, 2, 1, 2>
// CHECK-SAME: strides = array<i64: 1, 1>
// CHECK: %[[XT:.*]] = tera.transpose %arg0 {permutation = array<i64: 1, 0, 2, 3>}
// CHECK: %[[GT:.*]] = tera.transpose %arg2 {permutation = array<i64: 1, 0, 2, 3>}
// CHECK: %[[WIDE:.*]] = tera.conv %[[XT]], %[[GT]]
// CHECK-SAME: dilation = array<i64: 2, 2>
// CHECK-SAME: strides = array<i64: 1, 1>
// CHECK: %[[TRIM:.*]] = tera.slice %[[WIDE]]
// CHECK-SAME: limit_indices = array<i64: 2, 3, 3, 3>
// CHECK: %[[DKERNEL:.*]] = tera.transpose %[[TRIM]] {permutation = array<i64: 1, 0, 2, 3>}
// CHECK: return %[[DINPUT]], %[[DKERNEL]]
func.func @sliding(%x: tensor<1x2x8x8xf32>, %k: tensor<3x2x3x3xf32>)
    -> tensor<1x3x4x4xf32> attributes {tera.differentiable} {
  %0 = tera.conv %x, %k {strides = array<i64: 2, 2>,
                         padding = array<i64: 1, 1, 1, 1>,
                         dilation = array<i64: 1, 1>,
                         groups = 1 : i64}
      : (tensor<1x2x8x8xf32>, tensor<3x2x3x3xf32>) -> tensor<1x3x4x4xf32>
  return %0 : tensor<1x3x4x4xf32>
}

// -----

// A pool whose windows tile the input is undone by handing each window its
// adjoint back. Splitting each spatial axis into a window index and a position
// within it makes the adjoint's axes a subset of the result's, which is what a
// broadcast says; a reshape puts the halves back together.
//
// An average gives every element the same share of it.
// CHECK-LABEL: func @mean_window_vjp
// CHECK: %[[SPLIT:.*]] = tera.broadcast_in_dim %arg1 {broadcast_dimensions = array<i64: 0, 1, 2, 4>}
// CHECK-SAME: tensor<1x1x2x3xf32> -> tensor<1x1x2x2x3x2xf32>
// CHECK: %[[SHARED:.*]] = tera.reshape %[[SPLIT]] : tensor<1x1x2x2x3x2xf32> -> tensor<1x1x4x6xf32>
// CHECK: %[[COUNT:.*]] = tera.constant dense<4.000000e+00>
// CHECK: %[[GRAD:.*]] = tera.div %[[SHARED]], %[[COUNT]]
// CHECK: return %[[GRAD]]
func.func @mean_window(%x: tensor<1x1x4x6xf32>) -> tensor<1x1x2x3xf32>
    attributes {tera.differentiable} {
  %0 = tera.pool2d average, %x {kernel_size = array<i64: 2, 2>,
                                strides = array<i64: 2, 2>,
                                padding = array<i64: 0, 0, 0, 0>}
      : tensor<1x1x4x6xf32> -> tensor<1x1x2x3xf32>
  return %0 : tensor<1x1x2x3xf32>
}

// -----

// A maximum gives its whole share to the one element the window chose, and
// comparing against the answer is how that element is found again -- the same
// rule a max reduction uses, over a window rather than an axis.
// CHECK-LABEL: func @largest_window_vjp
// CHECK: %[[FWD:.*]] = tera.pool2d max, %arg0
// CHECK: %[[SHARED:.*]] = tera.reshape
// CHECK: %[[SPREAD:.*]] = tera.broadcast_in_dim %[[FWD]]
// CHECK: %[[BACK:.*]] = tera.reshape %[[SPREAD]]
// CHECK: %[[CHOSEN:.*]] = tera.compare eq, %arg0, %[[BACK]]
// CHECK: %[[NONE:.*]] = tera.constant dense<0.000000e+00> : tensor<1x1x4x6xf32>
// CHECK: %[[GRAD:.*]] = tera.select %[[CHOSEN]], %[[SHARED]], %[[NONE]]
// CHECK: return %[[GRAD]]
func.func @largest_window(%x: tensor<1x1x4x6xf32>) -> tensor<1x1x2x3xf32>
    attributes {tera.differentiable} {
  %0 = tera.pool2d max, %x {kernel_size = array<i64: 2, 2>,
                            strides = array<i64: 2, 2>,
                            padding = array<i64: 0, 0, 0, 0>}
      : tensor<1x1x4x6xf32> -> tensor<1x1x2x3xf32>
  return %0 : tensor<1x1x2x3xf32>
}

// -----

// Reading an axis from the far end is its own inverse, and padding is undone
// by reading back out the window the operand landed in -- which is the strided
// slice `tera.slice` takes, so the two rules are each other's.
// CHECK-LABEL: func @spaced_vjp
// CHECK: %[[BACK:.*]] = tera.reverse %arg2 {dimensions = array<i64: 0>}
// CHECK: %[[GRAD:.*]] = tera.slice %[[BACK]]
// CHECK-SAME: limit_indices = array<i64: 6>
// CHECK-SAME: start_indices = array<i64: 1>
// CHECK-SAME: strides = array<i64: 2>
// CHECK: return %[[GRAD]]
func.func @spaced(%x: tensor<3xf32>, %v: tensor<f32>) -> tensor<7xf32>
    attributes {tera.differentiable} {
  %0 = tera.pad %x, %v {low = array<i64: 1>, high = array<i64: 1>,
                        interior = array<i64: 1>}
      : (tensor<3xf32>, tensor<f32>) -> tensor<7xf32>
  %1 = tera.reverse %0 {dimensions = array<i64: 0>}
      : tensor<7xf32> -> tensor<7xf32>
  return %1 : tensor<7xf32>
}
