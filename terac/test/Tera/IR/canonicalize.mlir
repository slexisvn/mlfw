// RUN: tera-opt %s --canonicalize | FileCheck %s

// CHECK-LABEL: func @transpose_identity
// CHECK-SAME:    (%[[A:.*]]: tensor<2x3xf32>)
// CHECK-NEXT:    return %[[A]]
func.func @transpose_identity(%a: tensor<2x3xf32>) -> tensor<2x3xf32> {
  %0 = tera.transpose %a {permutation = array<i64: 0, 1>}
      : tensor<2x3xf32> -> tensor<2x3xf32>
  return %0 : tensor<2x3xf32>
}

// Two transposes collapse into the composed permutation: inner[outer[i]].
// CHECK-LABEL: func @transpose_of_transpose
// CHECK-SAME:    (%[[A:.*]]: tensor<2x3x4xf32>)
// CHECK-NEXT:    %[[T:.*]] = tera.transpose %[[A]] {permutation = array<i64: 2, 0, 1>}
// CHECK-NEXT:    return %[[T]]
func.func @transpose_of_transpose(%a: tensor<2x3x4xf32>) -> tensor<4x2x3xf32> {
  %0 = tera.transpose %a {permutation = array<i64: 1, 2, 0>}
      : tensor<2x3x4xf32> -> tensor<3x4x2xf32>
  %1 = tera.transpose %0 {permutation = array<i64: 1, 2, 0>}
      : tensor<3x4x2xf32> -> tensor<4x2x3xf32>
  return %1 : tensor<4x2x3xf32>
}

// CHECK-LABEL: func @reshape_identity
// CHECK-SAME:    (%[[A:.*]]: tensor<6xf32>)
// CHECK-NEXT:    return %[[A]]
func.func @reshape_identity(%a: tensor<6xf32>) -> tensor<6xf32> {
  %0 = tera.reshape %a : tensor<6xf32> -> tensor<6xf32>
  return %0 : tensor<6xf32>
}

// CHECK-LABEL: func @reshape_of_reshape
// CHECK-SAME:    (%[[A:.*]]: tensor<2x3xf32>)
// CHECK-NEXT:    %[[R:.*]] = tera.reshape %[[A]] : tensor<2x3xf32> -> tensor<3x2xf32>
// CHECK-NEXT:    return %[[R]]
func.func @reshape_of_reshape(%a: tensor<2x3xf32>) -> tensor<3x2xf32> {
  %0 = tera.reshape %a : tensor<2x3xf32> -> tensor<6xf32>
  %1 = tera.reshape %0 : tensor<6xf32> -> tensor<3x2xf32>
  return %1 : tensor<3x2xf32>
}

// CHECK-LABEL: func @convert_identity
// CHECK-SAME:    (%[[A:.*]]: tensor<4xf32>)
// CHECK-NEXT:    return %[[A]]
func.func @convert_identity(%a: tensor<4xf32>) -> tensor<4xf32> {
  %0 = tera.convert %a : tensor<4xf32> -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// Pure ops with no user must not survive.
// CHECK-LABEL: func @dead_code
// CHECK-NOT: tera.
func.func @dead_code(%a: tensor<2x3xf32>) {
  %0 = tera.exp %a : tensor<2x3xf32>
  %1 = tera.neg %0 : tensor<2x3xf32>
  return
}

// Reversing an axis with one element in it reads the same element, so the op
// is the tensor it was given. Reversing no axes at all is the same thing said
// with an empty list.
// CHECK-LABEL: func @reverse_of_one
// CHECK-SAME:    (%[[A:.*]]: tensor<3x1xf32>)
// CHECK-NEXT:    return %[[A]]
func.func @reverse_of_one(%a: tensor<3x1xf32>) -> tensor<3x1xf32> {
  %0 = tera.reverse %a {dimensions = array<i64: 1>}
      : tensor<3x1xf32> -> tensor<3x1xf32>
  return %0 : tensor<3x1xf32>
}

// An axis with more than one element in it is not the same read, so this one
// stays.
// CHECK-LABEL: func @reverse_of_many
// CHECK: tera.reverse
func.func @reverse_of_many(%a: tensor<3x2xf32>) -> tensor<3x2xf32> {
  %0 = tera.reverse %a {dimensions = array<i64: 1>}
      : tensor<3x2xf32> -> tensor<3x2xf32>
  return %0 : tensor<3x2xf32>
}

// Padding by nothing is the tensor it was given, whatever it was going to pad
// with. The rule reads the shape rather than the attributes, so a pad that
// spaces an axis of one element out -- which also adds nothing -- goes too.
// CHECK-LABEL: func @pad_nothing
// CHECK-SAME:    (%[[A:.*]]: tensor<1x3xf32>
// CHECK-NEXT:    return %[[A]]
func.func @pad_nothing(%a: tensor<1x3xf32>, %v: tensor<f32>) -> tensor<1x3xf32> {
  %0 = tera.pad %a, %v {low = array<i64: 0, 0>, high = array<i64: 0, 0>,
                        interior = array<i64: 4, 0>}
      : (tensor<1x3xf32>, tensor<f32>) -> tensor<1x3xf32>
  return %0 : tensor<1x3xf32>
}

// CHECK-LABEL: func @pad_something
// CHECK: tera.pad
func.func @pad_something(%a: tensor<2x3xf32>, %v: tensor<f32>) -> tensor<3x3xf32> {
  %0 = tera.pad %a, %v {low = array<i64: 1, 0>, high = array<i64: 0, 0>}
      : (tensor<2x3xf32>, tensor<f32>) -> tensor<3x3xf32>
  return %0 : tensor<3x3xf32>
}

// A transpose in front of a `dot` is a relabelling of axes the dot already
// names, so it is absorbed into those names: the contracting axis moves from 1
// to 0 and the tensor is never materialised.
// CHECK-LABEL: func @transpose_into_dot
// CHECK-SAME:    (%[[A:.*]]: tensor<4x3xf32>, %[[B:.*]]: tensor<4x2xf32>)
// CHECK-NOT: tera.transpose
// CHECK: tera.dot %[[A]], %[[B]]
// CHECK-SAME: lhs_contracting = array<i64: 0>
func.func @transpose_into_dot(%a: tensor<4x3xf32>, %b: tensor<4x2xf32>)
    -> tensor<3x2xf32> {
  %t = tera.transpose %a {permutation = array<i64: 1, 0>}
      : tensor<4x3xf32> -> tensor<3x4xf32>
  %0 = tera.dot %t, %b {lhs_batch = array<i64>, lhs_contracting = array<i64: 1>,
                        rhs_batch = array<i64>, rhs_contracting = array<i64: 0>}
      : (tensor<3x4xf32>, tensor<4x2xf32>) -> tensor<3x2xf32>
  return %0 : tensor<3x2xf32>
}

// The batch axes are a pairing, so they move through the permutation with
// their positions kept and the pairing with the rhs survives.
// CHECK-LABEL: func @transpose_into_batched_dot
// CHECK-NOT: tera.transpose
// CHECK: tera.dot
// CHECK-SAME: lhs_contracting = array<i64: 1>
func.func @transpose_into_batched_dot(%a: tensor<2x4x3xf32>,
                                      %b: tensor<2x4x5xf32>)
    -> tensor<2x3x5xf32> {
  %t = tera.transpose %a {permutation = array<i64: 0, 2, 1>}
      : tensor<2x4x3xf32> -> tensor<2x3x4xf32>
  %0 = tera.dot %t, %b {lhs_batch = array<i64: 0>,
                        lhs_contracting = array<i64: 2>,
                        rhs_batch = array<i64: 0>,
                        rhs_contracting = array<i64: 1>}
      : (tensor<2x3x4xf32>, tensor<2x4x5xf32>) -> tensor<2x3x5xf32>
  return %0 : tensor<2x3x5xf32>
}

// The free axes are the ones whose order in the operand the result inherits,
// so a permutation that swaps two of them is not a relabelling and the
// transpose stays where it is.
// CHECK-LABEL: func @transpose_reorders_the_free_axes
// CHECK: tera.transpose
// CHECK: tera.dot
func.func @transpose_reorders_the_free_axes(%a: tensor<2x3x4xf32>,
                                            %b: tensor<4x5xf32>)
    -> tensor<3x2x5xf32> {
  %t = tera.transpose %a {permutation = array<i64: 1, 0, 2>}
      : tensor<2x3x4xf32> -> tensor<3x2x4xf32>
  %0 = tera.dot %t, %b {lhs_batch = array<i64>, lhs_contracting = array<i64: 2>,
                        rhs_batch = array<i64>, rhs_contracting = array<i64: 0>}
      : (tensor<3x2x4xf32>, tensor<4x5xf32>) -> tensor<3x2x5xf32>
  return %0 : tensor<3x2x5xf32>
}

// Each broadcast maps operand axes to result axes strictly increasingly, so
// following one by the other does too and the two are one.
// CHECK-LABEL: func @broadcast_of_broadcast
// CHECK-SAME:    (%[[X:.*]]: tensor<3xf32>)
// CHECK-NEXT:    %[[B:.*]] = tera.broadcast_in_dim %[[X]] {broadcast_dimensions = array<i64: 1>}
// CHECK-NEXT:    return %[[B]]
func.func @broadcast_of_broadcast(%x: tensor<3xf32>) -> tensor<2x3x4xf32> {
  %0 = tera.broadcast_in_dim %x {broadcast_dimensions = array<i64: 0>}
      : tensor<3xf32> -> tensor<3x4xf32>
  %1 = tera.broadcast_in_dim %0 {broadcast_dimensions = array<i64: 1, 2>}
      : tensor<3x4xf32> -> tensor<2x3x4xf32>
  return %1 : tensor<2x3x4xf32>
}

// CHECK-LABEL: func @broadcast_that_broadcasts_nothing
// CHECK-SAME:    (%[[X:.*]]: tensor<2x3xf32>)
// CHECK-NEXT:    return %[[X]]
func.func @broadcast_that_broadcasts_nothing(%x: tensor<2x3xf32>)
    -> tensor<2x3xf32> {
  %0 = tera.broadcast_in_dim %x {broadcast_dimensions = array<i64: 0, 1>}
      : tensor<2x3xf32> -> tensor<2x3xf32>
  return %0 : tensor<2x3xf32>
}

// Reversing an axis twice reads it from the near end again, so what is left is
// the axes named an odd number of times.
// CHECK-LABEL: func @reverse_of_reverse
// CHECK-SAME:    (%[[X:.*]]: tensor<2x3x4xf32>)
// CHECK-NEXT:    %[[R:.*]] = tera.reverse %[[X]] {dimensions = array<i64: 0, 2>}
// CHECK-NEXT:    return %[[R]]
func.func @reverse_of_reverse(%x: tensor<2x3x4xf32>) -> tensor<2x3x4xf32> {
  %0 = tera.reverse %x {dimensions = array<i64: 0, 1>}
      : tensor<2x3x4xf32> -> tensor<2x3x4xf32>
  %1 = tera.reverse %0 {dimensions = array<i64: 1, 2>}
      : tensor<2x3x4xf32> -> tensor<2x3x4xf32>
  return %1 : tensor<2x3x4xf32>
}

// CHECK-LABEL: func @concat_of_one
// CHECK-SAME:    (%[[X:.*]]: tensor<2x3xf32>)
// CHECK-NEXT:    return %[[X]]
func.func @concat_of_one(%x: tensor<2x3xf32>) -> tensor<2x3xf32> {
  %0 = tera.concat %x {dimension = 0 : i64}
      : tensor<2x3xf32> -> tensor<2x3xf32>
  return %0 : tensor<2x3xf32>
}

// CHECK-LABEL: func @select_on_a_settled_predicate
// CHECK-SAME:    (%[[A:.*]]: tensor<4xf32>, %[[B:.*]]: tensor<4xf32>)
// CHECK-NEXT:    return %[[A]]
func.func @select_on_a_settled_predicate(%a: tensor<4xf32>, %b: tensor<4xf32>)
    -> tensor<4xf32> {
  %p = tera.constant dense<true> : tensor<4xi1>
  %0 = tera.select %p, %a, %b : tensor<4xi1>, tensor<4xf32>
  return %0 : tensor<4xf32>
}
