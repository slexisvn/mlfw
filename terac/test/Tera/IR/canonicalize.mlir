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
