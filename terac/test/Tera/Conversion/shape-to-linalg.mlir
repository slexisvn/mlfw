// RUN: tera-opt %s --convert-tera-to-linalg | FileCheck %s

// A broadcast that only adds axes reads the operand through a rank-reducing
// map; nothing is stretched.
// CHECK: #[[READ:.*]] = affine_map<(d0, d1) -> (d1)>
// CHECK: #[[IDENTITY:.*]] = affine_map<(d0, d1) -> (d0, d1)>
// A stretched axis reads index 0 forever, which is the constant expression.
// CHECK: #[[STRETCH:.*]] = affine_map<(d0, d1) -> (d0, 0)>

// CHECK-LABEL: func @broadcast_new_axis
func.func @broadcast_new_axis(%a: tensor<3xf32>) -> tensor<2x3xf32> {
  // CHECK: linalg.generic
  // CHECK-SAME: indexing_maps = [#[[READ]], #[[IDENTITY]]]
  // CHECK-SAME: iterator_types = ["parallel", "parallel"]
  %0 = tera.broadcast_in_dim %a {broadcast_dimensions = array<i64: 1>}
      : tensor<3xf32> -> tensor<2x3xf32>
  return %0 : tensor<2x3xf32>
}

// CHECK-LABEL: func @broadcast_stretch
func.func @broadcast_stretch(%a: tensor<2x1xf32>) -> tensor<2x3xf32> {
  // CHECK: linalg.generic
  // CHECK-SAME: indexing_maps = [#[[STRETCH]], #[[IDENTITY]]]
  %0 = tera.broadcast_in_dim %a {broadcast_dimensions = array<i64: 0, 1>}
      : tensor<2x1xf32> -> tensor<2x3xf32>
  return %0 : tensor<2x3xf32>
}

// linalg.transpose reads dim(result, i) = dim(input, permutation[i]), so the
// permutation carries over unchanged.
// CHECK-LABEL: func @transpose
func.func @transpose(%a: tensor<2x3x4xf32>) -> tensor<4x2x3xf32> {
  // CHECK: %[[E:.*]] = tensor.empty() : tensor<4x2x3xf32>
  // CHECK: linalg.transpose ins(%arg0 : tensor<2x3x4xf32>) outs(%[[E]] : tensor<4x2x3xf32>) permutation = [2, 0, 1]
  %0 = tera.transpose %a {permutation = array<i64: 2, 0, 1>}
      : tensor<2x3x4xf32> -> tensor<4x2x3xf32>
  return %0 : tensor<4x2x3xf32>
}

// A reshape that only folds adjacent axes together is a view, so it becomes
// `tensor.collapse_shape` rather than a `tensor.reshape` that copies.
// CHECK-LABEL: func @reshape
func.func @reshape(%a: tensor<2x3xf32>) -> tensor<6xf32> {
  // CHECK: tensor.collapse_shape %arg0 {{\[}}[0, 1]] : tensor<2x3xf32> into tensor<6xf32>
  %0 = tera.reshape %a : tensor<2x3xf32> -> tensor<6xf32>
  return %0 : tensor<6xf32>
}

// CHECK-LABEL: func @reshape_to_scalar
func.func @reshape_to_scalar(%a: tensor<1x1xf32>) -> tensor<f32> {
  // CHECK: tensor.collapse_shape %arg0 [] : tensor<1x1xf32> into tensor<f32>
  %0 = tera.reshape %a : tensor<1x1xf32> -> tensor<f32>
  return %0 : tensor<f32>
}

// A reshape that moves data between axes is not a view, and keeps the copy.
// CHECK-LABEL: func @reshape_across_axes
func.func @reshape_across_axes(%a: tensor<2x3xf32>) -> tensor<3x2xf32> {
  // CHECK: %[[S:.*]] = arith.constant dense<[3, 2]> : tensor<2xindex>
  // CHECK: tensor.reshape %arg0(%[[S]]) : (tensor<2x3xf32>, tensor<2xindex>) -> tensor<3x2xf32>
  %0 = tera.reshape %a : tensor<2x3xf32> -> tensor<3x2xf32>
  return %0 : tensor<3x2xf32>
}

// The slice sizes come from the result type, which inference already computed
// from the bounds and the stride.
// CHECK-LABEL: func @slice
func.func @slice(%a: tensor<8x6xf32>) -> tensor<3x2xf32> {
  // CHECK: tensor.extract_slice %arg0[1, 1] [3, 2] [2, 2] : tensor<8x6xf32> to tensor<3x2xf32>
  %0 = tera.slice %a {start_indices = array<i64: 1, 1>,
                      limit_indices = array<i64: 7, 5>,
                      strides = array<i64: 2, 2>}
      : tensor<8x6xf32> -> tensor<3x2xf32>
  return %0 : tensor<3x2xf32>
}

// Each input lands in its own band of one destination, at a running offset.
// CHECK-LABEL: func @concat
func.func @concat(%a: tensor<2x3xf32>, %b: tensor<4x3xf32>) -> tensor<6x3xf32> {
  // CHECK: %[[D:.*]] = tensor.empty() : tensor<6x3xf32>
  // CHECK: %[[A:.*]] = tensor.insert_slice %arg0 into %[[D]][0, 0] [2, 3] [1, 1]
  // CHECK: tensor.insert_slice %arg1 into %[[A]][2, 0] [4, 3] [1, 1]
  %0 = tera.concat %a, %b {dimension = 0 : i64}
      : tensor<2x3xf32>, tensor<4x3xf32> -> tensor<6x3xf32>
  return %0 : tensor<6x3xf32>
}
