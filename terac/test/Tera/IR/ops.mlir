// RUN: tera-opt %s | tera-opt | FileCheck %s

// CHECK-LABEL: func @elementwise
func.func @elementwise(%a: tensor<2x3xf32>, %b: tensor<2x3xf32>) -> tensor<2x3xf32> {
  // CHECK: %[[ADD:.*]] = tera.add %arg0, %arg1 : tensor<2x3xf32>
  %0 = tera.add %a, %b : tensor<2x3xf32>
  // CHECK: %[[SUB:.*]] = tera.sub %[[ADD]], %arg1 : tensor<2x3xf32>
  %1 = tera.sub %0, %b : tensor<2x3xf32>
  // CHECK: %[[MUL:.*]] = tera.mul %[[SUB]], %arg0 : tensor<2x3xf32>
  %2 = tera.mul %1, %a : tensor<2x3xf32>
  // CHECK: %[[DIV:.*]] = tera.div %[[MUL]], %arg1 : tensor<2x3xf32>
  %3 = tera.div %2, %b : tensor<2x3xf32>
  // CHECK: %[[MAX:.*]] = tera.maximum %[[DIV]], %arg0 : tensor<2x3xf32>
  %4 = tera.maximum %3, %a : tensor<2x3xf32>
  // CHECK: %[[NEG:.*]] = tera.neg %[[MAX]] : tensor<2x3xf32>
  %5 = tera.neg %4 : tensor<2x3xf32>
  // CHECK: tera.exp %[[NEG]] : tensor<2x3xf32>
  %6 = tera.exp %5 : tensor<2x3xf32>
  return %6 : tensor<2x3xf32>
}

// CHECK-LABEL: func @constant
func.func @constant() -> tensor<2xf32> {
  // CHECK: tera.constant dense<[1.000000e+00, 2.000000e+00]> : tensor<2xf32>
  %0 = tera.constant dense<[1.0, 2.0]> : tensor<2xf32>
  return %0 : tensor<2xf32>
}

// CHECK-LABEL: func @broadcast
func.func @broadcast(%a: tensor<3xf32>) -> tensor<2x3xf32> {
  // CHECK: tera.broadcast_in_dim %arg0 {broadcast_dimensions = array<i64: 1>} : tensor<3xf32> -> tensor<2x3xf32>
  %0 = tera.broadcast_in_dim %a {broadcast_dimensions = array<i64: 1>} : tensor<3xf32> -> tensor<2x3xf32>
  return %0 : tensor<2x3xf32>
}

// CHECK-LABEL: func @shape_ops
func.func @shape_ops(%a: tensor<2x3x4xf32>) -> tensor<24xf32> {
  // CHECK: %[[T:.*]] = tera.transpose %arg0 {permutation = array<i64: 2, 0, 1>} : tensor<2x3x4xf32> -> tensor<4x2x3xf32>
  %0 = tera.transpose %a {permutation = array<i64: 2, 0, 1>} : tensor<2x3x4xf32> -> tensor<4x2x3xf32>
  // CHECK: tera.reshape %[[T]] : tensor<4x2x3xf32> -> tensor<24xf32>
  %1 = tera.reshape %0 : tensor<4x2x3xf32> -> tensor<24xf32>
  return %1 : tensor<24xf32>
}

// CHECK-LABEL: func @slice
func.func @slice(%a: tensor<8x6xf32>) -> tensor<3x2xf32> {
  // CHECK: tera.slice %arg0 {limit_indices = array<i64: 7, 5>, start_indices = array<i64: 1, 1>, strides = array<i64: 2, 2>}
  %0 = tera.slice %a {start_indices = array<i64: 1, 1>,
                      limit_indices = array<i64: 7, 5>,
                      strides = array<i64: 2, 2>}
      : tensor<8x6xf32> -> tensor<3x2xf32>
  return %0 : tensor<3x2xf32>
}

// CHECK-LABEL: func @concat
func.func @concat(%a: tensor<2x3xf32>, %b: tensor<4x3xf32>) -> tensor<6x3xf32> {
  // CHECK: tera.concat %arg0, %arg1 {dimension = 0 : i64}
  %0 = tera.concat %a, %b {dimension = 0 : i64}
      : tensor<2x3xf32>, tensor<4x3xf32> -> tensor<6x3xf32>
  return %0 : tensor<6x3xf32>
}

// The two shapes a lookup takes. An embedding reads a whole row, so the row
// axis collapses and the width axis survives as an offset, and one position is
// a single number -- which is what an `index_vector_dim` equal to the index
// tensor's rank says. Indexing along an axis reads one element, so every axis
// collapses, nothing survives as an offset, and the coordinates are a real axis.

// CHECK-LABEL: func @gather_rows
func.func @gather_rows(%table: tensor<10x4xf32>, %ids: tensor<2x3xi32>) -> tensor<2x3x4xf32> {
  // CHECK: tera.gather %arg0, %arg1 {collapsed_slice_dims = array<i64: 0>, index_vector_dim = 2 : i64, offset_dims = array<i64: 2>, slice_sizes = array<i64: 1, 4>, start_index_map = array<i64: 0>}
  %0 = tera.gather %table, %ids {offset_dims = array<i64: 2>,
                                 collapsed_slice_dims = array<i64: 0>,
                                 start_index_map = array<i64: 0>,
                                 slice_sizes = array<i64: 1, 4>,
                                 index_vector_dim = 2 : i64}
      : (tensor<10x4xf32>, tensor<2x3xi32>) -> tensor<2x3x4xf32>
  return %0 : tensor<2x3x4xf32>
}

// CHECK-LABEL: func @gather_elements
func.func @gather_elements(%x: tensor<3x4xf32>, %at: tensor<3x3x2xi32>) -> tensor<3x3xf32> {
  // CHECK: tera.gather %arg0, %arg1 {collapsed_slice_dims = array<i64: 0, 1>, index_vector_dim = 2 : i64, offset_dims = array<i64>, slice_sizes = array<i64: 1, 1>, start_index_map = array<i64: 0, 1>}
  %0 = tera.gather %x, %at {offset_dims = array<i64>,
                            collapsed_slice_dims = array<i64: 0, 1>,
                            start_index_map = array<i64: 0, 1>,
                            slice_sizes = array<i64: 1, 1>,
                            index_vector_dim = 2 : i64}
      : (tensor<3x4xf32>, tensor<3x3x2xi32>) -> tensor<3x3xf32>
  return %0 : tensor<3x3xf32>
}

// CHECK-LABEL: func @scatter_rows
func.func @scatter_rows(%table: tensor<10x4xf32>, %ids: tensor<2x3xi32>,
                        %updates: tensor<2x3x4xf32>) -> tensor<10x4xf32> {
  // CHECK: tera.scatter %arg0, %arg1, %arg2 {index_vector_dim = 2 : i64, inserted_window_dims = array<i64: 0>, scatter_dims_to_operand_dims = array<i64: 0>, update_window_dims = array<i64: 2>}
  %0 = tera.scatter %table, %ids, %updates
      {update_window_dims = array<i64: 2>,
       inserted_window_dims = array<i64: 0>,
       scatter_dims_to_operand_dims = array<i64: 0>,
       index_vector_dim = 2 : i64}
      : (tensor<10x4xf32>, tensor<2x3xi32>, tensor<2x3x4xf32>) -> tensor<10x4xf32>
  return %0 : tensor<10x4xf32>
}

// CHECK-LABEL: func @scatter_elements
func.func @scatter_elements(%x: tensor<3x4xf32>, %at: tensor<3x3x2xi32>,
                            %updates: tensor<3x3xf32>) -> tensor<3x4xf32> {
  // CHECK: tera.scatter %arg0, %arg1, %arg2 {index_vector_dim = 2 : i64, inserted_window_dims = array<i64: 0, 1>, scatter_dims_to_operand_dims = array<i64: 0, 1>, update_window_dims = array<i64>}
  %0 = tera.scatter %x, %at, %updates
      {update_window_dims = array<i64>,
       inserted_window_dims = array<i64: 0, 1>,
       scatter_dims_to_operand_dims = array<i64: 0, 1>,
       index_vector_dim = 2 : i64}
      : (tensor<3x4xf32>, tensor<3x3x2xi32>, tensor<3x3xf32>) -> tensor<3x4xf32>
  return %0 : tensor<3x4xf32>
}

// CHECK-LABEL: func @iota
func.func @iota() -> tensor<2x3xi32> {
  // CHECK: tera.iota {iota_dimension = 1 : i64} : tensor<2x3xi32>
  %0 = tera.iota {iota_dimension = 1 : i64} : tensor<2x3xi32>
  return %0 : tensor<2x3xi32>
}

// CHECK-LABEL: func @dot
func.func @dot(%a: tensor<8x2x4xf32>, %b: tensor<8x4x3xf32>) -> tensor<8x2x3xf32> {
  // CHECK: tera.dot %arg0, %arg1 {lhs_batch = array<i64: 0>, lhs_contracting = array<i64: 2>, rhs_batch = array<i64: 0>, rhs_contracting = array<i64: 1>}
  %0 = tera.dot %a, %b {lhs_batch = array<i64: 0>,
                        lhs_contracting = array<i64: 2>,
                        rhs_batch = array<i64: 0>,
                        rhs_contracting = array<i64: 1>}
      : (tensor<8x2x4xf32>, tensor<8x4x3xf32>) -> tensor<8x2x3xf32>
  return %0 : tensor<8x2x3xf32>
}

// CHECK-LABEL: func @reduce
func.func @reduce(%a: tensor<2x3x4xf32>) -> tensor<3xf32> {
  // CHECK: tera.reduce maximum, %arg0 {dimensions = array<i64: 0, 2>} : tensor<2x3x4xf32> -> tensor<3xf32>
  %0 = tera.reduce maximum, %a {dimensions = array<i64: 0, 2>}
      : tensor<2x3x4xf32> -> tensor<3xf32>
  return %0 : tensor<3xf32>
}

// CHECK-LABEL: func @predicates
func.func @predicates(%a: tensor<4xf32>, %b: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: %[[P:.*]] = tera.compare lt, %arg0, %arg1 : tensor<4xf32> -> tensor<4xi1>
  %0 = tera.compare lt, %a, %b : tensor<4xf32> -> tensor<4xi1>
  // CHECK: tera.select %[[P]], %arg0, %arg1 : tensor<4xi1>, tensor<4xf32>
  %1 = tera.select %0, %a, %b : tensor<4xi1>, tensor<4xf32>
  return %1 : tensor<4xf32>
}

// CHECK-LABEL: func @convert
func.func @convert(%a: tensor<4xf32>) -> tensor<4xf16> {
  // CHECK: tera.convert %arg0 : tensor<4xf32> -> tensor<4xf16>
  %0 = tera.convert %a : tensor<4xf32> -> tensor<4xf16>
  return %0 : tensor<4xf16>
}

// CHECK-LABEL: func @stop_gradient
func.func @stop_gradient(%a: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: tera.stop_gradient %arg0 : tensor<4xf32>
  %0 = tera.stop_gradient %a : tensor<4xf32>
  return %0 : tensor<4xf32>
}

// CHECK-LABEL: func @scan
func.func @scan(%init: tensor<2xf32>, %xs: tensor<4x2xf32>, %w: tensor<2xf32>)
    -> (tensor<2xf32>, tensor<4x2xf32>) {
  // CHECK: tera.scan init(%arg0 : tensor<2xf32>) xs(%arg1 : tensor<4x2xf32>) consts(%arg2 : tensor<2xf32>) -> (tensor<2xf32>, tensor<4x2xf32>)
  %carry, %ys = tera.scan init(%init : tensor<2xf32>) xs(%xs : tensor<4x2xf32>)
      consts(%w : tensor<2xf32>) -> (tensor<2xf32>, tensor<4x2xf32>) {
  // CHECK: ^bb0(%[[H:.*]]: tensor<2xf32>, %[[X:.*]]: tensor<2xf32>, %[[W:.*]]: tensor<2xf32>):
  ^bb0(%h: tensor<2xf32>, %x: tensor<2xf32>, %weight: tensor<2xf32>):
    %0 = tera.mul %h, %weight : tensor<2xf32>
    %1 = tera.add %0, %x : tensor<2xf32>
    // CHECK: tera.yield
    tera.yield %1, %1 : tensor<2xf32>, tensor<2xf32>
  }
  return %carry, %ys : tensor<2xf32>, tensor<4x2xf32>
}

// CHECK-LABEL: func @scan_reverse
func.func @scan_reverse(%init: tensor<f32>, %xs: tensor<4xf32>) -> tensor<f32> {
  // CHECK: tera.scan reverse init(%arg0 : tensor<f32>) xs(%arg1 : tensor<4xf32>) -> (tensor<f32>)
  %carry = tera.scan reverse init(%init : tensor<f32>) xs(%xs : tensor<4xf32>)
      -> (tensor<f32>) {
  ^bb0(%acc: tensor<f32>, %x: tensor<f32>):
    %0 = tera.add %acc, %x : tensor<f32>
    tera.yield %0 : tensor<f32>
  }
  return %carry : tensor<f32>
}

// CHECK-LABEL: func @branch
func.func @branch(%p: tensor<i1>, %x: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: tera.if %arg0, %arg1 : (tensor<i1>, tensor<4xf32>) -> tensor<4xf32>
  %0 = tera.if %p, %x : (tensor<i1>, tensor<4xf32>) -> tensor<4xf32> {
  ^bb0(%a: tensor<4xf32>):
    %1 = tera.exp %a : tensor<4xf32>
    tera.yield %1 : tensor<4xf32>
  // CHECK: } else {
  } else {
  ^bb0(%a: tensor<4xf32>):
    tera.yield %a : tensor<4xf32>
  }
  return %0 : tensor<4xf32>
}

// A convolution and the two things a pool does with a window. The layout is
// NCHW and OIHW because that is what the ops mean, so it is not written down.

// CHECK-LABEL: func @conv
func.func @conv(%x: tensor<1x2x5x5xf32>, %k: tensor<3x2x2x2xf32>) -> tensor<1x3x4x4xf32> {
  // CHECK: tera.conv %arg0, %arg1 {dilation = array<i64: 1, 1>, groups = 1 : i64, padding = array<i64: 0, 0, 0, 0>, strides = array<i64: 1, 1>}
  %0 = tera.conv %x, %k {strides = array<i64: 1, 1>,
                         padding = array<i64: 0, 0, 0, 0>,
                         dilation = array<i64: 1, 1>,
                         groups = 1 : i64}
      : (tensor<1x2x5x5xf32>, tensor<3x2x2x2xf32>) -> tensor<1x3x4x4xf32>
  return %0 : tensor<1x3x4x4xf32>
}

// Padding, striding and dilation together, and a rectangular window, so the
// count of windows along each axis is a different number.
// CHECK-LABEL: func @conv_strided
func.func @conv_strided(%x: tensor<2x1x7x9xf32>, %k: tensor<4x1x3x2xf32>) -> tensor<2x4x3x5xf32> {
  // CHECK: tera.conv %arg0, %arg1 {dilation = array<i64: 2, 1>, groups = 1 : i64, padding = array<i64: 1, 1, 1, 1>, strides = array<i64: 2, 2>}
  %0 = tera.conv %x, %k {strides = array<i64: 2, 2>,
                         padding = array<i64: 1, 1, 1, 1>,
                         dilation = array<i64: 2, 1>,
                         groups = 1 : i64}
      : (tensor<2x1x7x9xf32>, tensor<4x1x3x2xf32>) -> tensor<2x4x3x5xf32>
  return %0 : tensor<2x4x3x5xf32>
}

// Groups: the kernel reads a band of the input channels per band of the output
// ones, so it is as deep as the input divided by the group count.
// CHECK-LABEL: func @conv_grouped
func.func @conv_grouped(%x: tensor<1x4x5x5xf32>, %k: tensor<6x2x2x2xf32>) -> tensor<1x6x4x4xf32> {
  // CHECK: tera.conv %arg0, %arg1 {dilation = array<i64: 1, 1>, groups = 2 : i64, padding = array<i64: 0, 0, 0, 0>, strides = array<i64: 1, 1>}
  %0 = tera.conv %x, %k {strides = array<i64: 1, 1>,
                         padding = array<i64: 0, 0, 0, 0>,
                         dilation = array<i64: 1, 1>,
                         groups = 2 : i64}
      : (tensor<1x4x5x5xf32>, tensor<6x2x2x2xf32>) -> tensor<1x6x4x4xf32>
  return %0 : tensor<1x6x4x4xf32>
}

// A one-dimensional convolution is the same op with one spatial axis, which is
// what the length of `strides` says.
// CHECK-LABEL: func @conv_1d
func.func @conv_1d(%x: tensor<1x2x8xf32>, %k: tensor<3x2x3xf32>) -> tensor<1x3x6xf32> {
  // CHECK: tera.conv %arg0, %arg1 {dilation = array<i64: 1>, groups = 1 : i64, padding = array<i64: 0, 0>, strides = array<i64: 1>}
  %0 = tera.conv %x, %k {strides = array<i64: 1>,
                         padding = array<i64: 0, 0>,
                         dilation = array<i64: 1>,
                         groups = 1 : i64}
      : (tensor<1x2x8xf32>, tensor<3x2x3xf32>) -> tensor<1x3x6xf32>
  return %0 : tensor<1x3x6xf32>
}

// CHECK-LABEL: func @pool_max
func.func @pool_max(%x: tensor<1x3x8x8xf32>) -> tensor<1x3x4x4xf32> {
  // CHECK: tera.pool2d max, %arg0 {kernel_size = array<i64: 2, 2>, padding = array<i64: 0, 0, 0, 0>, strides = array<i64: 2, 2>}
  %0 = tera.pool2d max, %x {kernel_size = array<i64: 2, 2>,
                            strides = array<i64: 2, 2>,
                            padding = array<i64: 0, 0, 0, 0>}
      : tensor<1x3x8x8xf32> -> tensor<1x3x4x4xf32>
  return %0 : tensor<1x3x4x4xf32>
}

// An average that counts the padding, and a window that hangs over the high
// edge because `ceil_mode` rounds the count up rather than down. Both flags
// default to false and are only written when they are not.
// CHECK-LABEL: func @pool_average
func.func @pool_average(%x: tensor<1x1x5x5xf32>) -> tensor<1x1x3x3xf32> {
  // CHECK: tera.pool2d average, %arg0 {ceil_mode = true, count_include_pad = true, kernel_size = array<i64: 2, 2>, padding = array<i64: 0, 0, 0, 0>, strides = array<i64: 2, 2>}
  %0 = tera.pool2d average, %x {kernel_size = array<i64: 2, 2>,
                                strides = array<i64: 2, 2>,
                                padding = array<i64: 0, 0, 0, 0>,
                                ceil_mode = true,
                                count_include_pad = true}
      : tensor<1x1x5x5xf32> -> tensor<1x1x3x3xf32>
  return %0 : tensor<1x1x3x3xf32>
}

// Padding a tensor, and reading an axis of it from the far end. `interior` is
// optional, so a border prints without it and a dilation prints with it.

// CHECK-LABEL: func @pad_border
func.func @pad_border(%x: tensor<2x3xf32>, %v: tensor<f32>) -> tensor<5x4xf32> {
  // CHECK: tera.pad %arg0, %arg1 {high = array<i64: 2, 1>, low = array<i64: 1, 0>}
  // CHECK-NOT: interior
  %0 = tera.pad %x, %v {low = array<i64: 1, 0>, high = array<i64: 2, 1>}
      : (tensor<2x3xf32>, tensor<f32>) -> tensor<5x4xf32>
  return %0 : tensor<5x4xf32>
}

// CHECK-LABEL: func @pad_interior
func.func @pad_interior(%x: tensor<3xf32>, %v: tensor<f32>) -> tensor<6xf32> {
  // CHECK: tera.pad %arg0, %arg1 {high = array<i64: 0>, interior = array<i64: 1>, low = array<i64: 1>}
  %0 = tera.pad %x, %v {low = array<i64: 1>, high = array<i64: 0>,
                        interior = array<i64: 1>}
      : (tensor<3xf32>, tensor<f32>) -> tensor<6xf32>
  return %0 : tensor<6xf32>
}

// CHECK-LABEL: func @reverse
func.func @reverse(%x: tensor<2x3x4xf32>) -> tensor<2x3x4xf32> {
  // CHECK: tera.reverse %arg0 {dimensions = array<i64: 0, 2>}
  %0 = tera.reverse %x {dimensions = array<i64: 0, 2>}
      : tensor<2x3x4xf32> -> tensor<2x3x4xf32>
  return %0 : tensor<2x3x4xf32>
}
