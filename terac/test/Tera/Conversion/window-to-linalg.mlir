// RUN: tera-opt %s --convert-tera-to-linalg --split-input-file | FileCheck %s

// A convolution is a contraction that slides: the parallel loops are the
// output's axes and the reductions are the input channel and the kernel's own
// axes, and the only thing that makes it a convolution rather than a dot is
// that the input's map reads `window * stride + position * dilation`.
// CHECK: #[[READ:.*]] = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d4, d2 + d5, d3 + d6)>
// CHECK: #[[TAPS:.*]] = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1, d4, d5, d6)>
// CHECK: #[[WRITE:.*]] = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3)>
// CHECK-LABEL: func @conv
// One group needs no group axis, so there is no reshape either way round.
// CHECK-NOT: tensor.expand_shape
// CHECK: %[[ZERO:.*]] = arith.constant 0.000000e+00
// CHECK: %[[DEST:.*]] = linalg.fill ins(%[[ZERO]]
// CHECK: linalg.generic
// CHECK-SAME: indexing_maps = [#[[READ]], #[[TAPS]], #[[WRITE]]]
// CHECK-SAME: iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]
// CHECK-SAME: ins(%arg0, %arg1
// CHECK-SAME: outs(%[[DEST]]
// CHECK: %[[PRODUCT:.*]] = arith.mulf %in, %in_0
// CHECK: %[[TOTAL:.*]] = arith.addf %out, %[[PRODUCT]]
// CHECK: linalg.yield %[[TOTAL]]
// CHECK-NOT: tensor.collapse_shape
func.func @conv(%x: tensor<1x2x5x5xf32>, %k: tensor<3x2x2x2xf32>) -> tensor<1x3x4x4xf32> {
  %0 = tera.conv %x, %k {strides = array<i64: 1, 1>,
                         padding = array<i64: 0, 0, 0, 0>,
                         dilation = array<i64: 1, 1>,
                         groups = 1 : i64}
      : (tensor<1x2x5x5xf32>, tensor<3x2x2x2xf32>) -> tensor<1x3x4x4xf32>
  return %0 : tensor<1x3x4x4xf32>
}

// -----

// The stride and the dilation are the two coefficients of that map.
// CHECK: #[[READ:.*]] = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d4, d2 * 2 + d5 * 3, d3 * 2 + d6 * 3)>
// CHECK-LABEL: func @conv_spaced
func.func @conv_spaced(%x: tensor<1x1x9x9xf32>, %k: tensor<1x1x2x2xf32>) -> tensor<1x1x3x3xf32> {
  %0 = tera.conv %x, %k {strides = array<i64: 2, 2>,
                         padding = array<i64: 0, 0, 0, 0>,
                         dilation = array<i64: 3, 3>,
                         groups = 1 : i64}
      : (tensor<1x1x9x9xf32>, tensor<1x1x2x2xf32>) -> tensor<1x1x3x3xf32>
  return %0 : tensor<1x1x3x3xf32>
}

// -----

// Padding is not part of the map: an index the map produces has to be inside
// the operand, so what a window would read outside it is put there first. One
// `linalg.generic` writes the whole padded tensor, reading either an element
// or the border, which keeps it a single kernel on a device -- a destination
// filled and then copied into is a copy the host makes, and a padded tensor
// between two kernels is a buffer the host may not touch.
// CHECK-LABEL: func @conv_padded
// CHECK: %[[BORDER:.*]] = arith.constant 0.000000e+00 : f32
// CHECK: %[[WIDE:.*]] = tensor.empty() : tensor<1x1x7x7xf32>
// CHECK: %[[PADDED:.*]] = linalg.generic
// CHECK-SAME: outs(%[[WIDE]]
// CHECK: arith.floordivsi
// CHECK: %[[ELEMENT:.*]] = tensor.extract %arg0
// CHECK: %[[EITHER:.*]] = arith.select %{{.*}}, %[[ELEMENT]], %[[BORDER]]
// CHECK: linalg.yield %[[EITHER]]
// CHECK-NOT: tensor.insert_slice
// CHECK: linalg.generic
// CHECK-SAME: ins(%[[PADDED]], %arg1
func.func @conv_padded(%x: tensor<1x1x5x5xf32>, %k: tensor<1x1x3x3xf32>) -> tensor<1x1x5x5xf32> {
  %0 = tera.conv %x, %k {strides = array<i64: 1, 1>,
                         padding = array<i64: 1, 1, 1, 1>,
                         dilation = array<i64: 1, 1>,
                         groups = 1 : i64}
      : (tensor<1x1x5x5xf32>, tensor<1x1x3x3xf32>) -> tensor<1x1x5x5xf32>
  return %0 : tensor<1x1x5x5xf32>
}

// -----

// More than one group splits the channel axis in two, which makes the group a
// loop of its own and leaves every map a permutation again.
// CHECK: #[[READ:.*]] = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1, d5, d3 + d6, d4 + d7)>
// CHECK: #[[TAPS:.*]] = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d1, d2, d5, d6, d7)>
// CHECK: #[[WRITE:.*]] = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1, d2, d3, d4)>
// CHECK-LABEL: func @conv_grouped
// CHECK: %[[INPUT:.*]] = tensor.expand_shape %arg0 {{\[}}[0], [1, 2], [3], [4]{{\]}} output_shape [1, 2, 2, 5, 5]
// CHECK: %[[KERNEL:.*]] = tensor.expand_shape %arg1 {{\[}}[0, 1], [2], [3], [4]{{\]}} output_shape [2, 3, 2, 2, 2]
// CHECK: %[[GROUPED:.*]] = linalg.generic
// CHECK-SAME: indexing_maps = [#[[READ]], #[[TAPS]], #[[WRITE]]]
// CHECK-SAME: ins(%[[INPUT]], %[[KERNEL]]
// CHECK: tensor.collapse_shape %[[GROUPED]] {{\[}}[0], [1, 2], [3], [4]{{\]}}
func.func @conv_grouped(%x: tensor<1x4x5x5xf32>, %k: tensor<6x2x2x2xf32>) -> tensor<1x6x4x4xf32> {
  %0 = tera.conv %x, %k {strides = array<i64: 1, 1>,
                         padding = array<i64: 0, 0, 0, 0>,
                         dilation = array<i64: 1, 1>,
                         groups = 2 : i64}
      : (tensor<1x4x5x5xf32>, tensor<6x2x2x2xf32>) -> tensor<1x6x4x4xf32>
  return %0 : tensor<1x6x4x4xf32>
}

// -----

// The same traversal reading a reduction out of each window. A maximum starts
// from a value no element can beat, and the second input is a tensor the body
// never reads: `linalg.generic` needs the indexing maps taken together to be
// invertible, and without something that maps the position loops on their own
// they are not.
// CHECK: #[[READ:.*]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2 * 2 + d4, d3 * 2 + d5)>
// CHECK: #[[WINDOW:.*]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d4, d5)>
// CHECK: #[[WRITE:.*]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3)>
// CHECK-LABEL: func @pool_max
// CHECK: %[[LEAST:.*]] = arith.constant 0xFF800000 : f32
// CHECK: %[[DEST:.*]] = linalg.fill ins(%[[LEAST]]
// CHECK: %[[SHAPE:.*]] = tensor.empty() : tensor<2x2xi1>
// CHECK: linalg.generic
// CHECK-SAME: indexing_maps = [#[[READ]], #[[WINDOW]], #[[WRITE]]]
// CHECK-SAME: iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction"]
// CHECK-SAME: ins(%arg0, %[[SHAPE]]
// CHECK: arith.maximumf
// CHECK-NOT: arith.mulf
func.func @pool_max(%x: tensor<1x3x8x8xf32>) -> tensor<1x3x4x4xf32> {
  %0 = tera.pool2d max, %x {kernel_size = array<i64: 2, 2>,
                            strides = array<i64: 2, 2>,
                            padding = array<i64: 0, 0, 0, 0>}
      : tensor<1x3x8x8xf32> -> tensor<1x3x4x4xf32>
  return %0 : tensor<1x3x4x4xf32>
}

// -----

// An average sums into zero and divides once at the end, so one rounding per
// window rather than one per element.
// CHECK-LABEL: func @pool_average
// CHECK: %[[ZERO:.*]] = arith.constant 0.000000e+00 : f32
// CHECK: linalg.fill ins(%[[ZERO]]
// CHECK: arith.addf
// CHECK: %[[SHARE:.*]] = arith.constant 1.250000e-01 : f32
// CHECK: linalg.generic
// CHECK: arith.mulf %in, %[[SHARE]]
func.func @pool_average(%x: tensor<1x1x8x8xf32>) -> tensor<1x1x4x2xf32> {
  %0 = tera.pool2d average, %x {kernel_size = array<i64: 2, 4>,
                                strides = array<i64: 2, 4>,
                                padding = array<i64: 0, 0, 0, 0>}
      : tensor<1x1x8x8xf32> -> tensor<1x1x4x2xf32>
  return %0 : tensor<1x1x4x2xf32>
}

// -----

// An axis read from the far end is an indexing map like any other.
// CHECK: #[[BACK:.*]] = affine_map<(d0, d1) -> (d0, -d1 + 3)>
// CHECK: #[[FORWARD:.*]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-LABEL: func @reverse
// CHECK: linalg.generic
// CHECK-SAME: indexing_maps = [#[[BACK]], #[[FORWARD]]]
func.func @reverse(%x: tensor<2x4xf32>) -> tensor<2x4xf32> {
  %0 = tera.reverse %x {dimensions = array<i64: 1>}
      : tensor<2x4xf32> -> tensor<2x4xf32>
  return %0 : tensor<2x4xf32>
}
