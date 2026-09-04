// RUN: tera-gradcheck %s --shared-libs=%mlir_c_runner_utils

// The numbers behind the convolution and pooling rules.
//
// A convolution differentiates into two more convolutions, and both of them
// rearrange something: the input's runs the adjoint back through the kernel
// read backwards, and the kernel's swaps the batch and channel axes of both
// operands so that the batch is what gets contracted. FileCheck can see that
// two convolutions were built. It cannot see that the padding either one was
// given puts the adjoint back exactly where it was read from, and every case
// below is a shape where getting that wrong is off by a different amount.

// The plain case: one window per position, nothing skipped, nothing padded.
func.func @valid(%x: tensor<1x2x5x5xf64>, %k: tensor<3x2x2x2xf64>)
    -> tensor<1x3x4x4xf64> attributes {tera.differentiable} {
  %0 = tera.conv %x, %k {strides = array<i64: 1, 1>,
                         padding = array<i64: 0, 0, 0, 0>,
                         dilation = array<i64: 1, 1>,
                         groups = 1 : i64}
      : (tensor<1x2x5x5xf64>, tensor<3x2x2x2xf64>) -> tensor<1x3x4x4xf64>
  return %0 : tensor<1x3x4x4xf64>
}

// Padding, and an odd input so that the two edges are not padded alike. The
// input rule has to subtract each side of the padding from its own end.
func.func @padded(%x: tensor<2x1x5x4xf64>, %k: tensor<2x1x3x3xf64>)
    -> tensor<2x2x5x4xf64> attributes {tera.differentiable} {
  %0 = tera.conv %x, %k {strides = array<i64: 1, 1>,
                         padding = array<i64: 1, 1, 1, 1>,
                         dilation = array<i64: 1, 1>,
                         groups = 1 : i64}
      : (tensor<2x1x5x4xf64>, tensor<2x1x3x3xf64>) -> tensor<2x2x5x4xf64>
  return %0 : tensor<2x2x5x4xf64>
}

// A stride, which is the branch of the rule that spaces the adjoint out before
// running it back, and an extent the stride does not divide, so the last window
// stops short and the padding has to make up the difference. Getting the
// trailing edge wrong shifts every gradient by one column.
func.func @strided(%x: tensor<1x1x6x7xf64>, %k: tensor<1x1x3x2xf64>)
    -> tensor<1x1x2x3xf64> attributes {tera.differentiable} {
  %0 = tera.conv %x, %k {strides = array<i64: 2, 2>,
                         padding = array<i64: 0, 0, 0, 0>,
                         dilation = array<i64: 1, 1>,
                         groups = 1 : i64}
      : (tensor<1x1x6x7xf64>, tensor<1x1x3x2xf64>) -> tensor<1x1x2x3xf64>
  return %0 : tensor<1x1x2x3xf64>
}

// A dilation, which the kernel rule and the input rule read in opposite
// places: it spaces the kernel out in the forward pass and the windows out in
// the one that differentiates the kernel.
func.func @dilated(%x: tensor<1x1x7x7xf64>, %k: tensor<2x1x2x2xf64>)
    -> tensor<1x2x5x5xf64> attributes {tera.differentiable} {
  %0 = tera.conv %x, %k {strides = array<i64: 1, 1>,
                         padding = array<i64: 0, 0, 0, 0>,
                         dilation = array<i64: 2, 2>,
                         groups = 1 : i64}
      : (tensor<1x1x7x7xf64>, tensor<2x1x2x2xf64>) -> tensor<1x2x5x5xf64>
  return %0 : tensor<1x2x5x5xf64>
}

// Groups, where each band of channels is its own convolution. A rule that
// joined the bands back in the wrong order, or read the kernel's bands along
// the input's axis, still produces a tensor of the right shape.
func.func @grouped(%x: tensor<1x4x5x5xf64>, %k: tensor<6x2x2x2xf64>)
    -> tensor<1x6x4x4xf64> attributes {tera.differentiable} {
  %0 = tera.conv %x, %k {strides = array<i64: 1, 1>,
                         padding = array<i64: 0, 0, 0, 0>,
                         dilation = array<i64: 1, 1>,
                         groups = 2 : i64}
      : (tensor<1x4x5x5xf64>, tensor<6x2x2x2xf64>) -> tensor<1x6x4x4xf64>
  return %0 : tensor<1x6x4x4xf64>
}

// Everything at once, and a rectangular window rather than a square one, so an
// axis swapped for the other shows up as a wrong number rather than a wrong
// shape.
func.func @together(%x: tensor<2x4x7x9xf64>, %k: tensor<4x2x3x2xf64>)
    -> tensor<2x4x4x5xf64> attributes {tera.differentiable} {
  %0 = tera.conv %x, %k {strides = array<i64: 2, 2>,
                         padding = array<i64: 1, 1, 1, 1>,
                         dilation = array<i64: 1, 1>,
                         groups = 2 : i64}
      : (tensor<2x4x7x9xf64>, tensor<4x2x3x2xf64>) -> tensor<2x4x4x5xf64>
  return %0 : tensor<2x4x4x5xf64>
}

// A max pool routes the adjoint to whichever element the window chose, and
// the tie between two equal elements is the case a rule that compares against
// the answer gets wrong in a way finite differences can see -- so the pool
// sits behind an `exp`, which makes ties unlikely and the surface smooth.
func.func @max_pool(%x: tensor<1x2x4x6xf64>) -> tensor<1x2x2x3xf64>
    attributes {tera.differentiable} {
  %0 = tera.exp %x : tensor<1x2x4x6xf64>
  %1 = tera.pool2d max, %0 {kernel_size = array<i64: 2, 2>,
                            strides = array<i64: 2, 2>,
                            padding = array<i64: 0, 0, 0, 0>}
      : tensor<1x2x4x6xf64> -> tensor<1x2x2x3xf64>
  return %1 : tensor<1x2x2x3xf64>
}

// An average pool gives every element of the window the same share, and a
// rectangular window is what tells `1/(kh*kw)` apart from `1/kh` or `1/kw`.
func.func @average_pool(%x: tensor<2x1x6x6xf64>) -> tensor<2x1x2x3xf64>
    attributes {tera.differentiable} {
  %0 = tera.pool2d average, %x {kernel_size = array<i64: 3, 2>,
                                strides = array<i64: 3, 2>,
                                padding = array<i64: 0, 0, 0, 0>}
      : tensor<2x1x6x6xf64> -> tensor<2x1x2x3xf64>
  return %0 : tensor<2x1x2x3xf64>
}

// The two together, which is the shape every convolutional model has: a
// convolution, a pool over its output, and the adjoint travelling back through
// both.
func.func @conv_then_pool(%x: tensor<1x1x6x6xf64>, %k: tensor<2x1x3x3xf64>)
    -> tensor<2x1x2x2xf64> attributes {tera.differentiable} {
  %0 = tera.conv %x, %k {strides = array<i64: 1, 1>,
                         padding = array<i64: 0, 0, 0, 0>,
                         dilation = array<i64: 1, 1>,
                         groups = 1 : i64}
      : (tensor<1x1x6x6xf64>, tensor<2x1x3x3xf64>) -> tensor<1x2x4x4xf64>
  %1 = tera.pool2d average, %0 {kernel_size = array<i64: 2, 2>,
                                strides = array<i64: 2, 2>,
                                padding = array<i64: 0, 0, 0, 0>}
      : tensor<1x2x4x4xf64> -> tensor<1x2x2x2xf64>
  %2 = tera.transpose %1 {permutation = array<i64: 1, 0, 2, 3>}
      : tensor<1x2x2x2xf64> -> tensor<2x1x2x2xf64>
  return %2 : tensor<2x1x2x2xf64>
}
