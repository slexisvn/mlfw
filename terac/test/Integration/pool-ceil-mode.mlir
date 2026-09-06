// RUN: tera-runner %s --entry=hangs_over --shared-libs=%mlir_c_runner_utils --check --data='{"inputs": [{"shape": [1, 1, 5, 5], "data": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]}], "output": {"shape": [1, 1, 3, 3], "data": [4, 6, 3.75, 14, 16, 8.75, 10.75, 11.75, 6.25]}}'
// RUN: %if cuda %{ tera-runner %s --entry=hangs_over --target=cuda --shared-libs=%mlir_cuda_runtime --shared-libs=%mlir_c_runner_utils --check --data='{"inputs": [{"shape": [1, 1, 5, 5], "data": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]}], "output": {"shape": [1, 1, 3, 3], "data": [4, 6, 3.75, 14, 16, 8.75, 10.75, 11.75, 6.25]}}' %}

// RUN: tera-runner %s --entry=agrees_with_floor --shared-libs=%mlir_c_runner_utils --check --data='{"inputs": [{"shape": [1, 1, 4, 4], "data": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]}], "output": {"shape": [1, 1, 2, 2], "data": [3.5, 5.5, 11.5, 13.5]}}'

// RUN: tera-runner %s --entry=mostly_padding --shared-libs=%mlir_c_runner_utils --check --data='{"inputs": [{"shape": [1, 1, 3, 3], "data": [1, 2, 3, 4, 5, 6, 7, 8, 9]}], "output": {"shape": [1, 1, 2, 2], "data": [3, 2.25, 3.75, 2.25]}}'

// `ceil_mode` rounding the window count up is only different from rounding it
// down when the stride does not divide what is left, and then the last window
// hangs over the high edge. That case used to be refused -- the verifier said
// the window read past the end of the input, which it does -- and what it
// reads there is now written: the padded operand is extended by however much
// less than a stride is missing, and the elements out there are zero like
// every other pad. The numbers are checked rather than the shape, because a
// window extended by the wrong amount still has the right shape.
//
// @hangs_over is the program that was the refusal, over 1..25 laid out five to
// a row. Three windows across and down, 2 by 2, striding 2, so the third of
// each reaches one element past the edge:
//
//   1  2 | 3  4 | 5  .        (1+2+6+7)/4 = 4     (3+4+8+9)/4 = 6
//   6  7 | 8  9 |10  .        (5+0+10+0)/4 = 3.75
//  ------+------+-----
//  11 12 |13 14 |15  .        14   16   (15+0+20+0)/4 = 8.75
//  16 17 |18 19 |20  .
//  ------+------+-----
//  21 22 |23 24 |25  .        (21+22)/4 = 10.75   (23+24)/4 = 11.75
//   .  . | .  . | .  .        25/4 = 6.25
//
// Every number is a quarter, so f32 holds all of them exactly and `--check`
// compares against the arithmetic rather than near it. The divisor is the
// whole window everywhere, including the windows that are mostly padding,
// which is what `count_include_pad` says and what the op means by it.
//
// @agrees_with_floor is the case that was always allowed: 2 divides 4, so
// there is no extra window and `ceil_mode` changes nothing. It is here so
// that the path the fix took cannot quietly start padding programs that never
// hung over anything.
//
// @mostly_padding is the extreme of it: three elements, a window of two and a
// stride of two, so rounding down takes one window and rounding up takes two,
// and the second is one element of input against three of padding. Its answer
// is 9/4 rather than 9, which is what counting the padding means and the
// clearest place to see that this op has picked a side.

func.func @hangs_over(%x: tensor<1x1x5x5xf32>) -> tensor<1x1x3x3xf32> {
  %0 = tera.pool2d average, %x {kernel_size = array<i64: 2, 2>,
                                strides = array<i64: 2, 2>,
                                padding = array<i64: 0, 0, 0, 0>,
                                ceil_mode = true,
                                count_include_pad = true}
      : tensor<1x1x5x5xf32> -> tensor<1x1x3x3xf32>
  return %0 : tensor<1x1x3x3xf32>
}

func.func @agrees_with_floor(%x: tensor<1x1x4x4xf32>) -> tensor<1x1x2x2xf32> {
  %0 = tera.pool2d average, %x {kernel_size = array<i64: 2, 2>,
                                strides = array<i64: 2, 2>,
                                padding = array<i64: 0, 0, 0, 0>,
                                ceil_mode = true,
                                count_include_pad = true}
      : tensor<1x1x4x4xf32> -> tensor<1x1x2x2xf32>
  return %0 : tensor<1x1x2x2xf32>
}

func.func @mostly_padding(%x: tensor<1x1x3x3xf32>) -> tensor<1x1x2x2xf32> {
  %0 = tera.pool2d average, %x {kernel_size = array<i64: 2, 2>,
                                strides = array<i64: 2, 2>,
                                padding = array<i64: 0, 0, 0, 0>,
                                ceil_mode = true,
                                count_include_pad = true}
      : tensor<1x1x3x3xf32> -> tensor<1x1x2x2xf32>
  return %0 : tensor<1x1x2x2xf32>
}
