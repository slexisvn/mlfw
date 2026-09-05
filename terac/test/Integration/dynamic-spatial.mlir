// RUN: tera-runner %s --entry=net --shared-libs=%mlir_c_runner_utils --data=%S/dynamic-spatial-four.json --check
// RUN: tera-runner %s --entry=net --shared-libs=%mlir_c_runner_utils --data=%S/dynamic-spatial-seven.json --check
// RUN: tera-runner %s --entry=padded --shared-libs=%mlir_c_runner_utils --data=%S/dynamic-spatial-padded.json --check
// RUN: %if cuda %{ tera-runner %s --entry=net --target=cuda --shared-libs=%mlir_cuda_runtime --shared-libs=%mlir_c_runner_utils --data=%S/dynamic-spatial-seven.json --check %}
// RUN: %if cuda %{ tera-runner %s --entry=padded --target=cuda --shared-libs=%mlir_cuda_runtime --shared-libs=%mlir_c_runner_utils --data=%S/dynamic-spatial-padded.json --check %}

// A window over an axis nobody has committed to. The extent of the result is
// not the extent of the input here, it is `floor((n + pad - reach) / stride) +
// 1` of it, and the destination is still materialised before the op that fills
// it -- so this only lowers because the op counts its own windows through
// `ReifyRankedShapedTypeOpInterface` rather than being read back axis for axis.
//
// The same module runs at two widths, which is the point: an extent baked in
// at compile time would be right for one of them.
//
// The kernel is powers of two and the inputs are small integers, so every
// product and every sum is exact in f32 and `--check` compares bit for bit.

func.func @net(%x: tensor<1x1x4x?xf32>, %k: tensor<1x1x2x2xf32>)
    -> tensor<1x1x2x?xf32> {
  %0 = tera.conv %x, %k {strides = array<i64: 1, 1>,
                         padding = array<i64: 0, 0, 0, 0>,
                         dilation = array<i64: 1, 1>, groups = 1 : i64}
      : (tensor<1x1x4x?xf32>, tensor<1x1x2x2xf32>) -> tensor<1x1x3x?xf32>
  %1 = tera.pool2d max, %0 {kernel_size = array<i64: 2, 2>,
                            strides = array<i64: 1, 1>,
                            padding = array<i64: 0, 0, 0, 0>,
                            ceil_mode = false, count_include_pad = true}
      : tensor<1x1x3x?xf32> -> tensor<1x1x2x?xf32>
  return %1 : tensor<1x1x2x?xf32>
}

// The other thing a window count is worked out from: padding, which widens the
// axis before the windows are laid along it and before anyone knows how wide
// it was. The border itself is built the same way, by an op that reads the
// operand's extent back rather than one that was told it.

func.func @padded(%x: tensor<1x1x4x?xf32>) -> tensor<1x1x3x?xf32> {
  %0 = tera.pool2d average, %x {kernel_size = array<i64: 2, 2>,
                                strides = array<i64: 2, 2>,
                                padding = array<i64: 1, 1, 1, 1>,
                                ceil_mode = false, count_include_pad = true}
      : tensor<1x1x4x?xf32> -> tensor<1x1x3x?xf32>
  return %0 : tensor<1x1x3x?xf32>
}
