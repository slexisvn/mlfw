// RUN: tera-runner %s --entry=net --shared-libs=%mlir_c_runner_utils --data=%S/dynamic-window-one.json --check
// RUN: tera-runner %s --entry=net --shared-libs=%mlir_c_runner_utils --data=%S/dynamic-window-three.json --check
// RUN: %if cuda %{ tera-runner %s --entry=net --target=cuda --shared-libs=%mlir_cuda_runtime --shared-libs=%mlir_c_runner_utils --data=%S/dynamic-window-three.json --check %}
// A convolution and a pooling over a batch nobody has committed to, which is
// the shape of every inference server: the weights are fixed and the number of
// images is not.
//
// The batch is the one axis a window carries through unchanged, so it is the
// one whose extent the destination can be handed by reading it back off the
// input. A `?` on a spatial axis is a window count instead, which is
// dynamic-spatial.mlir beside this; a `?` in the kernel is still refused, and
// test/Tera/Conversion/invalid.mlir has that.

func.func @net(%x: tensor<?x1x4x4xf32>, %k: tensor<2x1x2x2xf32>)
    -> tensor<?x2x2x2xf32> {
  %0 = tera.conv %x, %k {strides = array<i64: 1, 1>,
                         padding = array<i64: 0, 0, 0, 0>,
                         dilation = array<i64: 1, 1>, groups = 1 : i64}
      : (tensor<?x1x4x4xf32>, tensor<2x1x2x2xf32>) -> tensor<?x2x3x3xf32>
  %1 = tera.pool2d max, %0 {kernel_size = array<i64: 2, 2>,
                            strides = array<i64: 1, 1>,
                            padding = array<i64: 0, 0, 0, 0>,
                            ceil_mode = false, count_include_pad = true}
      : tensor<?x2x3x3xf32> -> tensor<?x2x2x2xf32>
  return %1 : tensor<?x2x2x2xf32>
}
