// RUN: %if cuda %{ tera-runner %s --entry=weighted --target=cuda --shared-libs=%mlir_cuda_runtime --shared-libs=%mlir_c_runner_utils --data=%S/device-resident.json --check %}
// RUN: not tera-runner %s --entry=weighted --shared-libs=%mlir_c_runner_utils 2>&1 | FileCheck %s

// A weight the caller puts on the device once and a batch that arrives fresh,
// which is the shape of a step of inference. The runner reads
// `tera.device_resident` off the signature, uploads that argument before the
// first call and hands every call the device pointer, so the answer is only
// right if the staging really left it alone.
//
// The host target has no device to leave it on and says so where the target is
// chosen, rather than further down where a memref has become a pointer and the
// promise is no longer about anything a diagnostic can name.

// CHECK: argument 1 is marked 'tera.device_resident', and target 'cpu' has no device to leave it on

func.func @weighted(%x: tensor<2x3xf32>,
                    %w: tensor<2x3xf32> {tera.device_resident},
                    %b: tensor<3xf32> {tera.device_resident})
    -> tensor<2x3xf32> {
  %0 = tera.mul %x, %w : tensor<2x3xf32>
  %1 = tera.broadcast_in_dim %b {broadcast_dimensions = array<i64: 1>}
      : tensor<3xf32> -> tensor<2x3xf32>
  %2 = tera.add %0, %1 : tensor<2x3xf32>
  %3 = tera.tanh %2 : tensor<2x3xf32>
  return %3 : tensor<2x3xf32>
}
