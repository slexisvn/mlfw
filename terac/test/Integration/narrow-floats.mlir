// RUN: tera-runner %s --entry=narrow --shared-libs=%mlir_c_runner_utils --data=%S/narrow-floats.json --check
// RUN: %if cuda %{ tera-runner %s --entry=narrow --target=cuda --shared-libs=%mlir_cuda_runtime --shared-libs=%mlir_c_runner_utils --data=%S/narrow-floats.json --check %}

// f16 and bf16 side by side, on numbers every step of which is exact in both,
// so the expected output is the answer in real arithmetic rather than a
// rounding of it that only one format or one target would agree with.
//
// The two are not the same problem. f16 arithmetic is native on both targets;
// bf16 is not on either, and narrowing an f32 back into it is a compiler-rt
// builtin the JIT has no library to find. `-tera-to-llvm` and `-tera-to-nvvm`
// therefore promote bf16 to f32 around every operation and expand the pair of
// conversions that leaves into the integer shifts they are. No math function
// exists at either width, so those are widened too -- which is why the answer
// here is the same on the CPU and on the device, bit for bit.

func.func @narrow(%h: tensor<4xf16>, %b: tensor<4xbf16>)
    -> (tensor<4xf16>, tensor<4xbf16>) {
  %hk = tera.constant dense<[5.000000e-01, -1.000000e+00, 2.500000e-01, 2.000000e+00]>
      : tensor<4xf16>
  %bk = tera.constant dense<[5.000000e-01, -1.000000e+00, 2.500000e-01, 2.000000e+00]>
      : tensor<4xbf16>

  %0 = tera.add %h, %hk : tensor<4xf16>
  %1 = tera.mul %0, %0 : tensor<4xf16>
  %2 = tera.sqrt %1 : tensor<4xf16>

  %3 = tera.add %b, %bk : tensor<4xbf16>
  %4 = tera.mul %3, %3 : tensor<4xbf16>
  %5 = tera.sqrt %4 : tensor<4xbf16>

  return %2, %5 : tensor<4xf16>, tensor<4xbf16>
}
