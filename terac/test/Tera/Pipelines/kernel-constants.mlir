// RUN: tera-opt %s --tera-to-nvvm --mlir-print-ir-after=gpu-kernel-outlining 2>&1 \
// RUN:   | FileCheck %s

// Outlining a launch hands the kernel everything its body read from around it,
// and a constant read from around it is a constant the kernel now has to be
// told. `gpu-launch-sink-index-computations` copies those inside first, which
// is why upstream ships it as a separate pass and why the pipeline has to name
// it: nothing about outlining implies it.
//
// What it decides is whether a loop bound reaches the device as a number or as
// an argument. This reduction runs 64 iterations per thread, and 64 is either
// written in the kernel -- where the unroller can see it, the induction
// variable can be strength-reduced against it, and the branch can go -- or it
// is a parameter loaded at entry, and then none of that happens however small
// the count is. The difference does not show up as a wrong answer, only as a
// kernel several times slower than the one that was asked for, so a check on
// the shape of the kernel is what there is to catch it with.
//
// The signature is the assertion: two buffers and nothing else. An index
// parameter here would be the loop bound, arriving the way it used to.

// CHECK:     gpu.func @row_sum_kernel(%{{[a-z0-9_]+}}: memref<256x64xf32>, %{{[a-z0-9_]+}}: memref<256xf32>) kernel
// CHECK-DAG:   %[[BOUND:[a-z0-9_]+]] = arith.constant 64 : index
// CHECK-DAG:   %[[FROM:[a-z0-9_]+]] = arith.constant 0 : index
// CHECK:       scf.for %{{[a-z0-9_]+}} = %[[FROM]] to %[[BOUND]]

func.func @row_sum(%x: tensor<256x64xf32>) -> tensor<256xf32> {
  %0 = tera.reduce sum, %x {dimensions = array<i64: 1>}
      : tensor<256x64xf32> -> tensor<256xf32>
  return %0 : tensor<256xf32>
}
