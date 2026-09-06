// RUN: tera-opt %s --tera-to-nvvm=schedule=%S/Inputs/matmul-schedule.mlir --mlir-print-ir-before=gpu-kernel-outlining 2>&1 | FileCheck %s --check-prefix=SCRIPT
// RUN: tera-opt %s --tera-to-nvvm --mlir-print-ir-before=gpu-kernel-outlining 2>&1 | FileCheck %s --check-prefix=DEFAULT

// RUN: %if cuda %{ tera-runner %s --entry=scheduled --target=cuda --target-options=schedule=%S/Inputs/matmul-schedule.mlir --shared-libs=%mlir_cuda_runtime --shared-libs=%mlir_c_runner_utils --seed=11 > %t.scripted %}
// RUN: %if cuda %{ tera-runner %s --entry=scheduled --target=cuda --shared-libs=%mlir_cuda_runtime --shared-libs=%mlir_c_runner_utils --seed=11 > %t.default %}
// RUN: %if cuda %{ diff %t.scripted %t.default %}

// RUN: not tera-opt %s --tera-to-nvvm=schedule=%S/Inputs/no-such-schedule.mlir 2>&1 | FileCheck %s --check-prefix=MISSING
// RUN: not tera-opt %s --tera-to-nvvm=schedule=%S/Inputs/nameless-schedule.mlir 2>&1 | FileCheck %s --check-prefix=NAMELESS

// A schedule from outside the compiler, applied to a real program on a real
// device. Two things have to be true at once for that to be worth anything,
// and neither implies the other: the schedule has to be the one that was
// asked for, and the answer has to be the one the compiler would have given.
//
// The first is read off the launch. Terac's own model cuts a 64 by 64 product
// into blocks of 32 by 32 and stages both operand tiles in shared memory, and
// the fill that seeds the accumulator is a second kernel; the script asks for
// blocks of 16 by 16, no shared memory and the fill left on the host. Those
// are different enough that no accident produces one from the other, which is
// the point of choosing them.
//
// The second is a `diff` of every number against the default schedule's, run
// on the device from the same seed. A schedule is a choice about where the
// arithmetic happens and not about what it is, so anything other than the same
// bits means the script changed the program rather than the plan for it.

// SCRIPT-DAG: %[[BLOCKS:[a-z0-9_]+]] = arith.constant 4 : index
// SCRIPT-DAG: %[[LANES:[a-z0-9_]+]] = arith.constant 16 : index
// SCRIPT: gpu.launch blocks({{[^)]*}}) in (%{{[a-z0-9_]+}} = %[[BLOCKS]], %{{[a-z0-9_]+}} = %[[BLOCKS]], {{[^)]*}}) threads({{[^)]*}}) in (%{{[a-z0-9_]+}} = %[[LANES]], %{{[a-z0-9_]+}} = %[[LANES]],
// SCRIPT-NOT: address_space<workgroup>

// Nothing about the module says which schedule it wants, so the same module
// with no script is the compiler as it was: two launches, a thread block of
// 32 by 32, and the operand tiles staged.
// DEFAULT-DAG: %[[BLOCKS:[a-z0-9_]+]] = arith.constant 2 : index
// DEFAULT-DAG: %[[LANES:[a-z0-9_]+]] = arith.constant 32 : index
// DEFAULT: gpu.launch
// DEFAULT: gpu.launch blocks({{[^)]*}}) in (%{{[a-z0-9_]+}} = %[[BLOCKS]], %{{[a-z0-9_]+}} = %[[BLOCKS]], {{[^)]*}}) threads({{[^)]*}}) in (%{{[a-z0-9_]+}} = %[[LANES]], %{{[a-z0-9_]+}} = %[[LANES]], {{[^)]*}}) workgroup

// A script that was asked for and could not be used is an error rather than a
// quiet fall back to the default, because the fall back is for the absence of
// a schedule and not for a broken one: a tuned schedule that silently does
// nothing is a compiler that reports a tuning it did not do.
// MISSING: could not read a schedule from
// NAMELESS: could not find a nested named sequence with name: __transform_main

func.func @scheduled(%a: tensor<64x64xf32>, %b: tensor<64x64xf32>)
    -> tensor<64x64xf32> {
  %0 = tera.dot %a, %b {lhs_batch = array<i64>, lhs_contracting = array<i64: 1>,
                        rhs_batch = array<i64>, rhs_contracting = array<i64: 0>}
      : (tensor<64x64xf32>, tensor<64x64xf32>) -> tensor<64x64xf32>
  return %0 : tensor<64x64xf32>
}
