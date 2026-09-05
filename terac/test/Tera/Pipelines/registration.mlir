// Both pipelines take options, and both take them the same way: a schedule is
// varied by a flag rather than by a rebuild. Nothing else in the tree names
// either pipeline, so these two lines are what keeps the names and the option
// spellings from drifting.

// RUN: tera-opt %s --tera-to-llvm | FileCheck %s
// RUN: tera-opt %s --tera-to-llvm=vector-width=8 | FileCheck %s
// RUN: tera-opt %s --tera-to-llvm=tile=false | FileCheck %s --check-prefix=WHOLE
// RUN: tera-opt %s --tera-to-nvvm=chip=sm_90 | FileCheck %s --check-prefix=CHIP

// The default schedule cuts the loop to one vector and leaves a loop around
// it, so the lowered function keeps a branch.
// CHECK-LABEL: llvm.func @scale
// CHECK: llvm.cond_br

// Tiling off hands the vectorizer the whole iteration space at once, which is
// 64 elements and under `max-vector-elements`, so what is left is one vector
// operation and no loop at all.
// WHOLE-LABEL: llvm.func @scale
// WHOLE-NOT: llvm.cond_br

// CHIP: .target sm_90

func.func @scale(%x: tensor<64xf32>, %w: tensor<64xf32>) -> tensor<64xf32> {
  %0 = tera.mul %x, %w : tensor<64xf32>
  return %0 : tensor<64xf32>
}
