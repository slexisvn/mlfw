// RUN: tera-opt %s --tera-tile-parallel-loops --cse --split-input-file \
// RUN:   | FileCheck %s
// RUN: tera-opt %s --tera-tile-parallel-loops=tile-sizes=8,8 --cse \
// RUN:   --split-input-file | FileCheck %s --check-prefix=GIVEN
// RUN: tera-opt %s --pass-pipeline="builtin.module(func.func(tera-tile-parallel-loops),func.func(gpu-map-parallel-loops{mapping-policy=innermost-first}),convert-parallel-loops-to-gpu,canonicalize)" \
// RUN:   --split-input-file --verify-diagnostics | FileCheck %s --check-prefix=LAUNCH

// The `LAUNCH` prefix carries the tiled loops the rest of the way the pipeline
// carries them, through the mapping and the GPU conversion that read what this
// pass wrote. A tile is only a claim about a grid and a block until those two
// say the numbers out loud, which is the only place they can be checked.

// The block is derived from the loop's own trip counts. The innermost
// dimension is the one `mapping-policy=innermost-first` gives to `thread_x`,
// so it takes the budget first: 256 divides 512 and is a multiple of the warp,
// which leaves one thread for the dimension above it. Both extents stay
// constant because the tile divides, so no thread carries a bounds test.

func.func @derives_a_block(%out: memref<64x512xf32>, %v: f32) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c64 = arith.constant 64 : index
  %c512 = arith.constant 512 : index
  scf.parallel (%i, %j) = (%c0, %c0) to (%c64, %c512) step (%c1, %c1) {
    memref.store %v, %out[%i, %j] : memref<64x512xf32>
    scf.reduce
  }
  return
}

// CHECK-LABEL: func.func @derives_a_block
// CHECK-DAG:     %[[ONE:.*]] = arith.constant 1 : index
// CHECK-DAG:     %[[BUDGET:.*]] = arith.constant 256 : index
// CHECK:         scf.parallel {{.*}} step (%[[ONE]], %[[BUDGET]])
// CHECK:           scf.parallel {{.*}} to (%[[ONE]], %[[BUDGET]])

// LAUNCH-LABEL: func.func @derives_a_block
// LAUNCH-DAG:     %[[ONE:.*]] = arith.constant 1 : index
// LAUNCH-DAG:     %[[GRID_X:.*]] = arith.constant 2 : index
// LAUNCH-DAG:     %[[GRID_Y:.*]] = arith.constant 64 : index
// LAUNCH-DAG:     %[[BLOCK_X:.*]] = arith.constant 256 : index
// LAUNCH:         gpu.launch blocks{{.*}} in (%{{[^ ]*}} = %[[GRID_X]], %{{[^ ]*}} = %[[GRID_Y]], %{{[^ ]*}} = %[[ONE]])
// LAUNCH-SAME:      threads{{.*}} in (%{{[^ ]*}} = %[[BLOCK_X]], %{{[^ ]*}} = %[[ONE]], %{{[^ ]*}} = %[[ONE]])

// GIVEN-LABEL: func.func @derives_a_block
// GIVEN-DAG:     %[[EIGHT:.*]] = arith.constant 8 : index
// GIVEN:         scf.parallel {{.*}} step (%[[EIGHT]], %[[EIGHT]])

// -----

// A dimension whose trip count is not a constant takes one thread, because a
// block extent has to be known; the budget it does not spend goes to the next
// dimension, which here can take all of it. The bound the tiling puts on the
// dynamic dimension is folded here rather than left to a later canonicalizer,
// because `convert-parallel-loops-to-gpu` reads the constant out of it and
// quietly declines the loop when it cannot.

func.func @leaves_a_dynamic_dimension_alone(%out: memref<?x512xf32>, %v: f32) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c512 = arith.constant 512 : index
  %n = memref.dim %out, %c0 : memref<?x512xf32>
  scf.parallel (%i, %j) = (%c0, %c0) to (%n, %c512) step (%c1, %c1) {
    memref.store %v, %out[%i, %j] : memref<?x512xf32>
    scf.reduce
  }
  return
}

// CHECK-LABEL: func.func @leaves_a_dynamic_dimension_alone
// CHECK-DAG:     %[[ONE:.*]] = arith.constant 1 : index
// CHECK-DAG:     %[[BUDGET:.*]] = arith.constant 256 : index
// CHECK:         scf.parallel {{.*}} step (%[[ONE]], %[[BUDGET]])
// CHECK:           %[[BOUND:.*]] = affine.min
// CHECK:           scf.parallel {{.*}} to (%[[BOUND]], %[[BUDGET]])

// LAUNCH-LABEL: func.func @leaves_a_dynamic_dimension_alone
// LAUNCH-DAG:     %[[ONE:.*]] = arith.constant 1 : index
// LAUNCH-DAG:     %[[GRID_X:.*]] = arith.constant 2 : index
// LAUNCH-DAG:     %[[BLOCK_X:.*]] = arith.constant 256 : index
// LAUNCH:         %[[N:.*]] = memref.dim
// LAUNCH:         gpu.launch blocks{{.*}} in (%{{[^ ]*}} = %[[GRID_X]], %{{[^ ]*}} = %[[N]], %{{[^ ]*}} = %[[ONE]])
// LAUNCH-SAME:      threads{{.*}} in (%{{[^ ]*}} = %[[BLOCK_X]], %{{[^ ]*}} = %[[ONE]], %{{[^ ]*}} = %[[ONE]])

// -----

// A trip count whose only divisors are one and itself takes what it can: 17
// threads on `x`, and a single thread above it, where 1021 is prime and larger
// than any block. A tile is never rounded up to a number that does not divide,
// which would buy threads with a bounds test.

func.func @prime_extents(%out: memref<1021x17xf32>, %v: f32) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c17 = arith.constant 17 : index
  %c1021 = arith.constant 1021 : index
  scf.parallel (%i, %j) = (%c0, %c0) to (%c1021, %c17) step (%c1, %c1) {
    memref.store %v, %out[%i, %j] : memref<1021x17xf32>
    scf.reduce
  }
  return
}

// CHECK-LABEL: func.func @prime_extents
// CHECK-DAG:     %[[ONE:.*]] = arith.constant 1 : index
// CHECK-DAG:     %[[SEVENTEEN:.*]] = arith.constant 17 : index
// CHECK:         scf.parallel {{.*}} step (%[[ONE]], %[[SEVENTEEN]])
// CHECK:           scf.parallel {{.*}} to (%[[ONE]], %[[SEVENTEEN]])

// LAUNCH-LABEL: func.func @prime_extents
// LAUNCH-DAG:     %[[ONE:.*]] = arith.constant 1 : index
// LAUNCH-DAG:     %[[GRID_Y:.*]] = arith.constant 1021 : index
// LAUNCH-DAG:     %[[BLOCK_X:.*]] = arith.constant 17 : index
// LAUNCH:         gpu.launch blocks{{.*}} in (%{{[^ ]*}} = %[[ONE]], %{{[^ ]*}} = %[[GRID_Y]], %{{[^ ]*}} = %[[ONE]])
// LAUNCH-SAME:      threads{{.*}} in (%{{[^ ]*}} = %[[BLOCK_X]], %{{[^ ]*}} = %[[ONE]], %{{[^ ]*}} = %[[ONE]])

// -----

// A loop carrying a reduction is left whole. The upstream tiling it would go
// through does not carry the reduction into the tile, so cutting one is a
// wrong answer rather than a slow one.
//
// Whole is as far as this one gets: the GPU conversion writes the reduction as
// a `gpu.all_reduce` inside the launch, and a value produced there cannot leave
// it, so the loop's result is left behind as an unresolved cast and the
// conversion fails rather than declining the loop. That is why the case has no
// `LAUNCH` expectation but an expected diagnostic instead. The pipeline never
// meets one of these, because a linalg reduction dimension arrives here as an
// `scf.for` and only a hand-written loop carries a reduction this far.

func.func @leaves_a_reduction_alone(%in: memref<64xf32>) -> f32 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c64 = arith.constant 64 : index
  %zero = arith.constant 0.0 : f32
  // expected-error @below {{failed to legalize unresolved materialization}}
  %total = scf.parallel (%i) = (%c0) to (%c64) step (%c1) init (%zero) -> f32 {
    %element = memref.load %in[%i] : memref<64xf32>
    scf.reduce(%element : f32) {
    ^bb0(%lhs: f32, %rhs: f32):
      %sum = arith.addf %lhs, %rhs : f32
      scf.reduce.return %sum : f32
    }
  }
  // expected-note @below {{see existing live user here}}
  return %total : f32
}

// CHECK-LABEL: func.func @leaves_a_reduction_alone
// CHECK:         scf.parallel
// CHECK-NOT:       scf.parallel

// -----

// A loop that already sits inside a parallel one is a thread level someone
// else made, and cutting it would make a third level. `-tera-tile-contraction-
// to-shared` writes both levels itself, and what it leaves behind has to reach
// the GPU conversion the shape it was written in.

func.func @leaves_a_thread_level_alone(%out: memref<64x64xf32>, %v: f32) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c32 = arith.constant 32 : index
  %c64 = arith.constant 64 : index
  scf.parallel (%bi, %bj) = (%c0, %c0) to (%c64, %c64) step (%c32, %c32) {
    scf.parallel (%ti, %tj) = (%c0, %c0) to (%c32, %c32) step (%c1, %c1) {
      %i = arith.addi %bi, %ti : index
      %j = arith.addi %bj, %tj : index
      memref.store %v, %out[%i, %j] : memref<64x64xf32>
      scf.reduce
    }
    scf.reduce
  }
  return
}

// CHECK-LABEL: func.func @leaves_a_thread_level_alone
// CHECK-DAG:     %[[ONE:.*]] = arith.constant 1 : index
// CHECK-DAG:     %[[TILE:.*]] = arith.constant 32 : index
// CHECK:         scf.parallel {{.*}} step (%[[TILE]], %[[TILE]])
// CHECK:           scf.parallel {{.*}} to (%[[TILE]], %[[TILE]]) step (%[[ONE]], %[[ONE]])
// CHECK-NOT:         scf.parallel
