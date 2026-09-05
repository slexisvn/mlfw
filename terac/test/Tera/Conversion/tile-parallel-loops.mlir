// RUN: tera-opt %s --tera-tile-parallel-loops --cse --split-input-file \
// RUN:   | FileCheck %s
// RUN: tera-opt %s --tera-tile-parallel-loops=tile-sizes=8,8 --cse \
// RUN:   --split-input-file | FileCheck %s --check-prefix=GIVEN

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
// LAUNCH:         gpu.launch blocks{{.*}} in (%{{.*}} = %[[GX:.*]], %{{.*}} = %[[GY:.*]], %{{.*}} = %[[GZ:.*]])
// LAUNCH-SAME:      threads{{.*}} in (%{{.*}} = %[[BX:.*]], %{{.*}} = %[[BY:.*]], %{{.*}} = %[[BZ:.*]])
// LAUNCH-DAG:     %[[BX]] = arith.constant 256 : index
// LAUNCH-DAG:     %[[GY]] = arith.constant 64 : index

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
// LAUNCH:         gpu.launch blocks

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
// LAUNCH:         gpu.launch blocks{{.*}} threads{{.*}} in (%{{.*}} = %[[BX:.*]],
// LAUNCH-DAG:     %[[BX]] = arith.constant 17 : index

// -----

// A loop carrying a reduction is left whole. The upstream tiling it would go
// through does not carry the reduction into the tile, so cutting one is a
// wrong answer rather than a slow one.

func.func @leaves_a_reduction_alone(%in: memref<64xf32>) -> f32 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c64 = arith.constant 64 : index
  %zero = arith.constant 0.0 : f32
  %total = scf.parallel (%i) = (%c0) to (%c64) step (%c1) init (%zero) -> f32 {
    %element = memref.load %in[%i] : memref<64xf32>
    scf.reduce(%element : f32) {
    ^bb0(%lhs: f32, %rhs: f32):
      %sum = arith.addf %lhs, %rhs : f32
      scf.reduce.return %sum : f32
    }
  }
  return %total : f32
}

// CHECK-LABEL: func.func @leaves_a_reduction_alone
// CHECK:         scf.parallel
// CHECK-NOT:       scf.parallel

// LAUNCH-LABEL: func.func @leaves_a_reduction_alone
// LAUNCH:         scf.parallel
