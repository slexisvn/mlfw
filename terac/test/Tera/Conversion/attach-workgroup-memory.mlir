// RUN: tera-opt %s --tera-attach-workgroup-memory --split-input-file \
// RUN:   | FileCheck %s

// The allocation becomes an attribution on the launch, which is where a
// kernel says how much shared memory its blocks need. Nothing is left to
// allocate, and every read and write goes to the block argument instead.

func.func @staged(%out: memref<64xf32>, %v: f32) {
  %c1 = arith.constant 1 : index
  %c64 = arith.constant 64 : index
  gpu.launch blocks(%bx, %by, %bz) in (%gx = %c1, %gy = %c1, %gz = %c1)
             threads(%tx, %ty, %tz) in (%sx = %c64, %sy = %c1, %sz = %c1) {
    %shared = memref.alloc() : memref<64xf32, #gpu.address_space<workgroup>>
    memref.store %v, %shared[%tx] : memref<64xf32, #gpu.address_space<workgroup>>
    gpu.barrier
    %read = memref.load %shared[%tx] : memref<64xf32, #gpu.address_space<workgroup>>
    memref.store %read, %out[%tx] : memref<64xf32>
    gpu.terminator
  }
  return
}

// CHECK-LABEL: func.func @staged
// CHECK:         gpu.launch
// CHECK-SAME:      workgroup(%[[SHARED:.*]] : memref<64xf32, #gpu.address_space<workgroup>>)
// CHECK-NOT:       memref.alloc
// CHECK:           memref.store %{{.*}}, %[[SHARED]]
// CHECK:           gpu.barrier
// CHECK:           memref.load %[[SHARED]]

// -----

// Two buffers become two attributions, in the order they were allocated.

func.func @two_buffers(%out: memref<64xf32>, %v: f32) {
  %c1 = arith.constant 1 : index
  %c64 = arith.constant 64 : index
  gpu.launch blocks(%bx, %by, %bz) in (%gx = %c1, %gy = %c1, %gz = %c1)
             threads(%tx, %ty, %tz) in (%sx = %c64, %sy = %c1, %sz = %c1) {
    %first = memref.alloc() : memref<64xf32, #gpu.address_space<workgroup>>
    %second = memref.alloc() : memref<32xf32, #gpu.address_space<workgroup>>
    memref.store %v, %first[%tx] : memref<64xf32, #gpu.address_space<workgroup>>
    %read = memref.load %second[%bx] : memref<32xf32, #gpu.address_space<workgroup>>
    memref.store %read, %out[%tx] : memref<64xf32>
    gpu.terminator
  }
  return
}

// CHECK-LABEL: func.func @two_buffers
// CHECK:         gpu.launch
// CHECK-SAME:      workgroup(%[[FIRST:.*]] : memref<64xf32, #gpu.address_space<workgroup>>,
// CHECK-SAME:                %[[SECOND:.*]] : memref<32xf32, #gpu.address_space<workgroup>>)
// CHECK:           memref.store %{{.*}}, %[[FIRST]]
// CHECK:           memref.load %[[SECOND]]

// -----

// An allocation outside a kernel is nothing to do with shared memory here, and
// one in the default space inside a kernel is thread-private scratch the
// memref lowering already knows what to do with. Neither is touched.

func.func @untouched(%v: f32) -> memref<8xf32> {
  %c1 = arith.constant 1 : index
  %outside = memref.alloc() : memref<8xf32>
  gpu.launch blocks(%bx, %by, %bz) in (%gx = %c1, %gy = %c1, %gz = %c1)
             threads(%tx, %ty, %tz) in (%sx = %c1, %sy = %c1, %sz = %c1) {
    %private = memref.alloc() : memref<4xf32>
    memref.store %v, %private[%tx] : memref<4xf32>
    memref.dealloc %private : memref<4xf32>
    gpu.terminator
  }
  return %outside : memref<8xf32>
}

// CHECK-LABEL: func.func @untouched
// CHECK:         memref.alloc() : memref<8xf32>
// CHECK:         gpu.launch
// CHECK-NOT:     workgroup(
// CHECK:           memref.alloc() : memref<4xf32>
