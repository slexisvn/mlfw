// RUN: tera-opt %s --tera-tile-contraction-to-shared --canonicalize --cse \
// RUN:   --split-input-file | FileCheck %s

// The shape the whole pass exists to build, and every line of it is load
// bearing: two block loops stepping by the tile, the shared buffers between
// them and the thread loop, the contracted loop *inside* the thread loop, and
// a barrier on each side of the reading of the tiles. The accumulator is an
// iteration argument rather than a buffer slot, so the destination is read
// once before the loop and written once after it.

#lhs = affine_map<(m, n, k) -> (m, k)>
#rhs = affine_map<(m, n, k) -> (k, n)>
#out = affine_map<(m, n, k) -> (m, n)>

func.func @matmul(%a: memref<64x64xf32>, %b: memref<64x64xf32>,
                  %c: memref<64x64xf32>) {
  linalg.generic {indexing_maps = [#lhs, #rhs, #out],
                  iterator_types = ["parallel", "parallel", "reduction"]}
      ins(%a, %b : memref<64x64xf32>, memref<64x64xf32>)
      outs(%c : memref<64x64xf32>) {
  ^bb0(%x: f32, %y: f32, %acc: f32):
    %0 = arith.mulf %x, %y : f32
    %1 = arith.addf %acc, %0 : f32
    linalg.yield %1 : f32
  }
  return
}

// CHECK-LABEL: func.func @matmul
// CHECK-DAG:     %[[ZERO:[a-z0-9_]+]] = arith.constant 0 : index
// CHECK-DAG:     %[[ONE:[a-z0-9_]+]] = arith.constant 1 : index
// CHECK-DAG:     %[[TILE:[a-z0-9_]+]] = arith.constant 32 : index
// CHECK-DAG:     %[[WHOLE:[a-z0-9_]+]] = arith.constant 64 : index
// CHECK:         scf.parallel (%[[BI:[a-z0-9_]+]], %[[BJ:[a-z0-9_]+]]) = (%[[ZERO]], %[[ZERO]])
// CHECK-SAME:      to (%[[WHOLE]], %[[WHOLE]]) step (%[[TILE]], %[[TILE]])
// CHECK:           %[[SHA:[a-z0-9_]+]] = memref.alloc() : memref<32x32xf32, #gpu.address_space<workgroup>>
// CHECK:           %[[SHB:[a-z0-9_]+]] = memref.alloc() : memref<32x32xf32, #gpu.address_space<workgroup>>
// CHECK:           scf.parallel (%[[TI:[a-z0-9_]+]], %[[TJ:[a-z0-9_]+]]) = (%[[ZERO]], %[[ZERO]])
// CHECK-SAME:        to (%[[TILE]], %[[TILE]]) step (%[[ONE]], %[[ONE]])
// CHECK:             %[[M:[a-z0-9_]+]] = arith.addi %[[BI]], %[[TI]]
// CHECK:             %[[N:[a-z0-9_]+]] = arith.addi %[[BJ]], %[[TJ]]
// CHECK:             %[[INIT:[a-z0-9_]+]] = memref.load %arg2[%[[M]], %[[N]]]
// CHECK:             %[[SUM:[a-z0-9_]+]] = scf.for %[[K:[a-z0-9_]+]] = %[[ZERO]] to %[[WHOLE]] step %[[TILE]]
// CHECK-SAME:            iter_args(%[[CARRY:[a-z0-9_]+]] = %[[INIT]])
// CHECK:               %[[AK:[a-z0-9_]+]] = arith.addi %[[K]], %[[TJ]]
// CHECK:               %[[AV:[a-z0-9_]+]] = memref.load %arg0[%[[M]], %[[AK]]]
// CHECK:               memref.store %[[AV]], %[[SHA]][%[[TI]], %[[TJ]]]
// CHECK:               %[[BK:[a-z0-9_]+]] = arith.addi %[[K]], %[[TI]]
// CHECK:               %[[BV:[a-z0-9_]+]] = memref.load %arg1[%[[BK]], %[[N]]]
// CHECK:               memref.store %[[BV]], %[[SHB]][%[[TI]], %[[TJ]]]
// CHECK:               gpu.barrier
// CHECK:               %[[TILED:[a-z0-9_]+]] = scf.for %[[KK:[a-z0-9_]+]] = %[[ZERO]] to %[[TILE]] step %[[ONE]]
// CHECK-SAME:              iter_args(%[[INNER:[a-z0-9_]+]] = %[[CARRY]])
// CHECK:                 %[[SA:[a-z0-9_]+]] = memref.load %[[SHA]][%[[TI]], %[[KK]]]
// CHECK:                 %[[SB:[a-z0-9_]+]] = memref.load %[[SHB]][%[[KK]], %[[TJ]]]
// CHECK:                 %[[PRODUCT:[a-z0-9_]+]] = arith.mulf %[[SA]], %[[SB]]
// CHECK:                 %[[NEXT:[a-z0-9_]+]] = arith.addf %[[INNER]], %[[PRODUCT]]
// CHECK:                 scf.yield %[[NEXT]]
// CHECK:               gpu.barrier
// CHECK:               scf.yield %[[TILED]]
// CHECK:             memref.store %[[SUM]], %arg2[%[[M]], %[[N]]]
// CHECK-NOT:       linalg.generic

// -----

// A batch axis becomes a third block loop stepping by one, which is the third
// grid dimension. The tile still cuts only the two axes it is a tile of.

#lhs = affine_map<(b, m, n, k) -> (b, m, k)>
#rhs = affine_map<(b, m, n, k) -> (b, k, n)>
#out = affine_map<(b, m, n, k) -> (b, m, n)>

func.func @batched(%a: memref<4x64x64xf32>, %b: memref<4x64x64xf32>,
                   %c: memref<4x64x64xf32>) {
  linalg.generic {indexing_maps = [#lhs, #rhs, #out],
                  iterator_types = ["parallel", "parallel", "parallel",
                                    "reduction"]}
      ins(%a, %b : memref<4x64x64xf32>, memref<4x64x64xf32>)
      outs(%c : memref<4x64x64xf32>) {
  ^bb0(%x: f32, %y: f32, %acc: f32):
    %0 = arith.mulf %x, %y : f32
    %1 = arith.addf %acc, %0 : f32
    linalg.yield %1 : f32
  }
  return
}

// CHECK-LABEL: func.func @batched
// CHECK-DAG:     %[[ONE:[a-z0-9_]+]] = arith.constant 1 : index
// CHECK-DAG:     %[[TILE:[a-z0-9_]+]] = arith.constant 32 : index
// CHECK:         scf.parallel (%[[BB:[a-z0-9_]+]], %[[BI:[a-z0-9_]+]], %[[BJ:[a-z0-9_]+]]) =
// CHECK-SAME:      step (%[[ONE]], %[[TILE]], %[[TILE]])
// CHECK:           scf.parallel
// CHECK:             memref.load %arg2[%[[BB]], %{{.*}}, %{{.*}}]
// CHECK:             scf.for
// CHECK:               memref.load %arg0[%[[BB]], %{{.*}}, %{{.*}}]
// CHECK:               memref.load %arg1[%[[BB]], %{{.*}}, %{{.*}}]

// -----

// Which operand is the left one is read off the indexing maps rather than the
// operand order, so a contraction storing its left operand transposed stages
// the same tiles from the axes that actually carry `m` and `k`.

#lhs = affine_map<(m, n, k) -> (k, m)>
#rhs = affine_map<(m, n, k) -> (k, n)>
#out = affine_map<(m, n, k) -> (m, n)>

func.func @transposed_lhs(%a: memref<64x64xf32>, %b: memref<64x64xf32>,
                          %c: memref<64x64xf32>) {
  linalg.generic {indexing_maps = [#lhs, #rhs, #out],
                  iterator_types = ["parallel", "parallel", "reduction"]}
      ins(%a, %b : memref<64x64xf32>, memref<64x64xf32>)
      outs(%c : memref<64x64xf32>) {
  ^bb0(%x: f32, %y: f32, %acc: f32):
    %0 = arith.mulf %x, %y : f32
    %1 = arith.addf %acc, %0 : f32
    linalg.yield %1 : f32
  }
  return
}

// CHECK-LABEL: func.func @transposed_lhs
// CHECK:         scf.parallel (%[[BI:[a-z0-9_]+]], %[[BJ:[a-z0-9_]+]])
// CHECK:           %[[SHA:[a-z0-9_]+]] = memref.alloc() : memref<32x32xf32, #gpu.address_space<workgroup>>
// CHECK:           scf.parallel (%[[TI:[a-z0-9_]+]], %[[TJ:[a-z0-9_]+]])
// CHECK:             %[[M:[a-z0-9_]+]] = arith.addi %[[BI]], %[[TI]]
// CHECK:             scf.for %[[K:[a-z0-9_]+]] =
// CHECK:               %[[AK:[a-z0-9_]+]] = arith.addi %[[K]], %[[TJ]]
// CHECK:               %[[AV:[a-z0-9_]+]] = memref.load %arg0[%[[AK]], %[[M]]]
// CHECK:               memref.store %[[AV]], %[[SHA]][%[[TI]], %[[TJ]]]

// -----

// A tile that does not divide an extent would leave a partial tile, and a
// thread reading one would need a bounds test the whole shape is here to
// avoid. 48 is not a multiple of the warp, so no tile the target allows
// divides it and the contraction keeps the lowering it had.

#lhs = affine_map<(m, n, k) -> (m, k)>
#rhs = affine_map<(m, n, k) -> (k, n)>
#out = affine_map<(m, n, k) -> (m, n)>

func.func @indivisible(%a: memref<48x48xf32>, %b: memref<48x48xf32>,
                       %c: memref<48x48xf32>) {
  linalg.generic {indexing_maps = [#lhs, #rhs, #out],
                  iterator_types = ["parallel", "parallel", "reduction"]}
      ins(%a, %b : memref<48x48xf32>, memref<48x48xf32>)
      outs(%c : memref<48x48xf32>) {
  ^bb0(%x: f32, %y: f32, %acc: f32):
    %0 = arith.mulf %x, %y : f32
    %1 = arith.addf %acc, %0 : f32
    linalg.yield %1 : f32
  }
  return
}

// CHECK-LABEL: func.func @indivisible
// CHECK:         linalg.generic
// CHECK-NOT:     gpu.barrier

// -----

// An extent that is not known cannot be shown to be a whole number of tiles,
// and the tile is chosen so that it is.

#lhs = affine_map<(m, n, k) -> (m, k)>
#rhs = affine_map<(m, n, k) -> (k, n)>
#out = affine_map<(m, n, k) -> (m, n)>

func.func @dynamic(%a: memref<?x64xf32>, %b: memref<64x64xf32>,
                   %c: memref<?x64xf32>) {
  linalg.generic {indexing_maps = [#lhs, #rhs, #out],
                  iterator_types = ["parallel", "parallel", "reduction"]}
      ins(%a, %b : memref<?x64xf32>, memref<64x64xf32>)
      outs(%c : memref<?x64xf32>) {
  ^bb0(%x: f32, %y: f32, %acc: f32):
    %0 = arith.mulf %x, %y : f32
    %1 = arith.addf %acc, %0 : f32
    linalg.yield %1 : f32
  }
  return
}

// CHECK-LABEL: func.func @dynamic
// CHECK:         linalg.generic
// CHECK-NOT:     gpu.barrier

// -----

// Two batch axes would be four block loops, and a grid has three dimensions.

#lhs = affine_map<(b0, b1, m, n, k) -> (b0, b1, m, k)>
#rhs = affine_map<(b0, b1, m, n, k) -> (b0, b1, k, n)>
#out = affine_map<(b0, b1, m, n, k) -> (b0, b1, m, n)>

func.func @two_batches(%a: memref<2x2x64x64xf32>, %b: memref<2x2x64x64xf32>,
                       %c: memref<2x2x64x64xf32>) {
  linalg.generic {indexing_maps = [#lhs, #rhs, #out],
                  iterator_types = ["parallel", "parallel", "parallel",
                                    "parallel", "reduction"]}
      ins(%a, %b : memref<2x2x64x64xf32>, memref<2x2x64x64xf32>)
      outs(%c : memref<2x2x64x64xf32>) {
  ^bb0(%x: f32, %y: f32, %acc: f32):
    %0 = arith.mulf %x, %y : f32
    %1 = arith.addf %acc, %0 : f32
    linalg.yield %1 : f32
  }
  return
}

// CHECK-LABEL: func.func @two_batches
// CHECK:         linalg.generic
// CHECK-NOT:     gpu.barrier

// -----

// On tensors there is no buffer to share: the tile every thread writes part of
// has one producer per value, and a cross-thread write is not a value.

#lhs = affine_map<(m, n, k) -> (m, k)>
#rhs = affine_map<(m, n, k) -> (k, n)>
#out = affine_map<(m, n, k) -> (m, n)>

func.func @on_tensors(%a: tensor<64x64xf32>, %b: tensor<64x64xf32>,
                      %c: tensor<64x64xf32>) -> tensor<64x64xf32> {
  %0 = linalg.generic {indexing_maps = [#lhs, #rhs, #out],
                       iterator_types = ["parallel", "parallel", "reduction"]}
      ins(%a, %b : tensor<64x64xf32>, tensor<64x64xf32>)
      outs(%c : tensor<64x64xf32>) {
  ^bb0(%x: f32, %y: f32, %acc: f32):
    %1 = arith.mulf %x, %y : f32
    %2 = arith.addf %acc, %1 : f32
    linalg.yield %2 : f32
  } -> tensor<64x64xf32>
  return %0 : tensor<64x64xf32>
}

// CHECK-LABEL: func.func @on_tensors
// CHECK:         linalg.generic
// CHECK-NOT:     gpu.barrier

// -----

// An elementwise op is not a contraction: there is no axis to stage a tile
// along, and every element is read once.

#same = affine_map<(d0, d1) -> (d0, d1)>

func.func @elementwise(%a: memref<64x64xf32>, %b: memref<64x64xf32>,
                       %c: memref<64x64xf32>) {
  linalg.generic {indexing_maps = [#same, #same, #same],
                  iterator_types = ["parallel", "parallel"]}
      ins(%a, %b : memref<64x64xf32>, memref<64x64xf32>)
      outs(%c : memref<64x64xf32>) {
  ^bb0(%x: f32, %y: f32, %out: f32):
    %0 = arith.addf %x, %y : f32
    linalg.yield %0 : f32
  }
  return
}

// CHECK-LABEL: func.func @elementwise
// CHECK:         linalg.generic
// CHECK-NOT:     gpu.barrier
