// RUN: tera-opt %s --tera-tile-and-fuse --split-input-file | FileCheck %s
// RUN: tera-opt %s --tera-tile-and-fuse --canonicalize --cse --split-input-file | FileCheck %s --check-prefix=FOLDED --allow-unused-prefixes

#identity = affine_map<(d0) -> (d0)>

func.func @elementwise(%a: tensor<64xf32>, %b: tensor<64xf32>) -> tensor<64xf32> {
  %empty = tensor.empty() : tensor<64xf32>
  %sum = linalg.generic {indexing_maps = [#identity, #identity, #identity],
                         iterator_types = ["parallel"]}
      ins(%a, %b : tensor<64xf32>, tensor<64xf32>)
      outs(%empty : tensor<64xf32>) {
  ^bb0(%x: f32, %y: f32, %out: f32):
    %add = arith.addf %x, %y : f32
    linalg.yield %add : f32
  } -> tensor<64xf32>
  return %sum : tensor<64xf32>
}

// CHECK-LABEL: func.func @elementwise
// CHECK-DAG:     %[[STEP:.*]] = arith.constant 16 : index
// CHECK:         scf.for %{{.*}} = %{{.*}} to %{{.*}} step %[[STEP]]
// CHECK:           tensor.extract_slice %{{.*}}[%{{.*}}] [16] [1]
// CHECK:           linalg.generic
// CHECK-SAME:        outs(%{{.*}} : tensor<16xf32>)
// CHECK:           tensor.insert_slice

// -----

func.func @matmul_fuses_its_fill(%a: tensor<32x64xf32>,
                                 %b: tensor<64x48xf32>) -> tensor<32x48xf32> {
  %zero = arith.constant 0.0 : f32
  %empty = tensor.empty() : tensor<32x48xf32>
  %init = linalg.fill ins(%zero : f32)
      outs(%empty : tensor<32x48xf32>) -> tensor<32x48xf32>
  %product = linalg.matmul
      ins(%a, %b : tensor<32x64xf32>, tensor<64x48xf32>)
      outs(%init : tensor<32x48xf32>) -> tensor<32x48xf32>
  return %product : tensor<32x48xf32>
}

// CHECK-LABEL: func.func @matmul_fuses_its_fill
// CHECK-NOT:     linalg.fill
// CHECK:         scf.for
// CHECK:           scf.for
// CHECK:             tensor.extract_slice %{{.*}}[%{{.*}}, %{{.*}}] [1, 16] [1, 1]
// CHECK:             linalg.fill
// CHECK-SAME:          -> tensor<1x16xf32>
// CHECK:             scf.for
// CHECK:               linalg.matmul
// CHECK-SAME:            ins(%{{.*}}, %{{.*}} : tensor<1x1xf32>, tensor<1x16xf32>)

// -----

#reversed = affine_map<(d0) -> (-d0 + 63)>
#identity = affine_map<(d0) -> (d0)>

func.func @negative_coefficient_is_left_alone(%a: tensor<64xf32>)
    -> tensor<64xf32> {
  %empty = tensor.empty() : tensor<64xf32>
  %flipped = linalg.generic {indexing_maps = [#reversed, #identity],
                             iterator_types = ["parallel"]}
      ins(%a : tensor<64xf32>) outs(%empty : tensor<64xf32>) {
  ^bb0(%x: f32, %out: f32):
    linalg.yield %x : f32
  } -> tensor<64xf32>
  return %flipped : tensor<64xf32>
}

// CHECK-LABEL: func.func @negative_coefficient_is_left_alone
// CHECK-NOT:     scf.for
// CHECK:         linalg.generic
// CHECK-SAME:      outs(%{{.*}} : tensor<64xf32>)

// -----

#identity = affine_map<(d0) -> (d0)>

// A loop over a `?` is cut to a vector the same way a loop over a known extent
// is. Nothing can say the tile divides it, so the loop is then split in two:
// one over as many whole tiles as fit, whose slices are `16` and not
// `min(16, n - i)`, and one iteration for whatever is left over. The first is
// what the vectorizer can take, and it is where nearly every element goes.

func.func @dynamic_extent_is_peeled(%a: tensor<?xf32>, %n: index)
    -> tensor<?xf32> {
  %empty = tensor.empty(%n) : tensor<?xf32>
  %squared = linalg.generic {indexing_maps = [#identity, #identity],
                             iterator_types = ["parallel"]}
      ins(%a : tensor<?xf32>) outs(%empty : tensor<?xf32>) {
  ^bb0(%x: f32, %out: f32):
    %product = arith.mulf %x, %x : f32
    linalg.yield %product : f32
  } -> tensor<?xf32>
  return %squared : tensor<?xf32>
}

// FOLDED-DAG:  #[[WHOLE:.*]] = affine_map<()[s0] -> ((s0 floordiv 16) * 16)>
// FOLDED-LABEL: func.func @dynamic_extent_is_peeled
// FOLDED:         %[[BOUND:.*]] = affine.apply #[[WHOLE]]()
// FOLDED:         scf.for %{{.*}} = %{{.*}} to %[[BOUND]] step %{{.*}}
// FOLDED:           tensor.extract_slice %{{.*}}[%{{.*}}] [16] [1]
// FOLDED:           linalg.generic
// FOLDED:           } -> tensor<16xf32>
// FOLDED:         scf.for %{{.*}} = %[[BOUND]] to
// FOLDED:           linalg.generic
// FOLDED:           } -> tensor<?xf32>

// -----

#identity = affine_map<(d0) -> (d0)>

func.func @shorter_than_a_vector_is_left_alone(%a: tensor<8xf32>)
    -> tensor<8xf32> {
  %empty = tensor.empty() : tensor<8xf32>
  %negated = linalg.generic {indexing_maps = [#identity, #identity],
                             iterator_types = ["parallel"]}
      ins(%a : tensor<8xf32>) outs(%empty : tensor<8xf32>) {
  ^bb0(%x: f32, %out: f32):
    %negative = arith.negf %x : f32
    linalg.yield %negative : f32
  } -> tensor<8xf32>
  return %negated : tensor<8xf32>
}

// CHECK-LABEL: func.func @shorter_than_a_vector_is_left_alone
// CHECK-NOT:     scf.for
// CHECK:         linalg.generic
