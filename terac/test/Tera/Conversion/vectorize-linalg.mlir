// RUN: tera-opt %s --tera-vectorize-linalg --split-input-file | FileCheck %s
// RUN: tera-opt %s --tera-vectorize-linalg=max-vector-elements=8 \
// RUN:   --split-input-file | FileCheck %s --check-prefix=NARROW

#identity = affine_map<(d0) -> (d0)>

func.func @elementwise(%a: tensor<16xf32>, %b: tensor<16xf32>) -> tensor<16xf32> {
  %empty = tensor.empty() : tensor<16xf32>
  %sum = linalg.generic {indexing_maps = [#identity, #identity, #identity],
                         iterator_types = ["parallel"]}
      ins(%a, %b : tensor<16xf32>, tensor<16xf32>)
      outs(%empty : tensor<16xf32>) {
  ^bb0(%x: f32, %y: f32, %out: f32):
    %add = arith.addf %x, %y : f32
    linalg.yield %add : f32
  } -> tensor<16xf32>
  return %sum : tensor<16xf32>
}

// CHECK-LABEL: func.func @elementwise
// CHECK-NOT:     linalg.generic
// CHECK:         vector.transfer_read
// CHECK:         vector.transfer_read
// CHECK:         arith.addf %{{.*}}, %{{.*}} : vector<16xf32>
// CHECK:         vector.transfer_write

// NARROW-LABEL: func.func @elementwise
// NARROW-NOT:    vector.transfer_read
// NARROW:        linalg.generic

// -----

#identity = affine_map<(d0) -> (d0)>

func.func @wider_than_the_budget(%a: tensor<4096xf32>) -> tensor<4096xf32> {
  %empty = tensor.empty() : tensor<4096xf32>
  %negated = linalg.generic {indexing_maps = [#identity, #identity],
                             iterator_types = ["parallel"]}
      ins(%a : tensor<4096xf32>) outs(%empty : tensor<4096xf32>) {
  ^bb0(%x: f32, %out: f32):
    %negative = arith.negf %x : f32
    linalg.yield %negative : f32
  } -> tensor<4096xf32>
  return %negated : tensor<4096xf32>
}

// CHECK-LABEL: func.func @wider_than_the_budget
// CHECK-NOT:     vector.transfer_read
// CHECK:         linalg.generic

// -----

#identity = affine_map<(d0) -> (d0)>

func.func @dynamic_extent(%a: tensor<?xf32>, %n: index) -> tensor<?xf32> {
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

// CHECK-LABEL: func.func @dynamic_extent
// CHECK-NOT:     vector.transfer_read
// CHECK:         linalg.generic

// -----

#row = affine_map<(d0, d1) -> (d0, d1)>
#column = affine_map<(d0, d1) -> (d0)>

func.func @reduction(%a: tensor<4x8xf32>) -> tensor<4xf32> {
  %zero = arith.constant 0.0 : f32
  %empty = tensor.empty() : tensor<4xf32>
  %init = linalg.fill ins(%zero : f32)
      outs(%empty : tensor<4xf32>) -> tensor<4xf32>
  %total = linalg.generic {indexing_maps = [#row, #column],
                           iterator_types = ["parallel", "reduction"]}
      ins(%a : tensor<4x8xf32>) outs(%init : tensor<4xf32>) {
  ^bb0(%x: f32, %out: f32):
    %add = arith.addf %x, %out : f32
    linalg.yield %add : f32
  } -> tensor<4xf32>
  return %total : tensor<4xf32>
}

// CHECK-LABEL: func.func @reduction
// CHECK-NOT:     linalg.generic
// CHECK:         vector.multi_reduction <add>
// CHECK-SAME:      vector<4x8xf32> to vector<4xf32>
