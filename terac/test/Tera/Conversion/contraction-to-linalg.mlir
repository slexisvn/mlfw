// RUN: tera-opt %s --convert-tera-to-linalg | FileCheck %s

// The iteration space runs batch, lhs free, rhs free, contracted; the result
// carries all but the last, which is the reduction.
// CHECK: #[[LHS:.*]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
// CHECK: #[[RHS:.*]] = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>
// CHECK: #[[OUT:.*]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>

// CHECK-LABEL: func @batched_dot
func.func @batched_dot(%a: tensor<8x2x4xf32>, %b: tensor<8x4x3xf32>) -> tensor<8x2x3xf32> {
  // CHECK: %[[Z:.*]] = arith.constant 0.000000e+00 : f32
  // CHECK: %[[E:.*]] = tensor.empty() : tensor<8x2x3xf32>
  // CHECK: %[[ACC:.*]] = linalg.fill ins(%[[Z]] : f32) outs(%[[E]] : tensor<8x2x3xf32>)
  // CHECK: linalg.generic
  // CHECK-SAME: indexing_maps = [#[[LHS]], #[[RHS]], #[[OUT]]]
  // CHECK-SAME: iterator_types = ["parallel", "parallel", "parallel", "reduction"]
  // CHECK-SAME: outs(%[[ACC]]
  // CHECK: %[[P:.*]] = arith.mulf %in, %in_0 : f32
  // CHECK: %[[S:.*]] = arith.addf %out, %[[P]] : f32
  // CHECK: linalg.yield %[[S]]
  %0 = tera.dot %a, %b {lhs_batch = array<i64: 0>,
                        lhs_contracting = array<i64: 2>,
                        rhs_batch = array<i64: 0>,
                        rhs_contracting = array<i64: 1>}
      : (tensor<8x2x4xf32>, tensor<8x4x3xf32>) -> tensor<8x2x3xf32>
  return %0 : tensor<8x2x3xf32>
}

// The accumulator starts at the combiner's identity: -inf for maximum, so the
// first real element always wins.
// CHECK-LABEL: func @reduce_maximum
func.func @reduce_maximum(%a: tensor<2x3x4xf32>) -> tensor<3xf32> {
  // CHECK: %[[NEGINF:.*]] = arith.constant 0xFF800000 : f32
  // CHECK: %[[ACC:.*]] = linalg.fill ins(%[[NEGINF]] : f32)
  // CHECK: linalg.reduce { arith.maximumf } ins(%arg0 : tensor<2x3x4xf32>) outs(%[[ACC]] : tensor<3xf32>) dimensions = [0, 2]
  %0 = tera.reduce maximum, %a {dimensions = array<i64: 0, 2>}
      : tensor<2x3x4xf32> -> tensor<3xf32>
  return %0 : tensor<3xf32>
}

// CHECK-LABEL: func @reduce_minimum
func.func @reduce_minimum(%a: tensor<2x3xf32>) -> tensor<2xf32> {
  // CHECK: %[[POSINF:.*]] = arith.constant 0x7F800000 : f32
  // CHECK: %[[ACC:.*]] = linalg.fill ins(%[[POSINF]] : f32)
  // CHECK: linalg.reduce { arith.minimumf } ins(%arg0 : tensor<2x3xf32>) outs(%[[ACC]] : tensor<2xf32>)
  %0 = tera.reduce minimum, %a {dimensions = array<i64: 1>}
      : tensor<2x3xf32> -> tensor<2xf32>
  return %0 : tensor<2xf32>
}

// CHECK-LABEL: func @reduce_sum
func.func @reduce_sum(%a: tensor<2x2xf32>) -> tensor<f32> {
  // CHECK: %[[Z:.*]] = arith.constant 0.000000e+00 : f32
  // CHECK: %[[ACC:.*]] = linalg.fill ins(%[[Z]] : f32)
  // CHECK: linalg.reduce { arith.addf } ins(%arg0 : tensor<2x2xf32>) outs(%[[ACC]] : tensor<f32>) dimensions = [0, 1]
  %0 = tera.reduce sum, %a {dimensions = array<i64: 0, 1>}
      : tensor<2x2xf32> -> tensor<f32>
  return %0 : tensor<f32>
}

// CHECK-LABEL: func @reduce_product
func.func @reduce_product(%a: tensor<2x3xi32>) -> tensor<2xi32> {
  // CHECK: %[[ONE:.*]] = arith.constant 1 : i32
  // CHECK: %[[ACC:.*]] = linalg.fill ins(%[[ONE]] : i32)
  // CHECK: linalg.reduce { arith.muli {overflowFlags = #arith.overflow<none>} } ins(%arg0 : tensor<2x3xi32>) outs(%[[ACC]] : tensor<2xi32>)
  %0 = tera.reduce product, %a {dimensions = array<i64: 1>}
      : tensor<2x3xi32> -> tensor<2xi32>
  return %0 : tensor<2xi32>
}

// linalg.reduce needs its dimensions sorted; tera does not, because the result
// shape is the same either way. Sorting is the whole difference.
// CHECK-LABEL: func @reduce_unsorted_dimensions
func.func @reduce_unsorted_dimensions(%a: tensor<2x3x4xf32>) -> tensor<3xf32> {
  // CHECK: linalg.reduce { arith.addf } ins(%arg0 : tensor<2x3x4xf32>) outs({{.*}}) dimensions = [0, 2]
  %0 = tera.reduce sum, %a {dimensions = array<i64: 2, 0>}
      : tensor<2x3x4xf32> -> tensor<3xf32>
  return %0 : tensor<3xf32>
}
