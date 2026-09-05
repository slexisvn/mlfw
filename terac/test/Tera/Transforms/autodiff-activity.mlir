// RUN: tera-opt %s --tera-autodiff --split-input-file | FileCheck %s

// Which arguments the pass differentiates is a question the function may
// answer for itself. Without `tera.diff_args` it takes every floating-point
// argument, which is a guess that costs a gradient nobody asked for; with it,
// the answer is the one the caller wrote down.

// CHECK-LABEL: func.func @weights_only_bwd
// CHECK-SAME:    -> tensor<4xf32>
// CHECK-SAME:    attributes {tera.diff_args = array<i64: 1>}
func.func @weights_only(%x: tensor<4xf32>, %w: tensor<4xf32>) -> tensor<4xf32>
    attributes {tera.differentiable, tera.diff_args = array<i64: 1>} {
  %0 = tera.mul %x, %w : tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----

// A subgraph no differentiable argument reaches takes no gradient, even where
// the reverse walk does reach it. Here the adjoint of the sum reaches the
// `arith.negf`, which has no rule; asking it for one would be an error, and
// the answer would be discarded, because nothing downstream of `%b` is
// differentiated.

// CHECK-LABEL: func.func @past_an_inactive_op_bwd
// CHECK-SAME:    (%{{.*}}: tensor<4xf32>, %{{.*}}: tensor<4xf32>, %[[SEED:.*]]: tensor<4xf32>)
// CHECK-NEXT:    return %[[SEED]]
func.func @past_an_inactive_op(%a: tensor<4xf32>, %b: tensor<4xf32>)
    -> tensor<4xf32>
    attributes {tera.differentiable, tera.diff_args = array<i64: 0>} {
  %0 = arith.negf %b : tensor<4xf32>
  %1 = tera.add %a, %0 : tensor<4xf32>
  return %1 : tensor<4xf32>
}

// -----

// A rule that reads none of the forward leaves nothing to carry between the
// two halves: `_fwd` returns the result alone, and `_bwd` takes the arguments
// and the seed. An addition is the smallest such rule -- both its gradients
// are the adjoint itself, and a value used twice sums its two contributions.

// CHECK-LABEL: func.func @nothing_to_save_fwd
// CHECK-SAME:    -> tensor<4xf32> {
// CHECK-LABEL: func.func @nothing_to_save_bwd
// CHECK-SAME:    (%{{.*}}: tensor<4xf32>, %[[SEED:.*]]: tensor<4xf32>)
// CHECK-NEXT:    %[[SUM:.*]] = tera.add %[[SEED]], %[[SEED]]
// CHECK-NEXT:    return %[[SUM]]
func.func @nothing_to_save(%a: tensor<4xf32>) -> tensor<4xf32>
    attributes {tera.differentiable} {
  %0 = tera.add %a, %a : tensor<4xf32>
  return %0 : tensor<4xf32>
}
