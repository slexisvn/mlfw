// RUN: tera-opt %s --tera-autodiff --split-input-file | FileCheck %s

// The pass leaves the original alone and adds a second function beside it. The
// derivative takes the original arguments plus the gradient of the result, and
// returns one gradient per argument it differentiates.

// CHECK-LABEL: func @plain(
// CHECK-SAME: attributes {tera.differentiable, tera.vjp = @plain_vjp}
// CHECK: tera.exp
// CHECK: return
//
// CHECK-LABEL: func @plain_vjp(
// CHECK-SAME: %[[X:.*]]: tensor<4xf32>, %[[SEED:.*]]: tensor<4xf32>)
// CHECK-SAME: -> tensor<4xf32>
// CHECK-SAME: attributes {tera.diff_args = array<i64: 0>}
// CHECK: %[[FWD:.*]] = tera.exp %[[X]]
// CHECK: %[[GRAD:.*]] = tera.mul %[[SEED]], %[[FWD]]
// CHECK: return %[[GRAD]]
func.func @plain(%x: tensor<4xf32>) -> tensor<4xf32>
    attributes {tera.differentiable} {
  %0 = tera.exp %x : tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----

// A value used more than once collects one contribution per use, and they are
// summed as a tree rather than chained. `a * a * a` gives three contributions
// to `a`, so two additions and no more.

// CHECK-LABEL: func @accumulate_vjp
// CHECK: %[[SQUARE:.*]] = tera.mul %arg0, %arg0
// CHECK: tera.mul %[[SQUARE]], %arg0
// CHECK: %[[OUTER:.*]] = tera.mul %arg1, %arg0
// CHECK: %[[C0:.*]] = tera.mul %arg1, %[[SQUARE]]
// CHECK: %[[C1:.*]] = tera.mul %[[OUTER]], %arg0
// CHECK: %[[C2:.*]] = tera.mul %[[OUTER]], %arg0
// CHECK: %[[HALF:.*]] = tera.add %[[C0]], %[[C1]]
// CHECK: %[[TOTAL:.*]] = tera.add %[[HALF]], %[[C2]]
// CHECK: return %[[TOTAL]]
func.func @accumulate(%a: tensor<4xf32>) -> tensor<4xf32>
    attributes {tera.differentiable} {
  %0 = tera.mul %a, %a : tensor<4xf32>
  %1 = tera.mul %0, %a : tensor<4xf32>
  return %1 : tensor<4xf32>
}

// -----

// An argument the result does not depend on still gets a gradient, and it is
// an explicit zero rather than a missing result.

// CHECK-LABEL: func @unused_vjp
// CHECK-SAME: -> (tensor<4xf32>, tensor<4xf32>)
// CHECK-SAME: {tera.diff_args = array<i64: 0, 1>}
// CHECK: %[[USED:.*]] = tera.mul
// CHECK: %[[ZERO:.*]] = tera.constant dense<0.000000e+00> : tensor<4xf32>
// CHECK: return %[[USED]], %[[ZERO]]
func.func @unused(%a: tensor<4xf32>, %b: tensor<4xf32>) -> tensor<4xf32>
    attributes {tera.differentiable} {
  %0 = tera.exp %a : tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----

// An integer argument carries no gradient, so it is not in diff_args and the
// derivative does not return one for it.

// CHECK-LABEL: func @mixed_vjp
// CHECK-SAME: (%{{.*}}: tensor<4xf32>, %{{.*}}: tensor<4xi32>, %{{.*}}: tensor<4xf32>)
// CHECK-SAME: -> tensor<4xf32>
// CHECK-SAME: {tera.diff_args = array<i64: 0>}
func.func @mixed(%a: tensor<4xf32>, %counts: tensor<4xi32>) -> tensor<4xf32>
    attributes {tera.differentiable} {
  %0 = tera.convert %counts : tensor<4xi32> -> tensor<4xf32>
  %1 = tera.mul %a, %0 : tensor<4xf32>
  return %1 : tensor<4xf32>
}

// -----

// The walk stops at a stop_gradient, so the argument behind one is left with
// the zero it started with.

// CHECK-LABEL: func @blocked_vjp
// CHECK: %[[ZERO:.*]] = tera.constant dense<0.000000e+00>
// CHECK: return %[[ZERO]]
func.func @blocked(%a: tensor<4xf32>) -> tensor<4xf32>
    attributes {tera.differentiable} {
  %0 = tera.stop_gradient %a : tensor<4xf32>
  %1 = tera.exp %0 : tensor<4xf32>
  return %1 : tensor<4xf32>
}

// -----

// A function that already has a derivative is left alone, so running the pass
// twice does not try to add the same symbol again.

// CHECK-LABEL: func @already_done
// CHECK-NOT: func @already_done_vjp
// CHECK-NOT: func @already_done_vjp_vjp
func.func @already_done(%a: tensor<4xf32>) -> tensor<4xf32>
    attributes {tera.differentiable, tera.vjp = @somewhere_else} {
  %0 = tera.exp %a : tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----

// A function with no marker is not differentiated.

// CHECK-LABEL: func @ordinary
// CHECK-NOT: _vjp
func.func @ordinary(%a: tensor<4xf32>) -> tensor<4xf32> {
  %0 = tera.exp %a : tensor<4xf32>
  return %0 : tensor<4xf32>
}
