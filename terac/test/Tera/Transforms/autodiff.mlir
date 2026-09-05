// RUN: tera-opt %s --tera-autodiff --split-input-file | FileCheck %s

// The pass leaves the original alone and adds three functions beside it: a
// forward that returns the result followed by the values the backward reads
// back, a backward that takes those values and the gradient of the result, and
// a wrapper that calls one and then the other. The wrapper keeps the signature
// the derivative always had: the original arguments plus the gradient of the
// result, returning one gradient per argument it differentiates. The original
// names all three.

// CHECK-LABEL: func @plain(
// CHECK-SAME: attributes {tera.bwd = @plain_bwd, tera.differentiable, tera.fwd = @plain_fwd, tera.vjp = @plain_vjp}
// CHECK: tera.exp
// CHECK: return
//
// CHECK-LABEL: func @plain_vjp
// CHECK-SAME: (%[[X:.*]]: tensor<4xf32>, %[[SEED:.*]]: tensor<4xf32>)
// CHECK-SAME: -> tensor<4xf32>
// CHECK-SAME: attributes {tera.diff_args = array<i64: 0>}
// CHECK: %[[FWD:.*]]:2 = call @plain_fwd(%[[X]])
// CHECK: %[[GRAD:.*]] = call @plain_bwd(%[[X]], %[[FWD]]#1, %[[SEED]])
// CHECK: return %[[GRAD]]
//
// CHECK-LABEL: func @plain_fwd
// CHECK-SAME: (%[[X:.*]]: tensor<4xf32>)
// CHECK-SAME: -> (tensor<4xf32>, tensor<4xf32>)
// CHECK: %[[EXP:.*]] = tera.exp %[[X]]
// CHECK: return %[[EXP]], %[[EXP]]
//
// CHECK-LABEL: func @plain_bwd
// CHECK-SAME: (%{{.*}}: tensor<4xf32>, %[[EXP:.*]]: tensor<4xf32>, %[[SEED:.*]]: tensor<4xf32>)
// CHECK-SAME: -> tensor<4xf32>
// CHECK-SAME: attributes {tera.diff_args = array<i64: 0>}
// CHECK: %[[GRAD:.*]] = tera.mul %[[SEED]], %[[EXP]]
// CHECK: return %[[GRAD]]
func.func @plain(%x: tensor<4xf32>) -> tensor<4xf32>
    attributes {tera.differentiable} {
  %0 = tera.exp %x : tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----

// A value used more than once collects one contribution per use, and they are
// summed as a tree rather than chained. `a * a * a` gives three contributions
// to `a`, so two additions and no more. The square is not recomputed in the
// backward; the forward hands it back as its one residual.

// CHECK-LABEL: func @accumulate_fwd
// CHECK-SAME: (%[[A:.*]]: tensor<4xf32>)
// CHECK: %[[SQUARE:.*]] = tera.mul %[[A]], %[[A]]
// CHECK: %[[CUBE:.*]] = tera.mul %[[SQUARE]], %[[A]]
// CHECK: return %[[CUBE]], %[[SQUARE]]
//
// CHECK-LABEL: func @accumulate_bwd
// CHECK-SAME: (%[[A:.*]]: tensor<4xf32>, %[[SQUARE:.*]]: tensor<4xf32>, %[[SEED:.*]]: tensor<4xf32>)
// CHECK: %[[OUTER:.*]] = tera.mul %[[SEED]], %[[A]]
// CHECK: %[[C0:.*]] = tera.mul %[[SEED]], %[[SQUARE]]
// CHECK: %[[C1:.*]] = tera.mul %[[OUTER]], %[[A]]
// CHECK: %[[C2:.*]] = tera.mul %[[OUTER]], %[[A]]
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
// an explicit zero rather than a missing result. The zero is made in the
// backward, so the wrapper returns whatever the backward returns.

// CHECK-LABEL: func @unused_vjp
// CHECK-SAME: -> (tensor<4xf32>, tensor<4xf32>)
// CHECK-SAME: {tera.diff_args = array<i64: 0, 1>}
//
// CHECK-LABEL: func @unused_bwd
// CHECK-SAME: (%{{.*}}: tensor<4xf32>, %{{.*}}: tensor<4xf32>, %[[EXP:.*]]: tensor<4xf32>, %[[SEED:.*]]: tensor<4xf32>)
// CHECK-SAME: -> (tensor<4xf32>, tensor<4xf32>)
// CHECK-SAME: {tera.diff_args = array<i64: 0, 1>}
// CHECK: %[[USED:.*]] = tera.mul %[[SEED]], %[[EXP]]
// CHECK: %[[ZERO:.*]] = tera.constant dense<0.000000e+00> : tensor<4xf32>
// CHECK: return %[[USED]], %[[ZERO]]
func.func @unused(%a: tensor<4xf32>, %b: tensor<4xf32>) -> tensor<4xf32>
    attributes {tera.differentiable} {
  %0 = tera.exp %a : tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----

// An integer argument carries no gradient, so it is not in diff_args and the
// derivative does not return one for it. It is still an argument of all three
// functions, because the forward needs it to compute with.

// CHECK-LABEL: func @mixed_vjp
// CHECK-SAME: (%{{.*}}: tensor<4xf32>, %{{.*}}: tensor<4xi32>, %{{.*}}: tensor<4xf32>)
// CHECK-SAME: -> tensor<4xf32>
// CHECK-SAME: {tera.diff_args = array<i64: 0>}
//
// CHECK-LABEL: func @mixed_fwd
// CHECK-SAME: (%[[A:.*]]: tensor<4xf32>, %[[COUNTS:.*]]: tensor<4xi32>)
// CHECK: %[[F:.*]] = tera.convert %[[COUNTS]]
// CHECK: %[[OUT:.*]] = tera.mul %[[A]], %[[F]]
// CHECK: return %[[OUT]], %[[F]]
//
// CHECK-LABEL: func @mixed_bwd
// CHECK-SAME: (%{{.*}}: tensor<4xf32>, %{{.*}}: tensor<4xi32>, %[[F:.*]]: tensor<4xf32>, %[[SEED:.*]]: tensor<4xf32>)
// CHECK-SAME: -> tensor<4xf32>
// CHECK-SAME: {tera.diff_args = array<i64: 0>}
// CHECK: %[[GRAD:.*]] = tera.mul %[[SEED]], %[[F]]
// CHECK: return %[[GRAD]]
func.func @mixed(%a: tensor<4xf32>, %counts: tensor<4xi32>) -> tensor<4xf32>
    attributes {tera.differentiable} {
  %0 = tera.convert %counts : tensor<4xi32> -> tensor<4xf32>
  %1 = tera.mul %a, %0 : tensor<4xf32>
  return %1 : tensor<4xf32>
}

// -----

// The walk stops at a stop_gradient, so the argument behind one is left with
// the zero it started with.

// CHECK-LABEL: func @blocked_bwd
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
// twice does not try to add the same symbols again.

// CHECK-LABEL: func @already_done
// CHECK-NOT: func @already_done_fwd
// CHECK-NOT: func @already_done_bwd
// CHECK-NOT: func @already_done_vjp
func.func @already_done(%a: tensor<4xf32>) -> tensor<4xf32>
    attributes {tera.differentiable, tera.vjp = @somewhere_else} {
  %0 = tera.exp %a : tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----

// A function with no marker is not differentiated.

// CHECK-LABEL: func @ordinary
// CHECK-NOT: _fwd
// CHECK-NOT: _bwd
// CHECK-NOT: _vjp
func.func @ordinary(%a: tensor<4xf32>) -> tensor<4xf32> {
  %0 = tera.exp %a : tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----

// A weight the caller left on the device is still there when the derivative
// runs, and the derivative's leading arguments are the primal's own -- so the
// mark carries across to all three. `_bwd` is the one that could go wrong: it
// takes the values the forward pass saved between the arguments and the seed,
// and the mark has to stay on the argument rather than follow the position.
//
// The gradients do not inherit it. They are results, and a result comes back.

// CHECK-LABEL: func @weighted(
// CHECK-SAME:    %{{.*}}: tensor<4xf32>, %{{.*}}: tensor<4xf32> {tera.device_resident})
// CHECK-LABEL: func @weighted_vjp(
// CHECK-SAME:    %{{.*}}: tensor<4xf32>, %{{.*}}: tensor<4xf32> {tera.device_resident},
// CHECK-SAME:    %{{.*}}: tensor<f32>)
// CHECK-LABEL: func @weighted_fwd(
// CHECK-SAME:    %{{.*}}: tensor<4xf32>, %{{.*}}: tensor<4xf32> {tera.device_resident})
// CHECK-LABEL: func @weighted_bwd(
// CHECK-SAME:    %{{.*}}: tensor<4xf32>, %{{.*}}: tensor<4xf32> {tera.device_resident},
// CHECK-SAME:    %{{.*}}: tensor<4xf32>, %{{.*}}: tensor<f32>)
func.func @weighted(%x: tensor<4xf32>, %w: tensor<4xf32> {tera.device_resident})
    -> tensor<f32> attributes {tera.differentiable} {
  %0 = tera.mul %x, %w : tensor<4xf32>
  %1 = tera.tanh %0 : tensor<4xf32>
  %2 = tera.mul %1, %1 : tensor<4xf32>
  %3 = tera.reduce sum, %2 {dimensions = array<i64: 0>}
      : tensor<4xf32> -> tensor<f32>
  return %3 : tensor<f32>
}

// -----

// A function with more than one result takes a seed for each, in result order,
// and still gives one gradient per differentiable argument: the contributions
// meet where the two results share a value. Nothing joins them into a scalar
// first, so no loss has to be invented to differentiate a function that
// answers with several things.

// Two results, so `_vjp` takes the two arguments and then a seed for each,
// and `_fwd` answers with both results before any residual.

// CHECK-LABEL: func @two(
// CHECK: func @two_vjp(%{{.*}}: tensor<3xf32>, %{{.*}}: tensor<3xf32>, %{{.*}}: tensor<3xf32>, %{{.*}}: tensor<3xf32>) -> (tensor<3xf32>, tensor<3xf32>)
// CHECK: func @two_fwd(%{{.*}}: tensor<3xf32>, %{{.*}}: tensor<3xf32>) -> (tensor<3xf32>, tensor<3xf32>,
// CHECK: func @two_bwd({{.*}}) -> (tensor<3xf32>, tensor<3xf32>)
func.func @two(%x: tensor<3xf32>, %w: tensor<3xf32>)
    -> (tensor<3xf32>, tensor<3xf32>) attributes {tera.differentiable} {
  %0 = tera.mul %x, %w : tensor<3xf32>
  %1 = tera.tanh %0 : tensor<3xf32>
  %2 = tera.add %x, %w : tensor<3xf32>
  return %1, %2 : tensor<3xf32>, tensor<3xf32>
}
