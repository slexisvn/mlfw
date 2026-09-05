// RUN: tera-opt %s --tera-forward-mode --split-input-file | FileCheck %s

// The pass adds one function returning the result and the direction it moves
// in, and points the original at it. The tangents come after the arguments,
// one per entry of `tera.diff_args`.

// CHECK-LABEL: func.func @linear(
// CHECK-SAME:    attributes {tera.differentiable, tera.jvp = @linear_jvp}
//
// CHECK-LABEL: func.func @linear_jvp
// CHECK-SAME:    -> (tensor<4xf32>, tensor<4xf32>)
// CHECK-SAME:    attributes {tera.diff_args = array<i64: 0, 1>}
// CHECK:         %[[OUT:.*]] = tera.add %arg0, %arg1
// CHECK:         %[[MOVED:.*]] = tera.add %arg2, %arg3
// CHECK:         return %[[OUT]], %[[MOVED]]
func.func @linear(%x: tensor<4xf32>, %w: tensor<4xf32>) -> tensor<4xf32>
    attributes {tera.differentiable} {
  %0 = tera.add %x, %w : tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----

// An op linear in each operand separately sums one term per operand, each with
// that operand swapped for its tangent. Nothing about the product rule is
// written down for `tera.mul`: the trait says which shape its derivative has
// and the pass builds it.

// CHECK-LABEL: func.func @multilinear_jvp
// CHECK:         %[[LEFT:.*]] = tera.mul %arg2, %arg1
// CHECK:         %[[RIGHT:.*]] = tera.mul %arg0, %arg3
// CHECK:         %[[MOVED:.*]] = tera.add %[[LEFT]], %[[RIGHT]]
// CHECK:         return %{{.*}}, %[[MOVED]]
func.func @multilinear(%x: tensor<4xf32>, %w: tensor<4xf32>) -> tensor<4xf32>
    attributes {tera.differentiable} {
  %0 = tera.mul %x, %w : tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----

// An elementwise op has a diagonal Jacobian, which is its own transpose, so
// the slope its reverse rule already builds is the one the forward pass wants.
// `tera.tanh` has no forward rule written for it anywhere; what appears here
// is `TanhOp::buildVjp` run with the tangent in place of the adjoint.

// CHECK-LABEL: func.func @elementwise_jvp
// CHECK:         %[[OUT:.*]] = tera.tanh %arg0
// CHECK:         %[[ONE:.*]] = tera.constant dense<1.000000e+00>
// CHECK:         %[[SQUARE:.*]] = tera.mul %[[OUT]], %[[OUT]]
// CHECK:         %[[SLOPE:.*]] = tera.sub %[[ONE]], %[[SQUARE]]
// CHECK:         %[[MOVED:.*]] = tera.mul %arg1, %[[SLOPE]]
// CHECK:         return %[[OUT]], %[[MOVED]]
func.func @elementwise(%x: tensor<4xf32>) -> tensor<4xf32>
    attributes {tera.differentiable} {
  %0 = tera.tanh %x : tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----

// A wall to the reverse pass is a wall to the forward one: the result does not
// move, so the tangent returned is the zero the pass fills a missing one with.

// CHECK-LABEL: func.func @blocked_jvp
// CHECK:         %[[ZERO:.*]] = tera.constant dense<0.000000e+00>
// CHECK:         return %{{.*}}, %[[ZERO]]
func.func @blocked(%x: tensor<4xf32>) -> tensor<4xf32>
    attributes {tera.differentiable} {
  %0 = tera.exp %x : tensor<4xf32>
  %1 = tera.stop_gradient %0 : tensor<4xf32>
  return %1 : tensor<4xf32>
}

// -----

// A function that says which arguments it is differentiated for takes one
// tangent for each of them and no more.

// CHECK-LABEL: func.func @weights_only_jvp
// CHECK-SAME:    (%arg0: tensor<4xf32>, %arg1: tensor<4xf32>, %arg2: tensor<4xf32>)
// CHECK-SAME:    attributes {tera.diff_args = array<i64: 1>}
func.func @weights_only(%x: tensor<4xf32>, %w: tensor<4xf32>) -> tensor<4xf32>
    attributes {tera.differentiable, tera.diff_args = array<i64: 1>} {
  %0 = tera.mul %x, %w : tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----

// A scan runs forwards, so its forward derivative is a scan of the body's:
// one carry beside each carry, one stacked output beside each output, and no
// second pass over the sequence. That is the whole reason forward mode is
// cheap on a loop and reverse mode is not.

// CHECK-LABEL: func.func @through_a_scan_jvp
// CHECK:         tera.scan
// CHECK-NOT:     tera.scan
// CHECK:         return
func.func @through_a_scan(%init: tensor<2xf32>, %xs: tensor<3x2xf32>)
    -> tensor<2xf32> attributes {tera.differentiable} {
  %carry = tera.scan init(%init : tensor<2xf32>) xs(%xs : tensor<3x2xf32>)
      -> (tensor<2xf32>) {
  ^bb0(%h: tensor<2xf32>, %x: tensor<2xf32>):
    %0 = tera.mul %h, %x : tensor<2xf32>
    tera.yield %0 : tensor<2xf32>
  }
  return %carry : tensor<2xf32>
}

// -----

// A conversion that lands on an integer is a step function, so nothing moves
// through it. The rule is read off the result, not the operand: the reverse
// pass asks whether the operand can take a gradient, the forward pass asks
// whether the result can carry one.

// CHECK-LABEL: func.func @through_an_integer_jvp
// CHECK-NOT:     tera.convert %arg1
// CHECK:         %[[ZERO:.*]] = tera.constant dense<0.000000e+00>
// CHECK:         return %{{.*}}, %[[ZERO]]
func.func @through_an_integer(%x: tensor<4xf32>) -> tensor<4xf32>
    attributes {tera.differentiable} {
  %0 = tera.convert %x : tensor<4xf32> -> tensor<4xi32>
  %1 = tera.convert %0 : tensor<4xi32> -> tensor<4xf32>
  return %1 : tensor<4xf32>
}
