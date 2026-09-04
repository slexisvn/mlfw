// RUN: tera-opt %s --split-input-file --sccp | FileCheck %s

// Constant propagation is upstream's, and it reaches into a tera region only
// because `tera.if` and `tera.scan` describe their edges. What runs here is
// `-sccp`, but what is being checked is the two hooks it needs: the edges that
// carry a value into a body and back out, and the dialect's ability to build a
// `tera.constant` from a value it proved.

// The input is constant, so the value each side yields is that same constant,
// so the branch has one answer and is not worth taking.
// CHECK-LABEL: func @through_both_sides
// CHECK-NEXT:    %[[C:.*]] = tera.constant dense<1.000000e+00>
// CHECK-NEXT:    return %[[C]]
// CHECK-NOT:     tera.if
func.func @through_both_sides(%p: tensor<i1>) -> tensor<2xf32> {
  %c = tera.constant dense<1.000000e+00> : tensor<2xf32>
  %0 = tera.if %p, %c : (tensor<i1>, tensor<2xf32>) -> tensor<2xf32> {
  ^bb0(%a: tensor<2xf32>):
    tera.yield %a : tensor<2xf32>
  } else {
  ^bb0(%a: tensor<2xf32>):
    tera.yield %a : tensor<2xf32>
  }
  return %0 : tensor<2xf32>
}

// -----

// The constants here are inside an `IsolatedFromAbove` region, which is a fold
// scope of its own, so there is none in the function to reuse: answering this
// means building one, which is `materializeConstant` and nothing else.
// CHECK-LABEL: func @from_inside
// CHECK-NEXT:    %[[C:.*]] = tera.constant dense<3.000000e+00>
// CHECK-NEXT:    return %[[C]]
func.func @from_inside(%p: tensor<i1>) -> tensor<2xf32> {
  %0 = tera.if %p : (tensor<i1>) -> tensor<2xf32> {
  ^bb0:
    %c = tera.constant dense<3.000000e+00> : tensor<2xf32>
    tera.yield %c : tensor<2xf32>
  } else {
  ^bb0:
    %d = tera.constant dense<3.000000e+00> : tensor<2xf32>
    tera.yield %d : tensor<2xf32>
  }
  return %0 : tensor<2xf32>
}

// -----

// A carry that starts constant and is yielded unchanged is constant at every
// step, and it is the scan's edges back into its own body that say so.
// CHECK-LABEL: func @through_a_scan
// CHECK:         %[[C:.*]] = tera.constant dense<2.000000e+00>
// CHECK:         return %[[C]]
func.func @through_a_scan(%xs: tensor<4xf32>) -> tensor<f32> {
  %c = tera.constant dense<2.000000e+00> : tensor<f32>
  %0 = tera.scan init(%c : tensor<f32>) xs(%xs : tensor<4xf32>)
      -> (tensor<f32>) {
  ^bb0(%acc: tensor<f32>, %x: tensor<f32>):
    tera.yield %acc : tensor<f32>
  }
  return %0 : tensor<f32>
}
