// RUN: not tera-gradcheck %s --shared-libs=%mlir_c_runner_utils 2>&1 | FileCheck %s

// A check that cannot fail is not a gate, so here is one that must.
//
// `stop_gradient` is the only op whose derivative is meant to disagree with
// the function it wraps. Forwards this computes `x * x`; the rules report
// `d/dx = x` where the truth is `2x`. Finite differences run the compiled
// forward function and see the `2x`, so the driver has to notice.

// CHECK: derivative says
// CHECK: gradients disagree
func.func @half_a_gradient(%x: tensor<3xf64>) -> tensor<3xf64>
    attributes {tera.differentiable} {
  %0 = tera.stop_gradient %x : tensor<3xf64>
  %1 = tera.mul %x, %0 : tensor<3xf64>
  return %1 : tensor<3xf64>
}
