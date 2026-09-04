// RUN: tera-opt %s --convert-tera-to-linalg | FileCheck %s

// The op has done its work by the time the lowering runs, and forwards its
// operand. Nothing at all is left behind: no linalg.map, no copy.
// CHECK-LABEL: func @stop_gradient
// CHECK-NOT: tera.stop_gradient
// CHECK-NOT: linalg
func.func @stop_gradient(%a: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: return %arg0
  %0 = tera.stop_gradient %a : tensor<4xf32>
  return %0 : tensor<4xf32>
}
