// RUN: tera-opt %s --convert-tera-to-linalg | FileCheck %s

// CHECK-LABEL: func @constant
func.func @constant() -> tensor<2xf32> {
  // CHECK: arith.constant dense<[1.000000e+00, 2.000000e+00]> : tensor<2xf32>
  // CHECK-NOT: tera.
  %0 = tera.constant dense<[1.0, 2.0]> : tensor<2xf32>
  return %0 : tensor<2xf32>
}

// An iota reads nothing: its body is the loop index alone, cast to the element
// type.
// CHECK-LABEL: func @iota_integer
func.func @iota_integer() -> tensor<2x3xi32> {
  // CHECK: %[[E:.*]] = tensor.empty() : tensor<2x3xi32>
  // CHECK: linalg.generic
  // CHECK-SAME: outs(%[[E]] : tensor<2x3xi32>)
  // CHECK: %[[I:.*]] = linalg.index 1 : index
  // CHECK: %[[W:.*]] = arith.index_cast %[[I]] : index to i64
  // CHECK: %[[C:.*]] = arith.trunci %[[W]] : i64 to i32
  // CHECK: linalg.yield %[[C]]
  %0 = tera.iota {iota_dimension = 1 : i64} : tensor<2x3xi32>
  return %0 : tensor<2x3xi32>
}

// CHECK-LABEL: func @iota_float
func.func @iota_float() -> tensor<4xf32> {
  // CHECK: linalg.index 0 : index
  // CHECK: arith.index_cast
  // CHECK: arith.sitofp
  %0 = tera.iota {iota_dimension = 0 : i64} : tensor<4xf32>
  return %0 : tensor<4xf32>
}
