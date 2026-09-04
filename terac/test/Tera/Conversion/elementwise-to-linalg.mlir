// RUN: tera-opt %s --convert-tera-to-linalg | FileCheck %s

// The pass is anchored on nothing, so a pipeline can nest it under the
// functions and convert them in parallel. Both spellings have to agree.
// RUN: tera-opt %s --pass-pipeline='builtin.module(func.func(convert-tera-to-linalg))' | FileCheck %s

// Every elementwise op becomes one linalg.map; only the body differs.
// CHECK-LABEL: func @float_arithmetic
func.func @float_arithmetic(%a: tensor<2x3xf32>, %b: tensor<2x3xf32>) -> tensor<2x3xf32> {
  // CHECK: linalg.map { arith.addf } ins(%arg0, %arg1 : tensor<2x3xf32>, tensor<2x3xf32>)
  %0 = tera.add %a, %b : tensor<2x3xf32>
  // CHECK: linalg.map { arith.subf }
  %1 = tera.sub %0, %b : tensor<2x3xf32>
  // CHECK: linalg.map { arith.mulf }
  %2 = tera.mul %1, %a : tensor<2x3xf32>
  // CHECK: linalg.map { arith.divf }
  %3 = tera.div %2, %b : tensor<2x3xf32>
  // CHECK: linalg.map { arith.maximumf }
  %4 = tera.maximum %3, %a : tensor<2x3xf32>
  // CHECK: linalg.map { arith.negf }
  %5 = tera.neg %4 : tensor<2x3xf32>
  // CHECK: linalg.map { math.exp }
  %6 = tera.exp %5 : tensor<2x3xf32>
  return %6 : tensor<2x3xf32>
}

// The same ops on integers pick the integer arith op instead. Division and
// maximum are signed: tera integers are signless and read as two's complement.
// CHECK-LABEL: func @integer_arithmetic
func.func @integer_arithmetic(%a: tensor<4xi32>, %b: tensor<4xi32>) -> tensor<4xi32> {
  // CHECK: linalg.map { arith.addi {overflowFlags = #arith.overflow<none>} }
  %0 = tera.add %a, %b : tensor<4xi32>
  // CHECK: linalg.map { arith.divsi }
  %1 = tera.div %0, %b : tensor<4xi32>
  // CHECK: linalg.map { arith.maxsi }
  %2 = tera.maximum %1, %a : tensor<4xi32>
  // CHECK: %[[Z:.*]] = arith.constant 0 : i32
  // CHECK: arith.subi %[[Z]], %in
  %3 = tera.neg %2 : tensor<4xi32>
  return %3 : tensor<4xi32>
}

// CHECK-LABEL: func @predicates
func.func @predicates(%a: tensor<4xf32>, %b: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: %[[P:.*]] = tensor.empty() : tensor<4xi1>
  // arith.CmpFPredicate::OLT
  // CHECK: linalg.map { arith.cmpf {predicate = 4 : i64} }
  // CHECK-SAME: outs(%[[P]] : tensor<4xi1>)
  %0 = tera.compare lt, %a, %b : tensor<4xf32> -> tensor<4xi1>
  // CHECK: linalg.map { arith.select }
  %1 = tera.select %0, %a, %b : tensor<4xi1>, tensor<4xf32>
  return %1 : tensor<4xf32>
}

// Not-equal is unordered so that a NaN operand compares unequal, matching the
// IEEE meaning of `!=`; every other direction is ordered.
// CHECK-LABEL: func @unordered_not_equal
func.func @unordered_not_equal(%a: tensor<4xf32>, %b: tensor<4xf32>) -> tensor<4xi1> {
  // arith.CmpFPredicate::UNE
  // CHECK: linalg.map { arith.cmpf {predicate = 13 : i64} }
  %0 = tera.compare ne, %a, %b : tensor<4xf32> -> tensor<4xi1>
  return %0 : tensor<4xi1>
}

// CHECK-LABEL: func @integer_predicates
func.func @integer_predicates(%a: tensor<4xi32>, %b: tensor<4xi32>) -> tensor<4xi1> {
  // arith.CmpIPredicate::sge
  // CHECK: linalg.map { arith.cmpi {predicate = 5 : i64} }
  %0 = tera.compare ge, %a, %b : tensor<4xi32> -> tensor<4xi1>
  return %0 : tensor<4xi1>
}

// CHECK-LABEL: func @convert
func.func @convert(%a: tensor<4xf32>) -> tensor<4xi64> {
  // CHECK: %[[W:.*]] = tensor.empty() : tensor<4xf64>
  // CHECK: linalg.map { arith.extf } ins(%arg0 : tensor<4xf32>) outs(%[[W]] : tensor<4xf64>)
  %0 = tera.convert %a : tensor<4xf32> -> tensor<4xf64>
  // CHECK: linalg.map { arith.fptosi }
  %1 = tera.convert %0 : tensor<4xf64> -> tensor<4xi64>
  return %1 : tensor<4xi64>
}
