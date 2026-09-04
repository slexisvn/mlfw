// RUN: tera-opt %s --convert-tera-to-linalg --split-input-file | FileCheck %s

// A scan becomes an scf.for: the carry is loop-carried, each input is read one
// slice per step, and each stacked output is a tensor the loop writes a slice
// of. A constant is neither sliced nor carried, so it crosses the loop
// boundary untouched.
// CHECK-LABEL: func @stepping
// CHECK-DAG: %[[FIRST:.*]] = arith.constant 0 : index
// CHECK-DAG: %[[BOUND:.*]] = arith.constant 3 : index
// CHECK-DAG: %[[STRIDE:.*]] = arith.constant 1 : index
// CHECK: %[[ACC:.*]] = tensor.empty() : tensor<3x2xf32>
// CHECK: scf.for %[[I:.*]] = %[[FIRST]] to %[[BOUND]] step %[[STRIDE]]
// CHECK-SAME: iter_args(%[[H:.*]] = %arg0, %[[YS:.*]] = %[[ACC]])
// CHECK: %[[X:.*]] = tensor.extract_slice %arg1[%[[I]], 0] [1, 2] [1, 1]
// CHECK-SAME: tensor<3x2xf32> to tensor<2xf32>
// CHECK: linalg.map { arith.mulf } ins(%[[H]], %arg2
// CHECK: %[[NEXT:.*]] = linalg.map { arith.addf }
// CHECK: %[[WRITTEN:.*]] = tensor.insert_slice %[[NEXT]] into %[[YS]][%[[I]], 0]
// CHECK: scf.yield %[[NEXT]], %[[WRITTEN]]
func.func @stepping(%init: tensor<2xf32>, %xs: tensor<3x2xf32>,
                    %w: tensor<2xf32>) -> tensor<3x2xf32> {
  %carry, %ys = tera.scan init(%init : tensor<2xf32>) xs(%xs : tensor<3x2xf32>)
      consts(%w : tensor<2xf32>) -> (tensor<2xf32>, tensor<3x2xf32>) {
  ^bb0(%h: tensor<2xf32>, %x: tensor<2xf32>, %weight: tensor<2xf32>):
    %0 = tera.mul %h, %weight : tensor<2xf32>
    %1 = tera.add %0, %x : tensor<2xf32>
    tera.yield %1, %1 : tensor<2xf32>, tensor<2xf32>
  }
  return %ys : tensor<3x2xf32>
}

// -----

// `reverse` counts the induction variable down from the last step instead of
// moving the data, so both the read and the write use the same computed index
// and the loop bounds do not change.
// CHECK-LABEL: func @backwards
// CHECK: scf.for %[[I:.*]] = %{{.*}} to %{{.*}} step
// CHECK: %[[LAST:.*]] = arith.constant 2 : index
// CHECK: %[[T:.*]] = arith.subi %[[LAST]], %[[I]]
// CHECK: tensor.extract_slice %arg1[%[[T]]]
// CHECK: tensor.insert_slice %{{.*}}[%[[T]]]
func.func @backwards(%init: tensor<f32>, %xs: tensor<3xf32>) -> tensor<3xf32> {
  %carry, %ys = tera.scan reverse init(%init : tensor<f32>)
      xs(%xs : tensor<3xf32>) -> (tensor<f32>, tensor<3xf32>) {
  ^bb0(%acc: tensor<f32>, %x: tensor<f32>):
    %0 = tera.add %acc, %x : tensor<f32>
    tera.yield %0, %0 : tensor<f32>, tensor<f32>
  }
  return %ys : tensor<3xf32>
}

// -----

// A branch becomes an scf.if. The condition is a rank-0 tensor, so it is read
// out of the tensor first: scf.if wants an i1, not a tensor of one.
// CHECK-LABEL: func @branch
// CHECK: %[[COND:.*]] = tensor.extract %arg0[] : tensor<i1>
// CHECK: scf.if %[[COND]] -> (tensor<4xf32>) {
// CHECK: %[[THEN:.*]] = linalg.map { math.exp } ins(%arg1
// CHECK: scf.yield %[[THEN]]
// CHECK: } else {
// CHECK: %[[ELSE:.*]] = linalg.map { arith.negf } ins(%arg1
// CHECK: scf.yield %[[ELSE]]
// CHECK-NOT: tera.
func.func @branch(%p: tensor<i1>, %x: tensor<4xf32>) -> tensor<4xf32> {
  %0 = tera.if %p, %x : (tensor<i1>, tensor<4xf32>) -> tensor<4xf32> {
  ^bb0(%a: tensor<4xf32>):
    %1 = tera.exp %a : tensor<4xf32>
    tera.yield %1 : tensor<4xf32>
  } else {
  ^bb0(%a: tensor<4xf32>):
    %1 = tera.neg %a : tensor<4xf32>
    tera.yield %1 : tensor<4xf32>
  }
  return %0 : tensor<4xf32>
}

// -----

// A body that yields one of its own arguments untouched. The argument is gone
// once the body moves, so the value standing in for it has to be picked up
// before the move rather than after.
// CHECK-LABEL: func @identity_branch
// CHECK: scf.if
// CHECK: scf.yield %arg1
func.func @identity_branch(%p: tensor<i1>, %x: tensor<4xf32>) -> tensor<4xf32> {
  %0 = tera.if %p, %x : (tensor<i1>, tensor<4xf32>) -> tensor<4xf32> {
  ^bb0(%a: tensor<4xf32>):
    tera.yield %a : tensor<4xf32>
  } else {
  ^bb0(%a: tensor<4xf32>):
    %1 = tera.neg %a : tensor<4xf32>
    tera.yield %1 : tensor<4xf32>
  }
  return %0 : tensor<4xf32>
}
