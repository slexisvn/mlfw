// RUN: tera-opt %s --tera-autodiff --split-input-file | FileCheck %s

// The reverse of a branch is a branch on the same condition. Each side carries
// the reverse of the side it mirrors, and the inputs come along so that each
// can recompute whatever forward values its own derivative needs. The condition
// itself takes no gradient, so the derivative returns one value per input and
// none for it.
// CHECK-LABEL: func @branch_vjp
// CHECK-SAME: (%[[P:.*]]: tensor<i1>, %[[X:.*]]: tensor<4xf32>, %[[SEED:.*]]: tensor<4xf32>)
// CHECK-SAME: -> tensor<4xf32>
// CHECK-SAME: {tera.diff_args = array<i64: 1>}
// CHECK: %[[REV:.*]] = tera.if %[[P]], %[[X]], %[[SEED]]
// CHECK-SAME: (tensor<i1>, tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
// CHECK: ^bb0(%[[A:.*]]: tensor<4xf32>, %[[G:.*]]: tensor<4xf32>):
// CHECK: %[[FWD:.*]] = tera.exp %[[A]]
// CHECK: %[[THEN:.*]] = tera.mul %[[G]], %[[FWD]]
// CHECK: tera.yield %[[THEN]]
// CHECK: } else {
// CHECK: ^bb0(%[[B:.*]]: tensor<4xf32>, %[[H:.*]]: tensor<4xf32>):
// CHECK: %[[ELSE:.*]] = tera.neg %[[H]]
// CHECK: tera.yield %[[ELSE]]
// CHECK: return %[[REV]]
func.func @branch(%p: tensor<i1>, %x: tensor<4xf32>) -> tensor<4xf32>
    attributes {tera.differentiable} {
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

// A scan differentiates into two more. The first runs the body again and
// stacks the carry as it entered each step, which is the one thing the reverse
// cannot recompute from where it stands. The second runs the body's derivative
// in the opposite direction, and the constant's gradient rides in a carry that
// starts at zero because it is a sum over steps, not a value per step.
// CHECK-LABEL: func @weighted_vjp
// CHECK-SAME: -> (tensor<2xf32>, tensor<2xf32>, tensor<3x2xf32>)
//
// Two zeros: the forward result had no gradient of its own, so the carry is
// seeded with one rather than with a seed that was never handed over, and the
// constant starts its running total at the other.
// CHECK: %[[CARRYSEED:.*]] = tera.constant dense<0.000000e+00> : tensor<2xf32>
// CHECK: %[[TOTAL:.*]] = tera.constant dense<0.000000e+00> : tensor<2xf32>
//
// CHECK: %[[STASH:.*]]:2 = tera.scan init(%arg0 {{.*}} consts(%arg1
// CHECK: ^bb0(%[[H:.*]]: tensor<2xf32>, %[[X:.*]]: tensor<2xf32>, %[[W:.*]]: tensor<2xf32>):
// CHECK: tera.yield %{{.*}}, %[[H]]
//
// CHECK: %[[REV:.*]]:3 = tera.scan reverse
// CHECK-SAME: init(%[[CARRYSEED]], %[[TOTAL]] : tensor<2xf32>, tensor<2xf32>)
// CHECK-SAME: xs(%arg3, %[[STASH]]#1, %arg2
// CHECK-SAME: consts(%arg1
// CHECK-SAME: -> (tensor<2xf32>, tensor<2xf32>, tensor<3x2xf32>)
// CHECK: ^bb0(%[[DC:.*]]: tensor<2xf32>, %[[DW:.*]]: tensor<2xf32>, %[[DY:.*]]: tensor<2xf32>, %[[C:.*]]: tensor<2xf32>, %[[XT:.*]]: tensor<2xf32>, %[[WT:.*]]: tensor<2xf32>):
// CHECK: %[[SEED:.*]] = tera.add %[[DC]], %[[DY]]
// CHECK: %[[DH:.*]] = tera.mul %[[SEED]], %[[WT]]
// CHECK: %[[STEP:.*]] = tera.mul %[[SEED]], %[[C]]
// CHECK: %[[RUNNING:.*]] = tera.add %[[DW]], %[[STEP]]
// CHECK: tera.yield %[[DH]], %[[RUNNING]], %[[SEED]]
//
// The carry's gradient answers for the init, the stacked output for the inputs,
// and the accumulator for the constant.
// CHECK: return %[[REV]]#0, %[[REV]]#1, %[[REV]]#2
func.func @weighted(%h0: tensor<2xf32>, %w: tensor<2xf32>, %xs: tensor<3x2xf32>)
    -> tensor<3x2xf32> attributes {tera.differentiable} {
  %carry, %ys = tera.scan init(%h0 : tensor<2xf32>) xs(%xs : tensor<3x2xf32>)
      consts(%w : tensor<2xf32>) -> (tensor<2xf32>, tensor<3x2xf32>) {
  ^bb0(%h: tensor<2xf32>, %x: tensor<2xf32>, %weight: tensor<2xf32>):
    %0 = tera.mul %h, %weight : tensor<2xf32>
    %1 = tera.add %0, %x : tensor<2xf32>
    tera.yield %1, %1 : tensor<2xf32>, tensor<2xf32>
  }
  return %ys : tensor<3x2xf32>
}

// -----

// A scan already running backwards differentiates into one running forwards.
// CHECK-LABEL: func @backwards_vjp
// CHECK: tera.scan reverse init
// CHECK: %[[REV:.*]]:2 = tera.scan init
// CHECK-NOT: tera.scan reverse
// CHECK: return %[[REV]]#0, %[[REV]]#1
func.func @backwards(%h0: tensor<f32>, %xs: tensor<3xf32>) -> tensor<3xf32>
    attributes {tera.differentiable} {
  %carry, %ys = tera.scan reverse init(%h0 : tensor<f32>)
      xs(%xs : tensor<3xf32>) -> (tensor<f32>, tensor<3xf32>) {
  ^bb0(%acc: tensor<f32>, %x: tensor<f32>):
    %0 = tera.mul %acc, %x : tensor<f32>
    tera.yield %0, %0 : tensor<f32>, tensor<f32>
  }
  return %ys : tensor<3xf32>
}

// -----

// A boolean constant is read every step and carries no gradient, so it gets no
// accumulator at all: the reverse scan has one carry, not two.
// CHECK-LABEL: func @gated_vjp
// CHECK: %[[REV:.*]]:2 = tera.scan reverse init(%{{[^,]*}} : tensor<2xf32>)
// CHECK-SAME: consts(%{{.*}} : tensor<i1>)
// CHECK-SAME: -> (tensor<2xf32>, tensor<4x2xf32>)
func.func @gated(%h0: tensor<2xf32>, %xs: tensor<4x2xf32>) -> tensor<4x2xf32>
    attributes {tera.differentiable} {
  %0 = tera.constant dense<true> : tensor<i1>
  %carry, %ys = tera.scan init(%h0 : tensor<2xf32>) xs(%xs : tensor<4x2xf32>)
      consts(%0 : tensor<i1>) -> (tensor<2xf32>, tensor<4x2xf32>) {
  ^bb0(%h: tensor<2xf32>, %x: tensor<2xf32>, %gate: tensor<i1>):
    %1 = tera.if %gate, %h, %x : (tensor<i1>, tensor<2xf32>, tensor<2xf32>)
        -> tensor<2xf32> {
    ^bb0(%a: tensor<2xf32>, %b: tensor<2xf32>):
      %2 = tera.mul %a, %b : tensor<2xf32>
      tera.yield %2 : tensor<2xf32>
    } else {
    ^bb0(%a: tensor<2xf32>, %b: tensor<2xf32>):
      %2 = tera.add %a, %b : tensor<2xf32>
      tera.yield %2 : tensor<2xf32>
    }
    tera.yield %1, %1 : tensor<2xf32>, tensor<2xf32>
  }
  return %ys : tensor<4x2xf32>
}
