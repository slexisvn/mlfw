// RUN: tera-opt %s --tera-autodiff --split-input-file | FileCheck %s

// What checkpointing buys, in the only place it is visible: the extent of the
// tensors the reverse pass keeps.
//
// Both functions below are the same six-step recurrence. Differentiated
// plainly, one scan stacks the carry from all six steps and the reverse runs
// over that. Checkpointed every three, one scan keeps two carries — one per
// group — and the reverse restacks three at a time from them. Six kept at once
// against three plus two, for one extra run of the body.

// CHECK-LABEL: func @plain_vjp
// CHECK: tera.scan init
// CHECK-SAME: -> (tensor<2xf32>, tensor<6x2xf32>)
// CHECK: tera.scan reverse init
// A CHECK-NOT runs to the next CHECK, so the return has to be here: without it
// the range would run on past the end of this function.
// CHECK-NOT: tera.scan
// CHECK: return
func.func @plain(%h0: tensor<2xf32>, %xs: tensor<6x2xf32>) -> tensor<6x2xf32>
    attributes {tera.differentiable} {
  %carry, %ys = tera.scan init(%h0 : tensor<2xf32>) xs(%xs : tensor<6x2xf32>)
      -> (tensor<2xf32>, tensor<6x2xf32>) {
  ^bb0(%h: tensor<2xf32>, %x: tensor<2xf32>):
    %0 = tera.mul %h, %x : tensor<2xf32>
    tera.yield %0, %0 : tensor<2xf32>, tensor<2xf32>
  }
  return %ys : tensor<6x2xf32>
}

// -----

// CHECK-LABEL: func @kept_vjp
//
// The sequence is seen as two groups of three. Nothing is copied to do it: a
// reshape is how a step axis becomes a group axis and a step-within-group axis.
// CHECK: %[[GROUPED:.*]] = tera.reshape %arg1 : tensor<6x2xf32> -> tensor<2x3x2xf32>
//
// One carry per group, so two, where the plain form kept six.
// CHECK: %[[CHECKPOINTS:.*]]:2 = tera.scan init(%arg0 : tensor<2xf32>)
// CHECK-SAME: xs(%[[GROUPED]] : tensor<2x3x2xf32>)
// CHECK-SAME: -> (tensor<2xf32>, tensor<2x2xf32>)
// CHECK: ^bb0(%[[ENTRY:.*]]: tensor<2xf32>, %[[GROUP:.*]]: tensor<3x2xf32>):
// CHECK: %[[END:.*]] = tera.scan init(%[[ENTRY]] {{.*}} -> (tensor<2xf32>)
// CHECK: tera.yield %[[END]], %[[ENTRY]]
//
// The reverse walks the groups backwards. Inside each, the three carries of
// that group are stacked again from its checkpoint, and only then is the body
// differentiated over them.
// CHECK: tera.scan reverse init
// CHECK-SAME: xs(%{{.*}}, %[[CHECKPOINTS]]#1, %[[GROUPED]]
// CHECK: %[[RESTACK:.*]]:2 = tera.scan init
// CHECK-SAME: -> (tensor<2xf32>, tensor<3x2xf32>)
// CHECK: tera.scan reverse init
// CHECK-SAME: %[[RESTACK]]#1
//
// CHECK: tera.reshape %{{.*}} : tensor<2x3x2xf32> -> tensor<6x2xf32>
func.func @kept(%h0: tensor<2xf32>, %xs: tensor<6x2xf32>) -> tensor<6x2xf32>
    attributes {tera.differentiable} {
  %carry, %ys = tera.scan init(%h0 : tensor<2xf32>) xs(%xs : tensor<6x2xf32>)
      {checkpoint = 3 : i64} -> (tensor<2xf32>, tensor<6x2xf32>) {
  ^bb0(%h: tensor<2xf32>, %x: tensor<2xf32>):
    %0 = tera.mul %h, %x : tensor<2xf32>
    tera.yield %0, %0 : tensor<2xf32>, tensor<2xf32>
  }
  return %ys : tensor<6x2xf32>
}
