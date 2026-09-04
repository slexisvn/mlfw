// RUN: %if mlfw-oracle %{ %node %tera_harness/mlfw_oracle.mjs dynrnn --emit-mlir --out %t.rnn.mlir --json %t.rnn.json %}
// RUN: %if mlfw-oracle %{ tera-opt %t.rnn.mlir -o %t.rnn.rt.mlir %}
// RUN: %if mlfw-oracle %{ FileCheck %s --check-prefix=SCAN --input-file=%t.rnn.mlir --implicit-check-not='"tera.' %}
// RUN: %if mlfw-oracle %{ tera-runner %t.rnn.mlir --entry=dynrnn --shared-libs=%mlir_c_runner_utils --data=%t.rnn.json --check %}
// RUN: %if mlfw-oracle %{ %if cuda %{ tera-runner %t.rnn.mlir --entry=dynrnn --target=cuda --shared-libs=%mlir_cuda_runtime --shared-libs=%mlir_c_runner_utils --data=%t.rnn.json --check %} %}

// The same module, a different batch. Three steps either way, so the trip
// count is not what changed.
// RUN: %if mlfw-oracle %{ %node %tera_harness/mlfw_oracle.mjs dynrnn_wide --out %t.rnnwide.json %}
// RUN: %if mlfw-oracle %{ tera-runner %t.rnn.mlir --entry=dynrnn --shared-libs=%mlir_c_runner_utils --data=%t.rnnwide.json --check %}
// RUN: %if mlfw-oracle %{ %if cuda %{ tera-runner %t.rnn.mlir --entry=dynrnn --target=cuda --shared-libs=%mlir_cuda_runtime --shared-libs=%mlir_c_runner_utils --data=%t.rnnwide.json --check %} %}

// And back, at both batches. The reader refines the module to the shapes the
// record carries rather than refusing it, which is the whole of what reading a
// dynamic module back means.
// RUN: %if mlfw-oracle %{ %node %tera_harness/mlfw_reader.mjs %t.rnn.rt.mlir --entry dynrnn --data %t.rnn.json %}
// RUN: %if mlfw-oracle %{ %node %tera_harness/mlfw_reader.mjs %t.rnn.rt.mlir --entry dynrnn --data %t.rnnwide.json %}

// RUN: %if mlfw-oracle %{ %node %tera_harness/mlfw_oracle.mjs dyncond --emit-mlir --out %t.cond.mlir --json %t.cond.json %}
// RUN: %if mlfw-oracle %{ tera-opt %t.cond.mlir -o %t.cond.rt.mlir %}
// RUN: %if mlfw-oracle %{ FileCheck %s --check-prefix=COND --input-file=%t.cond.mlir --implicit-check-not='"tera.' %}
// RUN: %if mlfw-oracle %{ tera-runner %t.cond.mlir --entry=dyncond --shared-libs=%mlir_c_runner_utils --data=%t.cond.json --check %}
// RUN: %if mlfw-oracle %{ %if cuda %{ tera-runner %t.cond.mlir --entry=dyncond --target=cuda --shared-libs=%mlir_cuda_runtime --shared-libs=%mlir_c_runner_utils --data=%t.cond.json --check %} %}

// The other body, at another batch. One record for each side and a different
// extent under each, against the one module.
// RUN: %if mlfw-oracle %{ %node %tera_harness/mlfw_oracle.mjs dyncond_else --out %t.condelse.json %}
// RUN: %if mlfw-oracle %{ tera-runner %t.cond.mlir --entry=dyncond --shared-libs=%mlir_c_runner_utils --data=%t.condelse.json --check %}
// RUN: %if mlfw-oracle %{ %if cuda %{ tera-runner %t.cond.mlir --entry=dyncond --target=cuda --shared-libs=%mlir_cuda_runtime --shared-libs=%mlir_c_runner_utils --data=%t.condelse.json --check %} %}
// RUN: %if mlfw-oracle %{ %node %tera_harness/mlfw_reader.mjs %t.cond.rt.mlir --entry dyncond --data %t.cond.json %}
// RUN: %if mlfw-oracle %{ %node %tera_harness/mlfw_reader.mjs %t.cond.rt.mlir --entry dyncond --data %t.condelse.json %}

// And their derivatives, dynamic all the way through. Two reverse-mode
// implementations sharing no code, asked for the derivative of a program whose
// extents neither of them knows until it runs.
// RUN: %if mlfw-oracle %{ %node %tera_harness/mlfw_oracle.mjs dynrnn --grad --emit-mlir --out %t.rnn.g.mlir --json %t.rnn.g.json %}
// RUN: %if mlfw-oracle %{ tera-opt %t.rnn.g.mlir --tera-autodiff -o %t.rnn.vjp.mlir %}
// RUN: %if mlfw-oracle %{ FileCheck %s --check-prefix=RNNVJP --input-file=%t.rnn.vjp.mlir %}
// RUN: %if mlfw-oracle %{ tera-runner %t.rnn.vjp.mlir --entry=dynrnn_vjp --shared-libs=%mlir_c_runner_utils --data=%t.rnn.g.json --check %}
// RUN: %if mlfw-oracle %{ %if cuda %{ tera-runner %t.rnn.vjp.mlir --entry=dynrnn_vjp --target=cuda --shared-libs=%mlir_cuda_runtime --shared-libs=%mlir_c_runner_utils --data=%t.rnn.g.json --check %} %}

// RUN: %if mlfw-oracle %{ %node %tera_harness/mlfw_oracle.mjs dyncond --grad --emit-mlir --out %t.cond.g.mlir --json %t.cond.g.json %}
// RUN: %if mlfw-oracle %{ tera-opt %t.cond.g.mlir --tera-autodiff -o %t.cond.vjp.mlir %}
// RUN: %if mlfw-oracle %{ tera-runner %t.cond.vjp.mlir --entry=dyncond_vjp --shared-libs=%mlir_c_runner_utils --data=%t.cond.g.json --check %}
// RUN: %if mlfw-oracle %{ %if cuda %{ tera-runner %t.cond.vjp.mlir --entry=dyncond_vjp --target=cuda --shared-libs=%mlir_cuda_runtime --shared-libs=%mlir_c_runner_utils --data=%t.cond.g.json --check %} %}

// A `?` inside a body, which is a different problem from a `?` beside one.
//
// mlfw-bridge-dynamic.mlir is a dynamic extent that every op can find by asking
// an operand for it. These two put the extent where no operand's type answers.
//
// A scan's body is isolated from what surrounds it, so a stacked output's shape
// is decided by values that do not exist outside the loop. That is what the
// `sizes` clause on `tera.scan` is: one extent per `?` across the results, in
// the same order and for the same reason `tera.broadcast_in_dim` has one. The
// step axis is not among them and stays refused by name -- it is a trip count,
// not a destination's width, and `Tera/Conversion/invalid.mlir` says so.
//
// A branch is the opposite case: its results are its bodies' and both bodies
// were handed the op's own inputs, so it needs telling nothing at all. It is
// here because the extent has to cross two regions that are separate scopes,
// and because the results arrive from inside one rather than through an
// operand.
//
// Running these found five real defects, none in the bridge, and four of them
// are the same mistake: a symbol is what ties one `?` to another and it rides
// beside the type rather than in it, so anything that builds a value from a
// shape instead of from the value that had it drops the tie.
//
//   1. mlfw's tracer dropped symbols at both region ops. `scan` and `cond`
//      built their results from the op's types, where a dynamic extent is a `?`
//      and the symbol is gone. The compiled forward then returned a tensor
//      whose own shape had a `-1` in it.
//   2. mlfw's scan lowering asked for its step buffers by type alone, so their
//      dynamic extent was a fresh unknown and the runtime fell back to guessing
//      which argument axis it came from -- picking the step count, which is
//      right whenever the batch happens to equal it. Correct at batch 3 of 3
//      and wrong at 4.
//   3. mlfw's reverse pass dropped them wholesale: the backward function's
//      arguments were fresh values standing for forward ones, and every zero
//      adjoint was a splat built from a shape. The weights' gradients came back
//      NaN. Fixed at the three producers, and then underneath by
//      `unifyShapeSymbols`, which recovers what any pass drops from what the
//      dataflow already says rather than asking every pass to carry it.
//   4. mlfw's `reshape` lowering flattened with strides multiplied out as
//      numbers, so it refused any dynamic extent but the leading one. The
//      strides are expressions now and fold back to constants over a static
//      shape.
//   5. terac's `tera-stage-gpu-buffers` counted `memref.dim` as a host read of
//      a buffer the kernels held. A shape query reads no element, and it is
//      also how a dynamic extent reaches a launch's grid, so every dynamic
//      program on the device had one between the copy in and the copy back.

// SCAN-LABEL: func.func @dynrnn
// SCAN-SAME:    (%{{.*}}: tensor<3x?x2xf32>, %{{.*}}: tensor<?x2xf32>
// SCAN: %[[N:.*]] = tera.dim %{{.*}} {dimension = 0 : i64} : tensor<?x2xf32> -> tensor<i64>
// SCAN: tera.scan init({{.*}}) xs({{.*}}) consts({{.*}}) sizes(%[[N]], %[[N]]) -> (tensor<?x2xf32>, tensor<3x?x2xf32>)
// The body carries the `?` too, and has a `tera.dim` of its own: the zero the
// relu compares against is shaped like a value that only exists in there.
// SCAN: tera.dim
// SCAN: tera.broadcast_in_dim %{{.*}} sizes(
// SCAN: tera.yield

// COND-LABEL: func.func @dyncond
// COND-SAME:    (%{{.*}}: tensor<?x2xf32>) -> (tensor<?x2xf32>)
// COND: tera.if %{{.*}} : (tensor<i1>, tensor<?x2xf32>) -> tensor<?x2xf32>
// COND: tera.mul
// COND: tera.neg

// The reverse pass keeps the `?` and gains a `sizes` clause of its own: the
// scan it builds to accumulate the weights' gradients has stacked outputs, and
// their shape is decided in a body too.
// RNNVJP: func.func @dynrnn_vjp
// RNNVJP-SAME: tensor<3x?x2xf32>
// RNNVJP: tera.scan
// RNNVJP: sizes(
