// RUN: %if mlfw-oracle %{ %node %tera_harness/mlfw_oracle.mjs branch --emit-mlir --out %t.mlir --json %t.json %}
// RUN: %if mlfw-oracle %{ tera-opt %t.mlir %}
// RUN: %if mlfw-oracle %{ FileCheck %s --check-prefix=MLIR --input-file=%t.mlir --implicit-check-not='"tera.' %}
// RUN: %if mlfw-oracle %{ tera-runner %t.mlir --shared-libs=%mlir_c_runner_utils --data=%t.json --check %}
// RUN: %if mlfw-oracle %{ %if cuda %{ tera-runner %t.mlir --target=cuda --shared-libs=%mlir_cuda_runtime --shared-libs=%mlir_c_runner_utils --data=%t.json --check %} %}

// The same program with inputs that take the other side. A lowering that
// dropped the else body would pass every check above.
// RUN: %if mlfw-oracle %{ %node %tera_harness/mlfw_oracle.mjs branch_else --emit-mlir --out %t.else.mlir --json %t.else.json %}
// RUN: %if mlfw-oracle %{ tera-runner %t.else.mlir --shared-libs=%mlir_c_runner_utils --data=%t.else.json --check %}
// RUN: %if mlfw-oracle %{ %if cuda %{ tera-runner %t.else.mlir --target=cuda --shared-libs=%mlir_cuda_runtime --shared-libs=%mlir_c_runner_utils --data=%t.else.json --check %} %}

// A conditional traced by mlfw and run here. controlflow.mlir's `branch_on_sign`
// is the same program written by hand, and this one is not written at all.
//
// mlfw had no traced conditional to reach `tera.if` with: the op existed in its
// graph IR, with a lowering and a VJP, but nothing in the language built one.
// `mlfw.cond` is that, and it is the same shape as `mlfw.scan` -- both branches
// are recorded and the choice stays in the graph, rather than the predicate
// being read and one side traced.
//
// The isolation this needs is the other half of the one mlfw-bridge-rnn.mlir
// describes. `x` is read by both bodies and defined outside them, so it becomes
// an operand of the op and an argument of *both* bodies -- not only of the ones
// that read it, because the two bodies of a branch have to agree on their
// signature. mlfw's `if` bodies used to take no arguments at all and read
// everything from around them, which `tera.if` cannot express.
//
// The condition is computed rather than passed in, which is the shape a real
// branch has, and is why the reduction and the compare are above the op.

// MLIR-LABEL: func.func @branch
// MLIR: tera.reduce sum
// MLIR: %[[COND:.*]] = tera.compare gt
// MLIR: tera.if %[[COND]], %{{.*}} : (tensor<i1>, tensor<4xf32>) -> tensor<4xf32> {

// Both bodies take the op's one input, and each reads it under its own name.
// MLIR-NEXT: ^bb0(%[[T:.*]]: tensor<4xf32>):
// MLIR-NEXT: tera.mul %[[T]], %[[T]]
// MLIR-NEXT: tera.yield
// MLIR-NEXT: } else {
// MLIR-NEXT: ^bb0(%[[E:.*]]: tensor<4xf32>):
// MLIR-NEXT: tera.neg %[[E]]
// MLIR-NEXT: tera.yield
// MLIR: return

// Running this on the device is what found the bug the copies back had: they
// were emitted beside the last launch that read a buffer, and both launches
// here are inside a branch, so the copy back ran only when that side was taken
// and the allocation was freed on one path and leaked on the other.
// stage-gpu-buffers.mlir now pins the shape of the fix.
