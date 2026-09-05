// RUN: %if mlfw-oracle %{ %node %tera_harness/mlfw_oracle.mjs demo --grad --emit-mlir --out %t.demo.mlir --json %t.demo.json %}
// RUN: %if mlfw-oracle %{ FileCheck %s --check-prefix=MLIR --input-file=%t.demo.mlir %}
// RUN: %if mlfw-oracle %{ tera-opt %t.demo.mlir --tera-autodiff -o %t.demo.vjp.mlir %}
// RUN: %if mlfw-oracle %{ FileCheck %s --check-prefix=VJP --input-file=%t.demo.vjp.mlir %}
// RUN: %if mlfw-oracle %{ tera-runner %t.demo.vjp.mlir --shared-libs=%mlir_c_runner_utils --data=%t.demo.json --check %}
// RUN: %if mlfw-oracle %{ %if cuda %{ tera-runner %t.demo.vjp.mlir --target=cuda --shared-libs=%mlir_cuda_runtime --shared-libs=%mlir_c_runner_utils --data=%t.demo.json --check %} %}

// RUN: %if mlfw-oracle %{ %node %tera_harness/mlfw_oracle.mjs mlp --grad --emit-mlir --out %t.mlp.mlir --json %t.mlp.json %}
// RUN: %if mlfw-oracle %{ tera-opt %t.mlp.mlir --tera-autodiff -o %t.mlp.vjp.mlir %}
// RUN: %if mlfw-oracle %{ tera-runner %t.mlp.vjp.mlir --shared-libs=%mlir_c_runner_utils --data=%t.mlp.json --check %}
// RUN: %if mlfw-oracle %{ %if cuda %{ tera-runner %t.mlp.vjp.mlir --target=cuda --shared-libs=%mlir_cuda_runtime --shared-libs=%mlir_c_runner_utils --data=%t.mlp.json --check %} %}

// RUN: %if mlfw-oracle %{ %node %tera_harness/mlfw_oracle.mjs attention --grad --emit-mlir --out %t.attn.mlir --json %t.attn.json %}
// RUN: %if mlfw-oracle %{ tera-opt %t.attn.mlir --tera-autodiff -o %t.attn.vjp.mlir %}
// RUN: %if mlfw-oracle %{ tera-runner %t.attn.vjp.mlir --shared-libs=%mlir_c_runner_utils --data=%t.attn.json --check %}
// RUN: %if mlfw-oracle %{ %if cuda %{ tera-runner %t.attn.vjp.mlir --target=cuda --shared-libs=%mlir_cuda_runtime --shared-libs=%mlir_c_runner_utils --data=%t.attn.json --check %} %}

// RUN: %if mlfw-oracle %{ %node %tera_harness/mlfw_oracle.mjs rnn --grad --emit-mlir --out %t.rnn.mlir --json %t.rnn.json %}
// RUN: %if mlfw-oracle %{ tera-opt %t.rnn.mlir --tera-autodiff -o %t.rnn.vjp.mlir %}
// RUN: %if mlfw-oracle %{ tera-runner %t.rnn.vjp.mlir --shared-libs=%mlir_c_runner_utils --data=%t.rnn.json --check %}
// RUN: %if mlfw-oracle %{ %if cuda %{ tera-runner %t.rnn.vjp.mlir --target=cuda --shared-libs=%mlir_cuda_runtime --shared-libs=%mlir_c_runner_utils --data=%t.rnn.json --check %} %}

// RUN: %if mlfw-oracle %{ %node %tera_harness/mlfw_oracle.mjs branch --grad --emit-mlir --out %t.branch.mlir --json %t.branch.json %}
// RUN: %if mlfw-oracle %{ tera-opt %t.branch.mlir --tera-autodiff -o %t.branch.vjp.mlir %}
// RUN: %if mlfw-oracle %{ tera-runner %t.branch.vjp.mlir --shared-libs=%mlir_c_runner_utils --data=%t.branch.json --check %}
// RUN: %if mlfw-oracle %{ %if cuda %{ tera-runner %t.branch.vjp.mlir --target=cuda --shared-libs=%mlir_cuda_runtime --shared-libs=%mlir_c_runner_utils --data=%t.branch.json --check %} %}

// Two reverse-mode implementations that share no code, asked for the same
// derivative and compared.
//
// The gradcheck tests beside this one compare terac's derivative against finite
// differences of terac's own forward, which catches a wrong rule but not a
// disagreement about what the program means. The bridge tests compare the two
// compilers' forwards. Neither says anything about mlfw's derivative and
// terac's agreeing, and they are two separate reverse passes written against
// two separate IRs -- so this is the only test that can tell.
//
// The oracle marks the entry `tera.differentiable`, which is what
// `-tera-autodiff` looks for; it emits `<entry>_vjp` taking the arguments and a
// seed, returning one gradient per floating-point argument. mlfw's own
// `compileWithBackward` is asked for the same thing with the same seed, and the
// record it writes names that function and expects those gradients in that
// order.
//
// The seed varies across the output rather than being ones: a derivative that
// is right up to a permutation of the output axes gives the same answer for a
// uniform seed and a different one for this.
//
// mlfw needed function attributes to say `tera.differentiable` at all -- its
// graph functions carried none, and the printer had nowhere to put one. A
// `true` attribute prints as its bare name, which is what MLIR's unit attribute
// is, so the round trip through the parser gets the same thing back.

// MLIR-LABEL: func.func @demo
// MLIR-SAME: attributes {tera.differentiable}

// The pass writes the derivative and points the original at it. The argument
// list is the original's plus the seed; the results are the two gradients, in
// argument order, which is the order the record lists them in. The forward and
// backward halves it splits that derivative into are named beside it, and the
// derivative is what calls them.
// VJP-DAG: func.func @demo({{.*}}) -> tensor<f32> attributes {tera.bwd = @demo_bwd, tera.differentiable, tera.fwd = @demo_fwd, tera.vjp = @demo_vjp}
// VJP-DAG: func.func @demo_vjp(%{{.*}}: tensor<2x4xf32>, %{{.*}}: tensor<4x2xf32>, %{{.*}}: tensor<f32>) -> (tensor<2x4xf32>, tensor<4x2xf32>) attributes {tera.diff_args = array<i64: 0, 1>}
// VJP-DAG: call @demo_fwd
// VJP-DAG: call @demo_bwd

// `rnn` and `branch` are here because a derivative through a region is where
// two reverse passes are most likely to differ and least likely to be caught by
// anything else: the scan's reverse runs the body backwards and accumulates the
// gradient of a constant across every step, and the branch's picks a side with
// a select. Running them on the device is what found the two GPU bugs
// stage-gpu-buffers.mlir and the pipeline now pin.
