// RUN: %if mlfw-oracle %{ %node %tera_harness/mlfw_oracle.mjs dynbatch --emit-mlir --out %t.mlir --json %t.json %}
// RUN: %if mlfw-oracle %{ tera-opt %t.mlir %}
// RUN: %if mlfw-oracle %{ FileCheck %s --check-prefix=MLIR --input-file=%t.mlir --implicit-check-not='"tera.' %}
// RUN: %if mlfw-oracle %{ tera-runner %t.mlir --entry=dynbatch --shared-libs=%mlir_c_runner_utils --data=%t.json --check %}
// RUN: %if mlfw-oracle %{ %if cuda %{ tera-runner %t.mlir --entry=dynbatch --target=cuda --shared-libs=%mlir_cuda_runtime --shared-libs=%mlir_c_runner_utils --data=%t.json --check %} %}

// The same module, a different batch. One compile, two shapes -- which is the
// whole of what a dynamic extent buys, and the only thing that says the extent
// is read rather than folded in.
// RUN: %if mlfw-oracle %{ %node %tera_harness/mlfw_oracle.mjs dynbatch_wide --out %t.wide.json %}
// RUN: %if mlfw-oracle %{ tera-runner %t.mlir --entry=dynbatch --shared-libs=%mlir_c_runner_utils --data=%t.wide.json --check %}
// RUN: %if mlfw-oracle %{ %if cuda %{ tera-runner %t.mlir --entry=dynbatch --target=cuda --shared-libs=%mlir_cuda_runtime --shared-libs=%mlir_c_runner_utils --data=%t.wide.json --check %} %}

// And its derivative, which is dynamic all the way through: the reverse pass
// broadcasts an adjoint back onto a shape it has to be told, the same way the
// forward does.
// RUN: %if mlfw-oracle %{ %node %tera_harness/mlfw_oracle.mjs dynbatch --grad --emit-mlir --out %t.grad.mlir --json %t.grad.json %}
// RUN: %if mlfw-oracle %{ tera-opt %t.grad.mlir --tera-autodiff -o %t.vjp.mlir %}
// RUN: %if mlfw-oracle %{ FileCheck %s --check-prefix=VJP --input-file=%t.vjp.mlir %}
// RUN: %if mlfw-oracle %{ tera-runner %t.vjp.mlir --entry=dynbatch_vjp --shared-libs=%mlir_c_runner_utils --data=%t.grad.json --check %}
// RUN: %if mlfw-oracle %{ %if cuda %{ tera-runner %t.vjp.mlir --entry=dynbatch_vjp --target=cuda --shared-libs=%mlir_cuda_runtime --shared-libs=%mlir_c_runner_utils --data=%t.grad.json --check %} %}
// RUN: %if mlfw-oracle %{ %node %tera_harness/mlfw_oracle.mjs dynbatch_wide --grad --out %t.gradwide.json %}
// RUN: %if mlfw-oracle %{ tera-runner %t.vjp.mlir --entry=dynbatch_vjp --shared-libs=%mlir_c_runner_utils --data=%t.gradwide.json --check %}

// And back through mlfw, at both batches.
// RUN: %if mlfw-oracle %{ tera-opt %t.mlir -o %t.rt.mlir %}
// RUN: %if mlfw-oracle %{ %node %tera_harness/mlfw_reader.mjs %t.rt.mlir --entry dynbatch --data %t.json %}
// RUN: %if mlfw-oracle %{ %node %tera_harness/mlfw_reader.mjs %t.rt.mlir --entry dynbatch --data %t.wide.json %}

// A second op that decides a shape rather than inheriting one, and is told the
// extents the same way. `tera.reshape` takes its target shape from its result
// type, so a `?` there is a number the type does not carry.
// RUN: %if mlfw-oracle %{ %node %tera_harness/mlfw_oracle.mjs dynreshape --emit-mlir --out %t.reshape.mlir --json %t.reshape.json %}
// RUN: %if mlfw-oracle %{ FileCheck %s --check-prefix=RESHAPE --input-file=%t.reshape.mlir --implicit-check-not='"tera.' %}
// RUN: %if mlfw-oracle %{ tera-opt %t.reshape.mlir -o %t.reshape.rt.mlir %}
// RUN: %if mlfw-oracle %{ tera-runner %t.reshape.mlir --entry=dynreshape --shared-libs=%mlir_c_runner_utils --data=%t.reshape.json --check %}
// RUN: %if mlfw-oracle %{ %if cuda %{ tera-runner %t.reshape.mlir --entry=dynreshape --target=cuda --shared-libs=%mlir_cuda_runtime --shared-libs=%mlir_c_runner_utils --data=%t.reshape.json --check %} %}
// RUN: %if mlfw-oracle %{ %node %tera_harness/mlfw_oracle.mjs dynreshape_wide --out %t.reshapewide.json %}
// RUN: %if mlfw-oracle %{ tera-runner %t.reshape.mlir --entry=dynreshape --shared-libs=%mlir_c_runner_utils --data=%t.reshapewide.json --check %}
// RUN: %if mlfw-oracle %{ %if cuda %{ tera-runner %t.reshape.mlir --entry=dynreshape --target=cuda --shared-libs=%mlir_cuda_runtime --shared-libs=%mlir_c_runner_utils --data=%t.reshapewide.json --check %} %}
// RUN: %if mlfw-oracle %{ %node %tera_harness/mlfw_reader.mjs %t.reshape.rt.mlir --entry dynreshape --data %t.reshape.json %}
// RUN: %if mlfw-oracle %{ %node %tera_harness/mlfw_reader.mjs %t.reshape.rt.mlir --entry dynreshape --data %t.reshapewide.json %}

// A batch nobody has decided yet, traced by mlfw and run here.
//
// Every other bridge program is one shape compiled for one shape, and a
// compiler that had quietly specialised on the example inputs would pass all of
// them. This one cannot be passed that way: the module says `tensor<?x4xf32>`
// and the two records above give it three rows and then five.
//
// The op that makes a dynamic extent hard is `broadcast_in_dim`, and it is in
// every traced program. Everything else is shaped like an operand -- an
// elementwise result has its operands' extents, a `dot`'s free axes have its
// operands' -- so the lowering finds those with `tensor.dim` on something it was
// already given. A broadcast decides a shape instead of inheriting one: the
// relu's zero is a scalar and has nothing to say how tall the activation it is
// being broadcast onto is.
//
// So the dialect gained the two things that answer it. `tera.dim` is the only
// op that turns a shape into a value, and `sizes` is where a broadcast is handed
// the extents it decided, one per `?` in its result type. `MaterializeShapesPass`
// is what fills that in on mlfw's side, and like the other two normalisation
// passes it is mlfw's own and not part of its default pipeline.
//
// Running it on the device found the last of it: `scf-parallel-loop-tiling`
// writes a tiled loop's step as the old step times the tile size, and
// `convert-parallel-loops-to-gpu` only takes a step it can read as a constant.
// Over a static extent that product is folded where it is built; over a dynamic
// one it is an `arith.muli` nothing had folded, and every loop silently stayed
// on the host -- a program that lowered, ran, and gave the right answer without
// ever reaching the GPU. Only a `--target=cuda` line over a dynamic shape could
// have found it, and only by looking at whether a kernel was launched at all.

// MLIR-LABEL: func.func @dynbatch
// MLIR-SAME:    (%{{.*}}: tensor<?x4xf32>, %{{.*}}: tensor<4x2xf32>) -> (tensor<?x2xf32>)
// MLIR: tera.dot
// MLIR: tera.constant
// MLIR: %[[N:.*]] = tera.dim %{{.*}} {dimension = 0 : i64} : tensor<?x2xf32> -> tensor<i64>
// MLIR: tera.broadcast_in_dim %{{.*}} sizes(%[[N]]) {broadcast_dimensions = array<i64>} : tensor<f32> -> tensor<?x2xf32>
// MLIR: tera.maximum
// MLIR: return

// The derivative keeps the `?` and gains a `tera.dim` of its own: the zero the
// maximum's rule compares against is shaped like the forward's result, which
// is a shape the reverse pass has to be told in exactly the same way.
// VJP: func.func @dynbatch_vjp(%{{.*}}: tensor<?x4xf32>, %{{.*}}: tensor<4x2xf32>, %{{.*}}: tensor<?x2xf32>) -> (tensor<?x4xf32>, tensor<4x2xf32>)
// VJP: tera.dim
// VJP: tera.broadcast_in_dim %{{.*}} sizes(

// The gradients the record holds are mlfw's own dynamic reverse pass, over the
// same `?` the forward carries. It answered zeros until mlfw stopped handing
// its runtime bare typed arrays for a compiled backward's buffers: an extent
// that is a kernel parameter is read off the shapes the arguments carry, and
// arrays that carry none left every one of them resolving to 1.

// RESHAPE-LABEL: func.func @dynreshape
// RESHAPE-SAME:    (%{{.*}}: tensor<?x2x4xf32>
// RESHAPE: %[[N:.*]] = tera.dim %{{.*}} {dimension = 0 : i64} : tensor<?x2x4xf32> -> tensor<i64>
// RESHAPE: tera.reshape %{{.*}} sizes(%[[N]]) : tensor<?x2x4xf32> -> tensor<?x8xf32>

// The reader runs over these now. A dynamic extent is `?` in MLIR's type and a
// symbol beside it in mlfw's, and only the first survives the text -- so a
// module read back has extents that have lost the identity tying them to each
// other. What ties them is the dataflow, not the type, and the record's inputs
// are concrete: `ShapeRefinementPass` walks the argument shapes forward through
// the program and puts every extent back. Both records go through the one
// module, so a reader that had baked one shape in would be caught by the other.
