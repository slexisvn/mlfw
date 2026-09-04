// RUN: %if mlfw-oracle %{ %node %tera_harness/mlfw_oracle.mjs rnn --emit-mlir --out %t.mlir --json %t.json %}
// RUN: %if mlfw-oracle %{ tera-opt %t.mlir %}
// RUN: %if mlfw-oracle %{ FileCheck %s --check-prefix=MLIR --input-file=%t.mlir --implicit-check-not='"tera.' %}
// RUN: %if mlfw-oracle %{ tera-runner %t.mlir --shared-libs=%mlir_c_runner_utils --data=%t.json --check %}
// RUN: %if mlfw-oracle %{ %if cuda %{ tera-runner %t.mlir --target=cuda --shared-libs=%mlir_cuda_runtime --shared-libs=%mlir_c_runner_utils --data=%t.json --check %} %}

// A recurrence traced by mlfw and run here. controlflow.mlir is the same shape
// of program written by hand.
//
// This is the first program to cross the bridge carrying a region, and it is
// here because of the third disagreement between the compilers -- the expensive
// one, and the one the two dialects disagree about at the level of what an op
// means rather than how it is spelt.
//
// mlfw's regions may read the block around them. `tera.scan` is
// `IsolatedFromAbove` on purpose: every value the body reads is an operand, so
// the reverse pass has somewhere to put that value's gradient. A recurrence's
// weights are exactly the values this is about -- read every step, sliced by
// none of them -- so a traced RNN captures two of them and could not be spelt
// in the dialect at all.
//
// `IsolateRegionsPass` is mlfw's answer, and it is a pass rather than a special
// case in the emitter for the same reason `ExplicitBroadcastPass` is: the
// captures are a fact about the graph, not about its printing. It lifts what
// the body reads into the op's `consts` clause and adds one body argument per
// value; a captured *constant* is cloned into the body instead, because
// rematerialising one is cheaper than threading it through and MLIR would sink
// it back in anyway.
//
// The operand and body-argument order is the other half. mlfw used to put the
// stacked inputs first and the carries after; `tera.scan` takes the carries
// first, then one slice of each input, then the constants, which is also JAX's
// order. mlfw was the odd one out and now is not, so the printer has nothing to
// permute -- a permutation in a printer is a place for a transposed answer to
// hide, and FileCheck cannot see one.
//
// Every value is a multiple of 1/8 and every step is a short sum, so the whole
// recurrence is exact in f32 across all three steps and `--check` compares the
// two compilers bit for bit.

// MLIR-LABEL: func.func @rnn
// MLIR: tera.scan init(%{{.*}} : tensor<2xf32>) xs(%{{.*}} : tensor<3x2xf32>) consts(%{{.*}}, %{{.*}} : tensor<2x2xf32>, tensor<2x2xf32>) -> (tensor<2xf32>, tensor<3x2xf32>) {

// The body takes the carry, then the step slice, then the two lifted weights.
// MLIR-NEXT: ^bb0(%{{.*}}: tensor<2xf32>, %{{.*}}: tensor<2xf32>, %{{.*}}: tensor<2x2xf32>, %{{.*}}: tensor<2x2xf32>):

// MLIR: tera.dot
// MLIR: tera.dot
// MLIR: tera.add

// The relu's zero was captured too, and was sunk rather than lifted: it appears
// inside the body, after the ops that came before it there.
// MLIR: tera.constant
// MLIR: tera.broadcast_in_dim
// MLIR: tera.maximum
// MLIR: tera.yield
// MLIR: return

// The `--implicit-check-not` above holds over the whole module and reads
// %t.mlir rather than tera-opt's output, for the reason mlfw-bridge.mlir gives:
// MLIR reprints a registered op in its custom form whichever way it arrived, so
// an op that fell back to the generic form is invisible on the far side of a
// round trip -- and it verifies perfectly well, so the tera-opt line cannot
// catch it either. `tera.scan` is the op where that mattered most: it is the
// only one whose custom form carries a region and three named operand clauses,
// and getting any of them wrong prints a module that still parses.
