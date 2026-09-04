// RUN: %if mlfw-oracle %{ %node %tera_harness/mlfw_oracle.mjs attention --emit-mlir --out %t.mlir --json %t.json %}
// RUN: %if mlfw-oracle %{ tera-opt %t.mlir %}
// RUN: %if mlfw-oracle %{ FileCheck %s --check-prefix=MLIR --input-file=%t.mlir --implicit-check-not=softmax --implicit-check-not='"tera.' %}
// RUN: %if mlfw-oracle %{ tera-runner %t.mlir --shared-libs=%mlir_c_runner_utils --data=%t.json --check %}
// RUN: %if mlfw-oracle %{ %if cuda %{ tera-runner %t.mlir --target=cuda --shared-libs=%mlir_cuda_runtime --shared-libs=%mlir_c_runner_utils --data=%t.json --check %} %}

// Causal single-head attention, traced by mlfw and run here. gradcheck/attention.mlir
// is the same shape of program written by hand.
//
// This one is about the second disagreement between the compilers. mlfw's
// graph carries composites: a traced `softmax` is one op, and the tera dialect
// has no op for it -- deliberately, because a dialect whose reduce is a closed
// enum cannot also have an op whose derivative depends on a fused numerical
// trick. mlfw already owns the answer, `DecompositionPass`, which re-expresses
// a composite in the primitives every target can lower. Run before the print,
// the five ops it produces -- a max reduction, a broadcast, a subtract, an
// exponential, a sum reduction and a divide -- are all ops the dialect has.
//
// The same pass turns `where` into `select`, so the mask reaches the dialect as
// `tera.compare` + `tera.select` rather than an op named after neither.
//
// What this program adds over mlfw-bridge-mlp.mlir is coverage: between them
// the three bridge tests exercise dot, constant, broadcast_in_dim, transpose,
// add, sub, mul, div, maximum, exp, compare, select, reduce over two combiners,
// and return. The seed operand and combiner region that mlfw's reduce carries
// and the dialect's does not are elided by the printer and rebuilt by the
// parser, and `tera-opt` accepting both reductions is what says that mapping is
// right.
//
// Unlike the other two this program is not exact in f32 -- it exponentiates and
// divides -- so `--check` leans on its relative tolerance rather than on bit
// equality. The masked positions are the exception: they are exactly zero on
// both sides or the causal structure is wrong.

// MLIR-LABEL: func.func @attention
// MLIR: tera.dot
// MLIR: %[[MASK:.*]] = tera.compare ge
// MLIR: tera.select %[[MASK]]
// MLIR: tera.reduce maximum
// MLIR: tera.sub
// MLIR: tera.exp
// MLIR: tera.reduce sum
// MLIR: tera.div
// MLIR: tera.dot
// MLIR: return

// The two --implicit-check-not patterns above hold over the whole module rather
// than between two CHECK lines: no composite survived to the dialect, and no op
// fell back to MLIR's generic form. Both read %t.mlir and not tera-opt's output,
// for the reason mlfw-bridge.mlir gives: MLIR reprints a registered op in its
// custom form whichever way it arrived, so a generic fallback is invisible on
// the far side of a round trip -- and it verifies perfectly well, so the
// tera-opt line cannot catch it either.
