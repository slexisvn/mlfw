// RUN: %if mlfw-oracle %{ %node %tera_harness/mlfw_oracle.mjs mlp --emit-mlir --out %t.mlir --json %t.json %}
// RUN: %if mlfw-oracle %{ tera-opt %t.mlir %}
// RUN: %if mlfw-oracle %{ FileCheck %s --check-prefix=MLIR --input-file=%t.mlir --implicit-check-not='"tera.' %}
// RUN: %if mlfw-oracle %{ tera-runner %t.mlir --shared-libs=%mlir_c_runner_utils --data=%t.json --check %}
// RUN: %if mlfw-oracle %{ %if cuda %{ tera-runner %t.mlir --target=cuda --shared-libs=%mlir_cuda_runtime --shared-libs=%mlir_c_runner_utils --data=%t.json --check %} %}

// A two-layer perceptron, traced by mlfw and run here. bench/mlp.mlir is the
// same program written by hand; this one is not written at all.
//
// It is here because of one disagreement between the two compilers, and the
// disagreement is instructive. mlfw's elementwise ops broadcast their operands
// implicitly: `add` of a `2x6` activation and a `6` bias is a well-formed op
// there, and its trait `SAME_OPERAND_AND_RESULT_TYPE` constrains the element
// type only. `tera.add` is `SameOperandsAndResultType` in MLIR's sense, which
// includes the shape, and its assembly format prints one type for all three
// operands -- so an implicitly broadcasting add cannot even be spelt in the
// dialect without claiming the bias has a shape it does not have.
//
// The fix is not in the printer. mlfw runs `ExplicitBroadcastPass` first, which
// materialises the broadcast as the `tera.broadcast_in_dim` the dialect already
// had, and the printer then has nothing left to lie about. Without that pass
// this program leaves mlfw in MLIR's generic form and `tera-opt` rejects it --
// which is the right failure, and the reason the pass exists rather than a
// special case in the emitter.
//
// The inputs are multiples of 1/8 and every contraction is short, so the whole
// program is exact in f32: `--check` compares against the oracle's answer with
// no rounding to hide behind, and the device has to produce the same bits.

// MLIR-LABEL: func.func @mlp
// MLIR: tera.dot
// MLIR: %[[BIAS:.*]] = tera.broadcast_in_dim %{{.*}} {broadcast_dimensions = array<i64: 1>} : tensor<6xf32> -> tensor<2x6xf32>
// MLIR: tera.add %{{.*}}, %[[BIAS]] : tensor<2x6xf32>
// MLIR: tera.maximum
// MLIR: tera.dot
// MLIR: tera.add
// MLIR: return
