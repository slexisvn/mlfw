// RUN: %if mlfw-oracle %{ %node %tera_harness/mlfw_oracle.mjs matmul --emit-mlir --out %t.mlir --json %t.json %}
// RUN: %if mlfw-oracle %{ FileCheck %s --check-prefix=MLIR --input-file=%t.mlir --implicit-check-not='"tera.' %}
// RUN: %if mlfw-oracle %{ tera-runner %t.mlir --shared-libs=%mlir_c_runner_utils --data=%t.json --check %}
// RUN: %if mlfw-oracle %{ %if cuda %{ tera-runner %t.mlir --target=cuda --shared-libs=%mlir_cuda_runtime --shared-libs=%mlir_c_runner_utils --data=%t.json --check %} %}
// RUN: %if mlfw-oracle %{ %if cuda %{ tera-runner %t.mlir --target=cuda --target-options=shared-tiles=false --shared-libs=%mlir_cuda_runtime --shared-libs=%mlir_c_runner_utils --data=%t.json --check %} %}

// RUN: %if mlfw-oracle %{ %node %tera_harness/mlfw_oracle.mjs matmul_batched --emit-mlir --out %t.b.mlir --json %t.b.json %}
// RUN: %if mlfw-oracle %{ FileCheck %s --check-prefix=BATCHED --input-file=%t.b.mlir --implicit-check-not='"tera.' %}
// RUN: %if mlfw-oracle %{ tera-runner %t.b.mlir --shared-libs=%mlir_c_runner_utils --data=%t.b.json --check %}
// RUN: %if mlfw-oracle %{ %if cuda %{ tera-runner %t.b.mlir --target=cuda --shared-libs=%mlir_cuda_runtime --shared-libs=%mlir_c_runner_utils --data=%t.b.json --check %} %}
// RUN: %if mlfw-oracle %{ %if cuda %{ tera-runner %t.b.mlir --target=cuda --target-options=shared-tiles=false --shared-libs=%mlir_cuda_runtime --shared-libs=%mlir_c_runner_utils --data=%t.b.json --check %} %}

// Two contractions at a size `-tera-tile-contraction-to-shared` cuts, traced
// by mlfw and answered by it. shared-tiles.mlir compares terac's two lowerings
// against each other, which catches a staged tile read at the wrong index but
// not a schedule that is wrong the same way twice; this compares against a
// compiler that shares no code with either.
//
// It is run three ways: on the host, on the device with the tiles staged, and
// on the device without. The last is not redundant. A number that is right
// only when the operands come from global memory says the staging is wrong; a
// number that is wrong both ways says something above the schedule is, and the
// two failures want telling apart before anyone looks at a barrier.
//
// `ramp` deals values that are multiples of an eighth, so a product is a
// multiple of a sixty-fourth and a sum of 128 of them is one too, well inside
// what f32 holds exactly. There is no rounding here for a wrong answer to hide
// in, and `--check` compares against the oracle's own bits.

// MLIR-LABEL: func.func @matmul
// MLIR: tera.dot
// MLIR: tera.dot
// MLIR: return

// BATCHED-LABEL: func.func @matmul_batched
// BATCHED: tera.dot {{.*}}lhs_batch = array<i64: 0>
// BATCHED: return
