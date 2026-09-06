// RUN: %if cuda %{ tera-runner %s --entry=square --target=cuda --seed=1 --shared-libs=%mlir_cuda_runtime --shared-libs=%mlir_c_runner_utils > %t.square.staged %}
// RUN: %if cuda %{ tera-runner %s --entry=square --target=cuda --target-options=shared-tiles=false --seed=1 --shared-libs=%mlir_cuda_runtime --shared-libs=%mlir_c_runner_utils > %t.square.plain %}
// RUN: %if cuda %{ diff %t.square.staged %t.square.plain %}

// RUN: %if cuda %{ tera-runner %s --entry=deep --target=cuda --seed=2 --shared-libs=%mlir_cuda_runtime --shared-libs=%mlir_c_runner_utils > %t.deep.staged %}
// RUN: %if cuda %{ tera-runner %s --entry=deep --target=cuda --target-options=shared-tiles=false --seed=2 --shared-libs=%mlir_cuda_runtime --shared-libs=%mlir_c_runner_utils > %t.deep.plain %}
// RUN: %if cuda %{ diff %t.deep.staged %t.deep.plain %}

// RUN: %if cuda %{ tera-runner %s --entry=oblong --target=cuda --seed=3 --shared-libs=%mlir_cuda_runtime --shared-libs=%mlir_c_runner_utils > %t.oblong.staged %}
// RUN: %if cuda %{ tera-runner %s --entry=oblong --target=cuda --target-options=shared-tiles=false --seed=3 --shared-libs=%mlir_cuda_runtime --shared-libs=%mlir_c_runner_utils > %t.oblong.plain %}
// RUN: %if cuda %{ diff %t.oblong.staged %t.oblong.plain %}

// RUN: %if cuda %{ tera-runner %s --entry=batched --target=cuda --seed=4 --shared-libs=%mlir_cuda_runtime --shared-libs=%mlir_c_runner_utils > %t.batched.staged %}
// RUN: %if cuda %{ tera-runner %s --entry=batched --target=cuda --target-options=shared-tiles=false --seed=4 --shared-libs=%mlir_cuda_runtime --shared-libs=%mlir_c_runner_utils > %t.batched.plain %}
// RUN: %if cuda %{ diff %t.batched.staged %t.batched.plain %}

// RUN: %if cuda %{ tera-runner %s --entry=transposed --target=cuda --seed=5 --shared-libs=%mlir_cuda_runtime --shared-libs=%mlir_c_runner_utils > %t.transposed.staged %}
// RUN: %if cuda %{ tera-runner %s --entry=transposed --target=cuda --target-options=shared-tiles=false --seed=5 --shared-libs=%mlir_cuda_runtime --shared-libs=%mlir_c_runner_utils > %t.transposed.plain %}
// RUN: %if cuda %{ diff %t.transposed.staged %t.transposed.plain %}

// RUN: %if cuda %{ tera-runner %s --entry=deep --target=cuda --seed=6 --shared-libs=%mlir_cuda_runtime --shared-libs=%mlir_c_runner_utils > %t.repeat.a %}
// RUN: %if cuda %{ tera-runner %s --entry=deep --target=cuda --seed=6 --shared-libs=%mlir_cuda_runtime --shared-libs=%mlir_c_runner_utils > %t.repeat.b %}
// RUN: %if cuda %{ tera-runner %s --entry=deep --target=cuda --seed=6 --shared-libs=%mlir_cuda_runtime --shared-libs=%mlir_c_runner_utils > %t.repeat.c %}
// RUN: %if cuda %{ diff %t.repeat.a %t.repeat.b %}
// RUN: %if cuda %{ diff %t.repeat.a %t.repeat.c %}

// What `-tera-tile-contraction-to-shared` does to a contraction has to be
// invisible in the answer, and every number here is compared rather than a
// summary of them: the same program is run on the device twice, once with the
// operand tiles staged in shared memory and once without, and the two are
// required to agree exactly.
//
// Exactly, and not within a tolerance, because the staging does not reorder
// the sum. Both lowerings add the products for k ascending; the staged one
// carries the running total in a register and reads its operands from shared
// memory rather than global, and neither of those is arithmetic. A tolerance
// here would hide the failure this test is for.
//
// The failure it is for is a barrier in the wrong place, and that is why the
// shapes are chosen the way they are rather than for coverage of the op. A
// race between the thread that writes a tile and the thread that reads it
// shows up once per staged tile and only when the two threads happen to be
// far enough apart in time, so what exposes it is a contraction with many
// tiles to stage and few blocks to stage them -- @deep is one block staging
// sixteen tiles in sequence, which is the whole barrier protocol run sixteen
// times over with nothing else on the device to hide behind. It is run three
// more times against itself at the end, because a race that gives the right
// answer nine times out of ten is the one that reaches a release.
//
// The rest are shapes rather than sizes: an oblong tile grid where the block
// count differs per axis, a batch axis taking the third grid dimension, and a
// left operand stored transposed so that the staging reads down a column.

func.func @square(%a: tensor<64x64xf32>, %b: tensor<64x64xf32>)
    -> tensor<64x64xf32> {
  %0 = tera.dot %a, %b {lhs_batch = array<i64>, lhs_contracting = array<i64: 1>,
                        rhs_batch = array<i64>, rhs_contracting = array<i64: 0>}
      : (tensor<64x64xf32>, tensor<64x64xf32>) -> tensor<64x64xf32>
  return %0 : tensor<64x64xf32>
}

// One block of threads, sixteen tiles staged one after another.
func.func @deep(%a: tensor<32x512xf32>, %b: tensor<512x32xf32>)
    -> tensor<32x32xf32> {
  %0 = tera.dot %a, %b {lhs_batch = array<i64>, lhs_contracting = array<i64: 1>,
                        rhs_batch = array<i64>, rhs_contracting = array<i64: 0>}
      : (tensor<32x512xf32>, tensor<512x32xf32>) -> tensor<32x32xf32>
  return %0 : tensor<32x32xf32>
}

// Two blocks down, five across, seven tiles deep: nothing about the grid is
// square, so an axis swapped anywhere in the staging reads the wrong element.
func.func @oblong(%a: tensor<64x224xf32>, %b: tensor<224x160xf32>)
    -> tensor<64x160xf32> {
  %0 = tera.dot %a, %b {lhs_batch = array<i64>, lhs_contracting = array<i64: 1>,
                        rhs_batch = array<i64>, rhs_contracting = array<i64: 0>}
      : (tensor<64x224xf32>, tensor<224x160xf32>) -> tensor<64x160xf32>
  return %0 : tensor<64x160xf32>
}

// The batch axis is the third grid dimension, and a block staging the tile of
// the wrong batch is a wrong answer rather than a slow one.
func.func @batched(%a: tensor<3x64x96xf32>, %b: tensor<3x96x64xf32>)
    -> tensor<3x64x64xf32> {
  %0 = tera.dot %a, %b {lhs_batch = array<i64: 0>,
                        lhs_contracting = array<i64: 2>,
                        rhs_batch = array<i64: 0>,
                        rhs_contracting = array<i64: 1>}
      : (tensor<3x64x96xf32>, tensor<3x96x64xf32>) -> tensor<3x64x64xf32>
  return %0 : tensor<3x64x64xf32>
}

// Contracting the left operand on its first axis stores it transposed, so the
// axis the staging walks along is the one it is contiguous in.
func.func @transposed(%a: tensor<128x64xf32>, %b: tensor<128x96xf32>)
    -> tensor<64x96xf32> {
  %0 = tera.dot %a, %b {lhs_batch = array<i64>, lhs_contracting = array<i64: 0>,
                        rhs_batch = array<i64>, rhs_contracting = array<i64: 0>}
      : (tensor<128x64xf32>, tensor<128x96xf32>) -> tensor<64x96xf32>
  return %0 : tensor<64x96xf32>
}
