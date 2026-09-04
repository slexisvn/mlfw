// RUN: %if mlfw-oracle %{ %node %tera_harness/mlfw_oracle.mjs demo --emit-mlir --out %t.mlir --json %t.json %}
// RUN: %if mlfw-oracle %{ tera-opt %t.mlir %}
// RUN: %if mlfw-oracle %{ FileCheck %s --check-prefix=MLIR --input-file=%t.mlir --implicit-check-not='"tera.' %}
// RUN: %if mlfw-oracle %{ tera-runner %t.mlir --shared-libs=%mlir_c_runner_utils --data=%t.json --check %}
// RUN: %if mlfw-oracle %{ %if cuda %{ tera-runner %t.mlir --target=cuda --shared-libs=%mlir_cuda_runtime --shared-libs=%mlir_c_runner_utils --data=%t.json --check %} %}

// The same program as demo.mlir, with nothing written by hand in the loop.
//
// demo.mlir is a transcription: someone read what the mlfw tracer prints and
// wrote the tera dialect that means the same thing. That proves the dialect can
// hold the program, and it is worth keeping for exactly that, but a
// transcription is also a place where the two compilers can disagree without
// either one noticing — the hand-written file agrees with whatever it was
// transcribed from on the day it was written.
//
// Here the module is not written at all. mlfw's own graph IR printer emits the
// tera dialect, so `--emit-mlir` hands over what that printer produced,
// `--json` hands over the answer the mlfw compiler computed for the same
// inputs, and the two arrive from one run of one program. Nothing in this file
// names a value, a shape or a number, so nothing in it can drift.
//
// What each RUN line adds:
//
//   tera-opt        the module parses and verifies as tera, not merely as text
//                   that looks like it. An op the dialect does not define, or
//                   an attribute of the wrong kind, fails here.
//   FileCheck       reads %t.mlir rather than tera-opt's output, because MLIR
//                   reprints every registered op in its custom form: an op that
//                   mlfw emitted in the generic fallback would come back out of
//                   tera-opt looking like one that had not, and the
//                   implicit-check-not would never fire.
//   tera-runner     it lowers, compiles and runs, and `--check` compares the
//                   result against the oracle's own output tensor. A mismatch
//                   is a non-zero exit, so the number itself stays out of this
//                   file.
//   --target=cuda   and again on the device, where the two targets share
//                   nothing below bufferized linalg.

// The printer is asserted on, not just the answer: an emitter that produced an
// empty module, or fell back to the generic form for every op, would still make
// `tera-opt` exit zero.
// MLIR-LABEL: func.func @demo
// MLIR-SAME:    tensor<2x4xf32>
// MLIR: tera.dot
// MLIR: tera.constant
// MLIR: tera.broadcast_in_dim
// MLIR: tera.maximum
// MLIR: tera.reduce sum
// MLIR: return
