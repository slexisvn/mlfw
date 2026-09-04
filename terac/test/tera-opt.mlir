// RUN: tera-opt --show-dialects | FileCheck %s
// CHECK: Available Dialects:
// CHECK-SAME: tera

// The conversions to LLVM arrive as dialect extensions rather than with the
// dialects themselves, so a driver that registers only dialects aborts the
// first time a pass asks `func` for its LLVM patterns — an abort, not a
// diagnostic, which is why it needs a test of its own rather than a
// `-verify-diagnostics` case.

// RUN: tera-opt %s --convert-to-llvm | FileCheck %s --check-prefix=EXTENSIONS
// EXTENSIONS-LABEL: llvm.func @needs_an_extension
// EXTENSIONS: llvm.add
func.func @needs_an_extension(%a: i32) -> i32 {
  %0 = arith.addi %a, %a : i32
  return %0 : i32
}

// Translating a `gpu.module` to LLVM IR goes through an interface that neither
// the dialect nor its extensions carry, so a driver holding only the two above
// serialises nothing: `--gpu-module-to-binary` reports a missing
// `LLVMTranslationDialectInterface` and the whole path below `gpu` is dead.
// That registration has been dropped from this driver once already, which is
// why it is checked here rather than left to whichever pipeline needs it next.
//
// `format=isa` stops at PTX, so this reads the NVPTX backend and nothing else —
// no `ptxas`, no CUDA toolkit, no device. The entry point is what proves the
// body was translated; a module that lost its kernel still prints `gpu.binary`.

// RUN: tera-opt %s --nvvm-attach-target=chip=sm_86 --gpu-module-to-binary=format=isa | FileCheck %s --check-prefix=PTX
// PTX-LABEL: gpu.binary @kernels
// PTX-SAME: .target sm_86
// PTX-SAME: .visible .entry kernel
gpu.module @kernels {
  llvm.func @kernel(%out: !llvm.ptr) attributes {gpu.kernel, nvvm.kernel} {
    %block = nvvm.read.ptx.sreg.ctaid.x : i32
    %slot = llvm.getelementptr %out[%block] : (!llvm.ptr, i32) -> !llvm.ptr, f32
    %one = llvm.mlir.constant(1.0 : f32) : f32
    llvm.store %one, %slot : f32, !llvm.ptr
    llvm.return
  }
}
