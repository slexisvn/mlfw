//===- Passes.td - Tera conversion passes ------------------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Passes that take tera out to another dialect. Generated with
// `-name TeraConversion`, so `registerTeraConversionPasses` sits beside
// `registerTeraPasses` instead of colliding with it.
//
//===----------------------------------------------------------------------===//

#ifndef TERA_CONVERSION_PASSES
#define TERA_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

// Anchored on nothing: the conversion adds no symbol and reads no other
// function, so it runs on whatever it is given and nests under `func.func`
// when a pipeline wants the functions converted in parallel.
def ConvertTeraToLinalg : Pass<"convert-tera-to-linalg"> {
  let summary = "Lower the tera dialect to linalg, tensor, arith and math.";
  let description = [{
    Replaces every tera op with its upstream equivalent. This is the floor of
    the compiler that terac owns: below it, tiling, fusion, vectorization and
    bufferization are all upstream passes.

    Destination tensors are materialised up front, so the conversion requires
    static shapes and rejects a dynamic extent rather than mislowering it.
  }];

  let dependentDialects = [
    "::mlir::arith::ArithDialect",
    "::mlir::bufferization::BufferizationDialect",
    "::mlir::linalg::LinalgDialect",
    "::mlir::math::MathDialect",
    "::mlir::scf::SCFDialect",
    "::mlir::tensor::TensorDialect",
  ];
}

def StageGpuBuffers
    : Pass<"tera-stage-gpu-buffers", "::mlir::func::FuncOp"> {
  let summary = "Give every kernel operand a buffer that lives on the device.";
  let description = [{
    A `gpu.launch_func` is otherwise handed the pointer the host allocated, and
    a kernel dereferencing that faults with an illegal address.

    The obvious cheaper fix is `gpu.host_register`, which page-locks the host
    buffer and lets the device address it directly, and it is what MLIR's own
    CUDA integration tests do. It does not work here: under the WDDM driver
    model a registered host pointer is still not valid on the device, so the
    kernel faults exactly as before. Copying is what is left.

    So each buffer a kernel reads gets a `gpu.alloc` beside it, a copy in
    before the launches and a copy back after the last one. Both directions,
    unconditionally: which kernel writes which operand is knowable, but not
    from the launch, and guessing it wrong is a silently wrong answer rather
    than a slow one.

    The copies sit around the launches rather than at the function boundary,
    which is only correct while the host does not touch the buffer in between.
    That holds because every computation is in a kernel by this point, and the
    pass checks it rather than trusting it: a host reader other than the
    terminator or a deallocation is an error.
  }];

  let dependentDialects = [
    "::mlir::gpu::GPUDialect",
    "::mlir::memref::MemRefDialect",
  ];
}

def LowerNestedLinalg
    : Pass<"tera-lower-nested-linalg", "::mlir::func::FuncOp"> {
  let summary = "Lower to scf loops the linalg ops affine cannot name the extents of.";
  let description = [{
    A dynamic extent reaches a loop nest as a value, and the affine dialect will
    only take one it can read as a dim or a symbol: the result of a `dim` op on
    a value that sits at the top level of the enclosing affine scope. A function
    is such a scope and an `scf.for` is not, so a `linalg` op inside a scan's
    body whose destination is dynamic asks `affine.for` for a bound it refuses,
    and `convert-linalg-to-affine-loops` fails on the whole function.

    This lowers exactly those ops to `scf` loops first, leaving the rest for the
    affine conversion and the vectorizer that reads it. It is not a fallback for
    dynamic shapes in general: an op at the top level of the function names its
    extents perfectly well in affine terms, and a static op inside a body does
    not name any.
  }];

  let dependentDialects = [
    "::mlir::memref::MemRefDialect",
    "::mlir::scf::SCFDialect",
  ];
}

def VectorizeAffine
    : Pass<"tera-vectorize-affine", "::mlir::func::FuncOp"> {
  let summary = "Vectorize the affine loops whose accesses the vectorizer reads correctly.";
  let description = [{
    `affine-super-vectorize` decides whether a loop is worth vectorizing by
    asking whether each access walks memory one element at a time along it, and
    the question it actually answers is narrower: which index of the access
    varies with the loop, not how fast. An access of `w * 2 + k` varies with `w`
    at the fastest axis, so it is taken for contiguous, and the vector read that
    comes out starts in the right place and then reads the wrong elements.

    Every op in the dialect indexed by a permutation of its loops until two did
    not. A window moves by its stride and an axis read from the far end moves
    backwards, and both are wrong under that analysis rather than merely
    unvectorized.

    So the pass runs where the analysis is right and stands aside where it is
    not, which is decided here rather than inside it. The grain is a function,
    because that is what the pass takes; a function holding one strided window
    keeps its scalar loops throughout, which is slower than it could be and
    is the answer it should have had.
  }];

  let options = [
    Option<"vectorSize", "vector-size", "int64_t", "16",
           "How many lanes wide the vectorizer should aim for.">,
  ];

  let dependentDialects = [
    "::mlir::affine::AffineDialect",
    "::mlir::vector::VectorDialect",
  ];
}

#endif // TERA_CONVERSION_PASSES
