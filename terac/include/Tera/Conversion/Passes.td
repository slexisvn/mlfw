//===- Passes.td - Tera conversion passes ------------------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TERA_CONVERSION_PASSES
#define TERA_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def ConvertTeraToLinalg : Pass<"convert-tera-to-linalg"> {
  let summary = "Lower the tera dialect to linalg, tensor, arith and math.";
  let description = [{
    Replaces every tera op with its upstream equivalent. This is the floor of
    the dialect that terac owns: below it the IR is linalg on tensors, which
    `-tera-tile-and-fuse` and `-tera-vectorize-linalg` schedule and upstream
    bufferizes.

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

def TileAndFuse
    : Pass<"tera-tile-and-fuse", "::mlir::func::FuncOp"> {
  let summary = "Cut every linalg op into tiles and fold its producers into them.";
  let description = [{
    Runs on tensors, above bufferization, because that is the only place a
    producer and its consumer are still an SSA edge. Below it they are two
    writes to a buffer and folding one into the other is an aliasing question.

    Two levels. The first cuts the parallel loops -- the innermost to one
    vector, the rest to one element -- and folds into that tile every producer
    it reaches which nothing else reads, and the destination whatever else
    reads it. The second cuts every reduction loop to one element, so what is
    left for the vectorizer is exactly one vector and a loop carrying the
    accumulator. It is separate because a producer cannot be folded into a tile
    holding a partial sum: fusion has to be finished before a reduction is cut.

    A tile shaped as a block instead of a vector is what a register-blocking
    cost model would choose, and there is none here. A 2-D tile also unrolls,
    once per row, into as many vector ops; measured on test/Integration/bench
    that is slower than the loop it replaces, so the tile stays one vector.

    An op whose indexing maps carry a non-positive coefficient is left alone.
    Slicing one means reasoning about an index that runs backwards, which the
    upstream tiling utilities assert against rather than answer.
  }];

  let options = [
    Option<"vectorWidth", "vector-width", "int64_t", "16",
           "Lanes the innermost tiled loop is cut to.">,
  ];

  let dependentDialects = [
    "::mlir::affine::AffineDialect",
    "::mlir::arith::ArithDialect",
    "::mlir::linalg::LinalgDialect",
    "::mlir::scf::SCFDialect",
    "::mlir::tensor::TensorDialect",
  ];
}

def VectorizeLinalg
    : Pass<"tera-vectorize-linalg", "::mlir::func::FuncOp"> {
  let summary = "Rewrite the linalg ops small enough to be one vector into vector ops.";
  let description = [{
    A linalg op carries an affine map per operand saying how it is indexed by
    the loops beside it, so vectorizing here reads the access instead of
    inferring it. `affine-super-vectorize`, which this replaces, had to recover
    the access from loop bounds and subscripts and answered which loop an index
    varies with as though that were how fast: a window moving by its stride and
    an axis read from the far end were both reported contiguous and read the
    wrong elements, which is why running it needed a gate. This needs none.

    The vector shape is the op's whole iteration space, so this belongs after
    `-tera-tile-and-fuse`. An op the tiling could not cut below
    `max-vector-elements` keeps its loops and is lowered as a scalar nest.
  }];

  let options = [
    Option<"maxVectorElements", "max-vector-elements", "int64_t", "1024",
           "Iteration-space size above which an op keeps its loops.">,
  ];

  let dependentDialects = [
    "::mlir::arith::ArithDialect",
    "::mlir::tensor::TensorDialect",
    "::mlir::vector::VectorDialect",
  ];
}

#endif
