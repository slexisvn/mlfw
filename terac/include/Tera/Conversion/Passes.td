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
    : Pass<"tera-stage-gpu-buffers", "::mlir::ModuleOp"> {
  let summary = "Give every kernel operand a buffer that lives on the device.";
  let description = [{
    A `gpu.launch_func` is otherwise handed the pointer the host allocated, and
    a kernel dereferencing that faults with an illegal address.

    The obvious cheaper fix is `gpu.host_register`, which page-locks the host
    buffer and lets the device address it directly, and it is what MLIR's own
    CUDA integration tests do. It does not work here: under the WDDM driver
    model a registered host pointer is still not valid on the device, so the
    kernel faults exactly as before. Copying is what is left.

    So each buffer a kernel touches gets a `gpu.alloc` beside it, and the
    copies between the two are placed where the current contents move. The
    pass walks the block the buffer lives in carrying one of four states --
    the contents are nowhere yet, on the host, on the device, or the same on
    both -- and emits a copy only at a transition: in before a kernel touches
    what the host last wrote, back before the host reads what a kernel last
    wrote. A freshly allocated buffer holds nothing worth copying in, and one
    whose last writer is a kernel nothing goes on to read is not copied back.

    A kernel that writes gets the copy in as well as one that reads, because
    what a kernel writes is not always all of it, and the elements it leaves
    alone are the ones the host put there.

    Which direction a kernel moves a buffer is read off the outlined
    `gpu.func` rather than assumed, through the memory effects of the ops that
    reach the matching argument. That is the question the earlier form of this
    pass could not answer from the launch alone, and it is why it had to copy
    both ways every time and forbid the host from touching a staged buffer
    between two launches. Both restrictions are gone: a host read between two
    kernels is a copy back and a copy in, which is what it means.

    An argument carrying `tera.device_resident` is skipped: the caller has
    already put it on the device and hands the kernels that pointer, which is
    what a weight read every step wants rather than an upload per call. The
    host cannot address it, so a host access to one is an error. The attribute
    is taken off the argument once it has been honoured: below here a memref
    is a pointer, and a promise about a pointer is one nothing can check.

    An op holding regions stands for everything inside it, because a copy
    placed in there would run once per execution of the region rather than
    once. So a buffer a loop's kernels use crosses once, before the loop and
    after it. The one case left over is a region that holds both a kernel and
    a host access to the same buffer: the order between them is not visible
    from outside, and that is an error rather than a guess.
  }];

  let dependentDialects = [
    "::mlir::gpu::GPUDialect",
    "::mlir::memref::MemRefDialect",
  ];
}

def TileParallelLoops
    : Pass<"tera-tile-parallel-loops", "::mlir::func::FuncOp"> {
  let summary = "Cut every parallel loop into the thread block it becomes.";
  let description = [{
    `convert-parallel-loops-to-gpu` turns a two-level parallel nest into a grid
    of blocks over a block of threads, so the inner level's extents are the
    thread block. This pass makes that inner level, and it picks the extents
    per loop from the loop's own trip counts rather than from one number given
    for the whole module.

    Each dimension is cut to the largest divisor of its trip count that the
    remaining thread budget and the hardware's limit on that axis allow, the
    dimension that becomes `thread_x` preferring a multiple of `warp-size`
    because that is the axis a warp runs along. A divisor is used rather than a
    round number so the inner loop keeps a constant bound: a tile that does not
    divide its loop still lowers, but every thread then carries a bounds test
    the block shape could have made unnecessary.

    Which dimension that is comes from `gpu-map-parallel-loops`, which this
    pass is written against at `mapping-policy=innermost-first`: the innermost
    of the first three loops takes `x`, so it is the innermost that is given
    the budget first. That policy is the one where the axis a warp runs along
    is also the axis a row is contiguous in.

    `tile-sizes` overrides the choice for every loop, which is what a schedule
    written by hand wants; a dimension whose trip count is not a constant is
    left at one, because a block extent has to be known to be one.
  }];

  let options = [
    Option<"threadsPerBlock", "threads-per-block", "int64_t", "256",
           "Threads a block is aimed at, before the divisors it must use.">,
    Option<"maxThreadsPerBlock", "max-threads-per-block", "int64_t", "1024",
           "Threads a block may not exceed.">,
    Option<"warpSize", "warp-size", "int64_t", "32",
           "Threads that run in lockstep, which the first dimension prefers "
           "to be a multiple of.">,
    ListOption<"tileSizes", "tile-sizes", "int64_t",
               "Thread-block extents to use for every loop instead of the "
               "ones derived from its trip counts.">,
  ];

  let dependentDialects = [
    "::mlir::affine::AffineDialect",
    "::mlir::arith::ArithDialect",
    "::mlir::scf::SCFDialect",
  ];
}

def VerifyGpuMapping
    : Pass<"tera-verify-gpu-mapping", "::mlir::func::FuncOp"> {
  let summary = "Report a parallel loop the GPU conversion left on the host.";
  let description = [{
    `convert-parallel-loops-to-gpu` marks a loop legal as soon as its pattern
    has been tried, whether or not the pattern succeeded, so a loop it cannot
    map is not an error there: the loop is simply still an `scf.parallel`
    afterwards, and everything below lowers it to a sequential host loop. A
    program compiled for the device then runs that part on one core, correctly,
    with nothing said. This pass says it.

    A loop the conversion took is gone by now, so what is left to look for is a
    loop still carrying the `mapping` attribute that `gpu-map-parallel-loops`
    put on it: it was asked for a processor and did not get one. A loop with no
    mapping was never a candidate, and on the host it is left alone -- a
    reduction over every axis never becomes a loop the mapper walks.

    Inside a `gpu.launch` the same loop means something else, and worse. The
    conversion builds the launch from the outermost mapped loop and then walks
    its body, but it only descends into a nested `scf.parallel`: every other
    op, `scf.for` included, is cloned whole. A parallel loop under a sequential
    one is therefore never reached, never mapped, and never even annotated --
    so the mapping attribute this pass was written to look for is not there to
    find. What is emitted is a launch whose thread block is one thread wide,
    running the loop's every iteration in sequence. It is the same failure as
    the host case and it is quieter, because nothing upstream reports it and
    the numbers still come out right. A parallel loop that survives into a
    kernel is that, whatever it carries.
  }];

  let dependentDialects = ["::mlir::gpu::GPUDialect"];
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

    A loop over a `?` is cut the same way one over a known extent is, and then
    peeled: the tile cannot be shown to divide the extent, so every slice taken
    inside carries an `affine.min` against what is left of it, and a tile whose
    own shape is dynamic is one `-tera-vectorize-linalg` has no vector for.
    Peeling separates the whole tiles from the short last one, and the loop
    holding nearly every element then has a static tile and is vectorized;
    the one iteration left over keeps the scalar code it needs. Without it the
    tiling would buy the loop overhead and none of the vectors it is for, which
    is why the dynamic case was skipped entirely before.

    Peeling is what this needs and not `ValueBoundsOpInterface`. Bounds would
    answer how large the extent can be, and that is a question with no answer
    here: a `?` arrives as a function argument and there is no fact anywhere in
    the module that constrains it. What the tiling actually wanted was not the
    extent but a tile whose shape is known, and peeling gives that without
    knowing the extent at all.
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
