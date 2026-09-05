//===- TileParallelLoops.cpp - Make the block a loop becomes ----*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Tera/Conversion/Passes.h"

#include "Tera/Analysis/BlockTiles.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::tera {
#define GEN_PASS_DEF_TILEPARALLELLOOPS
#include "Tera/Conversion/Passes.h.inc"

namespace {
struct TileParallelLoops
    : public impl::TileParallelLoopsBase<TileParallelLoops> {
  using impl::TileParallelLoopsBase<TileParallelLoops>::TileParallelLoopsBase;

  void runOnOperation() final {
    GpuTargetModel model;
    model.warpSize = warpSize;
    model.maxThreadsPerBlock = maxThreadsPerBlock;
    model.preferredThreadsPerBlock = threadsPerBlock;

    SmallVector<int64_t> given(tileSizes.begin(), tileSizes.end());
    if (llvm::any_of(given, [](int64_t size) { return size < 1; })) {
      getOperation().emitError()
          << "was given a thread-block extent below one, which no loop can "
             "be cut to";
      return signalPassFailure();
    }
    if (model.warpSize < 1 || model.maxThreadsPerBlock < 1 ||
        model.preferredThreadsPerBlock < 1) {
      getOperation().emitError()
          << "was given a thread budget below one, which leaves no block to "
             "choose";
      return signalPassFailure();
    }

    SmallVector<scf::ParallelOp> loops;
    getInnermostParallelLoops(getOperation(), loops);

    SmallVector<Operation *> bounds;
    for (scf::ParallelOp loop : loops) {
      if (loop.getNumReductions() != 0)
        continue;
      SmallVector<int64_t> tile =
          given.empty() ? chooseBlockTile(loop, model) : given;
      if (tile.empty())
        continue;
      std::pair<scf::ParallelOp, scf::ParallelOp> tiled =
          scf::tileParallelLoop(loop, tile, /*noMinMaxBounds=*/false);
      collectBounds(tiled.first, bounds);
      collectBounds(tiled.second, bounds);
    }
    if (bounds.empty())
      return;

    GreedyRewriteConfig config;
    config.setRegionSimplificationLevel(GreedySimplifyRegionLevel::Disabled);
    (void)applyOpPatternsGreedily(bounds, FrozenRewritePatternSet(), config);
  }

  static void collectBounds(scf::ParallelOp loop,
                            SmallVectorImpl<Operation *> &bounds) {
    for (Value bound : llvm::concat<Value>(loop.getLowerBound(),
                                           loop.getUpperBound(),
                                           loop.getStep()))
      if (Operation *defining = bound.getDefiningOp())
        bounds.push_back(defining);
  }
};

}
}
