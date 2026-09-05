//===- TileAndFuse.cpp - Cut linalg into tiles a register holds -*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Tera/Conversion/Passes.h"

#include "Tera/Analysis/VectorTiles.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/TilingInterface.h"
#include "llvm/ADT/DenseSet.h"

namespace mlir::tera {
#define GEN_PASS_DEF_TILEANDFUSE
#include "Tera/Conversion/Passes.h.inc"

namespace {
std::optional<scf::SCFTileAndFuseOptions::ControlFnResult>
fuseSingleUse(tensor::ExtractSliceOp, OpResult producer, bool destination) {
  if (!destination && !producer.hasOneUse())
    return std::nullopt;
  return scf::SCFTileAndFuseOptions::ControlFnResult{};
}

struct TileAndFuse : public impl::TileAndFuseBase<TileAndFuse> {
  using impl::TileAndFuseBase<TileAndFuse>::TileAndFuseBase;

  void runOnOperation() final {
    HostTargetModel model;
    model.vectorLanes = vectorWidth;
    tileParallelAndFuse(model);
    tileReductions();
  }

  void tileParallelAndFuse(const HostTargetModel &model) {
    SmallVector<linalg::LinalgOp> roots;
    getOperation().walk([&](linalg::LinalgOp op) { roots.push_back(op); });

    IRRewriter rewriter(&getContext());
    DenseSet<Operation *> handled;

    for (linalg::LinalgOp op : llvm::reverse(roots)) {
      if (handled.contains(op.getOperation()))
        continue;
      SmallVector<int64_t> sizes = chooseVectorTile(op, model);
      if (sizes.empty())
        continue;

      scf::SCFTileAndFuseOptions options;
      options.setTilingOptions(scf::SCFTilingOptions().setTileSizes(
          getAsIndexOpFoldResult(&getContext(), sizes)));
      options.setFusionControlFn(fuseSingleUse);

      rewriter.setInsertionPoint(op);
      FailureOr<scf::SCFTileAndFuseResult> tiled =
          scf::tileConsumerAndFuseProducersUsingSCF(
              rewriter, cast<TilingInterface>(op.getOperation()), options);
      if (failed(tiled))
        continue;

      SmallVector<Operation *> replaced{op.getOperation()};
      llvm::append_range(replaced, tiled->fusedProducers);
      for (Operation *original : replaced) {
        handled.insert(original);
        for (OpResult result : original->getResults())
          if (Value replacement = tiled->replacements.lookup(result))
            rewriter.replaceAllUsesWith(result, replacement);
        if (original->use_empty())
          rewriter.eraseOp(original);
      }
    }
  }

  void tileReductions() {
    SmallVector<linalg::LinalgOp> targets;
    getOperation().walk([&](linalg::LinalgOp op) { targets.push_back(op); });

    IRRewriter rewriter(&getContext());
    for (linalg::LinalgOp op : targets) {
      SmallVector<int64_t> sizes = chooseReductionTile(op);
      if (sizes.empty())
        continue;

      scf::SCFTilingOptions options;
      options.setTileSizes(getAsIndexOpFoldResult(&getContext(), sizes));

      rewriter.setInsertionPoint(op);
      FailureOr<scf::SCFTilingResult> tiled = scf::tileUsingSCF(
          rewriter, cast<TilingInterface>(op.getOperation()), options);
      if (failed(tiled))
        continue;
      rewriter.replaceOp(op, tiled->replacements);
    }
  }
};

}
}
