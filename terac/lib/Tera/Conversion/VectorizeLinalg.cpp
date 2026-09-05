//===- VectorizeLinalg.cpp - Vectorize where the maps are read --*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Tera/Conversion/Passes.h"

#include "Tera/Analysis/VectorTiles.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir::tera {
#define GEN_PASS_DEF_VECTORIZELINALG
#include "Tera/Conversion/Passes.h.inc"

namespace {
struct VectorizeLinalg : public impl::VectorizeLinalgBase<VectorizeLinalg> {
  using impl::VectorizeLinalgBase<VectorizeLinalg>::VectorizeLinalgBase;

  void runOnOperation() final {
    HostTargetModel model;
    model.maxVectorElements = maxVectorElements;

    SmallVector<linalg::LinalgOp> targets;
    getOperation().walk([&](linalg::LinalgOp op) {
      if (linalg::hasVectorizationImpl(op))
        targets.push_back(op);
    });

    IRRewriter rewriter(&getContext());
    for (linalg::LinalgOp op : targets) {
      if (!fitsOneVector(op, model))
        continue;
      if (failed(linalg::vectorizeOpPrecondition(op)))
        continue;

      rewriter.setInsertionPoint(op);
      FailureOr<linalg::VectorizationResult> vectorized =
          linalg::vectorize(rewriter, op);
      if (failed(vectorized))
        continue;
      rewriter.replaceOp(op, vectorized->replacements);
    }
  }
};

}
}
