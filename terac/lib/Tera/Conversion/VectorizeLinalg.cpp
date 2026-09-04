//===- VectorizeLinalg.cpp - Vectorize where the maps are read --*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Tera/Conversion/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/IndexingMapOpInterface.h"

namespace mlir::tera {
#define GEN_PASS_DEF_VECTORIZELINALG
#include "Tera/Conversion/Passes.h.inc"

namespace {
std::optional<int64_t> vectorElements(linalg::LinalgOp op) {
  auto indexed = dyn_cast<IndexingMapOpInterface>(op.getOperation());
  if (!indexed)
    return std::nullopt;
  int64_t elements = 1;
  for (int64_t extent : indexed.getStaticLoopRanges()) {
    if (ShapedType::isDynamic(extent))
      return std::nullopt;
    elements *= extent;
  }
  return elements;
}

struct VectorizeLinalg : public impl::VectorizeLinalgBase<VectorizeLinalg> {
  using impl::VectorizeLinalgBase<VectorizeLinalg>::VectorizeLinalgBase;

  void runOnOperation() final {
    SmallVector<linalg::LinalgOp> targets;
    getOperation().walk([&](linalg::LinalgOp op) {
      if (linalg::hasVectorizationImpl(op))
        targets.push_back(op);
    });

    IRRewriter rewriter(&getContext());
    for (linalg::LinalgOp op : targets) {
      std::optional<int64_t> elements = vectorElements(op);
      if (!elements || *elements > maxVectorElements)
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
