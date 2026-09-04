//===- VectorizeAffine.cpp - Vectorize where analysis is sound --*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Tera/Conversion/Passes.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/Pass/PassManager.h"

namespace mlir::tera {
#define GEN_PASS_DEF_VECTORIZEAFFINE
#include "Tera/Conversion/Passes.h.inc"

namespace {

bool movesOneAtATime(AffineExpr expr) {
  switch (expr.getKind()) {
  case AffineExprKind::Constant:
  case AffineExprKind::DimId:
  case AffineExprKind::SymbolId:
    return true;
  case AffineExprKind::Add: {
    auto sum = cast<AffineBinaryOpExpr>(expr);
    return movesOneAtATime(sum.getLHS()) && movesOneAtATime(sum.getRHS());
  }
  default:
    return false;
  }
}

bool everyAccessIsUnitStride(func::FuncOp func) {
  auto reads = [](AffineMap map) {
    return llvm::all_of(map.getResults(), movesOneAtATime);
  };
  WalkResult walk = func.walk([&](Operation *op) {
    if (auto read = dyn_cast<affine::AffineReadOpInterface>(op))
      return reads(read.getAffineMap()) ? WalkResult::advance()
                                        : WalkResult::interrupt();
    if (auto write = dyn_cast<affine::AffineWriteOpInterface>(op))
      return reads(write.getAffineMap()) ? WalkResult::advance()
                                         : WalkResult::interrupt();
    return WalkResult::advance();
  });
  return !walk.wasInterrupted();
}

struct VectorizeAffine : public impl::VectorizeAffineBase<VectorizeAffine> {
  using impl::VectorizeAffineBase<VectorizeAffine>::VectorizeAffineBase;

  void runOnOperation() final {
    if (!everyAccessIsUnitStride(getOperation()))
      return;

    affine::AffineVectorizeOptions options;
    options.vectorSizes = {vectorSize};
    OpPassManager vectorize(func::FuncOp::getOperationName());
    vectorize.addPass(affine::createAffineVectorize(options));
    if (failed(runPipeline(vectorize, getOperation())))
      signalPassFailure();
  }
};

} // namespace
} // namespace mlir::tera
