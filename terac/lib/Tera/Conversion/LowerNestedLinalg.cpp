//===- LowerNestedLinalg.cpp - scf loops where affine cannot go -*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The CPU pipeline lowers linalg to affine loops because the vectorizer reads
// affine maps, and that is where the whole of its measured speedup comes from.
// Affine will not take every loop, though: a bound has to be a value it can
// read as a dim or a symbol, and `mlir::affine::isValidDim` allows a `dim` op
// only when the value it measures sits at the top level of the enclosing affine
// scope. A function is such a scope; an `scf.for` is not.
//
// So a scan over a dynamic batch reaches the affine conversion with a
// `linalg.fill` whose destination was allocated inside the loop from
// `memref.dim` of a loop-carried value, and the conversion fails on the whole
// function rather than on that op. This lowers those ops -- and only those --
// to `scf` loops beforehand.
//
//===----------------------------------------------------------------------===//

#include "Tera/Conversion/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir::tera {
#define GEN_PASS_DEF_LOWERNESTEDLINALG
#include "Tera/Conversion/Passes.h.inc"

namespace {

bool atFunctionTopLevel(Operation *op, func::FuncOp function) {
  return op->getParentOp() == function.getOperation();
}

bool touchesDynamicShape(linalg::LinalgOp op) {
  return llvm::any_of(op->getOperands(), [](Value operand) {
    auto shaped = dyn_cast<ShapedType>(operand.getType());
    return shaped && !shaped.hasStaticShape();
  });
}

struct LowerNestedLinalg
    : public impl::LowerNestedLinalgBase<LowerNestedLinalg> {
  using impl::LowerNestedLinalgBase<
      LowerNestedLinalg>::LowerNestedLinalgBase;

  void runOnOperation() final {
    func::FuncOp function = getOperation();
    SmallVector<linalg::LinalgOp> nested;
    function.walk([&](linalg::LinalgOp op) {
      if (!atFunctionTopLevel(op, function) && touchesDynamicShape(op))
        nested.push_back(op);
    });

    IRRewriter rewriter(&getContext());
    for (linalg::LinalgOp op : nested) {
      rewriter.setInsertionPoint(op);
      if (failed(linalg::linalgOpToLoops(rewriter, op))) {
        op->emitError() << "could not be lowered to scf loops";
        return signalPassFailure();
      }
      rewriter.eraseOp(op);
    }
  }
};

} // namespace
} // namespace mlir::tera
