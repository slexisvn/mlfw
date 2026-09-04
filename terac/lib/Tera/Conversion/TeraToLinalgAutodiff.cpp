//===- TeraToLinalgAutodiff.cpp - Lower autodiff markers --------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Tera/IR/TeraOps.h"

#include "TeraToLinalgDetail.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;
using namespace mlir::tera;

namespace {
struct StopGradientOpLowering : public OpConversionPattern<StopGradientOp> {
  using OpConversionPattern<StopGradientOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(StopGradientOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOp(op, adaptor.getOperand());
    return success();
  }
};

}

void mlir::tera::detail::populateAutodiffPatterns(RewritePatternSet &patterns) {
  patterns.add<StopGradientOpLowering>(patterns.getContext());
}
