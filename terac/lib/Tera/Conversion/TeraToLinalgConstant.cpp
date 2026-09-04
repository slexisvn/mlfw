//===- TeraToLinalgConstant.cpp - Lower value-producing ops -----*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Tera/IR/TeraOps.h"

#include "TeraToLinalgDetail.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;
using namespace mlir::tera;
using namespace mlir::tera::detail;

namespace {

struct ConstantOpLowering : public OpConversionPattern<ConstantOp> {
  using OpConversionPattern<ConstantOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ConstantOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<arith::ConstantOp>(
        op, cast<TypedAttr>(op.getValue()));
    return success();
  }
};

/// A counting tensor is a `linalg.generic` over its own result with an empty
/// input list: every element is a function of the loop index alone.
struct IotaOpLowering : public OpConversionPattern<IotaOp> {
  using OpConversionPattern<IotaOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(IotaOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto resultType = cast<RankedTensorType>(op.getType());
    int64_t rank = resultType.getRank();
    int64_t dimension = op.getIotaDimension();
    Type elementType = resultType.getElementType();

    AffineMap identity =
        AffineMap::getMultiDimIdentityMap(rank, rewriter.getContext());
    SmallVector<utils::IteratorType> iterators(rank,
                                               utils::IteratorType::parallel);

    SmallVector<Value> sizes;
    for (Value extent : adaptor.getSizes()) {
      Value scalar =
          tensor::ExtractOp::create(rewriter, loc, extent, ValueRange{});
      sizes.push_back(arith::IndexCastOp::create(
          rewriter, loc, rewriter.getIndexType(), scalar));
    }

    rewriter.replaceOpWithNewOp<linalg::GenericOp>(
        op, TypeRange{resultType}, ValueRange{},
        ValueRange{emptyTensor(rewriter, loc, resultType, sizes)},
        ArrayRef<AffineMap>{identity}, iterators,
        [&](OpBuilder &builder, Location bodyLoc, ValueRange) {
          Value index = linalg::IndexOp::create(builder, bodyLoc, dimension);
          Value wide = arith::IndexCastOp::create(
              builder, bodyLoc, builder.getI64Type(), index);
          linalg::YieldOp::create(
              builder, bodyLoc,
              convertScalarToDtype(builder, bodyLoc, wide, elementType,
                                   /*isUnsignedCast=*/false));
        });
    return success();
  }
};

} // namespace

void mlir::tera::detail::populateConstantPatterns(RewritePatternSet &patterns) {
  patterns.add<ConstantOpLowering, IotaOpLowering>(patterns.getContext());
}
