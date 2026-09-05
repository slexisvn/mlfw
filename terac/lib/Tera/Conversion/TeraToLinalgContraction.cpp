//===- TeraToLinalgContraction.cpp - Lower dot and reduce -------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Tera/IR/TeraOps.h"

#include "TeraToLinalgDetail.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/APFloat.h"

using namespace mlir;
using namespace mlir::tera;
using namespace mlir::tera::detail;

namespace {
struct ContractionSpace {
  SmallVector<AffineExpr> lhs;
  SmallVector<AffineExpr> rhs;
  SmallVector<AffineExpr> result;
  SmallVector<utils::IteratorType> iterators;
  unsigned count = 0;
};

ContractionSpace buildContractionSpace(DotOp op, MLIRContext *context) {
  auto lhsType = cast<RankedTensorType>(op.getLhs().getType());
  auto rhsType = cast<RankedTensorType>(op.getRhs().getType());

  ContractionSpace space;
  space.lhs.resize(lhsType.getRank());
  space.rhs.resize(rhsType.getRank());

  auto claim = [&](utils::IteratorType kind) {
    space.iterators.push_back(kind);
    return getAffineDimExpr(space.count++, context);
  };

  for (auto [lhsAxis, rhsAxis] :
       llvm::zip_equal(op.getLhsBatch(), op.getRhsBatch())) {
    AffineExpr iterator = claim(utils::IteratorType::parallel);
    space.lhs[lhsAxis] = iterator;
    space.rhs[rhsAxis] = iterator;
    space.result.push_back(iterator);
  }
  for (int64_t axis : op.getLhsFreeAxes()) {
    AffineExpr iterator = claim(utils::IteratorType::parallel);
    space.lhs[axis] = iterator;
    space.result.push_back(iterator);
  }
  for (int64_t axis : op.getRhsFreeAxes()) {
    AffineExpr iterator = claim(utils::IteratorType::parallel);
    space.rhs[axis] = iterator;
    space.result.push_back(iterator);
  }
  for (auto [lhsAxis, rhsAxis] :
       llvm::zip_equal(op.getLhsContracting(), op.getRhsContracting())) {
    AffineExpr iterator = claim(utils::IteratorType::reduction);
    space.lhs[lhsAxis] = iterator;
    space.rhs[rhsAxis] = iterator;
  }
  return space;
}

struct DotOpLowering : public OpConversionPattern<DotOp> {
  using OpConversionPattern<DotOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(DotOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    MLIRContext *context = rewriter.getContext();
    auto resultType = cast<RankedTensorType>(op.getType());
    Type elementType = resultType.getElementType();

    ContractionSpace space = buildContractionSpace(op, context);
    SmallVector<AffineMap> maps = {
        AffineMap::get(space.count, 0, space.lhs, context),
        AffineMap::get(space.count, 0, space.rhs, context),
        AffineMap::get(space.count, 0, space.result, context)};

    SmallVector<Value> sizes = resultExtents(rewriter, loc, op);

    Value accumulator = filledTensor(rewriter, loc, resultType,
                                     zeroAttr(elementType), sizes);

    rewriter.replaceOpWithNewOp<linalg::GenericOp>(
        op, TypeRange{resultType},
        ValueRange{adaptor.getLhs(), adaptor.getRhs()},
        ValueRange{accumulator}, maps, space.iterators,
        [](OpBuilder &builder, Location bodyLoc, ValueRange args) {
          Value product =
              isa<FloatType>(args[0].getType())
                  ? arith::MulFOp::create(builder, bodyLoc, args[0], args[1])
                        .getResult()
                  : arith::MulIOp::create(builder, bodyLoc, args[0], args[1])
                        .getResult();
          Value sum =
              isa<FloatType>(args[2].getType())
                  ? arith::AddFOp::create(builder, bodyLoc, args[2], product)
                        .getResult()
                  : arith::AddIOp::create(builder, bodyLoc, args[2], product)
                        .getResult();
          linalg::YieldOp::create(builder, bodyLoc, sum);
        });
    return success();
  }
};

TypedAttr identityAttr(ReduceKind kind, Type elementType) {
  auto floatType = dyn_cast<FloatType>(elementType);
  const llvm::fltSemantics *semantics =
      floatType ? &floatType.getFloatSemantics() : nullptr;
  unsigned width = elementType.getIntOrFloatBitWidth();

  switch (kind) {
  case ReduceKind::Sum:
  case ReduceKind::Mean:
    return floatType ? TypedAttr(FloatAttr::get(
                           floatType, APFloat::getZero(*semantics)))
                     : TypedAttr(IntegerAttr::get(elementType, 0));
  case ReduceKind::Product:
    return floatType ? TypedAttr(FloatAttr::get(floatType, 1.0))
                     : TypedAttr(IntegerAttr::get(elementType, 1));
  case ReduceKind::Maximum:
    return floatType
               ? TypedAttr(FloatAttr::get(
                     floatType, APFloat::getInf(*semantics, /*Negative=*/true)))
               : TypedAttr(IntegerAttr::get(
                     elementType, APInt::getSignedMinValue(width)));
  case ReduceKind::Minimum:
    return floatType
               ? TypedAttr(FloatAttr::get(
                     floatType,
                     APFloat::getInf(*semantics, /*Negative=*/false)))
               : TypedAttr(IntegerAttr::get(
                     elementType, APInt::getSignedMaxValue(width)));
  }
  llvm_unreachable("unhandled reduce kind");
}

Value combine(OpBuilder &builder, Location loc, ReduceKind kind,
              Value accumulator, Value element) {
  bool isFloat = isa<FloatType>(element.getType());
  switch (kind) {
  case ReduceKind::Sum:
  case ReduceKind::Mean:
    return isFloat ? arith::AddFOp::create(builder, loc, accumulator, element)
                         .getResult()
                   : arith::AddIOp::create(builder, loc, accumulator, element)
                         .getResult();
  case ReduceKind::Product:
    return isFloat ? arith::MulFOp::create(builder, loc, accumulator, element)
                         .getResult()
                   : arith::MulIOp::create(builder, loc, accumulator, element)
                         .getResult();
  case ReduceKind::Maximum:
    return isFloat
               ? arith::MaximumFOp::create(builder, loc, accumulator, element)
                     .getResult()
               : arith::MaxSIOp::create(builder, loc, accumulator, element)
                     .getResult();
  case ReduceKind::Minimum:
    return isFloat
               ? arith::MinimumFOp::create(builder, loc, accumulator, element)
                     .getResult()
               : arith::MinSIOp::create(builder, loc, accumulator, element)
                     .getResult();
  }
  llvm_unreachable("unhandled reduce kind");
}

struct ReduceOpLowering : public OpConversionPattern<ReduceOp> {
  using OpConversionPattern<ReduceOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ReduceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto resultType = cast<RankedTensorType>(op.getType());
    ReduceKind kind = op.getKind();

    SmallVector<int64_t> dimensions(op.getDimensions());
    llvm::sort(dimensions);

    auto operandType = cast<RankedTensorType>(adaptor.getOperand().getType());
    SmallVector<Value> sizes = resultExtents(rewriter, loc, op);
    Value accumulator =
        filledTensor(rewriter, loc, resultType,
                     identityAttr(kind, resultType.getElementType()), sizes);

    auto reduce = linalg::ReduceOp::create(
        rewriter, loc, ValueRange{adaptor.getOperand()},
        ValueRange{accumulator}, dimensions,
        [kind](OpBuilder &builder, Location bodyLoc, ValueRange args) {
          linalg::YieldOp::create(
              builder, bodyLoc,
              combine(builder, bodyLoc, kind, args[1], args[0]));
        });

    if (kind != ReduceKind::Mean) {
      rewriter.replaceOp(op, reduce.getResults());
      return success();
    }

    Value elements;
    for (int64_t axis : dimensions) {
      Value extent =
          ShapedType::isDynamic(operandType.getDimSize(axis))
              ? tensor::DimOp::create(rewriter, loc, adaptor.getOperand(), axis)
                    .getResult()
              : arith::ConstantIndexOp::create(rewriter, loc,
                                               operandType.getDimSize(axis))
                    .getResult();
      elements = elements
                     ? arith::MulIOp::create(rewriter, loc, elements, extent)
                           .getResult()
                     : extent;
    }
    Value asInteger = arith::IndexCastOp::create(
        rewriter, loc, rewriter.getI64Type(), elements);
    Value divisor = arith::SIToFPOp::create(
        rewriter, loc, resultType.getElementType(), asInteger);

    rewriter.replaceOpWithNewOp<linalg::GenericOp>(
        op, TypeRange{resultType}, ValueRange{reduce.getResult(0)},
        ValueRange{emptyTensor(rewriter, loc, resultType, sizes)},
        SmallVector<AffineMap>(
            2, rewriter.getMultiDimIdentityMap(resultType.getRank())),
        SmallVector<utils::IteratorType>(resultType.getRank(),
                                        utils::IteratorType::parallel),
        [divisor](OpBuilder &builder, Location bodyLoc, ValueRange args) {
          linalg::YieldOp::create(
              builder, bodyLoc,
              arith::DivFOp::create(builder, bodyLoc, args[0], divisor)
                  .getResult());
        });
    return success();
  }
};

}

void mlir::tera::detail::populateContractionPatterns(
    RewritePatternSet &patterns) {
  patterns.add<DotOpLowering, ReduceOpLowering>(patterns.getContext());
}
