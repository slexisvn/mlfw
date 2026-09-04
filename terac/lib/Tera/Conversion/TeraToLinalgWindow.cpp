//===- TeraToLinalgWindow.cpp - Lower conv and pool -------------*- C++ -*-===//
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
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;
using namespace mlir::tera;
using namespace mlir::tera::detail;

namespace {
constexpr int64_t kLeadingAxes = 2;

Value bordered(OpBuilder &builder, Location loc, Value operand,
               ArrayRef<int64_t> low, ArrayRef<int64_t> high, TypedAttr fill) {
  auto operandType = cast<RankedTensorType>(operand.getType());
  int64_t rank = operandType.getRank();
  if (llvm::all_of(low, [](int64_t pad) { return pad == 0; }) &&
      llvm::all_of(high, [](int64_t pad) { return pad == 0; }))
    return operand;

  SmallVector<int64_t> shape(operandType.getShape());
  SmallVector<int64_t> before(rank, 0);
  SmallVector<Value> extents;
  for (int64_t axis = 0; axis < rank; ++axis) {
    int64_t added = axis < kLeadingAxes
                        ? 0
                        : low[axis - kLeadingAxes] + high[axis - kLeadingAxes];
    if (axis >= kLeadingAxes)
      before[axis] = low[axis - kLeadingAxes];
    if (ShapedType::isDynamic(shape[axis])) {
      Value width = tensor::DimOp::create(builder, loc, operand, axis);
      Value border = arith::ConstantIndexOp::create(builder, loc, added);
      extents.push_back(arith::AddIOp::create(builder, loc, width, border));
      continue;
    }
    shape[axis] += added;
  }

  Value scalar = arith::ConstantOp::create(builder, loc, fill);
  return spreadInto(builder, loc,
                    RankedTensorType::get(shape, operandType.getElementType()),
                    operand, before, SmallVector<int64_t>(rank, 1), scalar,
                    extents);
}

AffineExpr windowRead(MLIRContext *context, int64_t windowDim,
                      int64_t positionDim, int64_t stride, int64_t dilation) {
  return getAffineDimExpr(windowDim, context) * stride +
         getAffineDimExpr(positionDim, context) * dilation;
}

SmallVector<Value> destinationExtents(OpBuilder &builder, Location loc,
                                      RankedTensorType type, Value source,
                                      function_ref<int64_t(int64_t)> from) {
  return dynamicExtents(builder, loc, type, [&](int64_t axis) {
    return std::pair<Value, int64_t>{source, from(axis)};
  });
}

struct ConvOpLowering : public OpConversionPattern<ConvOp> {
  using OpConversionPattern<ConvOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ConvOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    MLIRContext *context = rewriter.getContext();
    auto resultType = cast<RankedTensorType>(op.getType());
    int64_t spatialRank = op.getSpatialRank();
    int64_t groups = op.getGroups();
    bool split = groups > 1;
    ArrayRef<int64_t> strides = op.getStrides();
    ArrayRef<int64_t> dilation = op.getDilation();

    Value input = bordered(rewriter, loc, adaptor.getInput(),
                           op.getPaddingLow(), op.getPaddingHigh(),
                           zeroAttr(resultType.getElementType()));

    int64_t batchDim = 0;
    int64_t groupDim = 1;
    int64_t outChannelDim = split ? 2 : 1;
    int64_t firstWindow = outChannelDim + 1;
    int64_t inChannelDim = firstWindow + spatialRank;
    int64_t firstPosition = inChannelDim + 1;
    int64_t loops = firstPosition + spatialRank;

    auto dim = [&](int64_t position) {
      return getAffineDimExpr(position, context);
    };
    SmallVector<AffineExpr> reads{dim(batchDim)};
    SmallVector<AffineExpr> taps;
    SmallVector<AffineExpr> writes{dim(batchDim)};
    if (split) {
      reads.push_back(dim(groupDim));
      taps.push_back(dim(groupDim));
      writes.push_back(dim(groupDim));
    }
    reads.push_back(dim(inChannelDim));
    taps.push_back(dim(outChannelDim));
    taps.push_back(dim(inChannelDim));
    writes.push_back(dim(outChannelDim));
    for (int64_t axis = 0; axis < spatialRank; ++axis) {
      reads.push_back(windowRead(context, firstWindow + axis,
                                 firstPosition + axis, strides[axis],
                                 dilation[axis]));
      taps.push_back(dim(firstPosition + axis));
      writes.push_back(dim(firstWindow + axis));
    }

    SmallVector<utils::IteratorType> iterators(loops,
                                               utils::IteratorType::parallel);
    iterators[inChannelDim] = utils::IteratorType::reduction;
    for (int64_t axis = 0; axis < spatialRank; ++axis)
      iterators[firstPosition + axis] = utils::IteratorType::reduction;

    Value kernel = adaptor.getKernel();
    RankedTensorType destinationType = resultType;
    if (split) {
      input = splitChannels(rewriter, loc, input, 1, groups);
      kernel = splitChannels(rewriter, loc, kernel, 0, groups);
      destinationType = groupedType(resultType, 1, groups);
    }

    Value destination =
        filledTensor(rewriter, loc, destinationType,
                     zeroAttr(resultType.getElementType()),
                     extentsLike(rewriter, loc, destinationType, input));

    SmallVector<AffineMap> maps = {AffineMap::get(loops, 0, reads, context),
                                   AffineMap::get(loops, 0, taps, context),
                                   AffineMap::get(loops, 0, writes, context)};
    Value convolved =
        linalg::GenericOp::create(
            rewriter, loc, TypeRange{destinationType},
            ValueRange{input, kernel}, ValueRange{destination}, maps, iterators,
            [](OpBuilder &builder, Location bodyLoc, ValueRange args) {
              Value product =
                  arith::MulFOp::create(builder, bodyLoc, args[0], args[1]);
              linalg::YieldOp::create(
                  builder, bodyLoc,
                  arith::AddFOp::create(builder, bodyLoc, args[2], product)
                      .getResult());
            })
            .getResult(0);

    if (!split) {
      rewriter.replaceOp(op, convolved);
      return success();
    }
    rewriter.replaceOpWithNewOp<tensor::CollapseShapeOp>(
        op, resultType, convolved, groupReassociation(resultType.getRank(), 1));
    return success();
  }

  static Value splitChannels(OpBuilder &builder, Location loc, Value value,
                             int64_t axis, int64_t groups) {
    auto type = cast<RankedTensorType>(value.getType());
    return tensor::ExpandShapeOp::create(
        builder, loc, groupedType(type, axis, groups), value,
        groupReassociation(type.getRank(), axis));
  }

  static RankedTensorType groupedType(RankedTensorType type, int64_t axis,
                                      int64_t groups) {
    SmallVector<int64_t> shape(type.getShape());
    int64_t extent = shape[axis];
    shape[axis] = ShapedType::isDynamic(extent) ? extent : extent / groups;
    shape.insert(shape.begin() + axis, groups);
    return RankedTensorType::get(shape, type.getElementType());
  }

  static SmallVector<ReassociationIndices>
  groupReassociation(int64_t rank, int64_t axis) {
    SmallVector<ReassociationIndices> grouping;
    for (int64_t at = 0; at < rank; ++at) {
      int64_t wide = at + (at > axis ? 1 : 0);
      if (at == axis) {
        grouping.push_back({wide, wide + 1});
        continue;
      }
      grouping.push_back({wide});
    }
    return grouping;
  }
};

struct Pool2dOpLowering : public OpConversionPattern<Pool2dOp> {
  using OpConversionPattern<Pool2dOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(Pool2dOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    MLIRContext *context = rewriter.getContext();
    auto resultType = cast<RankedTensorType>(op.getType());
    Type elementType = resultType.getElementType();
    ArrayRef<int64_t> window = op.getKernelSize();
    ArrayRef<int64_t> strides = op.getStrides();
    bool averaging = op.getKind() == PoolKind::Average;

    Value operand =
        bordered(rewriter, loc, adaptor.getOperand(), op.getPaddingLow(),
                 op.getPaddingHigh(), zeroAttr(elementType));

    int64_t batchDim = 0, channelDim = 1, firstWindow = 2;
    int64_t firstPosition = firstWindow + 2;
    int64_t loops = firstPosition + 2;

    SmallVector<AffineExpr> reads{getAffineDimExpr(batchDim, context),
                                  getAffineDimExpr(channelDim, context)};
    SmallVector<AffineExpr> writes{getAffineDimExpr(batchDim, context),
                                   getAffineDimExpr(channelDim, context)};
    SmallVector<AffineExpr> positions;
    for (int64_t axis = 0; axis < 2; ++axis) {
      reads.push_back(windowRead(context, firstWindow + axis,
                                 firstPosition + axis, strides[axis],
                                 /*dilation=*/1));
      writes.push_back(getAffineDimExpr(firstWindow + axis, context));
      positions.push_back(getAffineDimExpr(firstPosition + axis, context));
    }

    SmallVector<utils::IteratorType> iterators(loops,
                                               utils::IteratorType::parallel);
    iterators[firstPosition] = utils::IteratorType::reduction;
    iterators[firstPosition + 1] = utils::IteratorType::reduction;

    TypedAttr identity =
        averaging ? zeroAttr(elementType)
                  : TypedAttr(FloatAttr::get(
                        elementType,
                        APFloat::getInf(
                            cast<FloatType>(elementType).getFloatSemantics(),
                            /*Negative=*/true)));
    SmallVector<Value> extents = destinationExtents(
        rewriter, loc, resultType, operand, [](int64_t axis) { return axis; });
    Value destination =
        filledTensor(rewriter, loc, resultType, identity, extents);

    Value shape = emptyTensor(
        rewriter, loc,
        RankedTensorType::get(window, rewriter.getIntegerType(1)));

    SmallVector<AffineMap> maps = {AffineMap::get(loops, 0, reads, context),
                                   AffineMap::get(loops, 0, positions, context),
                                   AffineMap::get(loops, 0, writes, context)};
    Value pooled =
        linalg::GenericOp::create(
            rewriter, loc, TypeRange{resultType}, ValueRange{operand, shape},
            ValueRange{destination}, maps, iterators,
            [&](OpBuilder &builder, Location bodyLoc, ValueRange args) {
              Value combined =
                  averaging
                      ? arith::AddFOp::create(builder, bodyLoc, args[2],
                                              args[0])
                            .getResult()
                      : arith::MaximumFOp::create(builder, bodyLoc, args[2],
                                                  args[0])
                            .getResult();
              linalg::YieldOp::create(builder, bodyLoc, combined);
            })
            .getResult(0);

    if (!averaging) {
      rewriter.replaceOp(op, pooled);
      return success();
    }

    Value share = arith::ConstantOp::create(
        rewriter, loc,
        FloatAttr::get(elementType, 1.0 / (window[0] * window[1])));
    rewriter.replaceOpWithNewOp<linalg::GenericOp>(
        op, TypeRange{resultType}, ValueRange{pooled},
        ValueRange{emptyTensor(rewriter, loc, resultType, extents)},
        SmallVector<AffineMap>(
            2, AffineMap::getMultiDimIdentityMap(resultType.getRank(),
                                                 context)),
        SmallVector<utils::IteratorType>(resultType.getRank(),
                                         utils::IteratorType::parallel),
        [&](OpBuilder &builder, Location bodyLoc, ValueRange args) {
          linalg::YieldOp::create(
              builder, bodyLoc,
              arith::MulFOp::create(builder, bodyLoc, args[0], share)
                  .getResult());
        });
    return success();
  }
};

}

void mlir::tera::detail::populateWindowPatterns(RewritePatternSet &patterns) {
  patterns.add<ConvOpLowering, Pool2dOpLowering>(patterns.getContext());
}
