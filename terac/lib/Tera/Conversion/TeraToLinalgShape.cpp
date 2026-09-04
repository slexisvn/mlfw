//===- TeraToLinalgShape.cpp - Lower shape and layout ops -------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Tera/IR/TeraOps.h"

#include "TeraToLinalgDetail.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/ReshapeOpsUtils.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;
using namespace mlir::tera;
using namespace mlir::tera::detail;

namespace {
Value extentAsIndex(OpBuilder &builder, Location loc, Value extent) {
  Value scalar = tensor::ExtractOp::create(builder, loc, extent, ValueRange{});
  return arith::IndexCastOp::create(builder, loc, builder.getIndexType(),
                                    scalar);
}

struct DimOpLowering : public OpConversionPattern<DimOp> {
  using OpConversionPattern<DimOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(DimOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value extent = tensor::DimOp::create(rewriter, loc, adaptor.getOperand(),
                                         op.getDimension());
    Value widened = arith::IndexCastOp::create(
        rewriter, loc, rewriter.getIntegerType(64), extent);
    rewriter.replaceOpWithNewOp<tensor::FromElementsOp>(op, op.getType(),
                                                        ValueRange{widened});
    return success();
  }
};

struct BroadcastInDimOpLowering
    : public OpConversionPattern<BroadcastInDimOp> {
  using OpConversionPattern<BroadcastInDimOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(BroadcastInDimOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    MLIRContext *context = rewriter.getContext();
    auto operandType = cast<RankedTensorType>(op.getOperand().getType());
    auto resultType = cast<RankedTensorType>(op.getType());
    int64_t rank = resultType.getRank();

    SmallVector<AffineExpr> reads;
    for (auto [axis, target] : llvm::enumerate(op.getBroadcastDimensions())) {
      bool stretched = operandType.getDimSize(axis) == 1 &&
                       resultType.getDimSize(target) != 1;
      reads.push_back(stretched ? getAffineConstantExpr(0, context)
                                : getAffineDimExpr(target, context));
    }

    SmallVector<Value> sizes;
    for (Value extent : adaptor.getSizes())
      sizes.push_back(extentAsIndex(rewriter, loc, extent));

    SmallVector<AffineMap> maps = {
        AffineMap::get(rank, 0, reads, context),
        AffineMap::getMultiDimIdentityMap(rank, context)};
    SmallVector<utils::IteratorType> iterators(rank,
                                               utils::IteratorType::parallel);

    rewriter.replaceOpWithNewOp<linalg::GenericOp>(
        op, TypeRange{resultType}, ValueRange{adaptor.getOperand()},
        ValueRange{emptyTensor(rewriter, loc, resultType, sizes)}, maps,
        iterators,
        [](OpBuilder &builder, Location bodyLoc, ValueRange args) {
          linalg::YieldOp::create(builder, bodyLoc, args[0]);
        });
    return success();
  }
};

struct TransposeOpLowering : public OpConversionPattern<TransposeOp> {
  using OpConversionPattern<TransposeOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(TransposeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto resultType = cast<RankedTensorType>(op.getType());
    ArrayRef<int64_t> permutation = op.getPermutation();
    SmallVector<Value> sizes = dynamicExtents(
        rewriter, op.getLoc(), resultType, [&](int64_t axis) {
          return std::pair<Value, int64_t>{adaptor.getOperand(),
                                          permutation[axis]};
        });
    rewriter.replaceOpWithNewOp<linalg::TransposeOp>(
        op, adaptor.getOperand(),
        emptyTensor(rewriter, op.getLoc(), resultType, sizes), permutation);
    return success();
  }
};

struct ReshapeOpLowering : public OpConversionPattern<ReshapeOp> {
  using OpConversionPattern<ReshapeOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ReshapeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto operandType = cast<RankedTensorType>(adaptor.getOperand().getType());
    auto resultType = cast<RankedTensorType>(op.getType());

    SmallVector<OpFoldResult> extents;
    size_t at = 0;
    for (int64_t extent : resultType.getShape())
      extents.push_back(
          ShapedType::isDynamic(extent)
              ? OpFoldResult(
                    extentAsIndex(rewriter, loc, adaptor.getSizes()[at++]))
              : OpFoldResult(rewriter.getIndexAttr(extent)));

    if (operandType.getRank() != resultType.getRank()) {
      bool collapsing = resultType.getRank() < operandType.getRank();
      RankedTensorType wide = collapsing ? operandType : resultType;
      RankedTensorType narrow = collapsing ? resultType : operandType;
      if (std::optional<SmallVector<ReassociationIndices>> grouping =
              getReassociationIndicesForReshape(wide, narrow)) {
        if (collapsing) {
          rewriter.replaceOpWithNewOp<tensor::CollapseShapeOp>(
              op, resultType, adaptor.getOperand(), *grouping);
        } else {
          rewriter.replaceOpWithNewOp<tensor::ExpandShapeOp>(
              op, resultType, adaptor.getOperand(), *grouping, extents);
        }
        return success();
      }
    }

    auto shapeType = RankedTensorType::get({resultType.getRank()},
                                           rewriter.getIndexType());
    Value shape;
    if (resultType.hasStaticShape()) {
      shape = arith::ConstantOp::create(
          rewriter, loc,
          DenseIntElementsAttr::get(shapeType, resultType.getShape()));
    } else {
      SmallVector<Value> values;
      for (OpFoldResult extent : extents)
        values.push_back(getValueOrCreateConstantIndexOp(rewriter, loc, extent));
      shape = tensor::FromElementsOp::create(rewriter, loc, shapeType, values);
    }

    rewriter.replaceOpWithNewOp<tensor::ReshapeOp>(op, resultType,
                                                   adaptor.getOperand(), shape);
    return success();
  }
};

struct SliceOpLowering : public OpConversionPattern<SliceOp> {
  using OpConversionPattern<SliceOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(SliceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto resultType = cast<RankedTensorType>(op.getType());
    SmallVector<OpFoldResult> offsets =
        getAsIndexOpFoldResult(rewriter.getContext(), op.getStartIndices());
    SmallVector<OpFoldResult> strides =
        getAsIndexOpFoldResult(rewriter.getContext(), op.getStrides());

    SmallVector<OpFoldResult> sizes =
        getAsIndexOpFoldResult(rewriter.getContext(), resultType.getShape());

    rewriter.replaceOpWithNewOp<tensor::ExtractSliceOp>(
        op, resultType, adaptor.getOperand(), offsets, sizes, strides);
    return success();
  }
};

struct ReverseOpLowering : public OpConversionPattern<ReverseOp> {
  using OpConversionPattern<ReverseOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ReverseOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    MLIRContext *context = rewriter.getContext();
    auto resultType = cast<RankedTensorType>(op.getType());
    int64_t rank = resultType.getRank();

    llvm::SmallBitVector reversed(rank);
    for (int64_t axis : op.getDimensions())
      reversed.set(axis);

    SmallVector<AffineExpr> reads;
    for (int64_t axis = 0; axis < rank; ++axis) {
      AffineExpr read = getAffineDimExpr(axis, context);
      if (reversed.test(axis))
        read = getAffineConstantExpr(resultType.getDimSize(axis) - 1, context) -
               read;
      reads.push_back(read);
    }

    SmallVector<AffineMap> maps = {
        AffineMap::get(rank, 0, reads, context),
        AffineMap::getMultiDimIdentityMap(rank, context)};
    SmallVector<utils::IteratorType> iterators(rank,
                                               utils::IteratorType::parallel);

    rewriter.replaceOpWithNewOp<linalg::GenericOp>(
        op, TypeRange{resultType}, ValueRange{adaptor.getOperand()},
        ValueRange{emptyTensor(
            rewriter, loc, resultType,
            extentsLike(rewriter, loc, resultType, adaptor.getOperand()))},
        maps, iterators,
        [](OpBuilder &builder, Location bodyLoc, ValueRange args) {
          linalg::YieldOp::create(builder, bodyLoc, args[0]);
        });
    return success();
  }
};

struct PadOpLowering : public OpConversionPattern<PadOp> {
  using OpConversionPattern<PadOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(PadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto resultType = cast<RankedTensorType>(op.getType());
    ArrayRef<int64_t> low = op.getLow();
    ArrayRef<int64_t> high = op.getHigh();
    SmallVector<int64_t> spacing = op.getSpacing();

    SmallVector<Value> extents;
    for (int64_t axis = 0; axis < resultType.getRank(); ++axis) {
      if (!ShapedType::isDynamic(resultType.getDimSize(axis)))
        continue;
      Value width =
          tensor::DimOp::create(rewriter, loc, adaptor.getOperand(), axis);
      Value step = arith::ConstantIndexOp::create(rewriter, loc, spacing[axis]);
      Value border = arith::ConstantIndexOp::create(
          rewriter, loc, low[axis] + high[axis] - spacing[axis] + 1);
      extents.push_back(arith::AddIOp::create(
          rewriter, loc, arith::MulIOp::create(rewriter, loc, width, step),
          border));
    }

    Value fill = tensor::ExtractOp::create(
        rewriter, loc, adaptor.getPaddingValue(), ValueRange{});
    rewriter.replaceOp(op, spreadInto(rewriter, loc, resultType,
                                      adaptor.getOperand(), low, spacing, fill,
                                      extents));
    return success();
  }
};

struct ConcatOpLowering : public OpConversionPattern<ConcatOp> {
  using OpConversionPattern<ConcatOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ConcatOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto resultType = cast<RankedTensorType>(op.getType());
    int64_t rank = resultType.getRank();
    int64_t dimension = op.getDimension();

    SmallVector<Value> extents;
    for (auto [axis, extent] : llvm::enumerate(resultType.getShape())) {
      if (!ShapedType::isDynamic(extent))
        continue;
      if (static_cast<int64_t>(axis) != dimension) {
        extents.push_back(tensor::DimOp::create(rewriter, loc,
                                                adaptor.getInputs().front(),
                                                axis));
        continue;
      }
      Value total;
      for (Value input : adaptor.getInputs()) {
        Value band = tensor::DimOp::create(rewriter, loc, input, axis);
        total = total ? arith::AddIOp::create(rewriter, loc, total, band)
                      : band;
      }
      extents.push_back(total);
    }

    Value destination = emptyTensor(rewriter, loc, resultType, extents);
    SmallVector<OpFoldResult> offsets(rank, rewriter.getIndexAttr(0));
    SmallVector<OpFoldResult> strides(rank, rewriter.getIndexAttr(1));

    OpFoldResult running = rewriter.getIndexAttr(0);
    for (Value input : adaptor.getInputs()) {
      auto inputType = cast<RankedTensorType>(input.getType());
      offsets[dimension] = running;
      SmallVector<OpFoldResult> sizes;
      for (auto [axis, extent] : llvm::enumerate(inputType.getShape()))
        sizes.push_back(ShapedType::isDynamic(extent)
                            ? OpFoldResult(tensor::DimOp::create(
                                  rewriter, loc, input, axis).getResult())
                            : OpFoldResult(rewriter.getIndexAttr(extent)));
      destination = tensor::InsertSliceOp::create(
          rewriter, loc, input, destination, offsets, sizes, strides);
      auto known = [](OpFoldResult value) -> std::optional<int64_t> {
        if (auto attr = dyn_cast<Attribute>(value))
          return cast<IntegerAttr>(attr).getInt();
        return std::nullopt;
      };
      std::optional<int64_t> at = known(running);
      std::optional<int64_t> band = known(sizes[dimension]);
      if (at && band) {
        running = rewriter.getIndexAttr(*at + *band);
        continue;
      }
      Value atValue =
          getValueOrCreateConstantIndexOp(rewriter, loc, running);
      Value bandValue =
          getValueOrCreateConstantIndexOp(rewriter, loc, sizes[dimension]);
      running = arith::AddIOp::create(rewriter, loc, atValue, bandValue)
                    .getResult();
    }

    rewriter.replaceOp(op, destination);
    return success();
  }
};

}

void mlir::tera::detail::populateShapePatterns(RewritePatternSet &patterns) {
  patterns.add<DimOpLowering, BroadcastInDimOpLowering, TransposeOpLowering,
               ReshapeOpLowering, SliceOpLowering, ConcatOpLowering,
               ReverseOpLowering, PadOpLowering>(patterns.getContext());
}
