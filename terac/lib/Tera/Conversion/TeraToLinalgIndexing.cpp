//===- TeraToLinalgIndexing.cpp - Lower gather and scatter ------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Neither op has an indexing map, because neither knows at compile time which
// element it touches. A gather is still one write per result element and so
// stays a `linalg.generic`, with the read spelt as a `tensor.extract` the map
// could not have expressed. A scatter is not: two updates may land on the same
// element, so the writes have to happen in an order, and that order is a loop
// nest carrying the tensor from step to step.
//
//===----------------------------------------------------------------------===//

#include "Tera/IR/TeraOps.h"

#include "TeraToLinalgDetail.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/SmallBitVector.h"

using namespace mlir;
using namespace mlir::tera;
using namespace mlir::tera::detail;

namespace {

/// windowIndices follow operand-axis order; batchIndices follow index-tensor
/// axis order. Coordinates are added to addressedAxes; other axes start at
/// zero.
SmallVector<Value> operandPosition(OpBuilder &builder, Location loc,
                                   Value indices, int64_t indexVectorDim,
                                   ArrayRef<int64_t> addressedAxes,
                                   const llvm::SmallBitVector &degenerate,
                                   ValueRange windowIndices,
                                   ValueRange batchIndices) {
  SmallVector<Value> position(degenerate.size());
  size_t nextWindow = 0;
  for (int64_t axis = 0; axis < static_cast<int64_t>(position.size()); ++axis)
    if (!degenerate.test(axis))
      position[axis] = windowIndices[nextWindow++];

  auto indicesType = cast<RankedTensorType>(indices.getType());
  for (auto [coordinate, axis] : llvm::enumerate(addressedAxes)) {
    SmallVector<Value> read;
    size_t nextBatch = 0;
    for (int64_t at = 0; at < indicesType.getRank(); ++at)
      read.push_back(at == indexVectorDim
                         ? arith::ConstantIndexOp::create(builder, loc,
                                                          coordinate)
                               .getResult()
                         : batchIndices[nextBatch++]);
    Value start = tensor::ExtractOp::create(builder, loc, indices, read);
    Value widened = arith::IndexCastOp::create(builder, loc,
                                               builder.getIndexType(), start);
    position[axis] =
        position[axis]
            ? arith::AddIOp::create(builder, loc, position[axis], widened)
                  .getResult()
            : widened;
  }

  for (Value &at : position)
    if (!at)
      at = arith::ConstantIndexOp::create(builder, loc, 0);
  return position;
}

SmallVector<Value> unclaimed(const llvm::SmallBitVector &claimed,
                             ValueRange values) {
  SmallVector<Value> rest;
  for (auto [axis, value] : llvm::enumerate(values))
    if (!claimed.test(axis))
      rest.push_back(value);
  return rest;
}

SmallVector<Value> claimedBy(const llvm::SmallBitVector &claimed,
                             ValueRange values) {
  SmallVector<Value> taken;
  for (auto [axis, value] : llvm::enumerate(values))
    if (claimed.test(axis))
      taken.push_back(value);
  return taken;
}

llvm::SmallBitVector axisMask(ArrayRef<int64_t> axes, int64_t rank) {
  llvm::SmallBitVector mask(rank);
  for (int64_t axis : axes)
    mask.set(axis);
  return mask;
}

/// One write per result element, so the loop nest is the result's own shape
/// and `linalg.generic` supplies it. What the map cannot say is where the read
/// comes from, so the body says it: `linalg.index` recovers the position being
/// written and `tensor.extract` reads the element it was gathered from.
struct GatherOpLowering : public OpConversionPattern<GatherOp> {
  using OpConversionPattern<GatherOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(GatherOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto resultType = cast<RankedTensorType>(op.getType());
    int64_t rank = resultType.getRank();

    llvm::SmallBitVector isOffset = axisMask(op.getOffsetDims(), rank);
    SmallVector<int64_t> batchAxes = op.getBatchAxes();
    SmallVector<Value> sizes;
    size_t nextBatch = 0;
    for (int64_t axis = 0; axis < rank; ++axis) {
      if (isOffset.test(axis))
        continue;
      int64_t source = batchAxes[nextBatch++];
      if (ShapedType::isDynamic(resultType.getDimSize(axis)))
        sizes.push_back(tensor::DimOp::create(rewriter, loc,
                                              adaptor.getIndices(), source));
    }

    llvm::SmallBitVector collapsed =
        axisMask(op.getCollapsedSliceDims(),
                 cast<RankedTensorType>(op.getOperand().getType()).getRank());
    SmallVector<int64_t> addressed(op.getStartIndexMap());
    int64_t indexVectorDim = op.getIndexVectorDim();
    Value operand = adaptor.getOperand();
    Value indices = adaptor.getIndices();

    AffineMap identity =
        AffineMap::getMultiDimIdentityMap(rank, rewriter.getContext());
    SmallVector<utils::IteratorType> iterators(rank,
                                               utils::IteratorType::parallel);

    rewriter.replaceOpWithNewOp<linalg::GenericOp>(
        op, TypeRange{resultType}, ValueRange{},
        ValueRange{emptyTensor(rewriter, loc, resultType, sizes)},
        ArrayRef<AffineMap>{identity}, iterators,
        [&](OpBuilder &builder, Location bodyLoc, ValueRange) {
          SmallVector<Value> at;
          for (int64_t axis = 0; axis < rank; ++axis)
            at.push_back(linalg::IndexOp::create(builder, bodyLoc, axis));
          SmallVector<Value> position = operandPosition(
              builder, bodyLoc, indices, indexVectorDim, addressed, collapsed,
              claimedBy(isOffset, at), unclaimed(isOffset, at));
          linalg::YieldOp::create(
              builder, bodyLoc,
              tensor::ExtractOp::create(builder, bodyLoc, operand, position)
                  .getResult());
        });
    return success();
  }
};

/// The updates are walked in the order they are laid out and added into the
/// tensor one at a time, which is the only order that gives the same answer as
/// mlfw's own loop nest when two of them land on the same element. The tensor
/// travels through the nest as an iteration argument: a scatter that writes
/// nothing is a copy of the operand, and every write is one more
/// `tensor.insert` on top of that.
///
/// The copy is not something bufferization can be left to insert. The operand
/// is usually dead after the scatter, so writing into it in place is a legal
/// conclusion within the function -- and wrong across calls, because the buffer
/// belongs to the caller. The gradcheck is what finds that: it calls the same
/// function twice with the same inputs, and a scatter that kept its answer in
/// the caller's operand answers differently the second time.
///
/// `bufferization.alloc_tensor` is how the copy is asked for rather than a
/// `linalg.copy`, and the difference is not stylistic. A `linalg.copy` is a
/// parallel op and becomes a kernel on the GPU target, which would leave this
/// loop nest -- which has no parallel axis and stays on the host -- reading a
/// buffer between two launches that hold it on the device. An `alloc_tensor`
/// is a buffer and a copy into it, and never a launch.
struct ScatterOpLowering : public OpConversionPattern<ScatterOp> {
  using OpConversionPattern<ScatterOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ScatterOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto resultType = cast<RankedTensorType>(op.getType());
    auto updatesType = cast<RankedTensorType>(op.getUpdates().getType());
    int64_t updatesRank = updatesType.getRank();

    Value written = bufferization::AllocTensorOp::create(
        rewriter, loc, resultType,
        extentsLike(rewriter, loc, resultType, adaptor.getOperand()),
        adaptor.getOperand());

    Value zero = arith::ConstantIndexOp::create(rewriter, loc, 0);
    Value one = arith::ConstantIndexOp::create(rewriter, loc, 1);
    SmallVector<Value> lowerBounds(updatesRank, zero);
    SmallVector<Value> steps(updatesRank, one);
    SmallVector<Value> upperBounds;
    for (auto [axis, extent] : llvm::enumerate(updatesType.getShape()))
      upperBounds.push_back(
          ShapedType::isDynamic(extent)
              ? tensor::DimOp::create(rewriter, loc, adaptor.getUpdates(), axis)
                    .getResult()
              : arith::ConstantIndexOp::create(rewriter, loc, extent)
                    .getResult());

    llvm::SmallBitVector isWindow =
        axisMask(op.getUpdateWindowDims(), updatesRank);
    llvm::SmallBitVector inserted =
        axisMask(op.getInsertedWindowDims(),
                 cast<RankedTensorType>(op.getOperand().getType()).getRank());
    SmallVector<int64_t> addressed(op.getScatterDimsToOperandDims());
    int64_t indexVectorDim = op.getIndexVectorDim();
    Value indices = adaptor.getIndices();
    Value updates = adaptor.getUpdates();

    scf::LoopNest nest = scf::buildLoopNest(
        rewriter, loc, lowerBounds, upperBounds, steps,
        ValueRange{written},
        [&](OpBuilder &builder, Location bodyLoc, ValueRange at,
            ValueRange carried) -> scf::ValueVector {
          SmallVector<Value> position = operandPosition(
              builder, bodyLoc, indices, indexVectorDim, addressed, inserted,
              claimedBy(isWindow, at), unclaimed(isWindow, at));
          Value into = carried.front();
          Value standing =
              tensor::ExtractOp::create(builder, bodyLoc, into, position);
          Value update = tensor::ExtractOp::create(builder, bodyLoc, updates,
                                                   SmallVector<Value>(at));
          Value combined =
              isa<FloatType>(updatesType.getElementType())
                  ? arith::AddFOp::create(builder, bodyLoc, standing, update)
                        .getResult()
                  : arith::AddIOp::create(builder, bodyLoc, standing, update)
                        .getResult();
          return {tensor::InsertOp::create(builder, bodyLoc, combined, into,
                                           position)
                      .getResult()};
        });

    rewriter.replaceOp(op, nest.results);
    return success();
  }
};

} // namespace

void mlir::tera::detail::populateIndexingPatterns(RewritePatternSet &patterns) {
  patterns.add<GatherOpLowering, ScatterOpLowering>(patterns.getContext());
}
