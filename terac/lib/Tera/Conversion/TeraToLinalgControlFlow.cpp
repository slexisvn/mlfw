//===- TeraToLinalgControlFlow.cpp - Lower ops with a body ------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Tera/IR/TeraOps.h"

#include "TeraToLinalgDetail.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;
using namespace mlir::tera;
using namespace mlir::tera::detail;

namespace {
SmallVector<Value> resolveYield(Block &body, ValueRange arguments) {
  SmallVector<Value> yielded;
  for (Value value : cast<YieldOp>(body.getTerminator()).getResults()) {
    auto argument = dyn_cast<BlockArgument>(value);
    yielded.push_back(argument && argument.getOwner() == &body
                          ? arguments[argument.getArgNumber()]
                          : value);
  }
  return yielded;
}

SmallVector<Value> moveBody(ConversionPatternRewriter &rewriter, Block &from,
                            Block &into, ValueRange arguments) {
  SmallVector<Value> yielded = resolveYield(from, arguments);
  rewriter.eraseOp(from.getTerminator());
  if (!into.empty() && into.back().hasTrait<OpTrait::IsTerminator>())
    rewriter.eraseOp(&into.back());
  rewriter.inlineBlockBefore(&from, &into, into.end(), arguments);
  return yielded;
}

Value extentAsIndex(OpBuilder &builder, Location loc, Value extent) {
  Value scalar = tensor::ExtractOp::create(builder, loc, extent, ValueRange{});
  return arith::IndexCastOp::create(builder, loc, builder.getIndexType(),
                                    scalar);
}

SmallVector<SmallVector<Value>> sizesPerResult(OpBuilder &builder, Location loc,
                                               TypeRange results,
                                               ValueRange sizes) {
  SmallVector<SmallVector<Value>> split;
  size_t at = 0;
  for (Type result : results) {
    SmallVector<Value> extents;
    for (int64_t extent : cast<RankedTensorType>(result).getShape())
      if (ShapedType::isDynamic(extent))
        extents.push_back(extentAsIndex(builder, loc, sizes[at++]));
    split.push_back(std::move(extents));
  }
  return split;
}

Value sliceAt(OpBuilder &builder, Location loc, Value stacked, Value index) {
  auto type = cast<RankedTensorType>(stacked.getType());
  SmallVector<OpFoldResult> offsets{index}, sizes{builder.getIndexAttr(1)},
      strides{builder.getIndexAttr(1)};
  for (auto [axis, extent] : llvm::enumerate(type.getShape().drop_front())) {
    offsets.push_back(builder.getIndexAttr(0));
    sizes.push_back(ShapedType::isDynamic(extent)
                        ? OpFoldResult(tensor::DimOp::create(
                              builder, loc, stacked, axis + 1).getResult())
                        : OpFoldResult(builder.getIndexAttr(extent)));
    strides.push_back(builder.getIndexAttr(1));
  }
  return tensor::ExtractSliceOp::create(builder, loc,
                                        ScanOp::getSliceType(type), stacked,
                                        offsets, sizes, strides);
}

Value insertAt(OpBuilder &builder, Location loc, Value slice, Value stacked,
               Value index) {
  auto type = cast<RankedTensorType>(stacked.getType());
  SmallVector<OpFoldResult> offsets{index}, sizes{builder.getIndexAttr(1)},
      strides{builder.getIndexAttr(1)};
  for (auto [axis, extent] : llvm::enumerate(type.getShape().drop_front())) {
    offsets.push_back(builder.getIndexAttr(0));
    sizes.push_back(ShapedType::isDynamic(extent)
                        ? OpFoldResult(tensor::DimOp::create(
                              builder, loc, slice, axis).getResult())
                        : OpFoldResult(builder.getIndexAttr(extent)));
    strides.push_back(builder.getIndexAttr(1));
  }
  return tensor::InsertSliceOp::create(builder, loc, slice, stacked, offsets,
                                       sizes, strides);
}

struct IfOpLowering : public OpConversionPattern<IfOp> {
  using OpConversionPattern<IfOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(IfOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value condition = tensor::ExtractOp::create(
        rewriter, loc, adaptor.getCondition(), ValueRange{});
    auto branch = scf::IfOp::create(rewriter, loc, op.getResultTypes(),
                                    condition, /*withElseRegion=*/true);

    for (auto [from, into] :
         {std::pair{&op.getThenBody(), &branch.getThenRegion()},
          std::pair{&op.getElseBody(), &branch.getElseRegion()}}) {
      Block &target = into->front();
      SmallVector<Value> yielded =
          moveBody(rewriter, from->front(), target, adaptor.getInputs());
      rewriter.setInsertionPointToEnd(&target);
      scf::YieldOp::create(rewriter, loc, yielded);
    }

    rewriter.replaceOp(op, branch.getResults());
    return success();
  }
};

struct ScanOpLowering : public OpConversionPattern<ScanOp> {
  using OpConversionPattern<ScanOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ScanOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    int64_t steps = op.getTripCount();
    size_t carries = adaptor.getInits().size();

    Value first = arith::ConstantIndexOp::create(rewriter, loc, 0);
    Value bound = arith::ConstantIndexOp::create(rewriter, loc, steps);
    Value stride = arith::ConstantIndexOp::create(rewriter, loc, 1);

    SmallVector<SmallVector<Value>> extents = sizesPerResult(
        rewriter, loc, op.getResultTypes(), adaptor.getSizes());
    SmallVector<Value> initial(adaptor.getInits());
    for (auto [index, output] : llvm::enumerate(op.getOutputs()))
      initial.push_back(
          emptyTensor(rewriter, loc, cast<RankedTensorType>(output.getType()),
                      extents[carries + index]));

    auto loop =
        scf::ForOp::create(rewriter, loc, first, bound, stride, initial);
    Block &body = *loop.getBody();
    rewriter.setInsertionPointToEnd(&body);

    Value step = loop.getInductionVar();
    if (op.getReverse()) {
      Value last = arith::ConstantIndexOp::create(rewriter, loc, steps - 1);
      step = arith::SubIOp::create(rewriter, loc, last, step);
    }

    SmallVector<Value> arguments(loop.getRegionIterArgs().take_front(carries));
    for (Value input : adaptor.getInputs())
      arguments.push_back(sliceAt(rewriter, loc, input, step));
    arguments.append(adaptor.getConstants().begin(),
                     adaptor.getConstants().end());

    SmallVector<Value> yielded =
        moveBody(rewriter, op.getBody().front(), body, arguments);

    rewriter.setInsertionPointToEnd(&body);
    SmallVector<Value> next(yielded.begin(), yielded.begin() + carries);
    for (auto [slice, stacked] :
         llvm::zip_equal(ArrayRef<Value>(yielded).drop_front(carries),
                         loop.getRegionIterArgs().drop_front(carries)))
      next.push_back(insertAt(rewriter, loc, slice, stacked, step));
    scf::YieldOp::create(rewriter, loc, next);

    rewriter.replaceOp(op, loop.getResults());
    return success();
  }
};

}

void mlir::tera::detail::populateControlFlowPatterns(
    RewritePatternSet &patterns) {
  patterns.add<IfOpLowering, ScanOpLowering>(patterns.getContext());
}
