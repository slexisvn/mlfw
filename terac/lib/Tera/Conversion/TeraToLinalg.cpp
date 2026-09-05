//===- TeraToLinalg.cpp - tera to linalg conversion -------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Tera/Conversion/Passes.h"

#include "Tera/IR/TeraDialect.h"
#include "Tera/IR/TeraOps.h"
#include "TeraToLinalgDetail.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir::tera {
#define GEN_PASS_DEF_CONVERTTERATOLINALG
#include "Tera/Conversion/Passes.h.inc"

Value detail::emptyTensor(OpBuilder &builder, Location loc,
                          RankedTensorType type, ValueRange dynamicSizes) {
  return tensor::EmptyOp::create(builder, loc, type.getShape(),
                                 type.getElementType(), dynamicSizes);
}

Value detail::filledTensor(OpBuilder &builder, Location loc,
                           RankedTensorType type, TypedAttr init,
                           ValueRange dynamicSizes) {
  Value scalar = arith::ConstantOp::create(builder, loc, init);
  return linalg::FillOp::create(
             builder, loc, ValueRange{scalar},
             ValueRange{emptyTensor(builder, loc, type, dynamicSizes)})
      .getResult(0);
}

SmallVector<OpFoldResult> detail::resultShape(OpBuilder &builder, Location loc,
                                              Operation *op) {
  auto type = cast<RankedTensorType>(op->getResult(0).getType());
  if (type.hasStaticShape()) {
    SmallVector<OpFoldResult> shape;
    for (int64_t extent : type.getShape())
      shape.push_back(builder.getIndexAttr(extent));
    return shape;
  }

  ReifiedRankedShapedTypeDims reified;
  LogicalResult answered =
      cast<ReifyRankedShapedTypeOpInterface>(op).reifyResultShapes(builder,
                                                                   reified);
  assert(succeeded(answered) &&
         "an op whose destination is materialised has to reify its own "
         "result shapes; the ops that cannot -- the region ones, whose "
         "extents are only known inside the region -- take their results "
         "from the region rather than from a destination");
  (void)answered;
  return reified[0];
}

SmallVector<Value> detail::resultExtents(OpBuilder &builder, Location loc,
                                         Operation *op) {
  auto type = cast<RankedTensorType>(op->getResult(0).getType());
  if (type.hasStaticShape())
    return {};

  SmallVector<OpFoldResult> shape = resultShape(builder, loc, op);
  SmallVector<Value> extents;
  for (auto [axis, extent] : llvm::enumerate(type.getShape()))
    if (ShapedType::isDynamic(extent))
      extents.push_back(
          getValueOrCreateConstantIndexOp(builder, loc, shape[axis]));
  return extents;
}

Value detail::spreadInto(OpBuilder &builder, Location loc,
                         RankedTensorType resultType, Value operand,
                         ArrayRef<int64_t> low, ArrayRef<int64_t> spacing,
                         Value fill, ValueRange dynamicSizes) {
  MLIRContext *context = builder.getContext();
  int64_t rank = resultType.getRank();
  SmallVector<utils::IteratorType> iterators(rank,
                                             utils::IteratorType::parallel);

  return linalg::GenericOp::create(
             builder, loc, TypeRange{resultType}, ValueRange{},
             ValueRange{emptyTensor(builder, loc, resultType, dynamicSizes)},
             ArrayRef<AffineMap>{
                 AffineMap::getMultiDimIdentityMap(rank, context)},
             iterators,
             [&](OpBuilder &body, Location bodyLoc, ValueRange) {
               Value zero = arith::ConstantIndexOp::create(body, bodyLoc, 0);
               SmallVector<Value> position;
               Value inside;
               for (int64_t axis = 0; axis < rank; ++axis) {
                 Value at = linalg::IndexOp::create(body, bodyLoc, axis);
                 Value offset = arith::SubIOp::create(
                     body, bodyLoc, at,
                     arith::ConstantIndexOp::create(body, bodyLoc, low[axis]));
                 Value step = arith::ConstantIndexOp::create(body, bodyLoc,
                                                             spacing[axis]);
                 Value last = arith::SubIOp::create(
                     body, bodyLoc,
                     tensor::DimOp::create(body, bodyLoc, operand, axis),
                     arith::ConstantIndexOp::create(body, bodyLoc, 1));
                 Value from =
                     arith::FloorDivSIOp::create(body, bodyLoc, offset, step);
                 from = arith::MaxSIOp::create(body, bodyLoc, from, zero);
                 from = arith::MinSIOp::create(body, bodyLoc, from, last);
                 position.push_back(from);

                 Value lands = arith::CmpIOp::create(
                     body, bodyLoc, arith::CmpIPredicate::eq, offset,
                     arith::MulIOp::create(body, bodyLoc, from, step));
                 inside = inside ? arith::AndIOp::create(body, bodyLoc, inside,
                                                         lands)
                                       .getResult()
                                 : lands;
               }

               Value element =
                   tensor::ExtractOp::create(body, bodyLoc, operand, position);
               linalg::YieldOp::create(
                   body, bodyLoc,
                   arith::SelectOp::create(body, bodyLoc, inside, element, fill)
                       .getResult());
             })
      .getResult(0);
}

TypedAttr detail::zeroAttr(Type elementType) {
  if (auto floatType = dyn_cast<FloatType>(elementType))
    return FloatAttr::get(floatType,
                          APFloat::getZero(floatType.getFloatSemantics()));
  return IntegerAttr::get(elementType, 0);
}

void populateTeraToLinalgPatterns(RewritePatternSet &patterns) {
  detail::populateConstantPatterns(patterns);
  detail::populateElementwisePatterns(patterns);
  detail::populateShapePatterns(patterns);
  detail::populateIndexingPatterns(patterns);
  detail::populateWindowPatterns(patterns);
  detail::populateContractionPatterns(patterns);
  detail::populateAutodiffPatterns(patterns);
  detail::populateControlFlowPatterns(patterns);
}

namespace {
LogicalResult rejectUnlowerable(Operation *root) {
  auto isDynamic = [](Type type) {
    auto shaped = dyn_cast<ShapedType>(type);
    return !shaped || !shaped.hasStaticShape();
  };
  WalkResult walk = root->walk([&](TeraLoweringOpInterface op) {
    if (failed(op.verifyLoweringToLinalg()))
      return WalkResult::interrupt();
    if (llvm::none_of(op->getOperandTypes(), isDynamic) &&
        llvm::none_of(op->getResultTypes(), isDynamic))
      return WalkResult::advance();
    if (op.acceptsDynamicShapes())
      return WalkResult::advance();
    op->emitError() << "cannot be lowered to linalg with a dynamic shape: "
                       "this op materialises a destination from static "
                       "extents and no operand carries a dynamic one";
    return WalkResult::interrupt();
  });
  return failure(walk.wasInterrupted());
}

struct ConvertTeraToLinalg
    : public impl::ConvertTeraToLinalgBase<ConvertTeraToLinalg> {
  using impl::ConvertTeraToLinalgBase<
      ConvertTeraToLinalg>::ConvertTeraToLinalgBase;

  void runOnOperation() final {
    if (failed(rejectUnlowerable(getOperation())))
      return signalPassFailure();

    MLIRContext *context = &getContext();
    ConversionTarget target(*context);
    target.addIllegalDialect<TeraDialect>();
    target.addLegalOp<ModuleOp>();
    target.addLegalDialect<arith::ArithDialect,
                           bufferization::BufferizationDialect,
                           func::FuncDialect, linalg::LinalgDialect,
                           math::MathDialect, scf::SCFDialect,
                           tensor::TensorDialect>();

    target.addLegalOp<YieldOp>();

    RewritePatternSet patterns(context);
    populateTeraToLinalgPatterns(patterns);

    if (failed(
            applyFullConversion(getOperation(), target, std::move(patterns))))
      signalPassFailure();
  }
};

}
}
