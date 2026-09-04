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

SmallVector<Value> detail::dynamicExtents(
    OpBuilder &builder, Location loc, RankedTensorType type,
    function_ref<std::pair<Value, int64_t>(int64_t)> source) {
  SmallVector<Value> extents;
  for (auto [axis, extent] : llvm::enumerate(type.getShape())) {
    if (!ShapedType::isDynamic(extent))
      continue;
    auto [value, dimension] = source(axis);
    extents.push_back(tensor::DimOp::create(builder, loc, value, dimension));
  }
  return extents;
}

SmallVector<Value> detail::extentsLike(OpBuilder &builder, Location loc,
                                       RankedTensorType type, Value operand) {
  return dynamicExtents(builder, loc, type, [&](int64_t axis) {
    return std::pair<Value, int64_t>{operand, axis};
  });
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
bool acceptsDynamicShapes(Operation *op) {
  if (auto scan = dyn_cast<ScanOp>(op))
    return !ShapedType::isDynamic(scan.getTripCount());
  if (auto reverse = dyn_cast<ReverseOp>(op)) {
    auto type = cast<RankedTensorType>(reverse.getOperand().getType());
    return llvm::none_of(reverse.getDimensions(), [&](int64_t axis) {
      return ShapedType::isDynamic(type.getDimSize(axis));
    });
  }
  return op->hasTrait<AcceptsDynamicShapes>();
}

LogicalResult rejectDynamicShapes(Operation *root) {
  auto isDynamic = [](Type type) {
    auto shaped = dyn_cast<ShapedType>(type);
    return !shaped || !shaped.hasStaticShape();
  };
  WalkResult walk = root->walk([&](Operation *op) {
    if (!isa_and_present<TeraDialect>(op->getDialect()))
      return WalkResult::advance();
    if (llvm::none_of(op->getOperandTypes(), isDynamic) &&
        llvm::none_of(op->getResultTypes(), isDynamic))
      return WalkResult::advance();
    if (acceptsDynamicShapes(op))
      return WalkResult::advance();
    if (isa<ScanOp>(op)) {
      op->emitError() << "cannot be lowered to linalg with a dynamic step "
                         "axis: the trip count would have to be read at run "
                         "time";
      return WalkResult::interrupt();
    }
    op->emitError() << "cannot be lowered to linalg with a dynamic shape: "
                       "this op materialises a destination from static "
                       "extents and no operand carries a dynamic one";
    return WalkResult::interrupt();
  });
  return failure(walk.wasInterrupted());
}

LogicalResult rejectUnlowerablePools(Operation *root) {
  WalkResult walk = root->walk([&](Pool2dOp pool) {
    auto operandType = cast<RankedTensorType>(pool.getOperand().getType());
    auto resultType = cast<RankedTensorType>(pool.getType());
    ArrayRef<int64_t> window = pool.getKernelSize();
    ArrayRef<int64_t> strides = pool.getStrides();
    SmallVector<int64_t> low = pool.getPaddingLow();
    SmallVector<int64_t> high = pool.getPaddingHigh();
    bool padded = llvm::any_of(pool.getPadding(),
                               [](int64_t pad) { return pad != 0; });

    if (padded && pool.getKind() == PoolKind::Max) {
      pool.emitError() << "cannot be lowered with padding: a maximum has no "
                          "value to read outside the input that a window made "
                          "only of padding would not then answer with";
      return WalkResult::interrupt();
    }
    if (padded && !pool.getCountIncludePad()) {
      pool.emitError() << "cannot be lowered without counting the padding: "
                          "each window would be divided by how much of it fell "
                          "inside, which is a count per window rather than the "
                          "one number the traversal divides by";
      return WalkResult::interrupt();
    }

    for (int64_t axis = 0; axis < 2; ++axis) {
      int64_t extent = operandType.getDimSize(axis + 2);
      int64_t count = resultType.getDimSize(axis + 2);
      if (ShapedType::isDynamic(extent) || ShapedType::isDynamic(count))
        continue;
      if ((count - 1) * strides[axis] + window[axis] >
          extent + low[axis] + high[axis]) {
        pool.emitError() << "cannot be lowered with a window that hangs over "
                            "axis "
                         << axis
                         << ": it reads past the end of the input, and what is "
                            "there is padding the op does not carry";
        return WalkResult::interrupt();
      }
    }
    return WalkResult::advance();
  });
  return failure(walk.wasInterrupted());
}

struct ConvertTeraToLinalg
    : public impl::ConvertTeraToLinalgBase<ConvertTeraToLinalg> {
  using impl::ConvertTeraToLinalgBase<
      ConvertTeraToLinalg>::ConvertTeraToLinalgBase;

  void runOnOperation() final {
    if (failed(rejectDynamicShapes(getOperation())) ||
        failed(rejectUnlowerablePools(getOperation())))
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
