//===- TeraToLinalgElementwise.cpp - Lower elementwise ops ------*- C++ -*-===//
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
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;
using namespace mlir::tera;
using namespace mlir::tera::detail;

namespace {
using ScalarBuilder = Value (*)(OpBuilder &, Location, Operation *, ValueRange);

template <typename SourceOp>
struct MapOpLowering : public OpConversionPattern<SourceOp> {
  MapOpLowering(MLIRContext *context, ScalarBuilder scalar)
      : OpConversionPattern<SourceOp>(context), scalar(scalar) {}

  LogicalResult
  matchAndRewrite(SourceOp op, typename SourceOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto resultType = cast<RankedTensorType>(op.getType());
    ScalarBuilder scalarBuilder = scalar;
    Operation *source = op;

    SmallVector<Value> sizes = resultExtents(rewriter, loc, op);
    rewriter.replaceOpWithNewOp<linalg::MapOp>(
        op, adaptor.getOperands(),
        emptyTensor(rewriter, loc, resultType, sizes),
        [&](OpBuilder &builder, Location bodyLoc, ValueRange args) {
          linalg::YieldOp::create(
              builder, bodyLoc,
              scalarBuilder(builder, bodyLoc, source, args));
        });
    return success();
  }

  ScalarBuilder scalar;
};

bool isFloat(Value value) { return isa<FloatType>(value.getType()); }

template <typename FloatOp, typename IntOp>
Value binary(OpBuilder &builder, Location loc, Operation *, ValueRange args) {
  if (isFloat(args[0]))
    return FloatOp::create(builder, loc, args[0], args[1]);
  return IntOp::create(builder, loc, args[0], args[1]);
}

Value negate(OpBuilder &builder, Location loc, Operation *, ValueRange args) {
  if (isFloat(args[0]))
    return arith::NegFOp::create(builder, loc, args[0]);
  Value zero =
      arith::ConstantOp::create(builder, loc, zeroAttr(args[0].getType()));
  return arith::SubIOp::create(builder, loc, zero, args[0]);
}

template <typename MathOp>
Value unaryMath(OpBuilder &builder, Location loc, Operation *,
                ValueRange args) {
  return MathOp::create(builder, loc, args[0]);
}

arith::CmpFPredicate floatPredicate(ComparisonDirection direction) {
  switch (direction) {
  case ComparisonDirection::Eq:
    return arith::CmpFPredicate::OEQ;
  case ComparisonDirection::Ne:
    return arith::CmpFPredicate::UNE;
  case ComparisonDirection::Lt:
    return arith::CmpFPredicate::OLT;
  case ComparisonDirection::Le:
    return arith::CmpFPredicate::OLE;
  case ComparisonDirection::Gt:
    return arith::CmpFPredicate::OGT;
  case ComparisonDirection::Ge:
    return arith::CmpFPredicate::OGE;
  }
  llvm_unreachable("unhandled comparison direction");
}

arith::CmpIPredicate integerPredicate(ComparisonDirection direction) {
  switch (direction) {
  case ComparisonDirection::Eq:
    return arith::CmpIPredicate::eq;
  case ComparisonDirection::Ne:
    return arith::CmpIPredicate::ne;
  case ComparisonDirection::Lt:
    return arith::CmpIPredicate::slt;
  case ComparisonDirection::Le:
    return arith::CmpIPredicate::sle;
  case ComparisonDirection::Gt:
    return arith::CmpIPredicate::sgt;
  case ComparisonDirection::Ge:
    return arith::CmpIPredicate::sge;
  }
  llvm_unreachable("unhandled comparison direction");
}

Value compare(OpBuilder &builder, Location loc, Operation *source,
              ValueRange args) {
  ComparisonDirection direction = cast<CompareOp>(source).getDirection();
  if (isFloat(args[0]))
    return arith::CmpFOp::create(builder, loc, floatPredicate(direction),
                                 args[0], args[1]);
  return arith::CmpIOp::create(builder, loc, integerPredicate(direction),
                               args[0], args[1]);
}

Value choose(OpBuilder &builder, Location loc, Operation *, ValueRange args) {
  return arith::SelectOp::create(builder, loc, args[0], args[1], args[2]);
}

Value changeElementType(OpBuilder &builder, Location loc, Operation *source,
                        ValueRange args) {
  Type target =
      cast<RankedTensorType>(source->getResult(0).getType()).getElementType();
  return convertScalarToDtype(builder, loc, args[0], target,
                              /*isUnsignedCast=*/false);
}

}

void mlir::tera::detail::populateElementwisePatterns(
    RewritePatternSet &patterns) {
  MLIRContext *context = patterns.getContext();
  patterns.add<MapOpLowering<AddOp>>(context,
                                     binary<arith::AddFOp, arith::AddIOp>);
  patterns.add<MapOpLowering<SubOp>>(context,
                                     binary<arith::SubFOp, arith::SubIOp>);
  patterns.add<MapOpLowering<MulOp>>(context,
                                     binary<arith::MulFOp, arith::MulIOp>);
  patterns.add<MapOpLowering<DivOp>>(context,
                                     binary<arith::DivFOp, arith::DivSIOp>);
  patterns.add<MapOpLowering<MaximumOp>>(
      context, binary<arith::MaximumFOp, arith::MaxSIOp>);
  patterns.add<MapOpLowering<NegOp>>(context, negate);
  patterns.add<MapOpLowering<ExpOp>>(context, unaryMath<math::ExpOp>);
  patterns.add<MapOpLowering<SqrtOp>>(context, unaryMath<math::SqrtOp>);
  patterns.add<MapOpLowering<RsqrtOp>>(context, unaryMath<math::RsqrtOp>);
  patterns.add<MapOpLowering<TanhOp>>(context, unaryMath<math::TanhOp>);
  patterns.add<MapOpLowering<CompareOp>>(context, compare);
  patterns.add<MapOpLowering<SelectOp>>(context, choose);
  patterns.add<MapOpLowering<ConvertOp>>(context, changeElementType);
}
