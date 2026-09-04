//===- TeraOpsElementwise.cpp - Elementwise tera ops ------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Tera/IR/TeraOps.h"

#include "TeraOpsDetail.h"
#include "mlir/IR/BuiltinTypes.h"

using namespace mlir;
using namespace mlir::tera;

//===----------------------------------------------------------------------===//
// CompareOp
//===----------------------------------------------------------------------===//

LogicalResult CompareOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> location, Adaptor adaptor,
    SmallVectorImpl<Type> &inferredReturnTypes) {
  auto lhsType = dyn_cast<RankedTensorType>(adaptor.getLhs().getType());
  if (!lhsType)
    return emitOptionalError(location, "expects a ranked operand");
  inferredReturnTypes.push_back(
      RankedTensorType::get(lhsType.getShape(), IntegerType::get(context, 1)));
  return success();
}

//===----------------------------------------------------------------------===//
// ConvertOp
//===----------------------------------------------------------------------===//

OpFoldResult ConvertOp::fold(FoldAdaptor) {
  if (getOperand().getType() == getResult().getType())
    return getOperand();
  return {};
}

//===----------------------------------------------------------------------===//
// Vector-Jacobian products
//===----------------------------------------------------------------------===//

LogicalResult AddOp::buildVjp(OpBuilder &, ValueRange adjoints,
                              SmallVectorImpl<Value> &operandAdjoints) {
  operandAdjoints.assign({adjoints[0], adjoints[0]});
  return success();
}

LogicalResult SubOp::buildVjp(OpBuilder &builder, ValueRange adjoints,
                              SmallVectorImpl<Value> &operandAdjoints) {
  operandAdjoints.assign(
      {adjoints[0], NegOp::create(builder, getLoc(), adjoints[0])});
  return success();
}

LogicalResult MulOp::buildVjp(OpBuilder &builder, ValueRange adjoints,
                              SmallVectorImpl<Value> &operandAdjoints) {
  operandAdjoints.assign(
      {MulOp::create(builder, getLoc(), adjoints[0], getRhs()),
       MulOp::create(builder, getLoc(), adjoints[0], getLhs())});
  return success();
}

LogicalResult DivOp::buildVjp(OpBuilder &builder, ValueRange adjoints,
                              SmallVectorImpl<Value> &operandAdjoints) {
  Value overRhs = DivOp::create(builder, getLoc(), adjoints[0], getRhs());
  Value scaled = MulOp::create(builder, getLoc(), overRhs, getResult());
  operandAdjoints.assign({overRhs, NegOp::create(builder, getLoc(), scaled)});
  return success();
}

/// Routes the full adjoint to the lhs when operands tie.
LogicalResult MaximumOp::buildVjp(OpBuilder &builder, ValueRange adjoints,
                                  SmallVectorImpl<Value> &operandAdjoints) {
  Location loc = getLoc();
  auto resultType = cast<RankedTensorType>(getType());
  Value zero = createSplat(builder, loc, resultType, 0.0, getResult());
  Value fromLhs = CompareOp::create(builder, loc, getLhs(), getRhs(),
                                    ComparisonDirection::Ge);
  operandAdjoints.assign(
      {SelectOp::create(builder, loc, fromLhs, adjoints[0], zero),
       SelectOp::create(builder, loc, fromLhs, zero, adjoints[0])});
  return success();
}

LogicalResult NegOp::buildVjp(OpBuilder &builder, ValueRange adjoints,
                              SmallVectorImpl<Value> &operandAdjoints) {
  operandAdjoints.assign({NegOp::create(builder, getLoc(), adjoints[0])});
  return success();
}

LogicalResult ExpOp::buildVjp(OpBuilder &builder, ValueRange adjoints,
                              SmallVectorImpl<Value> &operandAdjoints) {
  operandAdjoints.assign(
      {MulOp::create(builder, getLoc(), adjoints[0], getResult())});
  return success();
}

LogicalResult SqrtOp::buildVjp(OpBuilder &builder, ValueRange adjoints,
                               SmallVectorImpl<Value> &operandAdjoints) {
  Location loc = getLoc();
  auto type = cast<RankedTensorType>(getResult().getType());
  Value two = createSplat(builder, loc, type, 2.0, getResult());
  Value denominator = MulOp::create(builder, loc, two, getResult());
  operandAdjoints.assign(
      {DivOp::create(builder, loc, adjoints[0], denominator)});
  return success();
}

LogicalResult RsqrtOp::buildVjp(OpBuilder &builder, ValueRange adjoints,
                                SmallVectorImpl<Value> &operandAdjoints) {
  Location loc = getLoc();
  auto type = cast<RankedTensorType>(getResult().getType());
  Value half = createSplat(builder, loc, type, -0.5, getResult());
  Value squared = MulOp::create(builder, loc, getResult(), getResult());
  Value cubed = MulOp::create(builder, loc, squared, getResult());
  Value slope = MulOp::create(builder, loc, half, cubed);
  operandAdjoints.assign({MulOp::create(builder, loc, adjoints[0], slope)});
  return success();
}

LogicalResult TanhOp::buildVjp(OpBuilder &builder, ValueRange adjoints,
                               SmallVectorImpl<Value> &operandAdjoints) {
  Location loc = getLoc();
  auto type = cast<RankedTensorType>(getResult().getType());
  Value one = createSplat(builder, loc, type, 1.0, getResult());
  Value squared = MulOp::create(builder, loc, getResult(), getResult());
  Value slope = SubOp::create(builder, loc, one, squared);
  operandAdjoints.assign({MulOp::create(builder, loc, adjoints[0], slope)});
  return success();
}

LogicalResult CompareOp::buildVjp(OpBuilder &, ValueRange,
                                  SmallVectorImpl<Value> &operandAdjoints) {
  operandAdjoints.assign({Value(), Value()});
  return success();
}

LogicalResult SelectOp::buildVjp(OpBuilder &builder, ValueRange adjoints,
                                 SmallVectorImpl<Value> &operandAdjoints) {
  Location loc = getLoc();
  auto resultType = cast<RankedTensorType>(getType());
  Value zero = createSplat(builder, loc, resultType, 0.0, getResult());
  operandAdjoints.assign(
      {Value(), SelectOp::create(builder, loc, getPred(), adjoints[0], zero),
       SelectOp::create(builder, loc, getPred(), zero, adjoints[0])});
  return success();
}

LogicalResult ConvertOp::buildVjp(OpBuilder &builder, ValueRange adjoints,
                                  SmallVectorImpl<Value> &operandAdjoints) {
  auto operandType = cast<RankedTensorType>(getOperand().getType());
  if (!isa<FloatType>(operandType.getElementType())) {
    operandAdjoints.assign({Value()});
    return success();
  }
  operandAdjoints.assign(
      {ConvertOp::create(builder, getLoc(), operandType, adjoints[0])});
  return success();
}
