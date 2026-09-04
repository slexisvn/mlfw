//===- TeraOpsContraction.cpp - Contraction and reduction -------*- C++ -*-===//
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
using namespace mlir::tera::detail;

LogicalResult
DotOp::inferReturnTypes(MLIRContext *, std::optional<Location> location,
                        Adaptor adaptor,
                        SmallVectorImpl<Type> &inferredReturnTypes) {
  auto lhsType = dyn_cast<RankedTensorType>(adaptor.getLhs().getType());
  auto rhsType = dyn_cast<RankedTensorType>(adaptor.getRhs().getType());
  if (!lhsType || !rhsType)
    return emitOptionalError(location, "expects ranked operands");
  if (lhsType.getElementType() != rhsType.getElementType())
    return emitOptionalError(location,
                             "expects both operands to share an element type");

  ArrayRef<int64_t> lhsBatch = adaptor.getLhsBatch();
  ArrayRef<int64_t> rhsBatch = adaptor.getRhsBatch();
  ArrayRef<int64_t> lhsContracting = adaptor.getLhsContracting();
  ArrayRef<int64_t> rhsContracting = adaptor.getRhsContracting();

  if (lhsBatch.size() != rhsBatch.size())
    return emitOptionalError(
        location, "expects the same number of batch axes on both operands");
  if (lhsContracting.size() != rhsContracting.size())
    return emitOptionalError(
        location,
        "expects the same number of contracting axes on both operands");

  llvm::SmallBitVector lhsUsed(lhsType.getRank());
  llvm::SmallBitVector rhsUsed(rhsType.getRank());
  if (failed(markAxes(location, lhsBatch, lhsType.getRank(), lhsUsed,
                      "lhs_batch")) ||
      failed(markAxes(location, lhsContracting, lhsType.getRank(), lhsUsed,
                      "lhs_contracting")) ||
      failed(markAxes(location, rhsBatch, rhsType.getRank(), rhsUsed,
                      "rhs_batch")) ||
      failed(markAxes(location, rhsContracting, rhsType.getRank(), rhsUsed,
                      "rhs_contracting")))
    return failure();

  for (size_t i = 0; i < lhsBatch.size(); ++i)
    if (!extentsAgree(lhsType.getDimSize(lhsBatch[i]),
                      rhsType.getDimSize(rhsBatch[i])))
      return emitOptionalError(location, "batch axis pair ", i,
                               " has mismatched extents");
  for (size_t i = 0; i < lhsContracting.size(); ++i)
    if (!extentsAgree(lhsType.getDimSize(lhsContracting[i]),
                      rhsType.getDimSize(rhsContracting[i])))
      return emitOptionalError(location, "contracting axis pair ", i,
                               " has mismatched extents");

  SmallVector<int64_t> shape;
  for (int64_t axis : lhsBatch)
    shape.push_back(lhsType.getDimSize(axis));
  for (int64_t axis : freeAxes(lhsUsed, lhsType.getRank()))
    shape.push_back(lhsType.getDimSize(axis));
  for (int64_t axis : freeAxes(rhsUsed, rhsType.getRank()))
    shape.push_back(rhsType.getDimSize(axis));

  inferredReturnTypes.push_back(
      RankedTensorType::get(shape, lhsType.getElementType()));
  return success();
}

SmallVector<int64_t> DotOp::getLhsFreeAxes() {
  int64_t rank = cast<RankedTensorType>(getLhs().getType()).getRank();
  return freeAxes(claimedAxes(getLhsBatch(), getLhsContracting(), rank), rank);
}

SmallVector<int64_t> DotOp::getRhsFreeAxes() {
  int64_t rank = cast<RankedTensorType>(getRhs().getType()).getRank();
  return freeAxes(claimedAxes(getRhsBatch(), getRhsContracting(), rank), rank);
}

LogicalResult
ReduceOp::inferReturnTypes(MLIRContext *, std::optional<Location> location,
                           Adaptor adaptor,
                           SmallVectorImpl<Type> &inferredReturnTypes) {
  auto operandType = dyn_cast<RankedTensorType>(adaptor.getOperand().getType());
  if (!operandType)
    return emitOptionalError(location, "expects a ranked operand");

  int64_t rank = operandType.getRank();
  llvm::SmallBitVector reduced(rank);
  if (failed(markAxes(location, adaptor.getDimensions(), rank, reduced,
                      "reduction")))
    return failure();

  SmallVector<int64_t> shape;
  for (int64_t axis : freeAxes(reduced, rank))
    shape.push_back(operandType.getDimSize(axis));

  inferredReturnTypes.push_back(
      RankedTensorType::get(shape, operandType.getElementType()));
  return success();
}

LogicalResult DotOp::buildVjp(OpBuilder &builder, ValueRange adjoints,
                              SmallVectorImpl<Value> &operandAdjoints) {
  Location loc = getLoc();
  SmallVector<int64_t> lhsFree = getLhsFreeAxes();
  SmallVector<int64_t> rhsFree = getRhsFreeAxes();
  int64_t batch = getLhsBatch().size();
  SmallVector<int64_t> adjointBatch = axisRange(0, batch);
  SmallVector<int64_t> adjointLhsFree =
      axisRange(batch, batch + lhsFree.size());
  SmallVector<int64_t> adjointRhsFree = axisRange(
      batch + lhsFree.size(), batch + lhsFree.size() + rhsFree.size());

  auto against = [&](Value other, ArrayRef<int64_t> adjointContracting,
                     ArrayRef<int64_t> otherBatch, ArrayRef<int64_t> otherFree,
                     ArrayRef<int64_t> otherContracting,
                     ArrayRef<int64_t> targetBatch,
                     ArrayRef<int64_t> targetFree,
                     ArrayRef<int64_t> targetContracting, int64_t targetRank) {
    Value product =
        DotOp::create(builder, loc, adjoints[0], other, adjointBatch,
                      adjointContracting, otherBatch, otherFree);

    SmallVector<int64_t> permutation(targetRank);
    for (auto [position, axis] : llvm::enumerate(targetBatch))
      permutation[axis] = position;
    for (auto [position, axis] : llvm::enumerate(targetFree))
      permutation[axis] = batch + position;
    SmallVector<int64_t> positions = sortedPositions(otherContracting);
    for (auto [pair, axis] : llvm::enumerate(targetContracting))
      permutation[axis] = batch + targetFree.size() + positions[pair];

    return TransposeOp::create(builder, loc, product, permutation);
  };

  int64_t lhsRank = cast<RankedTensorType>(getLhs().getType()).getRank();
  int64_t rhsRank = cast<RankedTensorType>(getRhs().getType()).getRank();
  operandAdjoints.assign({against(getRhs(), adjointRhsFree, getRhsBatch(),
                                  rhsFree, getRhsContracting(), getLhsBatch(),
                                  lhsFree, getLhsContracting(), lhsRank),
                          against(getLhs(), adjointLhsFree, getLhsBatch(),
                                  lhsFree, getLhsContracting(), getRhsBatch(),
                                  rhsFree, getRhsContracting(), rhsRank)});
  return success();
}

LogicalResult ReduceOp::buildVjp(OpBuilder &builder, ValueRange adjoints,
                                 SmallVectorImpl<Value> &operandAdjoints) {
  Location loc = getLoc();
  auto operandType = cast<RankedTensorType>(getOperand().getType());
  llvm::SmallBitVector reduced(operandType.getRank());
  for (int64_t axis : getDimensions())
    reduced.set(axis);
  SmallVector<int64_t> surviving = freeAxes(reduced, operandType.getRank());

  SmallVector<Value> sizes = dynamicExtentsOf(builder, loc, getOperand());
  auto spread = [&](Value value) {
    return BroadcastInDimOp::create(builder, loc, operandType, value, sizes,
                                    surviving);
  };

  switch (getKind()) {
  case ReduceKind::Sum:
    operandAdjoints.assign({spread(adjoints[0])});
    return success();
  case ReduceKind::Mean: {
    int64_t count = 1;
    for (int64_t axis : getDimensions()) {
      int64_t extent = operandType.getDimSize(axis);
      if (ShapedType::isDynamic(extent))
        return emitOpError() << "cannot be differentiated over a dynamic axis: "
                                "the share each element contributed is one "
                                "over a number that is not known here";
      count *= extent;
    }
    auto adjointType = cast<RankedTensorType>(adjoints[0].getType());
    Value share = createSplat(builder, loc, adjointType,
                              1.0 / static_cast<double>(count), adjoints[0]);
    operandAdjoints.assign(
        {spread(MulOp::create(builder, loc, adjoints[0], share))});
    return success();
  }
  case ReduceKind::Product: {
    Value scaled = MulOp::create(builder, loc, adjoints[0], getResult());
    operandAdjoints.assign(
        {DivOp::create(builder, loc, spread(scaled), getOperand())});
    return success();
  }
  case ReduceKind::Maximum:
  case ReduceKind::Minimum: {
    Value extremum =
        CompareOp::create(builder, loc, getOperand(), spread(getResult()),
                          ComparisonDirection::Eq);
    Value zero = createSplat(builder, loc, operandType, 0.0, getOperand());
    operandAdjoints.assign(
        {SelectOp::create(builder, loc, extremum, spread(adjoints[0]), zero)});
    return success();
  }
  }
  return failure();
}
