//===- TeraOpsIndexing.cpp - Gather and scatter tera ops --------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Tera/IR/TeraOps.h"

#include "TeraOpsDetail.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/STLExtras.h"

using namespace mlir;
using namespace mlir::tera;
using namespace mlir::tera::detail;

namespace {

int64_t indexVectorSize(RankedTensorType indices, int64_t indexVectorDim) {
  return indexVectorDim == indices.getRank()
             ? 1
             : indices.getDimSize(indexVectorDim);
}

SmallVector<int64_t> indexBatchAxes(RankedTensorType indices,
                                    int64_t indexVectorDim) {
  SmallVector<int64_t> axes;
  for (int64_t axis = 0; axis < indices.getRank(); ++axis)
    if (axis != indexVectorDim)
      axes.push_back(axis);
  return axes;
}

LogicalResult verifyIndexing(std::optional<Location> location,
                             RankedTensorType indices, int64_t indexVectorDim,
                             ArrayRef<int64_t> addressedAxes,
                             int64_t operandRank, StringRef what) {
  if (indexVectorDim < 0 || indexVectorDim > indices.getRank())
    return emitOptionalError(location, "index_vector_dim ", indexVectorDim,
                             " is out of range for indices of rank ",
                             indices.getRank());

  llvm::SmallBitVector addressed(operandRank);
  if (failed(markAxes(location, addressedAxes, operandRank, addressed, what)))
    return failure();

  int64_t coordinates = indexVectorSize(indices, indexVectorDim);
  if (!ShapedType::isDynamic(coordinates) &&
      static_cast<int64_t>(addressedAxes.size()) != coordinates)
    return emitOptionalError(location, "expects one ", what,
                             " entry per index coordinate: ", coordinates,
                             " expected, ", addressedAxes.size(), " given");
  return success();
}

} // namespace

//===----------------------------------------------------------------------===//
// GatherOp
//===----------------------------------------------------------------------===//

SmallVector<int64_t> GatherOp::getBatchAxes() {
  return indexBatchAxes(cast<RankedTensorType>(getIndices().getType()),
                        getIndexVectorDim());
}

LogicalResult
GatherOp::inferReturnTypes(MLIRContext *, std::optional<Location> location,
                           Adaptor adaptor,
                           SmallVectorImpl<Type> &inferredReturnTypes) {
  auto operandType = dyn_cast<RankedTensorType>(adaptor.getOperand().getType());
  auto indicesType = dyn_cast<RankedTensorType>(adaptor.getIndices().getType());
  if (!operandType || !indicesType)
    return emitOptionalError(location, "expects ranked operands");

  int64_t operandRank = operandType.getRank();
  int64_t indexVectorDim = adaptor.getIndexVectorDim();
  if (failed(verifyIndexing(location, indicesType, indexVectorDim,
                            adaptor.getStartIndexMap(), operandRank,
                            "start_index_map")))
    return failure();

  ArrayRef<int64_t> sliceSizes = adaptor.getSliceSizes();
  if (static_cast<int64_t>(sliceSizes.size()) != operandRank)
    return emitOptionalError(location,
                             "expects one slice size per operand axis: ",
                             operandRank, " expected, ", sliceSizes.size(),
                             " given");

  llvm::SmallBitVector collapsed(operandRank);
  if (failed(markAxes(location, adaptor.getCollapsedSliceDims(), operandRank,
                      collapsed, "collapsed_slice_dims")))
    return failure();

  for (int64_t axis = 0; axis < operandRank; ++axis) {
    if (sliceSizes[axis] < 0)
      return emitOptionalError(location, "slice size ", sliceSizes[axis],
                               " at axis ", axis, " is negative");
    if (collapsed.test(axis) && sliceSizes[axis] != 1)
      return emitOptionalError(location, "collapsed axis ", axis,
                               " has slice size ", sliceSizes[axis],
                               ", expected 1");
    int64_t extent = operandType.getDimSize(axis);
    if (!ShapedType::isDynamic(extent) && sliceSizes[axis] > extent)
      return emitOptionalError(location, "slice size ", sliceSizes[axis],
                               " at axis ", axis,
                               " exceeds the operand extent ", extent);
  }

  SmallVector<int64_t> batch;
  for (int64_t axis : indexBatchAxes(indicesType, indexVectorDim))
    batch.push_back(indicesType.getDimSize(axis));

  SmallVector<int64_t> offsets;
  for (int64_t axis = 0; axis < operandRank; ++axis)
    if (!collapsed.test(axis))
      offsets.push_back(sliceSizes[axis]);

  ArrayRef<int64_t> offsetDims = adaptor.getOffsetDims();
  if (offsetDims.size() != offsets.size())
    return emitOptionalError(
        location, "expects one offset dimension per surviving slice axis: ",
        offsets.size(), " expected, ", offsetDims.size(), " given");

  int64_t rank = batch.size() + offsets.size();
  llvm::SmallBitVector isOffset(rank);
  if (failed(markAxes(location, offsetDims, rank, isOffset, "offset_dims")))
    return failure();

  SmallVector<int64_t> shape;
  shape.reserve(rank);
  size_t nextBatch = 0, nextOffset = 0;
  for (int64_t axis = 0; axis < rank; ++axis)
    shape.push_back(isOffset.test(axis) ? offsets[nextOffset++]
                                        : batch[nextBatch++]);

  inferredReturnTypes.push_back(
      RankedTensorType::get(shape, operandType.getElementType()));
  return success();
}

//===----------------------------------------------------------------------===//
// ScatterOp
//===----------------------------------------------------------------------===//

SmallVector<int64_t> ScatterOp::getWindowSizes() {
  auto operandType = cast<RankedTensorType>(getOperand().getType());
  llvm::SmallBitVector inserted(operandType.getRank());
  for (int64_t axis : getInsertedWindowDims())
    inserted.set(axis);

  SmallVector<int64_t> sizes;
  for (auto [axis, extent] : llvm::enumerate(operandType.getShape()))
    sizes.push_back(inserted.test(axis) ? 1 : extent);
  return sizes;
}

LogicalResult
ScatterOp::inferReturnTypes(MLIRContext *, std::optional<Location> location,
                            Adaptor adaptor,
                            SmallVectorImpl<Type> &inferredReturnTypes) {
  auto operandType = dyn_cast<RankedTensorType>(adaptor.getOperand().getType());
  auto indicesType = dyn_cast<RankedTensorType>(adaptor.getIndices().getType());
  auto updatesType = dyn_cast<RankedTensorType>(adaptor.getUpdates().getType());
  if (!operandType || !indicesType || !updatesType)
    return emitOptionalError(location, "expects ranked operands");

  if (operandType.getElementType() != updatesType.getElementType())
    return emitOptionalError(location, "scatters ",
                             updatesType.getElementType(), " into ",
                             operandType.getElementType());

  int64_t operandRank = operandType.getRank();
  int64_t updatesRank = updatesType.getRank();
  int64_t indexVectorDim = adaptor.getIndexVectorDim();
  if (failed(verifyIndexing(location, indicesType, indexVectorDim,
                            adaptor.getScatterDimsToOperandDims(), operandRank,
                            "scatter_dims_to_operand_dims")))
    return failure();

  ArrayRef<int64_t> windowDims = adaptor.getUpdateWindowDims();
  llvm::SmallBitVector isWindow(updatesRank);
  if (failed(markAxes(location, windowDims, updatesRank, isWindow,
                      "update_window_dims")))
    return failure();

  ArrayRef<int64_t> insertedDims = adaptor.getInsertedWindowDims();
  llvm::SmallBitVector inserted(operandRank);
  if (failed(markAxes(location, insertedDims, operandRank, inserted,
                      "inserted_window_dims")))
    return failure();

  if (static_cast<int64_t>(windowDims.size() + insertedDims.size()) !=
      operandRank)
    return emitOptionalError(
        location,
        "expects one window axis per operand axis, inserted or not: ",
        operandRank, " expected, ", windowDims.size() + insertedDims.size(),
        " given");

  size_t batchAxes = indexBatchAxes(indicesType, indexVectorDim).size();
  if (updatesRank - static_cast<int64_t>(windowDims.size()) !=
      static_cast<int64_t>(batchAxes))
    return emitOptionalError(location, "expects ", batchAxes,
                             " update axes outside the window, one per index "
                             "batch axis, but the window leaves ",
                             updatesRank - windowDims.size());

  size_t nextWindow = 0;
  for (int64_t axis = 0; axis < operandRank; ++axis) {
    if (inserted.test(axis))
      continue;
    int64_t width = updatesType.getDimSize(windowDims[nextWindow++]);
    int64_t extent = operandType.getDimSize(axis);
    if (!ShapedType::isDynamic(width) && !ShapedType::isDynamic(extent) &&
        width > extent)
      return emitOptionalError(location, "window of width ", width,
                               " does not fit operand axis ", axis,
                               " of extent ", extent);
  }

  inferredReturnTypes.push_back(operandType);
  return success();
}

//===----------------------------------------------------------------------===//
// Vector-Jacobian products
//===----------------------------------------------------------------------===//

LogicalResult GatherOp::buildVjp(OpBuilder &builder, ValueRange adjoints,
                                 SmallVectorImpl<Value> &operandAdjoints) {
  auto operandType = cast<RankedTensorType>(getOperand().getType());
  Value zeros = createSplat(builder, getLoc(), operandType, 0.0, getOperand());
  operandAdjoints.assign(
      {ScatterOp::create(builder, getLoc(), zeros, getIndices(), adjoints[0],
                         getOffsetDims(), getCollapsedSliceDims(),
                         getStartIndexMap(), getIndexVectorDim()),
       Value()});
  return success();
}

LogicalResult ScatterOp::buildVjp(OpBuilder &builder, ValueRange adjoints,
                                  SmallVectorImpl<Value> &operandAdjoints) {
  auto operandType = cast<RankedTensorType>(getOperand().getType());
  auto updatesType = cast<RankedTensorType>(getUpdates().getType());
  SmallVector<int64_t> windowSizes = getWindowSizes();

  ArrayRef<int64_t> windowDims = getUpdateWindowDims();
  llvm::SmallBitVector inserted(operandType.getRank());
  for (int64_t axis : getInsertedWindowDims())
    inserted.set(axis);
  size_t nextWindow = 0;
  for (int64_t axis = 0; axis < operandType.getRank(); ++axis) {
    if (inserted.test(axis))
      continue;
    if (updatesType.getDimSize(windowDims[nextWindow++]) != windowSizes[axis])
      return emitOpError()
             << "cannot be differentiated: its window does not cover the whole "
                "of operand axis "
             << axis
             << ", so the adjoint of the updates is not a gather of the "
                "adjoint";
  }

  operandAdjoints.assign(
      {adjoints[0], Value(),
       GatherOp::create(builder, getLoc(), adjoints[0], getIndices(),
                        windowDims, getInsertedWindowDims(),
                        getScatterDimsToOperandDims(), windowSizes,
                        getIndexVectorDim())});
  return success();
}
