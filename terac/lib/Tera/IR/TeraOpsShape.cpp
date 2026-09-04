//===- TeraOpsShape.cpp - Shape and layout tera ops -------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Tera/IR/TeraOps.h"

#include "TeraOpsDetail.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/STLExtras.h"

using namespace mlir;
using namespace mlir::tera;
using namespace mlir::tera::detail;

SmallVector<Value> mlir::tera::detail::dynamicExtentsOf(OpBuilder &builder,
                                                        Location loc,
                                                        Value source) {
  auto type = cast<RankedTensorType>(source.getType());
  SmallVector<Value> extents;
  for (auto [axis, extent] : llvm::enumerate(type.getShape()))
    if (ShapedType::isDynamic(extent))
      extents.push_back(DimOp::create(builder, loc, source, axis));
  return extents;
}

//===----------------------------------------------------------------------===//
// DimOp
//===----------------------------------------------------------------------===//

LogicalResult DimOp::verify() {
  auto operandType = cast<RankedTensorType>(getOperand().getType());
  int64_t dimension = getDimension();
  if (dimension < 0 || dimension >= operandType.getRank())
    return emitOpError() << "dimension " << dimension
                         << " is out of range for rank "
                         << operandType.getRank();
  return success();
}

OpFoldResult DimOp::fold(FoldAdaptor) {
  auto operandType = cast<RankedTensorType>(getOperand().getType());
  int64_t extent = operandType.getDimSize(getDimension());
  if (ShapedType::isDynamic(extent))
    return {};
  return DenseElementsAttr::get(cast<ShapedType>(getResult().getType()),
                                extent);
}

//===----------------------------------------------------------------------===//
// BroadcastInDimOp
//===----------------------------------------------------------------------===//

LogicalResult BroadcastInDimOp::verify() {
  auto operandType = dyn_cast<RankedTensorType>(getOperand().getType());
  auto resultType = dyn_cast<RankedTensorType>(getResult().getType());
  if (!operandType || !resultType)
    return success();

  if (operandType.getElementType() != resultType.getElementType())
    return emitOpError() << "broadcasts " << operandType.getElementType()
                         << " to " << resultType.getElementType();

  if (failed(verifySizesClause(*this, getSizes())))
    return failure();

  ArrayRef<int64_t> dims = getBroadcastDimensions();
  if (static_cast<int64_t>(dims.size()) != operandType.getRank())
    return emitOpError() << "expects one broadcast dimension per operand axis: "
                         << operandType.getRank() << " expected, "
                         << dims.size() << " given";

  int64_t previous = -1;
  for (int64_t axis = 0; axis < static_cast<int64_t>(dims.size()); ++axis) {
    int64_t target = dims[axis];
    if (target < 0 || target >= resultType.getRank())
      return emitOpError() << "broadcast dimension " << target
                           << " is out of range for rank "
                           << resultType.getRank();
    if (target <= previous)
      return emitOpError()
             << "broadcast dimensions must be strictly increasing";
    previous = target;

    int64_t from = operandType.getDimSize(axis);
    int64_t to = resultType.getDimSize(target);
    if (!ShapedType::isDynamic(from) && from != 1 && !extentsAgree(from, to))
      return emitOpError() << "cannot broadcast extent " << from
                           << " at operand axis " << axis << " to " << to;
  }
  return success();
}

//===----------------------------------------------------------------------===//
// TransposeOp
//===----------------------------------------------------------------------===//

LogicalResult TransposeOp::inferReturnTypes(
    MLIRContext *, std::optional<Location> location, Adaptor adaptor,
    SmallVectorImpl<Type> &inferredReturnTypes) {
  auto operandType = dyn_cast<RankedTensorType>(adaptor.getOperand().getType());
  if (!operandType)
    return emitOptionalError(location, "expects a ranked operand");

  int64_t rank = operandType.getRank();
  ArrayRef<int64_t> permutation = adaptor.getPermutation();
  if (static_cast<int64_t>(permutation.size()) != rank)
    return emitOptionalError(location, "expects a permutation of length ", rank,
                             ", got ", permutation.size());

  llvm::SmallBitVector seen(rank);
  if (failed(markAxes(location, permutation, rank, seen, "permutation")))
    return failure();

  SmallVector<int64_t> shape;
  shape.reserve(rank);
  for (int64_t axis : permutation)
    shape.push_back(operandType.getDimSize(axis));

  inferredReturnTypes.push_back(
      RankedTensorType::get(shape, operandType.getElementType()));
  return success();
}

OpFoldResult TransposeOp::fold(FoldAdaptor) {
  if (llvm::is_sorted(getPermutation()) &&
      getOperand().getType() == getResult().getType())
    return getOperand();
  return {};
}

namespace {
/// transpose(transpose(x, inner), outer) -> transpose(x, inner o outer)
struct ComposeTransposes : public OpRewritePattern<TransposeOp> {
  using OpRewritePattern<TransposeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(TransposeOp op,
                                PatternRewriter &rewriter) const override {
    auto producer = op.getOperand().getDefiningOp<TransposeOp>();
    if (!producer)
      return failure();

    ArrayRef<int64_t> outer = op.getPermutation();
    ArrayRef<int64_t> inner = producer.getPermutation();
    if (outer.size() != inner.size())
      return failure();

    SmallVector<int64_t> composed;
    composed.reserve(outer.size());
    for (int64_t axis : outer)
      composed.push_back(inner[axis]);

    rewriter.replaceOpWithNewOp<TransposeOp>(
        op, op.getResult().getType(), producer.getOperand(),
        rewriter.getDenseI64ArrayAttr(composed));
    return success();
  }
};
} // namespace

void TransposeOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                              MLIRContext *context) {
  results.add<ComposeTransposes>(context);
}

//===----------------------------------------------------------------------===//
// ReshapeOp
//===----------------------------------------------------------------------===//

LogicalResult ReshapeOp::verify() {
  if (failed(verifySizesClause(*this, getSizes())))
    return failure();
  auto operandType = dyn_cast<RankedTensorType>(getOperand().getType());
  auto resultType = dyn_cast<RankedTensorType>(getResult().getType());
  if (!operandType || !resultType)
    return success();
  if (operandType.getElementType() != resultType.getElementType())
    return emitOpError() << "reshapes " << operandType.getElementType()
                         << " into " << resultType.getElementType();
  if (!operandType.hasStaticShape() || !resultType.hasStaticShape())
    return success();

  if (operandType.getNumElements() != resultType.getNumElements())
    return emitOpError() << "changes the element count from "
                         << operandType.getNumElements() << " to "
                         << resultType.getNumElements();
  return success();
}

OpFoldResult ReshapeOp::fold(FoldAdaptor) {
  if (getOperand().getType() == getResult().getType())
    return getOperand();
  return {};
}

namespace {
/// reshape(reshape(x)) -> reshape(x)
struct ComposeReshapes : public OpRewritePattern<ReshapeOp> {
  using OpRewritePattern<ReshapeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ReshapeOp op,
                                PatternRewriter &rewriter) const override {
    auto producer = op.getOperand().getDefiningOp<ReshapeOp>();
    if (!producer)
      return failure();
    rewriter.replaceOpWithNewOp<ReshapeOp>(op, op.getResult().getType(),
                                           producer.getOperand());
    return success();
  }
};
} // namespace

void ReshapeOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                            MLIRContext *context) {
  results.add<ComposeReshapes>(context);
}

//===----------------------------------------------------------------------===//
// SliceOp
//===----------------------------------------------------------------------===//

LogicalResult
SliceOp::inferReturnTypes(MLIRContext *, std::optional<Location> location,
                          Adaptor adaptor,
                          SmallVectorImpl<Type> &inferredReturnTypes) {
  auto operandType = dyn_cast<RankedTensorType>(adaptor.getOperand().getType());
  if (!operandType)
    return emitOptionalError(location, "expects a ranked operand");

  int64_t rank = operandType.getRank();
  ArrayRef<int64_t> starts = adaptor.getStartIndices();
  ArrayRef<int64_t> limits = adaptor.getLimitIndices();
  ArrayRef<int64_t> strides = adaptor.getStrides();

  if (static_cast<int64_t>(starts.size()) != rank ||
      static_cast<int64_t>(limits.size()) != rank ||
      static_cast<int64_t>(strides.size()) != rank)
    return emitOptionalError(location,
                             "expects start_indices, limit_indices and strides "
                             "to each have ",
                             rank, " entries");

  SmallVector<int64_t> shape;
  shape.reserve(rank);
  for (int64_t axis = 0; axis < rank; ++axis) {
    if (strides[axis] <= 0)
      return emitOptionalError(location, "stride ", strides[axis], " at axis ",
                               axis, " must be positive");
    if (starts[axis] < 0 || limits[axis] < starts[axis])
      return emitOptionalError(location, "axis ", axis,
                               " has an empty or inverted range [",
                               starts[axis], ", ", limits[axis], ")");

    int64_t extent = operandType.getDimSize(axis);
    if (!ShapedType::isDynamic(extent) && limits[axis] > extent)
      return emitOptionalError(location, "limit ", limits[axis], " at axis ",
                               axis, " exceeds the operand extent ", extent);

    shape.push_back((limits[axis] - starts[axis] + strides[axis] - 1) /
                    strides[axis]);
  }

  inferredReturnTypes.push_back(
      RankedTensorType::get(shape, operandType.getElementType()));
  return success();
}

//===----------------------------------------------------------------------===//
// ReverseOp
//===----------------------------------------------------------------------===//

LogicalResult
ReverseOp::inferReturnTypes(MLIRContext *, std::optional<Location> location,
                            Adaptor adaptor,
                            SmallVectorImpl<Type> &inferredReturnTypes) {
  auto operandType = dyn_cast<RankedTensorType>(adaptor.getOperand().getType());
  if (!operandType)
    return emitOptionalError(location, "expects a ranked operand");

  llvm::SmallBitVector seen(operandType.getRank());
  if (failed(markAxes(location, adaptor.getDimensions(), operandType.getRank(),
                      seen, "dimensions")))
    return failure();

  inferredReturnTypes.push_back(operandType);
  return success();
}

OpFoldResult ReverseOp::fold(FoldAdaptor) {
  auto operandType = cast<RankedTensorType>(getOperand().getType());
  for (int64_t axis : getDimensions())
    if (operandType.getDimSize(axis) != 1)
      return {};
  return getOperand();
}

//===----------------------------------------------------------------------===//
// PadOp
//===----------------------------------------------------------------------===//

LogicalResult
PadOp::inferReturnTypes(MLIRContext *, std::optional<Location> location,
                        Adaptor adaptor,
                        SmallVectorImpl<Type> &inferredReturnTypes) {
  auto operandType = dyn_cast<RankedTensorType>(adaptor.getOperand().getType());
  auto valueType =
      dyn_cast<RankedTensorType>(adaptor.getPaddingValue().getType());
  if (!operandType || !valueType)
    return emitOptionalError(location, "expects ranked operands");
  if (valueType.getRank() != 0)
    return emitOptionalError(location, "pads with a rank-", valueType.getRank(),
                             " tensor, which is not a single value");
  if (valueType.getElementType() != operandType.getElementType())
    return emitOptionalError(location, "pads ", operandType.getElementType(),
                             " with ", valueType.getElementType());

  int64_t rank = operandType.getRank();
  ArrayRef<int64_t> low = adaptor.getLow();
  ArrayRef<int64_t> high = adaptor.getHigh();
  ArrayRef<int64_t> interior = adaptor.getInterior().value_or(ArrayRef<int64_t>{});
  if (static_cast<int64_t>(low.size()) != rank ||
      static_cast<int64_t>(high.size()) != rank ||
      (!interior.empty() && static_cast<int64_t>(interior.size()) != rank))
    return emitOptionalError(
        location, "expects low, high and any interior to each have ", rank,
        " entries");

  SmallVector<int64_t> shape;
  shape.reserve(rank);
  for (int64_t axis = 0; axis < rank; ++axis) {
    int64_t holes = interior.empty() ? 0 : interior[axis];
    if (low[axis] < 0 || high[axis] < 0 || holes < 0)
      return emitOptionalError(location, "pads axis ", axis,
                               " by a negative amount; cropping is a slice");
    int64_t extent = operandType.getDimSize(axis);
    if (ShapedType::isDynamic(extent)) {
      shape.push_back(ShapedType::kDynamic);
      continue;
    }
    shape.push_back(low[axis] + extent + (extent - 1) * holes + high[axis]);
  }

  inferredReturnTypes.push_back(
      RankedTensorType::get(shape, operandType.getElementType()));
  return success();
}

SmallVector<int64_t> PadOp::getSpacing() {
  int64_t rank = cast<RankedTensorType>(getOperand().getType()).getRank();
  std::optional<ArrayRef<int64_t>> interior = getInterior();
  if (!interior)
    return SmallVector<int64_t>(rank, 1);
  return llvm::map_to_vector(*interior,
                             [](int64_t holes) { return holes + 1; });
}

OpFoldResult PadOp::fold(FoldAdaptor) {
  if (getOperand().getType() == getResult().getType())
    return getOperand();
  return {};
}

//===----------------------------------------------------------------------===//
// ConcatOp
//===----------------------------------------------------------------------===//

LogicalResult
ConcatOp::inferReturnTypes(MLIRContext *, std::optional<Location> location,
                           Adaptor adaptor,
                           SmallVectorImpl<Type> &inferredReturnTypes) {
  ValueRange inputs = adaptor.getInputs();
  if (inputs.empty())
    return emitOptionalError(location, "expects at least one input");

  auto firstType = dyn_cast<RankedTensorType>(inputs.front().getType());
  if (!firstType)
    return emitOptionalError(location, "expects ranked inputs");

  int64_t rank = firstType.getRank();
  int64_t dimension = adaptor.getDimension();
  if (dimension < 0 || dimension >= rank)
    return emitOptionalError(location, "concat dimension ", dimension,
                             " is out of range for rank ", rank);

  SmallVector<int64_t> shape(firstType.getShape());
  int64_t total = 0;
  bool totalIsStatic = true;

  for (auto [index, input] : llvm::enumerate(inputs)) {
    auto inputType = dyn_cast<RankedTensorType>(input.getType());
    if (!inputType)
      return emitOptionalError(location, "expects ranked inputs");
    if (inputType.getRank() != rank)
      return emitOptionalError(location, "input ", index, " has rank ",
                               inputType.getRank(), ", expected ", rank);
    if (inputType.getElementType() != firstType.getElementType())
      return emitOptionalError(location, "input ", index,
                               " has a different element type from input 0");

    for (int64_t axis = 0; axis < rank; ++axis) {
      if (axis == dimension)
        continue;
      if (!extentsAgree(inputType.getDimSize(axis), shape[axis]))
        return emitOptionalError(location, "input ", index,
                                 " disagrees with input 0 at axis ", axis);
      if (ShapedType::isDynamic(shape[axis]))
        shape[axis] = inputType.getDimSize(axis);
    }

    int64_t extent = inputType.getDimSize(dimension);
    if (ShapedType::isDynamic(extent))
      totalIsStatic = false;
    else
      total += extent;
  }

  shape[dimension] = totalIsStatic ? total : ShapedType::kDynamic;
  inferredReturnTypes.push_back(
      RankedTensorType::get(shape, firstType.getElementType()));
  return success();
}

//===----------------------------------------------------------------------===//
// Vector-Jacobian products
//===----------------------------------------------------------------------===//

LogicalResult
BroadcastInDimOp::buildVjp(OpBuilder &builder, ValueRange adjoints,
                           SmallVectorImpl<Value> &operandAdjoints) {
  auto operandType = cast<RankedTensorType>(getOperand().getType());
  auto resultType = cast<RankedTensorType>(getType());

  SmallVector<int64_t> source(resultType.getRank(), -1);
  for (auto [operandAxis, resultAxis] :
       llvm::enumerate(getBroadcastDimensions()))
    source[resultAxis] = operandAxis;

  SmallVector<int64_t> copied;
  for (int64_t axis = 0; axis < resultType.getRank(); ++axis) {
    int64_t operandAxis = source[axis];
    if (operandAxis < 0 || (operandType.getDimSize(operandAxis) == 1 &&
                            resultType.getDimSize(axis) != 1))
      copied.push_back(axis);
  }

  Value gathered =
      ReduceOp::create(builder, getLoc(), adjoints[0], ReduceKind::Sum, copied);
  operandAdjoints.assign(
      {ReshapeOp::create(builder, getLoc(), operandType, gathered)});
  operandAdjoints.append(getSizes().size(), Value());
  return success();
}

LogicalResult TransposeOp::buildVjp(OpBuilder &builder, ValueRange adjoints,
                                    SmallVectorImpl<Value> &operandAdjoints) {
  operandAdjoints.assign(
      {TransposeOp::create(builder, getLoc(), adjoints[0],
                           invertPermutationVector(getPermutation()))});
  return success();
}

LogicalResult ReshapeOp::buildVjp(OpBuilder &builder, ValueRange adjoints,
                                  SmallVectorImpl<Value> &operandAdjoints) {
  operandAdjoints.assign({ReshapeOp::create(
      builder, getLoc(), getOperand().getType(), adjoints[0])});
  return success();
}

LogicalResult SliceOp::buildVjp(OpBuilder &builder, ValueRange adjoints,
                                SmallVectorImpl<Value> &operandAdjoints) {
  auto operandType = cast<RankedTensorType>(getOperand().getType());
  auto resultType = cast<RankedTensorType>(getType());
  ArrayRef<int64_t> starts = getStartIndices();
  ArrayRef<int64_t> strides = getStrides();
  int64_t rank = operandType.getRank();

  SmallVector<int64_t> low(starts);
  SmallVector<int64_t> high(rank);
  SmallVector<int64_t> interior(rank);
  for (int64_t axis = 0; axis < rank; ++axis) {
    interior[axis] = strides[axis] - 1;
    high[axis] = operandType.getDimSize(axis) - starts[axis] -
                 (resultType.getDimSize(axis) - 1) * strides[axis] - 1;
  }

  Value zero = createSplat(
      builder, getLoc(),
      RankedTensorType::get({}, operandType.getElementType()), 0.0);
  operandAdjoints.assign(
      {PadOp::create(builder, getLoc(), adjoints[0], zero, low, high,
                     builder.getDenseI64ArrayAttr(interior))});
  return success();
}

LogicalResult ReverseOp::buildVjp(OpBuilder &builder, ValueRange adjoints,
                                  SmallVectorImpl<Value> &operandAdjoints) {
  operandAdjoints.assign(
      {ReverseOp::create(builder, getLoc(), adjoints[0], getDimensions())});
  return success();
}

/// Computes the operand gradient only; the padding value receives no gradient.
LogicalResult PadOp::buildVjp(OpBuilder &builder, ValueRange adjoints,
                              SmallVectorImpl<Value> &operandAdjoints) {
  auto operandType = cast<RankedTensorType>(getOperand().getType());
  int64_t rank = operandType.getRank();
  ArrayRef<int64_t> low = getLow();

  SmallVector<int64_t> strides = getSpacing();
  SmallVector<int64_t> limits(rank);
  for (int64_t axis = 0; axis < rank; ++axis)
    limits[axis] =
        low[axis] + (operandType.getDimSize(axis) - 1) * strides[axis] + 1;

  operandAdjoints.assign({SliceOp::create(builder, getLoc(), adjoints[0], low,
                                          limits, strides),
                          Value()});
  return success();
}

LogicalResult ConcatOp::buildVjp(OpBuilder &builder, ValueRange adjoints,
                                 SmallVectorImpl<Value> &operandAdjoints) {
  auto resultType = cast<RankedTensorType>(getType());
  int64_t dimension = getDimension();
  SmallVector<int64_t> starts(resultType.getRank(), 0);
  SmallVector<int64_t> limits(resultType.getShape());
  SmallVector<int64_t> strides(resultType.getRank(), 1);

  int64_t offset = 0;
  for (Value input : getInputs()) {
    starts[dimension] = offset;
    offset += cast<RankedTensorType>(input.getType()).getDimSize(dimension);
    limits[dimension] = offset;
    operandAdjoints.push_back(SliceOp::create(builder, getLoc(), adjoints[0],
                                              starts, limits, strides));
  }
  return success();
}
