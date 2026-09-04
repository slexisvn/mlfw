//===- TeraOpsWindow.cpp - Ops that slide a window --------------*- C++ -*-===//
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
constexpr int64_t kLeadingAxes = 2;

int64_t windowCount(int64_t extent, int64_t low, int64_t high, int64_t reach,
                    int64_t stride, bool ceilMode) {
  if (ShapedType::isDynamic(extent))
    return ShapedType::kDynamic;
  int64_t span = extent + low + high - reach;
  if (span < 0)
    return 0;
  return (ceilMode ? (span + stride - 1) / stride : span / stride) + 1;
}

SmallVector<int64_t> paddingSide(ArrayRef<int64_t> padding, int64_t side) {
  SmallVector<int64_t> half;
  for (size_t axis = 0; axis * 2 + side < padding.size(); ++axis)
    half.push_back(padding[axis * 2 + side]);
  return half;
}

LogicalResult verifyWindow(std::optional<Location> location,
                           int64_t spatialRank, ArrayRef<int64_t> strides,
                           ArrayRef<int64_t> padding,
                           ArrayRef<int64_t> dilation) {
  if (static_cast<int64_t>(strides.size()) != spatialRank ||
      static_cast<int64_t>(dilation.size()) != spatialRank)
    return emitOptionalError(location,
                             "expects one stride and one dilation per spatial "
                             "axis: ",
                             spatialRank, " expected, ", strides.size(),
                             " strides and ", dilation.size(),
                             " dilations given");
  if (static_cast<int64_t>(padding.size()) != spatialRank * 2)
    return emitOptionalError(location,
                             "expects a low and a high padding per spatial "
                             "axis: ",
                             spatialRank * 2, " expected, ", padding.size(),
                             " given");
  for (int64_t axis = 0; axis < spatialRank; ++axis) {
    if (strides[axis] < 1 || dilation[axis] < 1)
      return emitOptionalError(location, "stride ", strides[axis],
                               " and dilation ", dilation[axis], " at axis ",
                               axis, " must both be positive");
    if (padding[axis * 2] < 0 || padding[axis * 2 + 1] < 0)
      return emitOptionalError(location, "pads axis ", axis,
                               " by a negative amount; cropping is a slice");
  }
  return success();
}

}

int64_t ConvOp::getSpatialRank() { return getStrides().size(); }

static int64_t dilatedReach(int64_t width, int64_t dilation) {
  return ShapedType::isDynamic(width) ? ShapedType::kDynamic
                                      : (width - 1) * dilation + 1;
}

SmallVector<int64_t> ConvOp::getReach() {
  auto kernelType = cast<RankedTensorType>(getKernel().getType());
  SmallVector<int64_t> reach;
  for (auto [axis, dilation] : llvm::enumerate(getDilation()))
    reach.push_back(
        dilatedReach(kernelType.getDimSize(axis + kLeadingAxes), dilation));
  return reach;
}

SmallVector<int64_t> ConvOp::getPaddingLow() {
  return paddingSide(getPadding(), 0);
}

SmallVector<int64_t> ConvOp::getPaddingHigh() {
  return paddingSide(getPadding(), 1);
}

LogicalResult
ConvOp::inferReturnTypes(MLIRContext *, std::optional<Location> location,
                         Adaptor adaptor,
                         SmallVectorImpl<Type> &inferredReturnTypes) {
  auto inputType = dyn_cast<RankedTensorType>(adaptor.getInput().getType());
  auto kernelType = dyn_cast<RankedTensorType>(adaptor.getKernel().getType());
  if (!inputType || !kernelType)
    return emitOptionalError(location, "expects ranked operands");
  if (inputType.getElementType() != kernelType.getElementType())
    return emitOptionalError(location, "convolves ", inputType.getElementType(),
                             " with ", kernelType.getElementType());

  ArrayRef<int64_t> strides = adaptor.getStrides();
  ArrayRef<int64_t> padding = adaptor.getPadding();
  ArrayRef<int64_t> dilation = adaptor.getDilation();
  int64_t spatialRank = strides.size();
  if (failed(verifyWindow(location, spatialRank, strides, padding, dilation)))
    return failure();

  if (inputType.getRank() != spatialRank + kLeadingAxes ||
      kernelType.getRank() != spatialRank + kLeadingAxes)
    return emitOptionalError(location, "expects a batch axis, a channel axis "
                                       "and ",
                             spatialRank, " spatial axes on both operands, "
                                          "but they have rank ",
                             inputType.getRank(), " and ",
                             kernelType.getRank());

  int64_t groups = adaptor.getGroups();
  if (groups < 1)
    return emitOptionalError(location, "expects at least one group, got ",
                             groups);

  int64_t inChannels = inputType.getDimSize(1);
  int64_t outChannels = kernelType.getDimSize(0);
  if (!ShapedType::isDynamic(outChannels) && outChannels % groups != 0)
    return emitOptionalError(location, "splits ", outChannels,
                             " output channels into ", groups, " groups");
  if (!ShapedType::isDynamic(inChannels)) {
    if (inChannels % groups != 0)
      return emitOptionalError(location, "splits ", inChannels,
                               " input channels into ", groups, " groups");
    int64_t perGroup = kernelType.getDimSize(1);
    if (!ShapedType::isDynamic(perGroup) && perGroup != inChannels / groups)
      return emitOptionalError(location, "reads ", perGroup,
                               " input channels per group, but ", groups,
                               " groups of ", inChannels, " is ",
                               inChannels / groups);
  }

  SmallVector<int64_t> shape{inputType.getDimSize(0), outChannels};
  for (int64_t axis = 0; axis < spatialRank; ++axis) {
    int64_t reach = dilatedReach(kernelType.getDimSize(axis + kLeadingAxes),
                                 dilation[axis]);
    shape.push_back(windowCount(
        ShapedType::isDynamic(reach) ? ShapedType::kDynamic
                                     : inputType.getDimSize(axis + kLeadingAxes),
        padding[axis * 2], padding[axis * 2 + 1], reach, strides[axis],
        /*ceilMode=*/false));
  }

  inferredReturnTypes.push_back(
      RankedTensorType::get(shape, inputType.getElementType()));
  return success();
}

SmallVector<int64_t> Pool2dOp::getPaddingLow() {
  return paddingSide(getPadding(), 0);
}

SmallVector<int64_t> Pool2dOp::getPaddingHigh() {
  return paddingSide(getPadding(), 1);
}

LogicalResult
Pool2dOp::inferReturnTypes(MLIRContext *, std::optional<Location> location,
                           Adaptor adaptor,
                           SmallVectorImpl<Type> &inferredReturnTypes) {
  auto operandType = dyn_cast<RankedTensorType>(adaptor.getOperand().getType());
  if (!operandType)
    return emitOptionalError(location, "expects a ranked operand");
  if (operandType.getRank() != 2 + kLeadingAxes)
    return emitOptionalError(location,
                             "pools a batch, a channel and two spatial axes, "
                             "so expects rank 4, got ",
                             operandType.getRank());

  ArrayRef<int64_t> window = adaptor.getKernelSize();
  ArrayRef<int64_t> strides = adaptor.getStrides();
  ArrayRef<int64_t> padding = adaptor.getPadding();
  SmallVector<int64_t> dilation(2, 1);
  if (window.size() != 2)
    return emitOptionalError(location, "expects a two-axis window, got ",
                             window.size(), " extents");
  if (failed(verifyWindow(location, 2, strides, padding, dilation)))
    return failure();

  SmallVector<int64_t> shape{operandType.getDimSize(0),
                             operandType.getDimSize(1)};
  for (int64_t axis = 0; axis < 2; ++axis) {
    if (window[axis] < 1)
      return emitOptionalError(location, "window extent ", window[axis],
                               " at axis ", axis, " must be positive");
    shape.push_back(windowCount(operandType.getDimSize(axis + kLeadingAxes),
                                padding[axis * 2], padding[axis * 2 + 1],
                                window[axis], strides[axis],
                                adaptor.getCeilMode()));
  }

  inferredReturnTypes.push_back(
      RankedTensorType::get(shape, operandType.getElementType()));
  return success();
}

namespace {
struct ConvAdjoints {
  Value input;
  Value kernel;
};

SmallVector<int64_t> swapLeading(int64_t rank) {
  SmallVector<int64_t> order{1, 0};
  for (int64_t axis = kLeadingAxes; axis < rank; ++axis)
    order.push_back(axis);
  return order;
}

Value channelBand(OpBuilder &builder, Location loc, Value value, int64_t axis,
                  int64_t from, int64_t to) {
  auto type = cast<RankedTensorType>(value.getType());
  int64_t rank = type.getRank();
  SmallVector<int64_t> starts(rank, 0);
  SmallVector<int64_t> limits(type.getShape());
  SmallVector<int64_t> strides(rank, 1);
  starts[axis] = from;
  limits[axis] = to;
  return SliceOp::create(builder, loc, value, starts, limits, strides);
}

FailureOr<ConvAdjoints> convGroupVjp(ConvOp op, OpBuilder &builder, Value grad,
                                     Value input, Value kernel) {
  Location loc = op.getLoc();
  auto inputType = cast<RankedTensorType>(input.getType());
  auto kernelType = cast<RankedTensorType>(kernel.getType());
  auto gradType = cast<RankedTensorType>(grad.getType());
  int64_t rank = inputType.getRank();
  int64_t spatialRank = op.getSpatialRank();
  ArrayRef<int64_t> strides = op.getStrides();
  ArrayRef<int64_t> dilation = op.getDilation();
  SmallVector<int64_t> low = op.getPaddingLow();
  SmallVector<int64_t> high = op.getPaddingHigh();

  for (int64_t axis = kLeadingAxes; axis < rank; ++axis)
    if (ShapedType::isDynamic(inputType.getDimSize(axis)) ||
        ShapedType::isDynamic(kernelType.getDimSize(axis)) ||
        ShapedType::isDynamic(gradType.getDimSize(axis)))
      return op.emitOpError()
             << "cannot be differentiated over a dynamic spatial axis: the "
                "padding that puts the adjoint back where it was read from is "
                "arithmetic on extents that are not known here";

  Value zero = createSplat(
      builder, loc, RankedTensorType::get({}, inputType.getElementType()), 0.0);

  Value dilated = grad;
  if (llvm::any_of(strides, [](int64_t stride) { return stride > 1; })) {
    SmallVector<int64_t> holes(rank, 0);
    for (int64_t axis = 0; axis < spatialRank; ++axis)
      holes[axis + kLeadingAxes] = strides[axis] - 1;
    dilated = PadOp::create(builder, loc, grad, zero,
                            SmallVector<int64_t>(rank, 0),
                            SmallVector<int64_t>(rank, 0),
                            builder.getDenseI64ArrayAttr(holes));
  }

  SmallVector<int64_t> swap = swapLeading(rank);
  SmallVector<int64_t> spatialAxes = axisRange(kLeadingAxes, rank);
  Value swapped = TransposeOp::create(builder, loc, kernel, swap);
  Value flipped = ReverseOp::create(builder, loc, swapped, spatialAxes);

  SmallVector<int64_t> reach = op.getReach();
  SmallVector<int64_t> inputPadding;
  for (int64_t axis = 0; axis < spatialRank; ++axis) {
    int64_t trailing =
        inputType.getDimSize(axis + kLeadingAxes) + low[axis] + high[axis] -
        ((gradType.getDimSize(axis + kLeadingAxes) - 1) * strides[axis] +
         reach[axis]);
    inputPadding.push_back(reach[axis] - 1 - low[axis]);
    inputPadding.push_back(reach[axis] - 1 - high[axis] + trailing);
    if (inputPadding[axis * 2] < 0 || inputPadding[axis * 2 + 1] < 0)
      return op.emitOpError()
             << "cannot be differentiated at axis " << axis
             << ": it pads by more than the kernel reaches, so running the "
                "adjoint back through the kernel would have to crop rather "
                "than pad";
  }

  Value dInput = ConvOp::create(builder, loc, dilated, flipped,
                                SmallVector<int64_t>(spatialRank, 1),
                                inputPadding, dilation, 1);

  SmallVector<int64_t> weightPadding;
  for (int64_t axis = 0; axis < spatialRank; ++axis) {
    weightPadding.push_back(low[axis]);
    weightPadding.push_back(high[axis]);
  }
  Value inputAsKernel = TransposeOp::create(builder, loc, input, swap);
  Value gradAsKernel = TransposeOp::create(builder, loc, grad, swap);
  Value wide = ConvOp::create(builder, loc, inputAsKernel, gradAsKernel,
                              dilation, weightPadding, strides, 1);

  auto wideType = cast<RankedTensorType>(wide.getType());
  SmallVector<int64_t> limits(wideType.getShape());
  bool trimmed = false;
  for (int64_t axis = 0; axis < spatialRank; ++axis) {
    int64_t width = kernelType.getDimSize(axis + kLeadingAxes);
    trimmed |= limits[axis + kLeadingAxes] != width;
    limits[axis + kLeadingAxes] = width;
  }
  if (trimmed)
    wide = SliceOp::create(builder, loc, wide,
                           SmallVector<int64_t>(rank, 0), limits,
                           SmallVector<int64_t>(rank, 1));

  Value dKernel = TransposeOp::create(builder, loc, wide, swap);
  return ConvAdjoints{dInput, dKernel};
}

}

LogicalResult ConvOp::buildVjp(OpBuilder &builder, ValueRange adjoints,
                               SmallVectorImpl<Value> &operandAdjoints) {
  int64_t groups = getGroups();
  if (groups == 1) {
    FailureOr<ConvAdjoints> group =
        convGroupVjp(*this, builder, adjoints[0], getInput(), getKernel());
    if (failed(group))
      return failure();
    operandAdjoints.assign({group->input, group->kernel});
    return success();
  }

  Location loc = getLoc();
  auto inputType = cast<RankedTensorType>(getInput().getType());
  auto resultType = cast<RankedTensorType>(getType());
  int64_t inPerGroup = inputType.getDimSize(1) / groups;
  int64_t outPerGroup = resultType.getDimSize(1) / groups;

  SmallVector<Value> inputBands, kernelBands;
  for (int64_t group = 0; group < groups; ++group) {
    FailureOr<ConvAdjoints> band = convGroupVjp(
        *this, builder,
        channelBand(builder, loc, adjoints[0], 1, group * outPerGroup,
                    (group + 1) * outPerGroup),
        channelBand(builder, loc, getInput(), 1, group * inPerGroup,
                    (group + 1) * inPerGroup),
        channelBand(builder, loc, getKernel(), 0, group * outPerGroup,
                    (group + 1) * outPerGroup));
    if (failed(band))
      return failure();
    inputBands.push_back(band->input);
    kernelBands.push_back(band->kernel);
  }

  operandAdjoints.assign({ConcatOp::create(builder, loc, inputBands, 1),
                          ConcatOp::create(builder, loc, kernelBands, 0)});
  return success();
}

LogicalResult Pool2dOp::buildVjp(OpBuilder &builder, ValueRange adjoints,
                                 SmallVectorImpl<Value> &operandAdjoints) {
  Location loc = getLoc();
  auto operandType = cast<RankedTensorType>(getOperand().getType());
  auto resultType = cast<RankedTensorType>(getType());
  ArrayRef<int64_t> window = getKernelSize();
  ArrayRef<int64_t> strides = getStrides();

  if (llvm::any_of(getPadding(), [](int64_t pad) { return pad != 0; }) ||
      getCeilMode())
    return emitOpError() << "cannot be differentiated with padding or a window "
                            "that hangs over the edge: a position with no "
                            "element under it has nothing to take a share";

  SmallVector<int64_t> split{operandType.getDimSize(0),
                             operandType.getDimSize(1)};
  SmallVector<int64_t> spread{0, 1};
  for (int64_t axis = 0; axis < 2; ++axis) {
    if (strides[axis] != window[axis])
      return emitOpError() << "cannot be differentiated with windows that "
                              "overlap or leave gaps: at axis "
                           << axis << " the window is " << window[axis]
                           << " wide and the stride is " << strides[axis];
    spread.push_back(split.size());
    split.push_back(resultType.getDimSize(axis + 2));
    split.push_back(window[axis]);
  }

  auto splitType = RankedTensorType::get(split, operandType.getElementType());
  auto scatter = [&](Value value) -> Value {
    Value apart = BroadcastInDimOp::create(builder, loc, splitType, value,
                                           ValueRange{}, spread);
    return ReshapeOp::create(builder, loc, operandType, apart);
  };

  Value shared = scatter(adjoints[0]);
  if (getKind() == PoolKind::Average) {
    Value count = createSplat(builder, loc, operandType,
                              static_cast<double>(window[0] * window[1]));
    operandAdjoints.assign({DivOp::create(builder, loc, shared, count)});
    return success();
  }

  Value spreadResult = scatter(getResult());
  Value chosen = CompareOp::create(builder, loc, getOperand(), spreadResult,
                                   ComparisonDirection::Eq);
  Value zero = createSplat(builder, loc, operandType, 0.0);
  operandAdjoints.assign(
      {SelectOp::create(builder, loc, chosen, shared, zero)});
  return success();
}
