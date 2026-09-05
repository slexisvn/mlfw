//===- TeraOpsControlFlow.cpp - Region-carrying tera ops --------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Tera/IR/TeraOps.h"

#include "Tera/IR/TeraAutodiff.h"

#include "TeraOpsDetail.h"
#include "mlir/IR/BuiltinTypes.h"

using namespace mlir;
using namespace mlir::tera;
using namespace mlir::tera::detail;

namespace {
LogicalResult verifyBody(Operation *op, Block &body, TypeRange arguments,
                         TypeRange yielded, StringRef what) {
  if (body.getNumArguments() != arguments.size())
    return op->emitOpError() << what << " takes " << body.getNumArguments()
                             << " arguments, expected " << arguments.size();
  for (auto [index, given] : llvm::enumerate(body.getArgumentTypes()))
    if (given != arguments[index])
      return op->emitOpError() << what << " argument " << index << " is "
                               << given << ", expected " << arguments[index];

  if (body.empty())
    return success();
  auto terminator = dyn_cast<YieldOp>(body.back());
  if (!terminator)
    return success();
  if (terminator.getResults().size() != yielded.size())
    return op->emitOpError()
           << what << " yields " << terminator.getResults().size()
           << " values, expected " << yielded.size();
  for (auto [index, given] :
       llvm::enumerate(terminator.getResults().getTypes()))
    if (given != yielded[index])
      return op->emitOpError() << what << " yields " << given << " at " << index
                               << ", expected " << yielded[index];
  return success();
}

}

LogicalResult IfOp::verify() {
  auto conditionType = dyn_cast<RankedTensorType>(getCondition().getType());
  if (!conditionType || conditionType.getRank() != 0)
    return emitOpError() << "expects a rank-0 condition, got "
                         << getCondition().getType();

  TypeRange arguments = getInputs().getTypes();
  TypeRange results = getResults().getTypes();
  if (failed(verifyBody(*this, getThenBody().front(), arguments, results,
                        "the then body")))
    return failure();
  return verifyBody(*this, getElseBody().front(), arguments, results,
                    "the else body");
}

RankedTensorType ScanOp::getSliceType(RankedTensorType type) {
  return RankedTensorType::get(type.getShape().drop_front(),
                               type.getElementType());
}

int64_t ScanOp::getTripCount() {
  return cast<RankedTensorType>(getInputs().front().getType()).getDimSize(0);
}

LogicalResult ScanOp::verifyLoweringToLinalg() {
  if (ShapedType::isDynamic(getTripCount()))
    return emitError() << "cannot be lowered to linalg with a dynamic step "
                          "axis: the trip count would have to be read at run "
                          "time";
  return success();
}

SmallVector<Type> ScanOp::getBodyArgumentTypes() {
  SmallVector<Type> arguments(getInits().getTypes());
  for (Value input : getInputs())
    arguments.push_back(getSliceType(cast<RankedTensorType>(input.getType())));
  arguments.append(getConstants().getTypes().begin(),
                   getConstants().getTypes().end());
  return arguments;
}

LogicalResult ScanOp::verify() {
  if (getInputs().empty())
    return emitOpError()
           << "expects at least one input, which is what sets the trip count";

  int64_t steps = 0;
  for (auto [index, input] : llvm::enumerate(getInputs())) {
    auto type = dyn_cast<RankedTensorType>(input.getType());
    if (!type || type.getRank() == 0)
      return emitOpError() << "input " << index << " is " << input.getType()
                           << ", which has no step axis";
    if (index == 0)
      steps = type.getDimSize(0);
    else if (!extentsAgree(type.getDimSize(0), steps))
      return emitOpError() << "input " << index << " runs for "
                           << type.getDimSize(0) << " steps, but input 0 runs "
                           << "for " << steps;
  }
  SmallVector<Type> arguments = getBodyArgumentTypes();

  if (getResults().size() < getInits().size())
    return emitOpError() << "returns " << getResults().size()
                         << " values, fewer than its " << getInits().size()
                         << " carries";
  for (auto [index, carry] : llvm::enumerate(getCarries().getTypes()))
    if (carry != getInits().getTypes()[index])
      return emitOpError() << "carry " << index << " leaves as " << carry
                           << " but entered as "
                           << getInits().getTypes()[index];

  SmallVector<Type> yielded(getInits().getTypes());
  for (auto [index, output] : llvm::enumerate(getOutputs())) {
    auto type = dyn_cast<RankedTensorType>(output.getType());
    if (!type || type.getRank() == 0)
      return emitOpError() << "output " << index << " is " << output.getType()
                           << ", which has no step axis";
    if (!extentsAgree(type.getDimSize(0), steps))
      return emitOpError() << "output " << index << " stacks "
                           << type.getDimSize(0) << " steps, but the scan runs "
                           << "for " << steps;
    yielded.push_back(getSliceType(type));
  }

  if (std::optional<int64_t> chunk = getCheckpoint())
    if (!ShapedType::isDynamic(steps) && steps % *chunk != 0)
      return emitOpError() << "checkpoints every " << *chunk
                           << " steps, which does not divide the " << steps
                           << " steps it runs for";

  if (failed(verifySizesClause(*this, getSizes())))
    return failure();

  return verifyBody(*this, getBody().front(), arguments, yielded, "the body");
}

MutableOperandRange
YieldOp::getMutableSuccessorOperands(RegionSuccessor) {
  if (auto scan = dyn_cast<ScanOp>((*this)->getParentOp()))
    return getResultsMutable().slice(0, scan.getInits().size());
  return getResultsMutable();
}

void IfOp::getSuccessorRegions(RegionBranchPoint point,
                               SmallVectorImpl<RegionSuccessor> &regions) {
  if (point.isParent()) {
    regions.push_back(RegionSuccessor(&getThenBody()));
    regions.push_back(RegionSuccessor(&getElseBody()));
    return;
  }
  regions.push_back(RegionSuccessor(getOperation()));
}

OperandRange IfOp::getEntrySuccessorOperands(RegionSuccessor) {
  return getInputs();
}

ValueRange IfOp::getSuccessorInputs(RegionSuccessor successor) {
  if (Region *region = successor.getSuccessor())
    return region->getArguments();
  return getResults();
}

void IfOp::getRegionInvocationBounds(
    ArrayRef<Attribute>, SmallVectorImpl<InvocationBounds> &invocationBounds) {
  invocationBounds.assign(2, InvocationBounds(0, 1));
}

void ScanOp::getSuccessorRegions(RegionBranchPoint,
                                 SmallVectorImpl<RegionSuccessor> &regions) {
  regions.push_back(RegionSuccessor(&getBody()));
  regions.push_back(RegionSuccessor(getOperation()));
}

OperandRange ScanOp::getEntrySuccessorOperands(RegionSuccessor) {
  return getInits();
}

ValueRange ScanOp::getSuccessorInputs(RegionSuccessor successor) {
  if (Region *region = successor.getSuccessor())
    return region->getArguments().take_front(getInits().size());
  return getCarries();
}

void ScanOp::getRegionInvocationBounds(
    ArrayRef<Attribute>, SmallVectorImpl<InvocationBounds> &invocationBounds) {
  int64_t steps = getInputs().empty() ? ShapedType::kDynamic : getTripCount();
  if (ShapedType::isDynamic(steps)) {
    invocationBounds.push_back(InvocationBounds::getUnknown());
    return;
  }
  unsigned count = static_cast<unsigned>(steps);
  invocationBounds.push_back(InvocationBounds(count, count));
}

namespace {
Block *openBody(OpBuilder &builder, Region &region, TypeRange types,
                Location loc) {
  SmallVector<Location> locations(types.size(), loc);
  return builder.createBlock(&region, region.end(), types, locations);
}

SmallVector<Value>
sizesLike(OpBuilder &builder, Location loc, TypeRange results,
          ArrayRef<std::pair<Value, int64_t>> sources) {
  SmallVector<Value> sizes;
  auto extentType = RankedTensorType::get({}, builder.getIntegerType(64));
  for (auto [index, result] : llvm::enumerate(results)) {
    auto type = cast<RankedTensorType>(result);
    auto [source, shift] = sources[index];
    for (auto [axis, extent] : llvm::enumerate(type.getShape()))
      if (ShapedType::isDynamic(extent))
        sizes.push_back(DimOp::create(builder, loc, extentType, source,
                                      static_cast<int64_t>(axis) - shift));
  }
  return sizes;
}

SmallVector<std::pair<Value, int64_t>> shapedAfter(ValueRange values,
                                                   int64_t shift = 0) {
  SmallVector<std::pair<Value, int64_t>> sources;
  for (Value value : values)
    sources.push_back({value, shift});
  return sources;
}

ScanOp createScan(OpBuilder &builder, Location loc, TypeRange results,
                  ValueRange inits, ValueRange inputs, ValueRange constants,
                  ValueRange sizes, bool reverse) {
  return ScanOp::create(builder, loc, results, inits, inputs, constants, sizes,
                        reverse ? builder.getUnitAttr() : UnitAttr(),
                        IntegerAttr());
}

RankedTensorType stackedType(Type type, int64_t count) {
  auto tensor = cast<RankedTensorType>(type);
  SmallVector<int64_t> shape{count};
  shape.append(tensor.getShape().begin(), tensor.getShape().end());
  return RankedTensorType::get(shape, tensor.getElementType());
}

Value inChunks(OpBuilder &builder, Location loc, Value stacked, int64_t chunks,
               int64_t size) {
  auto type = cast<RankedTensorType>(stacked.getType());
  SmallVector<int64_t> shape{chunks, size};
  ArrayRef<int64_t> inner = type.getShape().drop_front();
  shape.append(inner.begin(), inner.end());
  return ReshapeOp::create(builder, loc,
                           RankedTensorType::get(shape, type.getElementType()),
                           stacked);
}

SmallVector<Type> stepTypes(ValueRange carries, ValueRange inputs,
                            ValueRange constants) {
  SmallVector<Type> types(carries.getTypes());
  for (Value input : inputs)
    types.push_back(
        ScanOp::getSliceType(cast<RankedTensorType>(input.getType())));
  types.append(constants.getTypes().begin(), constants.getTypes().end());
  return types;
}

ScanOp buildRun(OpBuilder &builder, Location loc, Block &body, ValueRange inits,
                ValueRange inputs, ValueRange constants, bool reverse) {
  auto scan = createScan(
      builder, loc, inits.getTypes(), inits, inputs, constants,
      sizesLike(builder, loc, inits.getTypes(), shapedAfter(inits)), reverse);
  OpBuilder::InsertionGuard guard(builder);
  Block *block = openBody(builder, scan.getBody(),
                          stepTypes(inits, inputs, constants), loc);
  SmallVector<Value> yielded;
  cloneBlock(builder, body, block->getArguments(), yielded);
  YieldOp::create(builder, loc,
                  ArrayRef<Value>(yielded).take_front(inits.size()));
  return scan;
}

ScanOp buildStash(OpBuilder &builder, Location loc, Block &body,
                  ValueRange inits, ValueRange inputs, ValueRange constants,
                  bool reverse, int64_t steps) {
  SmallVector<Type> results(inits.getTypes());
  for (Type type : inits.getTypes())
    results.push_back(stackedType(type, steps));

  SmallVector<std::pair<Value, int64_t>> sources = shapedAfter(inits);
  for (auto &entry : shapedAfter(inits, /*shift=*/1))
    sources.push_back(entry);
  auto scan =
      createScan(builder, loc, results, inits, inputs, constants,
                 sizesLike(builder, loc, results, sources), reverse);
  OpBuilder::InsertionGuard guard(builder);
  Block *block = openBody(builder, scan.getBody(),
                          stepTypes(inits, inputs, constants), loc);
  SmallVector<Value> yielded;
  cloneBlock(builder, body, block->getArguments(), yielded);
  SmallVector<Value> stacked(yielded.begin(), yielded.begin() + inits.size());
  stacked.append(block->getArguments().begin(),
                 block->getArguments().begin() + inits.size());
  YieldOp::create(builder, loc, stacked);
  return scan;
}

FailureOr<ScanOp> buildReverse(OpBuilder &builder, Location loc, Block &body,
                               ValueRange carrySeeds,
                               ValueRange outputSeeds, ValueRange residuals,
                               ValueRange inputs, ValueRange constants,
                               ArrayRef<int64_t> tracked, ValueRange totalSeeds,
                               bool reverse) {
  size_t carries = carrySeeds.size();
  size_t inputCount = inputs.size();

  SmallVector<Value> inits(carrySeeds);
  inits.append(totalSeeds.begin(), totalSeeds.end());

  SmallVector<Value> sequence(outputSeeds);
  sequence.append(residuals.begin(), residuals.end());
  sequence.append(inputs.begin(), inputs.end());

  SmallVector<Type> results(ValueRange(inits).getTypes());
  results.append(inputs.getTypes().begin(), inputs.getTypes().end());

  SmallVector<std::pair<Value, int64_t>> sources = shapedAfter(inits);
  for (auto &entry : shapedAfter(inputs))
    sources.push_back(entry);
  auto scan =
      createScan(builder, loc, results, inits, sequence, constants,
                 sizesLike(builder, loc, results, sources), reverse);

  OpBuilder::InsertionGuard guard(builder);
  Block *block = openBody(builder, scan.getBody(),
                          stepTypes(inits, sequence, constants), loc);

  size_t totals = totalSeeds.size();
  ValueRange arguments = block->getArguments();
  ValueRange runningTotals = arguments.slice(carries, totals);
  SmallVector<Value> seeds(arguments.take_front(carries));
  ValueRange outputAdjoints =
      arguments.slice(carries + totals, outputSeeds.size());
  seeds.append(outputAdjoints.begin(), outputAdjoints.end());

  SmallVector<Value> visible(arguments.slice(
      carries + totals + outputSeeds.size(), carries + inputCount));
  ValueRange invariants = arguments.take_back(constants.size());
  visible.append(invariants.begin(), invariants.end());

  SmallVector<Value> forward, gradients;
  if (failed(differentiateBlock(builder, body, visible, seeds, forward,
                                gradients)))
    return failure();
  SmallVector<Value> filled =
      fillMissingWithZero(builder, loc, gradients, visible);

  SmallVector<Value> next(filled.begin(), filled.begin() + carries);
  for (auto [total, index] : llvm::zip_equal(runningTotals, tracked))
    next.push_back(AddOp::create(builder, loc, total,
                                 filled[carries + inputCount + index]));
  next.append(filled.begin() + carries, filled.begin() + carries + inputCount);
  YieldOp::create(builder, loc, next);
  return scan;
}

}

LogicalResult IfOp::buildVjp(OpBuilder &builder, ValueRange adjoints,
                             SmallVectorImpl<Value> &operandAdjoints) {
  Location loc = getLoc();
  SmallVector<Type> inputTypes(getInputs().getTypes());

  SmallVector<Value> reverseInputs(getInputs());
  reverseInputs.append(adjoints.begin(), adjoints.end());
  SmallVector<Type> bodyTypes(inputTypes);
  bodyTypes.append(adjoints.getTypes().begin(), adjoints.getTypes().end());

  auto reverse =
      IfOp::create(builder, loc, inputTypes, getCondition(), reverseInputs);

  OpBuilder::InsertionGuard guard(builder);
  for (auto [from, into] :
       {std::pair{&getThenBody(), &reverse.getThenBody()},
        std::pair{&getElseBody(), &reverse.getElseBody()}}) {
    Block *body = openBody(builder, *into, bodyTypes, loc);
    ValueRange arguments = body->getArguments();
    SmallVector<Value> forward, gradients;
    if (failed(differentiateBlock(
            builder, from->front(), arguments.take_front(inputTypes.size()),
            arguments.drop_front(inputTypes.size()), forward, gradients)))
      return failure();
    YieldOp::create(builder, loc,
                    fillMissingWithZero(builder, loc, gradients,
                                        arguments.take_front(inputTypes.size())));
  }

  operandAdjoints.push_back(Value());
  operandAdjoints.append(reverse.getResults().begin(),
                         reverse.getResults().end());
  return success();
}

LogicalResult ScanOp::buildVjp(OpBuilder &builder, ValueRange adjoints,
                               SmallVectorImpl<Value> &operandAdjoints) {
  Location loc = getLoc();
  size_t carries = getInits().size();
  size_t inputs = getInputs().size();
  size_t constants = getConstants().size();
  int64_t steps = getTripCount();
  Block &body = getBody().front();

  SmallVector<int64_t> tracked;
  SmallVector<Value> totals;
  for (auto [index, constant] : llvm::enumerate(getConstants())) {
    auto type = cast<RankedTensorType>(constant.getType());
    if (isa<FloatType>(type.getElementType())) {
      tracked.push_back(index);
      totals.push_back(createSplat(builder, loc, type, 0.0, constant));
    }
  }

  ValueRange carrySeeds = adjoints.take_front(carries);
  ValueRange outputSeeds = adjoints.drop_front(carries);

  SmallVector<Value> initGradients, inputGradients, constantGradients;
  if (std::optional<int64_t> chunk = getCheckpoint()) {
    int64_t groups = steps / *chunk;

    SmallVector<Value> grouped;
    for (Value input : getInputs())
      grouped.push_back(inChunks(builder, loc, input, groups, *chunk));

    SmallVector<Type> checkpointTypes(getInits().getTypes());
    for (Type type : getInits().getTypes())
      checkpointTypes.push_back(stackedType(type, groups));
    SmallVector<std::pair<Value, int64_t>> checkpointSources =
        shapedAfter(getInits());
    for (auto &entry : shapedAfter(getInits(), /*shift=*/1))
      checkpointSources.push_back(entry);
    auto checkpoints = createScan(
        builder, loc, checkpointTypes, getInits(), grouped, getConstants(),
        sizesLike(builder, loc, checkpointTypes, checkpointSources),
        getReverse());
    {
      OpBuilder::InsertionGuard guard(builder);
      Block *block =
          openBody(builder, checkpoints.getBody(),
                   stepTypes(getInits(), grouped, getConstants()), loc);
      ValueRange arguments = block->getArguments();
      ScanOp group = buildRun(builder, loc, body, arguments.take_front(carries),
                              arguments.slice(carries, inputs),
                              arguments.take_back(constants), getReverse());
      SmallVector<Value> yielded(group.getResults());
      yielded.append(arguments.begin(), arguments.begin() + carries);
      YieldOp::create(builder, loc, yielded);
    }

    SmallVector<Value> groupedSeeds;
    for (Value seed : outputSeeds)
      groupedSeeds.push_back(inChunks(builder, loc, seed, groups, *chunk));

    SmallVector<Value> outerInits(carrySeeds);
    outerInits.append(totals.begin(), totals.end());
    SmallVector<Value> outerInputs(groupedSeeds);
    outerInputs.append(checkpoints.getOutputs().begin(),
                       checkpoints.getOutputs().end());
    outerInputs.append(grouped.begin(), grouped.end());

    SmallVector<Type> outerResults(ValueRange(outerInits).getTypes());
    for (Value value : grouped)
      outerResults.push_back(value.getType());

    SmallVector<std::pair<Value, int64_t>> outerSources =
        shapedAfter(outerInits);
    for (auto &entry : shapedAfter(grouped))
      outerSources.push_back(entry);
    auto outer = createScan(
        builder, loc, outerResults, outerInits, outerInputs, getConstants(),
        sizesLike(builder, loc, outerResults, outerSources), !getReverse());
    {
      OpBuilder::InsertionGuard guard(builder);
      Block *block =
          openBody(builder, outer.getBody(),
                   stepTypes(outerInits, outerInputs, getConstants()), loc);
      ValueRange arguments = block->getArguments();
      size_t seen = carries + totals.size();
      ValueRange groupSeeds = arguments.slice(seen, outputSeeds.size());
      ValueRange entry = arguments.slice(seen + outputSeeds.size(), carries);
      ValueRange sequence =
          arguments.slice(seen + outputSeeds.size() + carries, inputs);
      ValueRange invariants = arguments.take_back(constants);

      ScanOp stash = buildStash(builder, loc, body, entry, sequence, invariants,
                                getReverse(), *chunk);
      FailureOr<ScanOp> group = buildReverse(
          builder, loc, body, arguments.take_front(carries),
          groupSeeds, stash.getOutputs(), sequence, invariants, tracked,
          arguments.slice(carries, totals.size()), !getReverse());
      if (failed(group))
        return failure();
      YieldOp::create(builder, loc, group->getResults());
    }

    initGradients.assign(outer.getCarries().begin(),
                         outer.getCarries().begin() + carries);
    constantGradients.assign(outer.getCarries().begin() + carries,
                             outer.getCarries().end());
    for (auto [gradient, input] :
         llvm::zip_equal(outer.getOutputs(), getInputs()))
      inputGradients.push_back(
          ReshapeOp::create(builder, loc, input.getType(), gradient));
  } else {
    ScanOp stash = buildStash(builder, loc, body, getInits(), getInputs(),
                              getConstants(), getReverse(), steps);
    FailureOr<ScanOp> reverse =
        buildReverse(builder, loc, body, carrySeeds, outputSeeds,
                     stash.getOutputs(), getInputs(), getConstants(), tracked,
                     totals, !getReverse());
    if (failed(reverse))
      return failure();
    initGradients.assign(reverse->getCarries().begin(),
                         reverse->getCarries().begin() + carries);
    constantGradients.assign(reverse->getCarries().begin() + carries,
                             reverse->getCarries().end());
    inputGradients.assign(reverse->getOutputs().begin(),
                          reverse->getOutputs().end());
  }

  operandAdjoints.append(initGradients.begin(), initGradients.end());
  operandAdjoints.append(inputGradients.begin(), inputGradients.end());
  operandAdjoints.resize(carries + inputs + constants + getSizes().size());
  for (auto [slot, index] : llvm::enumerate(tracked))
    operandAdjoints[carries + inputs + index] = constantGradients[slot];
  return success();
}

LogicalResult IfOp::buildJvp(OpBuilder &builder, ValueRange tangents,
                             SmallVectorImpl<Value> &resultTangents,
                             SmallVectorImpl<Value> &primalResults) {
  Location loc = getLoc();
  ValueRange inputs = getInputs();
  ValueRange inputTangents = tangents.drop_front();

  SmallVector<Value> moving;
  SmallVector<int64_t> carried;
  for (auto [index, tangent] : llvm::enumerate(inputTangents))
    if (tangent) {
      carried.push_back(index);
      moving.push_back(tangent);
    }

  SmallVector<Value> forwardInputs(inputs);
  llvm::append_range(forwardInputs, moving);
  SmallVector<Type> bodyTypes(inputs.getTypes());
  for (Value tangent : moving)
    bodyTypes.push_back(tangent.getType());

  SmallVector<Type> results(getResultTypes());
  llvm::append_range(results, getResultTypes());
  auto forward =
      IfOp::create(builder, loc, results, getCondition(), forwardInputs);

  OpBuilder::InsertionGuard guard(builder);
  for (auto [from, into] : {std::pair{&getThenBody(), &forward.getThenBody()},
                            std::pair{&getElseBody(), &forward.getElseBody()}}) {
    Block *body = openBody(builder, *into, bodyTypes, loc);
    ValueRange arguments = body->getArguments();
    ValueRange visible = arguments.take_front(inputs.size());

    SmallVector<Value> argumentTangents(inputs.size());
    for (auto [slot, index] : llvm::enumerate(carried))
      argumentTangents[index] = arguments[inputs.size() + slot];

    SmallVector<Value> yielded, yieldedTangents;
    if (failed(jvpBlock(builder, from->front(), visible, argumentTangents,
                        yielded, yieldedTangents)))
      return failure();
    SmallVector<Value> both(yielded);
    llvm::append_range(both, fillMissingWithZero(builder, loc, yieldedTangents,
                                                 yielded));
    YieldOp::create(builder, loc, both);
  }

  size_t count = getNumResults();
  ValueRange produced = forward.getResults();
  llvm::append_range(primalResults, produced.take_front(count));
  llvm::append_range(resultTangents, produced.drop_front(count));
  return success();
}

LogicalResult ScanOp::buildJvp(OpBuilder &builder, ValueRange tangents,
                               SmallVectorImpl<Value> &resultTangents,
                               SmallVectorImpl<Value> &primalResults) {
  Location loc = getLoc();
  size_t carries = getInits().size();
  size_t inputs = getInputs().size();
  size_t constants = getConstants().size();

  ValueRange initTangents = tangents.take_front(carries);
  ValueRange inputTangents = tangents.slice(carries, inputs);
  ValueRange constantTangents = tangents.slice(carries + inputs, constants);

  SmallVector<Value> movedInits =
      fillMissingWithZero(builder, loc, initTangents, getInits());
  SmallVector<Value> movedInputs =
      fillMissingWithZero(builder, loc, inputTangents, getInputs());
  SmallVector<Value> movedConstants =
      fillMissingWithZero(builder, loc, constantTangents, getConstants());

  SmallVector<Value> forwardInits(getInits());
  llvm::append_range(forwardInits, movedInits);
  SmallVector<Value> forwardInputs(getInputs());
  llvm::append_range(forwardInputs, movedInputs);
  SmallVector<Value> forwardConstants(getConstants());
  llvm::append_range(forwardConstants, movedConstants);

  SmallVector<Type> results(getCarries().getTypes());
  for (Value init : movedInits)
    results.push_back(init.getType());
  for (Value output : getOutputs())
    results.push_back(output.getType());
  for (Value output : getOutputs())
    results.push_back(output.getType());

  SmallVector<std::pair<Value, int64_t>> sources = shapedAfter(forwardInits);
  for (auto &entry : shapedAfter(getOutputs()))
    sources.push_back(entry);
  for (auto &entry : shapedAfter(getOutputs()))
    sources.push_back(entry);

  auto forward = createScan(builder, loc, results, forwardInits, forwardInputs,
                            forwardConstants,
                            sizesLike(builder, loc, results, sources),
                            getReverse());

  {
    OpBuilder::InsertionGuard guard(builder);
    Block *block = openBody(
        builder, forward.getBody(),
        stepTypes(forwardInits, forwardInputs, forwardConstants), loc);
    ValueRange arguments = block->getArguments();

    SmallVector<Value> visible(arguments.take_front(carries));
    llvm::append_range(visible, arguments.slice(2 * carries, inputs));
    llvm::append_range(
        visible, arguments.slice(2 * carries + 2 * inputs, constants));

    SmallVector<Value> visibleTangents(arguments.slice(carries, carries));
    llvm::append_range(visibleTangents,
                       arguments.slice(2 * carries + inputs, inputs));
    llvm::append_range(visibleTangents, arguments.take_back(constants));

    SmallVector<Value> yielded, yieldedTangents;
    if (failed(jvpBlock(builder, getBody().front(), visible, visibleTangents,
                        yielded, yieldedTangents)))
      return failure();

    SmallVector<Value> filled =
        fillMissingWithZero(builder, loc, yieldedTangents, yielded);
    SmallVector<Value> next(yielded.begin(), yielded.begin() + carries);
    next.append(filled.begin(), filled.begin() + carries);
    next.append(yielded.begin() + carries, yielded.end());
    next.append(filled.begin() + carries, filled.end());
    YieldOp::create(builder, loc, next);
  }

  ValueRange produced = forward.getResults();
  size_t outputs = getOutputs().size();

  SmallVector<Value> primal(produced.begin(), produced.begin() + carries);
  primal.append(produced.begin() + 2 * carries,
                produced.begin() + 2 * carries + outputs);
  llvm::append_range(primalResults, primal);

  resultTangents.append(produced.begin() + carries,
                        produced.begin() + 2 * carries);
  resultTangents.append(produced.begin() + 2 * carries + outputs,
                        produced.end());
  return success();
}
