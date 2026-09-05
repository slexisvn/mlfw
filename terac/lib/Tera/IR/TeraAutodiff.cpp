//===- TeraAutodiff.cpp - Reverse-mode AD over a block ----------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Tera/IR/TeraAutodiff.h"

#include "Tera/IR/TeraInterfaces.h"
#include "mlir/IR/IRMapping.h"
#include "llvm/ADT/DenseMap.h"

using namespace mlir;
using namespace mlir::tera;

namespace {
class AdjointMap {
public:
  void addContribution(Value value, Value contribution) {
    contributions[value].push_back(contribution);
  }

  Value getAdjoint(OpBuilder &builder, Location loc, Value value) {
    auto entry = contributions.find(value);
    if (entry == contributions.end())
      return {};

    SmallVector<Value> level = entry->second;
    while (level.size() > 1) {
      SmallVector<Value> summed;
      summed.reserve(level.size() / 2 + 1);
      for (size_t index = 0; index + 1 < level.size(); index += 2)
        summed.push_back(
            AddOp::create(builder, loc, level[index], level[index + 1]));
      if (level.size() % 2 != 0)
        summed.push_back(level.back());
      level = std::move(summed);
    }
    entry->second.assign(1, level.front());
    return level.front();
  }

private:
  DenseMap<Value, SmallVector<Value>> contributions;
};

SmallVector<Operation *> cloneInto(OpBuilder &builder, Block &block,
                                   ValueRange arguments,
                                   SmallVectorImpl<Value> &forward) {
  IRMapping mapping;
  for (auto [argument, value] :
       llvm::zip_equal(block.getArguments(), arguments))
    mapping.map(argument, value);

  SmallVector<Operation *> cloned;
  for (Operation &op : block.without_terminator())
    cloned.push_back(builder.clone(op, mapping));
  for (Value yielded : block.getTerminator()->getOperands())
    forward.push_back(mapping.lookup(yielded));
  return cloned;
}

}

void mlir::tera::cloneBlock(OpBuilder &builder, Block &block,
                            ValueRange arguments,
                            SmallVectorImpl<Value> &forward,
                            SmallVectorImpl<Operation *> *cloned) {
  SmallVector<Operation *> copies =
      cloneInto(builder, block, arguments, forward);
  if (cloned)
    cloned->assign(copies.begin(), copies.end());
}

LogicalResult mlir::tera::differentiateBlock(
    OpBuilder &builder, Block &block, ValueRange arguments,
    ValueRange resultAdjoints, SmallVectorImpl<Value> &forward,
    SmallVectorImpl<Value> &argumentAdjoints,
    function_ref<bool(Operation *)> isActive,
    SmallVectorImpl<Operation *> *clonedForward) {
  SmallVector<Operation *> originals;
  for (Operation &op : block.without_terminator())
    originals.push_back(&op);

  SmallVector<Operation *> cloned =
      cloneInto(builder, block, arguments, forward);
  if (clonedForward)
    clonedForward->assign(cloned.begin(), cloned.end());

  AdjointMap adjoints;
  for (auto [value, adjoint] : llvm::zip_equal(forward, resultAdjoints))
    if (adjoint)
      adjoints.addContribution(value, adjoint);

  for (int64_t index = cloned.size() - 1; index >= 0; --index) {
    Operation *op = cloned[index];
    if (isActive && !isActive(originals[index]))
      continue;

    SmallVector<Value> results;
    bool reached = false;
    for (Value result : op->getResults()) {
      Value adjoint = adjoints.getAdjoint(builder, op->getLoc(), result);
      reached |= static_cast<bool>(adjoint);
      results.push_back(adjoint);
    }
    if (!reached)
      continue;

    for (auto [index, result] : llvm::enumerate(op->getResults()))
      if (!results[index])
        results[index] =
            createSplat(builder, op->getLoc(),
                        cast<RankedTensorType>(result.getType()), 0.0, result);

    auto differentiable = dyn_cast<TeraVjpOpInterface>(op);
    if (!differentiable)
      return op->emitError() << "has no derivative: '" << op->getName()
                             << "' does not implement TeraVjpOpInterface";

    SmallVector<Value> operandAdjoints;
    if (failed(differentiable.buildVjp(builder, results, operandAdjoints)))
      return failure();
    if (operandAdjoints.size() != op->getNumOperands())
      return op->emitError()
             << "its derivative produced " << operandAdjoints.size()
             << " gradients for " << op->getNumOperands() << " operands";

    for (auto [operand, adjoint] :
         llvm::zip_equal(op->getOperands(), operandAdjoints))
      if (adjoint)
        adjoints.addContribution(operand, adjoint);
  }

  Location loc = block.getParentOp() ? block.getParentOp()->getLoc()
                                     : builder.getUnknownLoc();
  for (Value argument : arguments)
    argumentAdjoints.push_back(adjoints.getAdjoint(builder, loc, argument));
  return success();
}

SmallVector<Value> mlir::tera::fillMissingWithZero(OpBuilder &builder,
                                                   Location loc,
                                                   ValueRange values,
                                                   ValueRange like) {
  SmallVector<Value> filled;
  for (auto [value, shaped] : llvm::zip_equal(values, like))
    filled.push_back(
        value ? value
              : createSplat(builder, loc,
                            cast<RankedTensorType>(shaped.getType()), 0.0,
                            shaped));
  return filled;
}

namespace {
bool movesLikeItsOperands(Operation *op) {
  if (!op->hasTrait<OpTrait::Elementwise>() || op->getNumResults() != 1)
    return false;
  Type resultType = op->getResult(0).getType();
  return llvm::all_of(op->getOperandTypes(), [&](Type type) {
    auto shaped = dyn_cast<RankedTensorType>(type);
    return !shaped || !isa<FloatType>(shaped.getElementType()) ||
           type == resultType;
  });
}

Value tangentOrZero(OpBuilder &builder, Location loc, Value operand,
                    Value tangent) {
  if (tangent)
    return tangent;
  auto type = dyn_cast<RankedTensorType>(operand.getType());
  if (!type || !isa<FloatType>(type.getElementType()))
    return operand;
  return createSplat(builder, loc, type, 0.0, operand);
}

Operation *cloneWithOperands(OpBuilder &builder, Operation *op,
                             ValueRange operands) {
  OperationState state(op->getLoc(), op->getName());
  state.addOperands(operands);
  state.addTypes(op->getResultTypes());
  state.addAttributes(op->getAttrs());
  return builder.create(state);
}

LogicalResult jvpFromLinearity(Operation *op, OpBuilder &builder,
                               ValueRange tangents,
                               SmallVectorImpl<Value> &resultTangents) {
  Location loc = op->getLoc();
  if (llvm::none_of(tangents, [](Value tangent) { return tangent != Value(); }))
    return success();

  if (op->hasTrait<LinearOperands>()) {
    SmallVector<Value> moved;
    for (auto [operand, tangent] : llvm::zip_equal(op->getOperands(), tangents))
      moved.push_back(tangentOrZero(builder, loc, operand, tangent));
    llvm::append_range(resultTangents,
                       cloneWithOperands(builder, op, moved)->getResults());
    return success();
  }

  SmallVector<Value> summed(op->getNumResults());
  for (auto [position, tangent] : llvm::enumerate(tangents)) {
    if (!tangent)
      continue;
    SmallVector<Value> moved(op->getOperands());
    moved[position] = tangent;
    Operation *term = cloneWithOperands(builder, op, moved);
    for (auto [slot, value] : llvm::enumerate(term->getResults())) {
      if (!summed[slot]) {
        summed[slot] = value;
        continue;
      }
      summed[slot] = AddOp::create(builder, loc, summed[slot], value);
    }
  }
  llvm::append_range(resultTangents, summed);
  return success();
}

LogicalResult jvpFromVjp(Operation *op, OpBuilder &builder, ValueRange tangents,
                         SmallVectorImpl<Value> &resultTangents) {
  auto differentiable = cast<TeraVjpOpInterface>(op);
  Location loc = op->getLoc();

  Value summed;
  for (auto [position, tangent] : llvm::enumerate(tangents)) {
    if (!tangent)
      continue;
    SmallVector<Value> contributions;
    if (failed(differentiable.buildVjp(builder, {tangent}, contributions)))
      return failure();
    Value moved = contributions[position];
    if (!moved)
      continue;
    summed = summed ? AddOp::create(builder, loc, summed, moved).getResult()
                    : moved;
  }
  resultTangents.push_back(summed);
  return success();
}

}

LogicalResult mlir::tera::buildStructuralJvp(
    Operation *op, OpBuilder &builder, ValueRange tangents,
    SmallVectorImpl<Value> &resultTangents) {
  if (llvm::none_of(op->getResultTypes(), [](Type type) {
        auto shaped = dyn_cast<RankedTensorType>(type);
        return shaped && isa<FloatType>(shaped.getElementType());
      })) {
    resultTangents.assign(op->getNumResults(), Value());
    return success();
  }
  if (op->hasTrait<LinearOperands>() || op->hasTrait<MultilinearOperands>())
    return jvpFromLinearity(op, builder, tangents, resultTangents);
  if (movesLikeItsOperands(op) && isa<TeraVjpOpInterface>(op))
    return jvpFromVjp(op, builder, tangents, resultTangents);
  return op->emitError() << "has no forward derivative: '" << op->getName()
                         << "' is neither linear nor elementwise and does not "
                            "implement buildJvp";
}

LogicalResult mlir::tera::jvpBlock(OpBuilder &builder, Block &block,
                                   ValueRange arguments,
                                   ValueRange argumentTangents,
                                   SmallVectorImpl<Value> &forward,
                                   SmallVectorImpl<Value> &forwardTangents) {
  IRMapping mapping;
  DenseMap<Value, Value> moves;
  for (auto [argument, value, tangent] :
       llvm::zip_equal(block.getArguments(), arguments, argumentTangents)) {
    mapping.map(argument, value);
    if (tangent)
      moves.try_emplace(argument, tangent);
  }

  for (Operation &op : block.without_terminator()) {
    Operation *copy = builder.clone(op, mapping);

    SmallVector<Value> tangents;
    bool moved = false;
    for (Value operand : op.getOperands()) {
      Value tangent = moves.lookup(operand);
      moved |= static_cast<bool>(tangent);
      tangents.push_back(tangent);
    }
    if (!moved)
      continue;

    auto differentiable = dyn_cast<TeraJvpOpInterface>(copy);
    if (!differentiable)
      return op.emitError() << "has no forward derivative: '" << op.getName()
                            << "' does not implement TeraJvpOpInterface";

    SmallVector<Value> results, primal;
    if (failed(differentiable.buildJvp(builder, tangents, results, primal)))
      return failure();
    if (results.size() != op.getNumResults())
      return op.emitError()
             << "its forward derivative produced " << results.size()
             << " tangents for " << op.getNumResults() << " results";

    if (!primal.empty()) {
      for (auto [original, better] : llvm::zip_equal(op.getResults(), primal))
        mapping.map(original, better);
      copy->erase();
    }

    for (auto [result, tangent] : llvm::zip_equal(op.getResults(), results))
      if (tangent)
        moves.try_emplace(result, tangent);
  }

  for (Value yielded : block.getTerminator()->getOperands()) {
    forward.push_back(mapping.lookup(yielded));
    forwardTangents.push_back(moves.lookup(yielded));
  }
  return success();
}
