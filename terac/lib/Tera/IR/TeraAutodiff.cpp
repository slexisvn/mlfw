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
                            SmallVectorImpl<Value> &forward) {
  cloneInto(builder, block, arguments, forward);
}

LogicalResult mlir::tera::differentiateBlock(
    OpBuilder &builder, Block &block, ValueRange arguments,
    ValueRange resultAdjoints, SmallVectorImpl<Value> &forward,
    SmallVectorImpl<Value> &argumentAdjoints) {
  SmallVector<Operation *> cloned =
      cloneInto(builder, block, arguments, forward);

  AdjointMap adjoints;
  for (auto [value, adjoint] : llvm::zip_equal(forward, resultAdjoints))
    if (adjoint)
      adjoints.addContribution(value, adjoint);

  for (Operation *op : llvm::reverse(cloned)) {
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
