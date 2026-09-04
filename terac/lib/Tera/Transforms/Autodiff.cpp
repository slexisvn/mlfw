//===- Autodiff.cpp - Reverse-mode AD over the tera dialect -----*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The pass decides which functions get a derivative and what its signature is.
// Emitting the derivative itself is `differentiateBlock`, which the bodies of
// `tera.scan` and `tera.if` reach for too.
//
//===----------------------------------------------------------------------===//

#include "Tera/Transforms/Passes.h"

#include "Tera/IR/TeraAutodiff.h"
#include "llvm/ADT/StringSet.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/SymbolTable.h"

namespace mlir::tera {
#define GEN_PASS_DEF_TERAAUTODIFF
#include "Tera/Transforms/Passes.h.inc"

namespace {

SmallVector<int64_t> differentiableArguments(func::FuncOp func) {
  SmallVector<int64_t> arguments;
  for (auto [position, type] : llvm::enumerate(func.getArgumentTypes())) {
    auto tensorType = dyn_cast<RankedTensorType>(type);
    if (tensorType && isa<FloatType>(tensorType.getElementType()))
      arguments.push_back(position);
  }
  return arguments;
}

/// taken must contain existing module symbols and is updated with new names.
LogicalResult differentiate(func::FuncOp func, llvm::StringSet<> &taken) {
  if (func.isExternal())
    return func.emitError() << "is marked differentiable but has no body";
  if (!func.getBody().hasOneBlock())
    return func.emitError()
           << "has control flow between blocks, which -tera-autodiff does not "
              "handle; use tera.if or tera.scan";
  if (func.getNumResults() != 1)
    return func.emitError()
           << "must return exactly one tensor to be differentiated, not "
           << func.getNumResults();

  auto resultType = dyn_cast<RankedTensorType>(func.getResultTypes().front());
  if (!resultType || !isa<FloatType>(resultType.getElementType()))
    return func.emitError()
           << "returns " << func.getResultTypes().front()
           << ", which carries no gradient to seed the reverse pass with";

  SmallVector<int64_t> arguments = differentiableArguments(func);
  if (arguments.empty())
    return func.emitError()
           << "has no argument with a floating-point element type";

  std::string name = (func.getName() + "_vjp").str();
  if (!taken.insert(name).second)
    return func.emitError()
           << "cannot add '" << name << "': that name is already taken";

  SmallVector<Type> inputs(func.getArgumentTypes());
  inputs.push_back(resultType);
  SmallVector<Type> results;
  for (int64_t position : arguments)
    results.push_back(func.getArgumentTypes()[position]);

  Location loc = func.getLoc();
  OpBuilder builder(func->getContext());
  builder.setInsertionPointAfter(func);
  auto vjp = func::FuncOp::create(builder, loc, name,
                                  builder.getFunctionType(inputs, results));
  vjp->setAttr(TeraDialect::kDiffArgsAttrName,
               builder.getDenseI64ArrayAttr(arguments));

  Block *body = vjp.addEntryBlock();
  builder.setInsertionPointToStart(body);
  Value seed = body->getArguments().back();

  SmallVector<Value> forward, argumentAdjoints;
  if (failed(differentiateBlock(builder, func.getBody().front(),
                                body->getArguments().drop_back(), {seed},
                                forward, argumentAdjoints))) {
    vjp.erase();
    return failure();
  }

  SmallVector<Value> gradients;
  for (int64_t position : arguments) {
    Value gradient = argumentAdjoints[position];
    if (!gradient)
      gradient = createSplat(builder, loc,
                             cast<RankedTensorType>(inputs[position]), 0.0,
                             body->getArgument(position));
    gradients.push_back(gradient);
  }
  func::ReturnOp::create(builder, loc, gradients);

  func->setAttr(TeraDialect::kVjpAttrName,
                FlatSymbolRefAttr::get(builder.getContext(), name));
  return success();
}

struct TeraAutodiff : public impl::TeraAutodiffBase<TeraAutodiff> {
  using impl::TeraAutodiffBase<TeraAutodiff>::TeraAutodiffBase;

  void runOnOperation() final {
    SmallVector<func::FuncOp> differentiable;
    llvm::StringSet<> taken;
    for (Operation &op : getOperation().getBody()->getOperations()) {
      if (auto symbol = dyn_cast<SymbolOpInterface>(op))
        taken.insert(symbol.getName());
      auto func = dyn_cast<func::FuncOp>(op);
      if (func && func->hasAttr(TeraDialect::kDifferentiableAttrName) &&
          !func->hasAttr(TeraDialect::kVjpAttrName))
        differentiable.push_back(func);
    }

    for (func::FuncOp func : differentiable)
      if (failed(differentiate(func, taken)))
        return signalPassFailure();
  }
};

} // namespace
} // namespace mlir::tera
