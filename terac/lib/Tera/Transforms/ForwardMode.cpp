//===- ForwardMode.cpp - Forward-mode AD over the tera dialect --*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Tera/Transforms/Passes.h"

#include "Tera/IR/TeraAutodiff.h"
#include "AutodiffDetail.h"
#include "llvm/ADT/StringSet.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/SymbolTable.h"

namespace mlir::tera {
#define GEN_PASS_DEF_TERAFORWARDMODE
#include "Tera/Transforms/Passes.h.inc"

namespace {
LogicalResult linearize(func::FuncOp func, llvm::StringSet<> &taken) {
  if (func.isExternal())
    return func.emitError() << "is marked differentiable but has no body";
  if (!func.getBody().hasOneBlock())
    return func.emitError()
           << "has control flow between blocks, which -tera-forward-mode does "
              "not handle; use tera.if or tera.scan";

  FailureOr<SmallVector<int64_t>> asked = differentiableArguments(func);
  if (failed(asked))
    return failure();
  SmallVector<int64_t> arguments = *asked;
  if (arguments.empty())
    return func.emitError()
           << "has no argument with a floating-point element type";

  std::string name = (func.getName() + "_jvp").str();
  if (!taken.insert(name).second)
    return func.emitError()
           << "cannot add '" << name << "': that name is already taken";

  SmallVector<Type> inputs(func.getArgumentTypes());
  for (int64_t position : arguments)
    inputs.push_back(func.getArgumentTypes()[position]);

  SmallVector<Type> results(func.getResultTypes());
  llvm::append_range(results, func.getResultTypes());

  Location loc = func.getLoc();
  OpBuilder builder(func->getContext());
  builder.setInsertionPointAfter(func);
  auto jvp = func::FuncOp::create(builder, loc, name,
                                  builder.getFunctionType(inputs, results));
  jvp->setAttr(TeraDialect::kDiffArgsAttrName,
               builder.getDenseI64ArrayAttr(arguments));

  Block *body = jvp.addEntryBlock();
  builder.setInsertionPointToStart(body);
  ValueRange given = body->getArguments();
  ValueRange primal = given.take_front(func.getNumArguments());

  SmallVector<Value> tangents(func.getNumArguments());
  for (auto [slot, position] : llvm::enumerate(arguments))
    tangents[position] = given[func.getNumArguments() + slot];

  SmallVector<Value> forward, forwardTangents;
  if (failed(jvpBlock(builder, func.getBody().front(), primal, tangents,
                      forward, forwardTangents))) {
    jvp.erase();
    return failure();
  }

  SmallVector<Value> returned(forward);
  llvm::append_range(returned, fillMissingWithZero(builder, loc,
                                                   forwardTangents, forward));
  func::ReturnOp::create(builder, loc, returned);
  eraseDeadOperations(*body);
  inheritResidence(func, jvp);

  func->setAttr(TeraDialect::kJvpAttrName,
                FlatSymbolRefAttr::get(builder.getContext(), name));
  return success();
}

struct TeraForwardMode : public impl::TeraForwardModeBase<TeraForwardMode> {
  using impl::TeraForwardModeBase<TeraForwardMode>::TeraForwardModeBase;

  void runOnOperation() final {
    SmallVector<func::FuncOp> differentiable;
    llvm::StringSet<> taken;
    for (Operation &op : getOperation().getBody()->getOperations()) {
      if (auto symbol = dyn_cast<SymbolOpInterface>(op))
        taken.insert(symbol.getName());
      auto func = dyn_cast<func::FuncOp>(op);
      if (func && func->hasAttr(TeraDialect::kDifferentiableAttrName) &&
          !func->hasAttr(TeraDialect::kJvpAttrName))
        differentiable.push_back(func);
    }

    for (func::FuncOp func : differentiable)
      if (failed(linearize(func, taken)))
        return signalPassFailure();
  }
};

}
}
