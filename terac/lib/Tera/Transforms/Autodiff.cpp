//===- Autodiff.cpp - Reverse-mode AD over the tera dialect -----*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Tera/Transforms/Passes.h"

#include "Tera/Analysis/Activity.h"
#include "Tera/IR/TeraAutodiff.h"
#include "AutodiffDetail.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/StringSet.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

namespace mlir::tera {
#define GEN_PASS_DEF_TERAAUTODIFF
#include "Tera/Transforms/Passes.h.inc"

namespace {
bool carriesAGradient(Type type) {
  auto tensorType = dyn_cast<RankedTensorType>(type);
  return tensorType && isa<FloatType>(tensorType.getElementType());
}

}

FailureOr<SmallVector<int64_t>> differentiableArguments(func::FuncOp func) {
  auto asked =
      func->getAttrOfType<DenseI64ArrayAttr>(TeraDialect::kDiffArgsAttrName);
  if (!asked) {
    SmallVector<int64_t> arguments;
    for (auto [position, type] : llvm::enumerate(func.getArgumentTypes()))
      if (carriesAGradient(type))
        arguments.push_back(position);
    return arguments;
  }

  SmallVector<int64_t> arguments(asked.asArrayRef());
  for (int64_t position : arguments) {
    if (position < 0 || position >= func.getNumArguments())
      return func.emitError()
             << "asks for the gradient of argument " << position
             << ", which it does not take";
    Type type = func.getArgumentTypes()[position];
    if (!carriesAGradient(type))
      return func.emitError()
             << "asks for the gradient of argument " << position << ", which is "
             << type << " and carries none";
  }
  return arguments;
}

DenseSet<Operation *> deadOperations(Block &block) {
  DenseSet<Operation *> dead;
  for (Operation &op : llvm::reverse(block)) {
    if (!isMemoryEffectFree(&op) || op.hasTrait<OpTrait::IsTerminator>())
      continue;
    if (llvm::any_of(op.getUsers(),
                     [&](Operation *user) { return !dead.contains(user); }))
      continue;
    dead.insert(&op);
  }
  return dead;
}

void eraseDeadOperations(Block &block) {
  DenseSet<Operation *> dead = deadOperations(block);
  for (Operation &op : llvm::make_early_inc_range(llvm::reverse(block)))
    if (dead.contains(&op))
      op.erase();
}

void inheritResidence(func::FuncOp primal, func::FuncOp derived) {
  for (unsigned index = 0; index < primal.getNumArguments(); ++index)
    if (Attribute mark = primal.getArgAttr(
            index, TeraDialect::kDeviceResidentAttrName))
      derived.setArgAttr(index, TeraDialect::kDeviceResidentAttrName, mark);
}

namespace {
struct ForwardSplit {
  DenseSet<Operation *> redone;
  SetVector<Value> saved;
};

bool yieldsOnlyFloats(Operation *op) {
  return llvm::all_of(op->getResultTypes(), [](Type type) {
    auto shaped = dyn_cast<ShapedType>(type);
    return shaped && isa<FloatType>(shaped.getElementType());
  });
}

ForwardSplit splitForward(Block &block, ArrayRef<Operation *> forward,
                          const DenseSet<Operation *> &dead) {
  DenseSet<Operation *> produced(forward.begin(), forward.end());
  ForwardSplit split;

  for (Operation *op : forward) {
    bool fromNothing = llvm::all_of(op->getOperands(), [&](Value operand) {
      return split.redone.contains(operand.getDefiningOp());
    });
    if (fromNothing || !yieldsOnlyFloats(op))
      split.redone.insert(op);
  }

  for (Operation &op : block) {
    if (dead.contains(&op))
      continue;
    if (produced.contains(&op) && !split.redone.contains(&op))
      continue;
    op.walk([&](Operation *reader) {
      for (Value operand : reader->getOperands()) {
        Operation *defining = operand.getDefiningOp();
        if (produced.contains(defining) && !split.redone.contains(defining))
          split.saved.insert(operand);
      }
    });
  }
  return split;
}

func::FuncOp openFunction(OpBuilder &builder, func::FuncOp after,
                          StringRef name, TypeRange inputs,
                          TypeRange results) {
  builder.setInsertionPointAfter(after);
  auto func = func::FuncOp::create(builder, after.getLoc(), name,
                                   builder.getFunctionType(inputs, results));
  builder.setInsertionPointToStart(func.addEntryBlock());
  return func;
}

func::FuncOp buildForward(OpBuilder &builder, func::FuncOp primal,
                          StringRef name, ArrayRef<Operation *> forward,
                          const SetVector<Value> &residuals) {
  SmallVector<Type> results(primal.getResultTypes());
  for (Value residual : residuals)
    results.push_back(residual.getType());

  func::FuncOp fwd =
      openFunction(builder, primal, name, primal.getArgumentTypes(), results);
  Block &body = fwd.getBody().front();

  SmallVector<Value> yielded;
  SmallVector<Operation *> cloned;
  cloneBlock(builder, primal.getBody().front(), body.getArguments(), yielded,
             &cloned);

  DenseMap<Value, Value> here;
  for (auto [original, copy] : llvm::zip_equal(forward, cloned))
    for (auto [from, to] :
         llvm::zip_equal(original->getResults(), copy->getResults()))
      here.try_emplace(from, to);

  SmallVector<Value> returned(yielded);
  for (Value residual : residuals)
    returned.push_back(here.lookup(residual));
  func::ReturnOp::create(builder, fwd.getLoc(), returned);
  return fwd;
}

void makeBackward(func::FuncOp scratch, StringRef name,
                  ArrayRef<Operation *> forward, const ForwardSplit &split,
                  const DenseSet<Operation *> &dead, unsigned seeds) {
  Block &body = scratch.getBody().front();
  unsigned seed = body.getNumArguments() - seeds;

  for (Operation &op : llvm::make_early_inc_range(llvm::reverse(body)))
    if (dead.contains(&op))
      op.erase();

  for (auto [offset, value] : llvm::enumerate(split.saved.getArrayRef())) {
    Value residual = value;
    BlockArgument argument = body.insertArgument(
        seed + static_cast<unsigned>(offset), residual.getType(),
        residual.getLoc());
    residual.replaceAllUsesWith(argument);
  }

  for (Operation *op : llvm::reverse(forward))
    if (!dead.contains(op) && !split.redone.contains(op))
      op->erase();

  for (Operation &op : llvm::make_early_inc_range(llvm::reverse(body)))
    if (isMemoryEffectFree(&op) && !op.hasTrait<OpTrait::IsTerminator>() &&
        op.use_empty())
      op.erase();

  scratch.setName(name);
  scratch.setType(FunctionType::get(scratch.getContext(),
                                    body.getArgumentTypes(),
                                    scratch.getResultTypes()));
}

func::FuncOp buildWrapper(OpBuilder &builder, func::FuncOp primal,
                          StringRef name, func::FuncOp fwd, func::FuncOp bwd,
                          unsigned seeds, size_t residuals) {
  SmallVector<Type> inputs(primal.getArgumentTypes());
  llvm::append_range(inputs, primal.getResultTypes());

  func::FuncOp vjp =
      openFunction(builder, primal, name, inputs, bwd.getResultTypes());
  Location loc = vjp.getLoc();
  ValueRange arguments = vjp.getBody().front().getArguments();

  auto forward =
      func::CallOp::create(builder, loc, fwd, arguments.drop_back(seeds));
  SmallVector<Value> given(arguments.drop_back(seeds));
  llvm::append_range(given, forward.getResults().take_back(residuals));
  llvm::append_range(given, arguments.take_back(seeds));

  auto backward = func::CallOp::create(builder, loc, bwd, given);
  func::ReturnOp::create(builder, loc, backward.getResults());
  return vjp;
}

LogicalResult differentiate(func::FuncOp func, llvm::StringSet<> &taken) {
  if (func.isExternal())
    return func.emitError() << "is marked differentiable but has no body";
  if (!func.getBody().hasOneBlock())
    return func.emitError()
           << "has control flow between blocks, which -tera-autodiff does not "
              "handle; use tera.if or tera.scan";
  if (func.getNumResults() == 0)
    return func.emitError() << "returns nothing to seed the reverse pass with";

  for (Type type : func.getResultTypes()) {
    auto tensorType = dyn_cast<RankedTensorType>(type);
    if (!tensorType || !isa<FloatType>(tensorType.getElementType()))
      return func.emitError()
             << "returns " << type
             << ", which carries no gradient to seed the reverse pass with";
  }
  unsigned seeds = func.getNumResults();

  FailureOr<SmallVector<int64_t>> asked = differentiableArguments(func);
  if (failed(asked))
    return failure();
  SmallVector<int64_t> arguments = *asked;
  if (arguments.empty())
    return func.emitError()
           << "has no argument with a floating-point element type";

  SmallVector<std::string> names;
  for (StringRef suffix : {"_vjp", "_fwd", "_bwd"}) {
    names.push_back((func.getName() + suffix).str());
    if (!taken.insert(names.back()).second)
      return func.emitError() << "cannot add '" << names.back()
                              << "': that name is already taken";
  }

  SmallVector<Type> inputs(func.getArgumentTypes());
  llvm::append_range(inputs, func.getResultTypes());
  SmallVector<Type> results;
  for (int64_t position : arguments)
    results.push_back(func.getArgumentTypes()[position]);

  Location loc = func.getLoc();
  OpBuilder builder(func->getContext());
  func::FuncOp bwd = openFunction(builder, func, names[2], inputs, results);
  Block &body = bwd.getBody().front();
  ValueRange seed = body.getArguments().take_back(seeds);

  DenseSet<Operation *> active =
      activeOperations(func.getBody().front(), arguments);
  auto reachesAGradient = [&](Operation *op) { return active.contains(op); };

  SmallVector<Value> forward, argumentAdjoints;
  SmallVector<Operation *> cloned;
  if (failed(differentiateBlock(builder, func.getBody().front(),
                                body.getArguments().drop_back(seeds), seed,
                                forward, argumentAdjoints, reachesAGradient,
                                &cloned))) {
    bwd.erase();
    return failure();
  }

  SmallVector<Value> gradients;
  for (int64_t position : arguments) {
    Value gradient = argumentAdjoints[position];
    if (!gradient)
      gradient = createSplat(builder, loc,
                             cast<RankedTensorType>(inputs[position]), 0.0,
                             body.getArgument(position));
    gradients.push_back(gradient);
  }
  func::ReturnOp::create(builder, loc, gradients);

  DenseSet<Operation *> dead = deadOperations(body);
  ForwardSplit split = splitForward(body, cloned, dead);
  func::FuncOp fwd =
      buildForward(builder, func, names[1], cloned, split.saved);
  makeBackward(bwd, names[2], cloned, split, dead, seeds);
  func::FuncOp vjp = buildWrapper(builder, func, names[0], fwd, bwd, seeds,
                                 split.saved.size());

  for (func::FuncOp derived : {fwd, bwd, vjp})
    inheritResidence(func, derived);

  auto positions = builder.getDenseI64ArrayAttr(arguments);
  bwd->setAttr(TeraDialect::kDiffArgsAttrName, positions);
  vjp->setAttr(TeraDialect::kDiffArgsAttrName, positions);

  MLIRContext *context = builder.getContext();
  func->setAttr(TeraDialect::kVjpAttrName,
                FlatSymbolRefAttr::get(context, names[0]));
  func->setAttr(TeraDialect::kFwdAttrName,
                FlatSymbolRefAttr::get(context, names[1]));
  func->setAttr(TeraDialect::kBwdAttrName,
                FlatSymbolRefAttr::get(context, names[2]));
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

}
}
