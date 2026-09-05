//===- tera-gradcheck.cpp - Finite-difference gate for autodiff -*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Tera/Execution/JitInvoker.h"
#include "Tera/IR/TeraDialect.h"
#include "Tera/Transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllExtensions.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Dialect/All.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"

#include <cmath>
#include <random>

using namespace mlir;
using namespace mlir::tera;

namespace {
llvm::cl::opt<std::string> inputFilename(llvm::cl::Positional,
                                         llvm::cl::desc("<input .mlir file>"),
                                         llvm::cl::init("-"));

llvm::cl::opt<std::string> entryOption(
    "entry",
    llvm::cl::desc("Check only this function, rather than every function "
                   "marked tera.differentiable"),
    llvm::cl::init(""));

llvm::cl::opt<uint64_t> seedOption(
    "seed", llvm::cl::desc("Seed for the inputs and the cotangent"),
    llvm::cl::init(20260901));

llvm::cl::opt<double> epsilon(
    "epsilon",
    llvm::cl::desc("Step of the central difference. The default suits f64; an "
                   "f32 module needs a larger step and a looser tolerance"),
    llvm::cl::init(1e-5));

llvm::cl::opt<double> tolerance(
    "tolerance",
    llvm::cl::desc("Largest relative disagreement that still passes"),
    llvm::cl::init(1e-6));

llvm::cl::opt<double> spreadOption(
    "spread",
    llvm::cl::desc("Inputs are drawn uniformly from [-spread, spread]"),
    llvm::cl::init(1.0));

llvm::cl::opt<unsigned> optLevel("O", llvm::cl::desc("JIT optimisation level"),
                                 llvm::cl::init(3));

llvm::cl::list<std::string> sharedLibs(
    "shared-libs",
    llvm::cl::desc("Shared libraries to load into the JIT, for the runtime "
                   "helpers a lowered memref program calls"),
    llvm::cl::MiscFlags::CommaSeparated);

struct Model {
  std::string primal;
  std::string vjp;
  std::string jvp;
  SmallVector<RankedTensorType> inputs;
  RankedTensorType output;
  SmallVector<int64_t> arguments;
};

FailureOr<Model> readModel(func::FuncOp func) {
  Model model;
  model.primal = func.getName().str();

  auto vjp = func->getAttrOfType<FlatSymbolRefAttr>(TeraDialect::kVjpAttrName);
  if (!vjp)
    return func.emitError() << "has no derivative; did -tera-autodiff run?";
  model.vjp = vjp.getValue().str();

  auto derivative =
      func->getParentOfType<ModuleOp>().lookupSymbol<func::FuncOp>(model.vjp);
  if (!derivative)
    return func.emitError() << "names a derivative '" << model.vjp
                            << "' that the module does not hold";
  auto arguments = derivative->getAttrOfType<DenseI64ArrayAttr>(
      TeraDialect::kDiffArgsAttrName);
  if (!arguments)
    return derivative.emitError()
           << "does not say which arguments it differentiates";
  model.arguments.assign(arguments.asArrayRef().begin(),
                         arguments.asArrayRef().end());

  if (auto jvp = func->getAttrOfType<FlatSymbolRefAttr>(
          TeraDialect::kJvpAttrName))
    model.jvp = jvp.getValue().str();

  for (Type type : func.getArgumentTypes()) {
    auto tensorType = dyn_cast<RankedTensorType>(type);
    if (!tensorType || !tensorType.hasStaticShape())
      return func.emitError()
             << "takes " << type << ", which the gradcheck cannot materialise";
    if (failed(TensorBuffer::checkElementType(tensorType.getElementType(),
                                              "an argument")))
      return failure();
    model.inputs.push_back(tensorType);
  }

  model.output = cast<RankedTensorType>(func.getResultTypes().front());
  if (!model.output.hasStaticShape())
    return func.emitError() << "returns a dynamic shape";
  return model;
}

FailureOr<double> project(JitInvoker &invoker, const Model &model,
                          MutableArrayRef<TensorBuffer> inputs,
                          const TensorBuffer &cotangent) {
  SmallVector<TensorBuffer> results;
  results.push_back(TensorBuffer::forResult(model.output));
  if (failed(invoker.invoke(model.primal, inputs, results)))
    return failure();

  double total = 0.0;
  for (int64_t index = 0; index < results.front().getNumElements(); ++index)
    total += results.front().getElement(index) * cotangent.getElement(index);
  return total;
}

LogicalResult checkForwardMode(JitInvoker &invoker, const Model &model,
                               ArrayRef<TensorBuffer> inputs,
                               ArrayRef<TensorBuffer> tangents,
                               const TensorBuffer &cotangent,
                               ArrayRef<TensorBuffer> reverse) {
  auto like = [](const TensorBuffer &source, RankedTensorType type) {
    TensorBuffer copy(type);
    for (int64_t index = 0; index < source.getNumElements(); ++index)
      copy.setElement(index, source.getElement(index));
    return copy;
  };

  SmallVector<TensorBuffer> given;
  for (auto [type, input] : llvm::zip_equal(model.inputs, inputs))
    given.push_back(like(input, type));
  for (auto [slot, position] : llvm::enumerate(model.arguments))
    given.push_back(like(tangents[slot], model.inputs[position]));

  SmallVector<TensorBuffer> produced;
  produced.push_back(TensorBuffer::forResult(model.output));
  produced.push_back(TensorBuffer::forResult(model.output));
  if (failed(invoker.invoke(model.jvp, given, produced)))
    return failure();

  double forward = 0.0;
  for (int64_t index = 0; index < produced[1].getNumElements(); ++index)
    forward += produced[1].getElement(index) * cotangent.getElement(index);

  double backward = 0.0;
  for (auto [slot, position] : llvm::enumerate(model.arguments))
    for (int64_t index = 0; index < reverse[slot].getNumElements(); ++index)
      backward +=
          tangents[slot].getElement(index) * reverse[slot].getElement(index);

  double error = std::abs(forward - backward) /
                 std::max({1.0, std::abs(forward), std::abs(backward)});
  llvm::outs() << "tera-gradcheck: " << model.primal
               << ": forward and reverse agree to "
               << llvm::format("%.3g", error) << "\n";
  if (error <= tolerance)
    return success();

  llvm::errs() << "tera-gradcheck: " << model.primal
               << ": the forward derivative projected onto the cotangent is "
               << llvm::format("%.12g", forward)
               << ", but the reverse one projected onto the tangent is "
               << llvm::format("%.12g", backward)
               << "; one of the two directions is wrong\n";
  return failure();
}

LogicalResult checkModel(JitInvoker &invoker, const Model &model) {
  std::mt19937_64 generator(seedOption);

  SmallVector<TensorBuffer> inputs;
  for (RankedTensorType type : model.inputs) {
    inputs.emplace_back(type);
    inputs.back().fill(generator, spreadOption);
  }

  TensorBuffer cotangent(model.output);
  cotangent.fill(generator, spreadOption);

  SmallVector<TensorBuffer> analytic;
  for (int64_t position : model.arguments)
    analytic.push_back(TensorBuffer::forResult(model.inputs[position]));

  SmallVector<TensorBuffer> vjpInputs;
  for (auto [type, given] : llvm::zip_equal(model.inputs, inputs)) {
    vjpInputs.emplace_back(type);
    for (int64_t index = 0; index < given.getNumElements(); ++index)
      vjpInputs.back().setElement(index, given.getElement(index));
  }
  vjpInputs.emplace_back(model.output);
  for (int64_t index = 0; index < cotangent.getNumElements(); ++index)
    vjpInputs.back().setElement(index, cotangent.getElement(index));

  if (failed(invoker.invoke(model.vjp, vjpInputs, analytic)))
    return failure();

  if (!model.jvp.empty()) {
    SmallVector<TensorBuffer> tangents;
    for (int64_t position : model.arguments) {
      tangents.emplace_back(model.inputs[position]);
      tangents.back().fill(generator, spreadOption);
    }
    if (failed(checkForwardMode(invoker, model, inputs, tangents, cotangent,
                                analytic)))
      return failure();
  }

  int64_t checked = 0;
  unsigned reported = 0;
  double worst = 0.0;
  for (auto [slot, position] : llvm::enumerate(model.arguments)) {
    TensorBuffer &input = inputs[position];
    for (int64_t index = 0; index < input.getNumElements(); ++index) {
      double centre = input.getElement(index);

      input.setElement(index, centre + epsilon);
      FailureOr<double> above = project(invoker, model, inputs, cotangent);
      input.setElement(index, centre - epsilon);
      FailureOr<double> below = project(invoker, model, inputs, cotangent);
      input.setElement(index, centre);
      if (failed(above) || failed(below))
        return failure();

      double numeric = (*above - *below) / (2.0 * epsilon);
      double exact = analytic[slot].getElement(index);
      double error = std::abs(numeric - exact) /
                     std::max({1.0, std::abs(numeric), std::abs(exact)});
      worst = std::max(worst, error);
      ++checked;

      if (error > tolerance && ++reported <= 8)
        llvm::errs() << "tera-gradcheck: " << model.primal << " argument "
                     << position << " element " << index << ": derivative says "
                     << llvm::format("%.12g", exact)
                     << ", finite differences say "
                     << llvm::format("%.12g", numeric) << " (relative error "
                     << llvm::format("%.3g", error) << ")\n";
    }
  }

  llvm::outs() << "tera-gradcheck: " << model.primal << ": " << checked
               << " elements over " << model.arguments.size()
               << " arguments, worst relative error "
               << llvm::format("%.3g", worst) << "\n";
  if (reported == 0)
    return success();
  llvm::errs() << "tera-gradcheck: " << model.primal << ": " << reported
               << " of " << checked << " gradients disagree\n";
  return failure();
}

LogicalResult run(ModuleOp module) {
  PassManager pm(module.getContext());
  pm.addPass(createTeraAutodiff());
  pm.addPass(createTeraForwardMode());
  if (failed(pm.run(module)))
    return failure();

  SmallVector<Model> models;
  for (auto func : module.getOps<func::FuncOp>()) {
    if (!func->hasAttr(TeraDialect::kDifferentiableAttrName))
      continue;
    if (!entryOption.empty() && func.getName() != entryOption)
      continue;
    FailureOr<Model> model = readModel(func);
    if (failed(model))
      return failure();
    models.push_back(std::move(*model));
  }
  if (models.empty()) {
    llvm::errs() << "tera-gradcheck: nothing to check; no function carries "
                 << TeraDialect::kDifferentiableAttrName << "\n";
    return failure();
  }

  const TargetBackend *host = lookupTargetBackend("cpu");
  if (!host) {
    llvm::errs() << "tera-gradcheck: this build has no host target to run "
                    "the finite differences on\n";
    return failure();
  }

  FailureOr<std::unique_ptr<JitInvoker>> invoker =
      JitInvoker::create(module, *host, "", optLevel, sharedLibs);
  if (failed(invoker))
    return failure();

  LogicalResult outcome = success();
  for (const Model &model : models)
    if (failed(checkModel(**invoker, model)))
      outcome = failure();
  return outcome;
}

}

int main(int argc, char **argv) {
  llvm::InitLLVM initialiser(argc, argv);
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();
  registerPassManagerCLOptions();
  registerTeraTargets();
  llvm::cl::ParseCommandLineOptions(argc, argv,
                                    "Tera finite-difference gradient check\n");

  DialectRegistry registry;
  registerAllDialects(registry);
  registerAllExtensions(registry);
  registry.insert<TeraDialect>();
  registerAllGPUToLLVMIRTranslations(registry);

  MLIRContext context(registry);
  OwningOpRef<ModuleOp> module =
      parseSourceFile<ModuleOp>(inputFilename, &context);
  if (!module)
    return 1;

  return failed(run(*module)) ? 1 : 0;
}
