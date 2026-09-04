//===- tera-gradcheck.cpp - Finite-difference gate for autodiff -*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Runs `-tera-autodiff` over a module, compiles the original function and its
// derivative together, and checks one against the other numerically.
//
// The derivative is asked for `J^T v` at a random cotangent `v`, which is the
// gradient of the scalar `s(x) = <f(x), v>`. Central differences of `s`, one
// input element at a time, give the same gradient a second way. Nothing here
// inspects the IR, so a rule that is wrong cannot also be wrong in the check.
//
// A disagreement is an exit code, not a report: this is the gate for Phase 3.
//
//===----------------------------------------------------------------------===//

#include "Tera/Execution/JitInvoker.h"
#include "Tera/IR/TeraDialect.h"
#include "Tera/Transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
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

/// What the driver needs to know about one function, read before the lowering
/// erases the tensor types it is written in.
struct Model {
  std::string primal;
  std::string vjp;
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

/// Returns the scalar inner product of f(x) and the cotangent.
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

  FailureOr<std::unique_ptr<JitInvoker>> invoker =
      JitInvoker::create(module, Target::CPU, optLevel, sharedLibs);
  if (failed(invoker))
    return failure();

  LogicalResult outcome = success();
  for (const Model &model : models)
    if (failed(checkModel(**invoker, model)))
      outcome = failure();
  return outcome;
}

} // namespace

int main(int argc, char **argv) {
  llvm::InitLLVM initialiser(argc, argv);
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();
  registerPassManagerCLOptions();
  llvm::cl::ParseCommandLineOptions(argc, argv,
                                    "Tera finite-difference gradient check\n");

  DialectRegistry registry;
  registerAllDialects(registry);
  registry.insert<TeraDialect>();
  registerAllGPUToLLVMIRTranslations(registry);

  MLIRContext context(registry);
  OwningOpRef<ModuleOp> module =
      parseSourceFile<ModuleOp>(inputFilename, &context);
  if (!module)
    return 1;

  return failed(run(*module)) ? 1 : 0;
}
