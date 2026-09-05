//===- Execution.cpp - Calling the tera JIT from a C caller -----*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Tera/CAPI/Execution.h"

#include "Tera/Execution/JitInvoker.h"
#include "Tera/IR/TeraDialect.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllExtensions.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Dialect/All.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>

using namespace mlir;
using namespace mlir::tera;

namespace {
thread_local std::string lastError;
thread_local std::string lastTargets;

void startUp() {
  static std::once_flag initialised;
  std::call_once(initialised, [] {
    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();
    registerPassManagerCLOptions();
    registerTeraTargets();
  });
}

std::string registeredTargets() {
  startUp();
  std::string names;
  for (StringRef name : getTargetBackendNames()) {
    if (!names.empty())
      names += ",";
    names += name.str();
  }
  return names;
}

struct Signature {
  SmallVector<RankedTensorType> inputs;
  SmallVector<RankedTensorType> results;
};

int64_t elementsIn(ArrayRef<int64_t> shape) {
  int64_t count = 1;
  for (int64_t extent : shape)
    count *= extent;
  return count;
}

SmallVector<int64_t> extentsOf(ArrayRef<uint64_t> descriptor, int64_t rank) {
  ArrayRef<uint64_t> sizes = descriptor.slice(3, rank);
  return SmallVector<int64_t>(sizes.begin(), sizes.end());
}

ArrayRef<int64_t> takeShape(const int64_t *&shapes, int64_t rank) {
  ArrayRef<int64_t> shape(shapes, rank);
  shapes += rank;
  return shape;
}

}

struct TeraModule {
  std::unique_ptr<MLIRContext> context;
  OwningOpRef<ModuleOp> module;
  std::unique_ptr<JitInvoker> invoker;
  llvm::StringMap<Signature> signatures;
};

TeraModule *teraCompileFor(const char *mlir, const char *target,
                           const char *targetOptions, unsigned optLevel,
                           const char *const *sharedLibs,
                           size_t numSharedLibs) {
  lastError.clear();
  if (!mlir) {
    lastError = "teraCompile: no module text";
    return nullptr;
  }
  if (!target) {
    lastError = "teraCompile: no target name";
    return nullptr;
  }

  startUp();

  const TargetBackend *backend = lookupTargetBackend(target);
  if (!backend) {
    lastError = std::string("teraCompile: no target named '") + target +
                "'; this build has " + registeredTargets();
    return nullptr;
  }

  DialectRegistry registry;
  registerAllDialects(registry);
  registerAllExtensions(registry);
  registry.insert<TeraDialect>();
  registerAllGPUToLLVMIRTranslations(registry);

  auto handle = std::make_unique<TeraModule>();
  handle->context = std::make_unique<MLIRContext>(registry);

  std::string diagnostics;
  llvm::raw_string_ostream stream(diagnostics);
  ScopedDiagnosticHandler collect(
      handle->context.get(), [&](Diagnostic &diagnostic) {
        diagnostic.print(stream);
        stream << "\n";
        return success();
      });

  auto fail = [&](const Twine &message) -> TeraModule * {
    lastError = (message + (diagnostics.empty() ? "" : "\n" + diagnostics))
                    .str();
    return nullptr;
  };

  handle->module = parseSourceString<ModuleOp>(mlir, handle->context.get());
  if (!handle->module)
    return fail("teraCompile: the module does not parse");

  for (auto function : handle->module->getOps<func::FuncOp>()) {
    if (function.isExternal())
      continue;
    Signature signature;
    StringRef name = function.getName();
    auto record = [&](TypeRange types, StringRef what,
                      SmallVectorImpl<RankedTensorType> &into) {
      for (auto [index, type] : llvm::enumerate(types)) {
        auto tensor = dyn_cast<RankedTensorType>(type);
        std::string described =
            (name + " " + what + " " + Twine(index)).str();
        if (!tensor) {
          stream << described << " is " << type
                 << "; only ranked tensors cross the JIT boundary\n";
          return failure();
        }
        if (failed(TensorBuffer::checkElementType(tensor.getElementType(),
                                                  described, stream)))
          return failure();
        into.push_back(tensor);
      }
      return success();
    };
    if (failed(record(function.getFunctionType().getInputs(), "argument",
                      signature.inputs)) ||
        failed(record(function.getFunctionType().getResults(), "result",
                      signature.results)))
      return fail("teraCompile: " + name + " cannot be called from here");
    handle->signatures[name] = std::move(signature);
  }

  SmallVector<std::string> libraries;
  for (size_t index = 0; index < numSharedLibs; index++)
    libraries.push_back(sharedLibs[index]);

  FailureOr<std::unique_ptr<JitInvoker>> invoker =
      JitInvoker::create(*handle->module, *backend,
                         targetOptions ? targetOptions : "", optLevel,
                         libraries);
  if (failed(invoker))
    return fail("teraCompile: the module does not lower");
  handle->invoker = std::move(*invoker);

  return handle.release();
}

TeraModule *teraCompile(const char *mlir, int target, unsigned optLevel,
                        const char *const *sharedLibs, size_t numSharedLibs) {
  static const char *const legacy[] = {"cpu", "cuda"};
  if (target < 0 || target >= static_cast<int>(std::size(legacy))) {
    lastError = "teraCompile: no target numbered " + std::to_string(target);
    return nullptr;
  }
  return teraCompileFor(mlir, legacy[target], "", optLevel, sharedLibs,
                        numSharedLibs);
}

const char *teraTargets(void) {
  lastTargets = registeredTargets();
  return lastTargets.c_str();
}

const char *teraTargetRuntimeLibraries(const char *target) {
  lastError.clear();
  lastTargets.clear();
  startUp();
  const TargetBackend *backend = target ? lookupTargetBackend(target) : nullptr;
  if (!backend) {
    lastError = "teraTargetRuntimeLibraries: no target named '" +
                std::string(target ? target : "") + "'";
    return nullptr;
  }
  for (StringRef library : backend->getRuntimeLibraries()) {
    if (!lastTargets.empty())
      lastTargets += ",";
    lastTargets += library.str();
  }
  return lastTargets.c_str();
}

void teraRelease(TeraModule *module) { delete module; }

const char *teraLastError(void) { return lastError.c_str(); }

int teraInvoke(TeraModule *module, const char *entry, void *const *inputs,
               const int64_t *inputShapes, int64_t numInputs,
               void *const *results, const int64_t *resultShapes,
               int64_t numResults) {
  lastError.clear();
  if (!module || !entry) {
    lastError = "teraInvoke: no module";
    return -1;
  }

  auto found = module->signatures.find(entry);
  if (found == module->signatures.end()) {
    lastError = ("teraInvoke: no function named " + Twine(entry)).str();
    return -1;
  }
  const Signature &signature = found->second;
  if (static_cast<int64_t>(signature.inputs.size()) != numInputs ||
      static_cast<int64_t>(signature.results.size()) != numResults) {
    lastError = (Twine(entry) + " takes " + Twine(signature.inputs.size()) +
                 " tensors and returns " + Twine(signature.results.size()) +
                 ", but the call passes " + Twine(numInputs) + " and " +
                 Twine(numResults))
                    .str();
    return -1;
  }

  SmallVector<SmallVector<uint64_t>> descriptors;
  SmallVector<MutableArrayRef<uint64_t>> inputDescriptors;
  SmallVector<MutableArrayRef<uint64_t>> resultDescriptors;
  descriptors.reserve(numInputs + numResults);

  for (auto [index, type] : llvm::enumerate(signature.inputs)) {
    if (!inputs[index]) {
      lastError = ("teraInvoke: input " + Twine(index) + " is null").str();
      return -1;
    }
    descriptors.push_back(TensorBuffer::describe(
        inputs[index], takeShape(inputShapes, type.getRank())));
  }
  for (RankedTensorType type : signature.results) {
    descriptors.push_back(SmallVector<uint64_t>(3 + 2 * type.getRank(), 0));
  }
  for (int64_t index = 0; index < numInputs; index++)
    inputDescriptors.push_back(descriptors[index]);
  for (int64_t index = 0; index < numResults; index++)
    resultDescriptors.push_back(descriptors[numInputs + index]);

  std::string diagnostics;
  llvm::raw_string_ostream stream(diagnostics);
  ScopedDiagnosticHandler collect(module->context.get(),
                                  [&](Diagnostic &diagnostic) {
                                    diagnostic.print(stream);
                                    stream << "\n";
                                    return success();
                                  });

  if (failed(module->invoker->invoke(entry, inputDescriptors,
                                     resultDescriptors))) {
    lastError = (Twine(entry) + " did not run\n" + diagnostics).str();
    return -1;
  }

  int status = 0;
  for (auto [index, type] : llvm::enumerate(signature.results)) {
    MutableArrayRef<uint64_t> descriptor = resultDescriptors[index];
    SmallVector<int64_t> extents = extentsOf(descriptor, type.getRank());
    ArrayRef<int64_t> declared = takeShape(resultShapes, type.getRank());

    void *allocated = reinterpret_cast<void *>(descriptor[0]);
    const char *data = reinterpret_cast<const char *>(descriptor[1]) +
                       descriptor[2] * (type.getElementTypeBitWidth() / 8);

    if (status == 0) {
      if (ArrayRef<int64_t>(extents) != declared) {
        lastError = (Twine(entry) + " result " + Twine(index) +
                     " came back with a shape the caller did not expect")
                        .str();
        status = -1;
      } else {
        SmallVector<uint64_t> expected =
            TensorBuffer::describe(allocated, extents);
        ArrayRef<uint64_t> strides =
            ArrayRef<uint64_t>(descriptor).slice(3 + type.getRank());
        if (strides != ArrayRef<uint64_t>(expected).slice(3 + type.getRank())) {
          lastError = (Twine(entry) + " result " + Twine(index) +
                       " is not contiguous")
                          .str();
          status = -1;
        } else {
          std::memcpy(results[index], data,
                      static_cast<size_t>(elementsIn(extents)) *
                          (type.getElementTypeBitWidth() / 8));
        }
      }
    }

    bool aliasesInput = false;
    for (int64_t other = 0; other < numInputs; other++)
      aliasesInput |= inputs[other] == allocated;
    if (allocated && !aliasesInput)
      module->invoker->releaseAllocation(allocated);
  }
  return status;
}
