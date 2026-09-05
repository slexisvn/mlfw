//===- JitInvoker.cpp - Lower, JIT and call a tera module -------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Tera/Execution/JitInvoker.h"

#include "Tera/IR/TeraDialect.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Pass/PassManager.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdlib>

using namespace mlir;
using namespace mlir::tera;

/// A buffer can only be left somewhere there is a device to leave it on. The
/// question is asked here, where the module first meets the target it is being
/// built for, so the answer names both rather than arriving as a type error
/// from the bottom of the lowering.
static LogicalResult checkResidence(ModuleOp module,
                                    const TargetBackend &target) {
  if (target.hasDeviceMemory())
    return success();

  WalkResult walked = module.walk([&](FunctionOpInterface function) {
    for (unsigned index = 0; index < function.getNumArguments(); ++index) {
      if (!function.getArgAttr(index, TeraDialect::kDeviceResidentAttrName))
        continue;
      function.emitError() << "argument " << index << " is marked '"
                           << TeraDialect::kDeviceResidentAttrName
                           << "', and target '" << target.getName()
                           << "' has no device to leave it on";
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return failure(walked.wasInterrupted());
}

FailureOr<std::unique_ptr<JitInvoker>>
JitInvoker::create(ModuleOp module, const TargetBackend &target,
                   StringRef targetOptions, unsigned optLevel,
                   ArrayRef<std::string> sharedLibs) {
  if (failed(checkResidence(module, target)))
    return failure();

  PassManager pm(module.getContext());
  if (failed(applyPassManagerCLOptions(pm)))
    return failure();
  if (failed(target.buildPipeline(pm, targetOptions, llvm::errs())))
    return failure();
  if (failed(pm.run(module)))
    return failure();

  SmallVector<StringRef> libraries(sharedLibs.begin(), sharedLibs.end());
  ExecutionEngineOptions options;
  options.transformer = makeOptimizingTransformer(optLevel, /*sizeLevel=*/0,
                                                  /*targetMachine=*/nullptr);
  options.sharedLibPaths = libraries;

  llvm::Expected<std::unique_ptr<ExecutionEngine>> engine =
      ExecutionEngine::create(module, options);
  if (!engine) {
    llvm::errs() << toString(engine.takeError()) << "\n";
    return failure();
  }
  (*engine)->initialize();
  return std::unique_ptr<JitInvoker>(
      new JitInvoker(std::move(*engine), target.hasDeviceMemory()));
}

DeviceMemory *JitInvoker::getDeviceMemory() {
  if (!deviceResolved) {
    if (onADevice)
      device = DeviceMemory::resolve(*engine);
    deviceResolved = true;
  }
  return device.get();
}

void JitInvoker::releaseAllocation(void *pointer) {
  if (!pointer)
    return;
  if (!deallocate) {
    llvm::Expected<void *> found = engine->lookup("free");
    if (found) {
      deallocate = reinterpret_cast<void (*)(void *)>(*found);
    } else {
      llvm::consumeError(found.takeError());
      deallocate = std::free;
    }
  }
  deallocate(pointer);
}

LogicalResult JitInvoker::invoke(StringRef name,
                                 MutableArrayRef<TensorBuffer> inputs,
                                 MutableArrayRef<TensorBuffer> results) {
  SmallVector<MutableArrayRef<uint64_t>> inputDescriptors;
  SmallVector<MutableArrayRef<uint64_t>> resultDescriptors;
  for (TensorBuffer &input : inputs)
    inputDescriptors.push_back(input.getDescriptor());
  for (TensorBuffer &result : results)
    resultDescriptors.push_back(result.getDescriptor());
  return invoke(name, inputDescriptors, resultDescriptors);
}

LogicalResult
JitInvoker::invoke(StringRef name, ArrayRef<MutableArrayRef<uint64_t>> inputs,
                   MutableArrayRef<MutableArrayRef<uint64_t>> results) {
  SmallVector<uint64_t> packed;
  for (MutableArrayRef<uint64_t> result : results)
    packed.append(result.begin(), result.end());

  SmallVector<void *> values;
  if (!results.empty())
    values.push_back(packed.data());
  for (MutableArrayRef<uint64_t> input : inputs)
    values.push_back(input.data());

  SmallVector<void *> arguments;
  arguments.reserve(values.size());
  for (void *&value : values)
    arguments.push_back(&value);

  auto entry = resolved.find(name);
  if (entry == resolved.end()) {
    llvm::Expected<void (*)(void **)> found =
        engine->lookupPacked(("_mlir_ciface_" + name).str());
    if (!found) {
      llvm::errs() << toString(found.takeError()) << "\n";
      return failure();
    }
    entry = resolved.try_emplace(name, *found).first;
  }
  entry->second(arguments.data());

  size_t offset = 0;
  for (MutableArrayRef<uint64_t> result : results) {
    llvm::copy(ArrayRef<uint64_t>(packed).slice(offset, result.size()),
               result.begin());
    offset += result.size();
  }
  return success();
}
