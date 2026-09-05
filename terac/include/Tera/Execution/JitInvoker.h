//===- JitInvoker.h - Lower, JIT and call a tera module ---------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TERA_EXECUTION_JITINVOKER_H
#define TERA_EXECUTION_JITINVOKER_H

#include "Tera/Pipelines/TargetBackend.h"
#include "Tera/Execution/TensorBuffer.h"

#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/StringMap.h"

#include <memory>
#include <string>

namespace mlir::tera {
class JitInvoker {
public:
  static FailureOr<std::unique_ptr<JitInvoker>>
  create(ModuleOp module, const TargetBackend &target, StringRef targetOptions,
         unsigned optLevel, ArrayRef<std::string> sharedLibs);

  LogicalResult invoke(StringRef name, MutableArrayRef<TensorBuffer> inputs,
                       MutableArrayRef<TensorBuffer> results);

  LogicalResult invoke(StringRef name,
                       ArrayRef<MutableArrayRef<uint64_t>> inputs,
                       MutableArrayRef<MutableArrayRef<uint64_t>> results);

  void releaseAllocation(void *pointer);

private:
  explicit JitInvoker(std::unique_ptr<ExecutionEngine> engine)
      : engine(std::move(engine)) {}

  llvm::StringMap<void (*)(void **)> resolved;
  void (*deallocate)(void *) = nullptr;
  std::unique_ptr<ExecutionEngine> engine;
};

}

#endif
