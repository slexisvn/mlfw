//===- JitInvoker.h - Lower, JIT and call a tera module ---------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TERA_EXECUTION_JITINVOKER_H
#define TERA_EXECUTION_JITINVOKER_H

#include "Tera/Conversion/Pipelines.h"
#include "Tera/Execution/TensorBuffer.h"

#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/StringMap.h"

#include <memory>
#include <string>

namespace mlir::tera {

/// Lowers a tera module to the LLVM dialect, JITs it, and calls its functions
/// on tensors. Every driver goes through this, so a module reaching the JIT
/// has been through exactly one pipeline and one calling convention.
class JitInvoker {
public:
  /// Lowers \p module in place and creates a JIT engine for \p target.
  /// Failure is diagnosed. CUDA requires mlir_cuda_runtime in \p sharedLibs.
  static FailureOr<std::unique_ptr<JitInvoker>>
  create(ModuleOp module, Target target, unsigned optLevel,
         ArrayRef<std::string> sharedLibs);

  /// Invokes \p name with tensor buffers. Result allocations are not freed
  /// automatically; release them with releaseAllocation.
  LogicalResult invoke(StringRef name, MutableArrayRef<TensorBuffer> inputs,
                       MutableArrayRef<TensorBuffer> results);

  /// Invokes \p name with memref descriptors over caller-owned input storage.
  /// The caller must release returned allocations with releaseAllocation.
  LogicalResult invoke(StringRef name,
                       ArrayRef<MutableArrayRef<uint64_t>> inputs,
                       MutableArrayRef<MutableArrayRef<uint64_t>> results);

  /// Frees a returned allocation using the JIT runtime allocator.
  void releaseAllocation(void *pointer);

private:
  explicit JitInvoker(std::unique_ptr<ExecutionEngine> engine)
      : engine(std::move(engine)) {}

  /// Entry points already resolved. Looking a symbol up is what makes the JIT
  /// compile it, so a caller timing a function would otherwise pay for that
  /// compilation inside its first measurement.
  llvm::StringMap<void (*)(void **)> resolved;
  void (*deallocate)(void *) = nullptr;
  std::unique_ptr<ExecutionEngine> engine;
};

} // namespace mlir::tera

#endif // TERA_EXECUTION_JITINVOKER_H
