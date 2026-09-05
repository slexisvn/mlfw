//===- CudaTarget.cpp - Building for an NVIDIA device -----------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Tera/Pipelines/TargetBackend.h"

#include "Tera/Pipelines/Pipelines.h"
#include "TargetsDetail.h"

using namespace mlir;
using namespace mlir::tera;

namespace {
struct CudaTarget : TargetBackend {
  StringRef getName() const override { return "cuda"; }

  StringRef getDescription() const override {
    return "Lower to NVVM, serialise the kernels, and JIT the host";
  }

  LogicalResult buildPipeline(OpPassManager &pm, StringRef text,
                              raw_ostream &errorStream) const override {
    TeraToNVVMOptions options;
    if (failed(options.parseFromString(text, errorStream)))
      return failure();
    buildTeraToNVVMPipeline(pm, options);
    return success();
  }

  bool hasDeviceMemory() const override { return true; }

  ArrayRef<StringRef> getRuntimeLibraries() const override {
    static const StringRef libraries[] = {"mlir_c_runner_utils",
                                          "mlir_cuda_runtime"};
    return libraries;
  }
};

}

void mlir::tera::detail::registerCudaTarget() {
  registerTargetBackend(std::make_unique<CudaTarget>());
}
