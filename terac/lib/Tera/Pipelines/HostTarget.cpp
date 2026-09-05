//===- HostTarget.cpp - Building for the machine this runs on ---*- C++ -*-===//
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
struct HostTarget : TargetBackend {
  StringRef getName() const override { return "cpu"; }

  StringRef getDescription() const override {
    return "Lower to the LLVM dialect and JIT it";
  }

  LogicalResult buildPipeline(OpPassManager &pm, StringRef text,
                              raw_ostream &errorStream) const override {
    TeraToLLVMOptions options;
    if (failed(options.parseFromString(text, errorStream)))
      return failure();
    buildTeraToLLVMPipeline(pm, options);
    return success();
  }

  ArrayRef<StringRef> getRuntimeLibraries() const override {
    static const StringRef libraries[] = {"mlir_c_runner_utils"};
    return libraries;
  }
};

}

void mlir::tera::detail::registerHostTarget() {
  registerTargetBackend(std::make_unique<HostTarget>());
}
