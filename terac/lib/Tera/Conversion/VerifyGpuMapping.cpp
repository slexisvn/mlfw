//===- VerifyGpuMapping.cpp - Find the loops still on the host --*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Tera/Conversion/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/GPU/Transforms/ParallelLoopMapper.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

namespace mlir::tera {
#define GEN_PASS_DEF_VERIFYGPUMAPPING
#include "Tera/Conversion/Passes.h.inc"

namespace {
struct VerifyGpuMapping
    : public impl::VerifyGpuMappingBase<VerifyGpuMapping> {
  using impl::VerifyGpuMappingBase<VerifyGpuMapping>::VerifyGpuMappingBase;

  void runOnOperation() final {
    WalkResult walked = getOperation().walk([](scf::ParallelOp loop) {
      if (!loop->hasAttr(gpu::getMappingAttrName()))
        return WalkResult::advance();
      loop.emitError() << "carries a processor mapping but is still a loop, "
                          "so it stayed on the host while the rest of the "
                          "function went to the device";
      return WalkResult::interrupt();
    });
    if (walked.wasInterrupted())
      signalPassFailure();
  }
};

}
}
