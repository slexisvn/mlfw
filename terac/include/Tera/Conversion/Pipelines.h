//===- Pipelines.h - Tera lowering pipelines --------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TERA_CONVERSION_PIPELINES_H
#define TERA_CONVERSION_PIPELINES_H

#include "mlir/Pass/PassOptions.h"

namespace mlir {
class OpPassManager;

namespace tera {
enum class Target { CPU, CUDA };

struct TeraToNVVMOptions : public PassPipelineOptions<TeraToNVVMOptions> {
  Option<std::string> chip{
      *this, "chip",
      llvm::cl::desc("NVIDIA architecture the kernels are compiled for"),
      llvm::cl::init("sm_86")};

  ListOption<int64_t> tileSizes{
      *this, "tile-sizes",
      llvm::cl::desc("Thread-block tile, one extent per parallel dimension")};
};

void buildTeraToLLVMPipeline(OpPassManager &pm);

void buildTeraToNVVMPipeline(OpPassManager &pm,
                             const TeraToNVVMOptions &options);

void registerTeraPipelines();

}
}

#endif
