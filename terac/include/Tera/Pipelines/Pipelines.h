//===- Pipelines.h - Tera lowering pipelines --------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TERA_PIPELINES_PIPELINES_H
#define TERA_PIPELINES_PIPELINES_H

#include "mlir/Pass/PassOptions.h"

#include <string>

namespace mlir {
class OpPassManager;

namespace tera {
struct TeraToLLVMOptions : public PassPipelineOptions<TeraToLLVMOptions> {
  Option<int64_t> vectorWidth{
      *this, "vector-width",
      llvm::cl::desc("Lanes the innermost tiled loop is cut to"),
      llvm::cl::init(16)};

  Option<int64_t> maxVectorElements{
      *this, "max-vector-elements",
      llvm::cl::desc("Iteration-space size above which an op keeps its loops"),
      llvm::cl::init(1024)};

  Option<bool> tile{
      *this, "tile",
      llvm::cl::desc("Cut the linalg ops into tiles before vectorising them"),
      llvm::cl::init(true)};

  Option<std::string> schedule{
      *this, "schedule",
      llvm::cl::desc("Path to a transform module scheduling the module, "
                     "instead of the passes that would have"),
      llvm::cl::init("")};
};

struct TeraToNVVMOptions : public PassPipelineOptions<TeraToNVVMOptions> {
  Option<std::string> chip{
      *this, "chip",
      llvm::cl::desc("NVIDIA architecture the kernels are compiled for"),
      llvm::cl::init("sm_86")};

  ListOption<int64_t> tileSizes{
      *this, "tile-sizes",
      llvm::cl::desc("Thread-block tile, one extent per parallel dimension, "
                     "instead of the one derived from each loop")};

  Option<int64_t> threadsPerBlock{
      *this, "threads-per-block",
      llvm::cl::desc("Threads a derived thread block is aimed at"),
      llvm::cl::init(256)};

  Option<bool> sharedTiles{
      *this, "shared-tiles",
      llvm::cl::desc("Cut every contraction into blocks that stage their "
                     "operands in shared memory"),
      llvm::cl::init(true)};

  Option<std::string> schedule{
      *this, "schedule",
      llvm::cl::desc("Path to a transform module scheduling the module, "
                     "instead of the passes that would have"),
      llvm::cl::init("")};
};

void buildTeraToLLVMPipeline(OpPassManager &pm,
                             const TeraToLLVMOptions &options);

void buildTeraToNVVMPipeline(OpPassManager &pm,
                             const TeraToNVVMOptions &options);

void registerTeraPipelines();

}
}

#endif
