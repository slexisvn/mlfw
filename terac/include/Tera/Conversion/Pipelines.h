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

/// Which machine a module is lowered for. The two pipelines below share
/// everything down to bufferized linalg and agree on nothing under it, so the
/// choice is made once by the driver and carried, never re-derived from the IR.
enum class Target { CPU, CUDA };

struct TeraToNVVMOptions : public PassPipelineOptions<TeraToNVVMOptions> {
  Option<std::string> chip{
      *this, "chip",
      llvm::cl::desc("NVIDIA architecture the kernels are compiled for"),
      llvm::cl::init("sm_86")};

  /// A thread block is a tile of the parallel loop nest, so with no tiling at
  /// all every kernel launches with one thread per block and uses one lane of
  /// each warp. The catch is that `convert-parallel-loops-to-gpu` gives up
  /// SILENTLY on a loop whose extent a tile exceeds — no diagnostic, the loop
  /// simply stays on the host — so a tile that suits one model's shapes can
  /// quietly un-map another's. Hence a conservative default rather than a fast
  /// one: it is the largest that maps every model in test/Integration.
  ListOption<int64_t> tileSizes{
      *this, "tile-sizes",
      llvm::cl::desc("Thread-block tile, one extent per parallel dimension")};
};

/// Adds passes to lower tera to the LLVM dialect.
void buildTeraToLLVMPipeline(OpPassManager &pm);

/// Adds passes to lower tera to NVVM and serialize kernels into gpu.binary.
/// The driver must register GPU translations before translating to LLVM IR.
void buildTeraToNVVMPipeline(OpPassManager &pm,
                             const TeraToNVVMOptions &options);

/// Registers the tera-to-llvm and tera-to-nvvm pipelines.
void registerTeraPipelines();

} // namespace tera
} // namespace mlir

#endif // TERA_CONVERSION_PIPELINES_H
