//===- Pipelines.cpp - Tera lowering pipelines ------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Tera/Conversion/Pipelines.h"

#include "Tera/Conversion/Passes.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/Affine/Transforms/Passes.h"
#include "mlir/Dialect/Bufferization/Pipelines/Passes.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/Pipelines/Passes.h"
#include "mlir/Dialect/GPU/Transforms/Passes.h"
#include "mlir/Dialect/LLVMIR/Transforms/Passes.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/SCF/Transforms/Passes.h"
#include "mlir/Dialect/Vector/Transforms/Passes.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;

static void buildTeraToLinalg(OpPassManager &pm) {
  pm.addPass(createCanonicalizerPass());
  pm.addPass(tera::createConvertTeraToLinalg());
}

static void buildBufferize(OpPassManager &pm) {
  bufferization::OneShotBufferizePassOptions bufferizeOptions;
  bufferizeOptions.bufferizeFunctionBoundaries = true;
  bufferizeOptions.allowReturnAllocsFromLoops = true;
  bufferizeOptions.functionBoundaryTypeConversion =
      bufferization::LayoutMapOption::IdentityLayoutMap;
  pm.addPass(bufferization::createOneShotBufferizePass(bufferizeOptions));
  bufferization::buildBufferDeallocationPipeline(pm);

  pm.addPass(createConvertBufferizationToMemRefPass());
}

static void buildCleanup(OpPassManager &pm) {
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());
}

void tera::buildTeraToLLVMPipeline(OpPassManager &pm) {
  buildTeraToLinalg(pm);

  pm.addPass(createLinalgElementwiseOpFusionPass());
  buildCleanup(pm);

  pm.nest<func::FuncOp>().addPass(tera::createTileAndFuse());
  buildCleanup(pm);

  pm.nest<func::FuncOp>().addPass(tera::createVectorizeLinalg());
  buildCleanup(pm);

  buildBufferize(pm);

  pm.addPass(createConvertLinalgToLoopsPass());
  pm.addPass(createCanonicalizerPass());

  pm.nest<func::FuncOp>().addPass(
      vector::createLowerVectorMultiReductionPass());
  pm.addPass(createLowerAffinePass());
  pm.addPass(createConvertVectorToSCFPass());
  pm.addPass(createSCFToControlFlowPass());
  pm.addPass(memref::createExpandStridedMetadataPass());
  pm.addPass(createLowerAffinePass());
  pm.nest<func::FuncOp>().addPass(LLVM::createLLVMRequestCWrappersPass());

  pm.addPass(createConvertVectorToLLVMPass());
  pm.addPass(createUBToLLVMConversionPass());
  pm.addPass(createFinalizeMemRefToLLVMConversionPass());
  pm.addPass(createConvertFuncToLLVMPass());
  pm.addPass(createArithToLLVMConversionPass());
  pm.addPass(createConvertMathToLLVMPass());
  pm.addPass(createConvertControlFlowToLLVMPass());
  pm.addPass(createReconcileUnrealizedCastsPass());
}

void tera::buildTeraToNVVMPipeline(OpPassManager &pm,
                                   const TeraToNVVMOptions &options) {
  buildTeraToLinalg(pm);
  buildBufferize(pm);

  pm.addPass(createConvertLinalgToParallelLoopsPass());

  pm.addPass(memref::createFoldMemRefAliasOpsPass());

  SmallVector<int64_t> tileSizes(options.tileSizes.begin(),
                                 options.tileSizes.end());
  if (tileSizes.empty())
    tileSizes = {2, 2};
  pm.addPass(createParallelLoopTilingPass(tileSizes));

  pm.addPass(createCanonicalizerPass());

  pm.nest<func::FuncOp>().addPass(createGpuMapParallelLoopsPass());
  pm.addPass(createConvertParallelLoopToGpuPass());

  pm.addPass(createGpuKernelOutliningPass());

  pm.nest<func::FuncOp>().addPass(tera::createStageGpuBuffers());

  pm.nest<func::FuncOp>().addPass(createGpuAsyncRegionPass());

  pm.nest<func::FuncOp>().addPass(LLVM::createLLVMRequestCWrappersPass());

  pm.addPass(memref::createExpandStridedMetadataPass());

  gpu::GPUToNVVMPipelineOptions nvvmOptions;
  nvvmOptions.cubinChip = options.chip;
  nvvmOptions.cubinFormat = "fatbin";
  gpu::buildLowerToNVVMPassPipeline(pm, nvvmOptions);
}

void tera::registerTeraPipelines() {
  PassPipelineRegistration<>(
      "tera-to-llvm",
      "Lower a tera module all the way to the LLVM dialect, ready to JIT.",
      buildTeraToLLVMPipeline);

  PassPipelineRegistration<TeraToNVVMOptions>(
      "tera-to-nvvm",
      "Lower a tera module to NVVM with its kernels serialised, ready to JIT.",
      buildTeraToNVVMPipeline);
}
