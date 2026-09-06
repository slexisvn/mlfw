//===- Pipelines.cpp - Tera lowering pipelines ------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Tera/Pipelines/Pipelines.h"

#include "Tera/Conversion/Passes.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/Affine/Transforms/Passes.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/Dialect/Bufferization/Pipelines/Passes.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/Pipelines/Passes.h"
#include "mlir/Dialect/GPU/Transforms/Passes.h"
#include "mlir/Dialect/LLVMIR/Transforms/Passes.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Math/Transforms/Passes.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/SCF/Transforms/Passes.h"
#include "mlir/Dialect/Vector/Transforms/Passes.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;

static void buildTeraToLinalg(OpPassManager &pm) {
  pm.addPass(createInlinerPass());
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

/// Nothing below here computes in bfloat16, and no machine has a `tanh` for a
/// float narrower than f32. Both are promoted to f32 and rounded back after
/// every operation, which is what the narrow arithmetic meant anyway; what is
/// left is the pair of conversions, and bfloat16's are expanded into the
/// integer shifts they are rather than left as a call to a compiler-rt builtin
/// the JIT has no library to find it in.
static void buildNarrowFloats(OpPassManager &pm) {
  math::MathExtendToSupportedTypesOptions widening;
  widening.targetTypeStr = "f32";
  pm.addPass(math::createMathExtendToSupportedTypes(widening));

  arith::ArithEmulateUnsupportedFloatsOptions promoting;
  promoting.sourceTypeStrs = {"bf16"};
  promoting.targetTypeStr = "f32";
  pm.addPass(arith::createArithEmulateUnsupportedFloats(promoting));

  arith::ArithExpandOpsPassOptions expanding;
  expanding.includeBf16 = true;
  pm.addPass(arith::createArithExpandOpsPass(expanding));
}

static void buildFuseOnTensors(OpPassManager &pm) {
  pm.addPass(createLinalgGeneralizeNamedOpsPass());
  pm.addPass(createLinalgElementwiseOpFusionPass());
  buildCleanup(pm);
}

void tera::buildTeraToLLVMPipeline(OpPassManager &pm,
                                   const TeraToLLVMOptions &options) {
  buildTeraToLinalg(pm);
  buildCleanup(pm);
  buildFuseOnTensors(pm);

  if (!options.schedule.empty()) {
    ApplyScheduleOptions scheduling;
    scheduling.schedule = options.schedule;
    pm.addPass(tera::createApplySchedule(scheduling));
    buildCleanup(pm);
  } else if (options.tile) {
    TileAndFuseOptions tiling;
    tiling.vectorWidth = options.vectorWidth;
    pm.nest<func::FuncOp>().addPass(tera::createTileAndFuse(tiling));
    buildCleanup(pm);
  }

  VectorizeLinalgOptions vectorizing;
  vectorizing.maxVectorElements = options.maxVectorElements;
  pm.nest<func::FuncOp>().addPass(tera::createVectorizeLinalg(vectorizing));
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

  buildNarrowFloats(pm);

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
  buildCleanup(pm);
  buildFuseOnTensors(pm);

  buildBufferize(pm);

  // What a script stands in for here is not one step but the whole run from
  // the tiling to the launch: it cuts the loops and maps them to processors
  // itself, through the GPU transform ops, where the passes below do the first
  // half and leave the second to `convert-parallel-loops-to-gpu`. Either way
  // what is left underneath is the same, which is what makes the two
  // interchangeable rather than two pipelines.
  //
  // Both run below bufferization, unlike the host pipeline's, because that is
  // where a shared tile can exist at all: it is written by every thread of a
  // block and read by every thread, and a value in SSA has one producer.
  if (!options.schedule.empty()) {
    ApplyScheduleOptions scheduling;
    scheduling.schedule = options.schedule;
    pm.addPass(tera::createApplySchedule(scheduling));
  } else {
    if (options.sharedTiles)
      pm.nest<func::FuncOp>().addPass(tera::createTileContractionToShared());

    pm.addPass(createConvertLinalgToParallelLoopsPass());

    pm.addPass(memref::createFoldMemRefAliasOpsPass());

    TileParallelLoopsOptions tiling;
    tiling.tileSizes = SmallVector<int64_t>(options.tileSizes.begin(),
                                            options.tileSizes.end());
    tiling.threadsPerBlock = options.threadsPerBlock;
    pm.nest<func::FuncOp>().addPass(tera::createTileParallelLoops(tiling));

    GpuMapParallelLoopsPassOptions mapping;
    mapping.mappingPolicyStr = "innermost-first";
    pm.nest<func::FuncOp>().addPass(createGpuMapParallelLoopsPass(mapping));
    pm.addPass(createConvertParallelLoopToGpuPass());
  }
  pm.addPass(createConvertLinalgToLoopsPass());
  pm.nest<func::FuncOp>().addPass(tera::createVerifyGpuMapping());

  pm.addPass(createCanonicalizerPass());

  pm.addPass(tera::createAttachWorkgroupMemory());

  // Outlining hands the kernel everything the launch body read from around it
  // as an argument, and this pass copies the constants among them inside
  // first. What that decides is whether a loop bound reaches the device as a
  // number or as an argument, and a trip count nothing can read is one nothing
  // unrolls: the loop over a staged tile counts to the tile, which is small
  // enough that the branch and the induction variable are most of it.
  pm.addPass(createGpuLaunchSinkIndexComputationsPass());
  pm.addPass(createGpuKernelOutliningPass());

  pm.addPass(tera::createStageGpuBuffers());

  pm.nest<func::FuncOp>().addPass(createGpuAsyncRegionPass());

  pm.nest<func::FuncOp>().addPass(LLVM::createLLVMRequestCWrappersPass());

  pm.addPass(memref::createExpandStridedMetadataPass());

  buildNarrowFloats(pm);

  gpu::GPUToNVVMPipelineOptions nvvmOptions;
  nvvmOptions.cubinChip = options.chip;
  nvvmOptions.cubinFormat = "fatbin";
  gpu::buildLowerToNVVMPassPipeline(pm, nvvmOptions);
}

void tera::registerTeraPipelines() {
  PassPipelineRegistration<TeraToLLVMOptions>(
      "tera-to-llvm",
      "Lower a tera module all the way to the LLVM dialect, ready to JIT.",
      buildTeraToLLVMPipeline);

  PassPipelineRegistration<TeraToNVVMOptions>(
      "tera-to-nvvm",
      "Lower a tera module to NVVM with its kernels serialised, ready to JIT.",
      buildTeraToNVVMPipeline);
}
