//===- tera-opt.cpp - Optimizer driver for the tera dialect -----*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/InitAllDialects.h"
#include "mlir/InitAllExtensions.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Target/LLVMIR/Dialect/All.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

#include "Tera/Conversion/Passes.h"
#include "Tera/Pipelines/Pipelines.h"
#include "Tera/IR/TeraDialect.h"
#include "Tera/Transforms/Passes.h"

int main(int argc, char **argv) {
  mlir::registerAllPasses();
  mlir::tera::registerTeraPasses();
  mlir::tera::registerTeraConversionPasses();
  mlir::tera::registerTeraPipelines();

  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  mlir::registerAllExtensions(registry);
  mlir::registerAllGPUToLLVMIRTranslations(registry);
  registry.insert<mlir::tera::TeraDialect>();

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "Tera optimizer driver\n", registry));
}
