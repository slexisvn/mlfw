//===- TeraOpsAutodiff.cpp - Ops that steer autodiff ------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Tera/IR/TeraOps.h"

using namespace mlir;
using namespace mlir::tera;

LogicalResult StopGradientOp::buildVjp(OpBuilder &, ValueRange,
                                       SmallVectorImpl<Value> &operandAdjoints) {
  operandAdjoints.assign({Value()});
  return success();
}

LogicalResult StopGradientOp::buildJvp(OpBuilder &, ValueRange,
                                       SmallVectorImpl<Value> &resultTangents,
                                       SmallVectorImpl<Value> &) {
  resultTangents.assign({Value()});
  return success();
}
