//===- TeraAutodiff.h - Reverse-mode AD over a block ------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TERA_IR_TERAAUTODIFF_H
#define TERA_IR_TERAAUTODIFF_H

#include "Tera/IR/TeraOps.h"

namespace mlir::tera {
void cloneBlock(OpBuilder &builder, Block &block, ValueRange arguments,
                SmallVectorImpl<Value> &forward);

LogicalResult differentiateBlock(OpBuilder &builder, Block &block,
                                 ValueRange arguments,
                                 ValueRange resultAdjoints,
                                 SmallVectorImpl<Value> &forward,
                                 SmallVectorImpl<Value> &argumentAdjoints);

SmallVector<Value> fillMissingWithZero(OpBuilder &builder, Location loc,
                                       ValueRange values, ValueRange like);

}

#endif
