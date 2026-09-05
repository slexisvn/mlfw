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
                SmallVectorImpl<Value> &forward,
                SmallVectorImpl<Operation *> *cloned = nullptr);

LogicalResult differentiateBlock(
    OpBuilder &builder, Block &block, ValueRange arguments,
    ValueRange resultAdjoints, SmallVectorImpl<Value> &forward,
    SmallVectorImpl<Value> &argumentAdjoints,
    function_ref<bool(Operation *)> isActive = nullptr,
    SmallVectorImpl<Operation *> *clonedForward = nullptr);

SmallVector<Value> fillMissingWithZero(OpBuilder &builder, Location loc,
                                       ValueRange values, ValueRange like);

LogicalResult jvpBlock(OpBuilder &builder, Block &block, ValueRange arguments,
                       ValueRange argumentTangents,
                       SmallVectorImpl<Value> &forward,
                       SmallVectorImpl<Value> &forwardTangents);

}

#endif
