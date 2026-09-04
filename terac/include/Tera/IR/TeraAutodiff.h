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

/// Clones \p block at the insertion point, substituting \p arguments for block
/// arguments and appending the cloned yielded values to \p forward.
void cloneBlock(OpBuilder &builder, Block &block, ValueRange arguments,
                SmallVectorImpl<Value> &forward);

/// Clones and differentiates \p block, substituting \p arguments for its
/// inputs.
/// Result adjoints follow yield order; null entries mean no gradient.
/// Outputs are cloned yields in \p forward and gradients in block-argument
/// order in \p argumentAdjoints, with null for unreached arguments.
LogicalResult differentiateBlock(OpBuilder &builder, Block &block,
                                 ValueRange arguments,
                                 ValueRange resultAdjoints,
                                 SmallVectorImpl<Value> &forward,
                                 SmallVectorImpl<Value> &argumentAdjoints);

/// Replaces null entries in \p values with zeros shaped like matching \p like
/// values, including their runtime dimensions.
SmallVector<Value> fillMissingWithZero(OpBuilder &builder, Location loc,
                                       ValueRange values, ValueRange like);

} // namespace mlir::tera

#endif // TERA_IR_TERAAUTODIFF_H
