//===- AutodiffDetail.h - What both directions agree on ---------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TERA_LIB_TRANSFORMS_AUTODIFFDETAIL_H
#define TERA_LIB_TRANSFORMS_AUTODIFFDETAIL_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "llvm/ADT/DenseSet.h"

namespace mlir::tera {
FailureOr<SmallVector<int64_t>> differentiableArguments(func::FuncOp func);

DenseSet<Operation *> deadOperations(Block &block);

void eraseDeadOperations(Block &block);

}

#endif
