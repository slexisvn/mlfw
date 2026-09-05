//===- Activity.h - Which values a gradient can reach -----------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TERA_ANALYSIS_ACTIVITY_H
#define TERA_ANALYSIS_ACTIVITY_H

#include "mlir/IR/Block.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/DenseSet.h"

namespace mlir::tera {
DenseSet<Operation *> activeOperations(Block &block,
                                       ArrayRef<int64_t> activeArguments);

}

#endif
