//===- Activity.cpp - Which values a gradient can reach ---------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Tera/Analysis/Activity.h"

using namespace mlir;
using namespace mlir::tera;

DenseSet<Operation *>
mlir::tera::activeOperations(Block &block, ArrayRef<int64_t> activeArguments) {
  DenseSet<Value> carriers;
  for (int64_t position : activeArguments)
    if (position >= 0 && position < block.getNumArguments())
      carriers.insert(block.getArgument(position));

  DenseSet<Operation *> active;
  for (Operation &op : block) {
    if (llvm::none_of(op.getOperands(),
                      [&](Value operand) { return carriers.contains(operand); }))
      continue;
    active.insert(&op);
    for (Value result : op.getResults())
      carriers.insert(result);
  }
  return active;
}
