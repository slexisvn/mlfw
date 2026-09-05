//===- TeraInterfaces.h - Tera op interfaces --------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TERA_IR_TERAINTERFACES_H
#define TERA_IR_TERAINTERFACES_H

#include "Tera/IR/TeraTraits.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpDefinition.h"

namespace mlir::tera {
LogicalResult buildStructuralJvp(Operation *op, OpBuilder &builder,
                                 ValueRange tangents,
                                 SmallVectorImpl<Value> &resultTangents);

}

#include "Tera/IR/TeraInterfaces.h.inc"

#endif
