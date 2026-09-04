//===- TeraOps.h - Tera dialect ops -----------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TERA_IR_TERAOPS_H
#define TERA_IR_TERAOPS_H

#include "Tera/IR/TeraEnums.h"
#include "Tera/IR/TeraInterfaces.h"
#include "Tera/IR/TeraTraits.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#define GET_OP_CLASSES
#include "Tera/IR/TeraOps.h.inc"

namespace mlir::tera {
LogicalResult verifySizesClause(Operation *op, ValueRange sizes);

Value createSplat(OpBuilder &builder, Location loc, RankedTensorType type,
                  double value, Value like = {});

}

#endif
