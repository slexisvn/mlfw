//===- TeraEnums.h - Tera dialect enums and attributes ----------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TERA_IR_TERAENUMS_H
#define TERA_IR_TERAENUMS_H

#include "Tera/IR/TeraDialect.h"
#include "mlir/IR/BuiltinAttributes.h"

#include "Tera/IR/TeraEnums.h.inc"

#define GET_ATTRDEF_CLASSES
#include "Tera/IR/TeraEnumAttrs.h.inc"

#endif // TERA_IR_TERAENUMS_H
