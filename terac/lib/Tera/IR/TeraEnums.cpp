//===- TeraEnums.cpp - Tera dialect enums and attributes --------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Tera/IR/TeraEnums.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::tera;

#include "Tera/IR/TeraEnums.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "Tera/IR/TeraEnumAttrs.cpp.inc"

void TeraDialect::registerAttributes() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "Tera/IR/TeraEnumAttrs.cpp.inc"
      >();
}
