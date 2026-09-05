//===- TeraDialect.cpp - Tera dialect ---------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Tera/IR/TeraDialect.h"
#include "Tera/IR/TeraEnums.h"
#include "Tera/IR/TeraOps.h"

#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Transforms/InliningUtils.h"

using namespace mlir;
using namespace mlir::tera;

#include "Tera/IR/TeraOpsDialect.cpp.inc"

namespace {
struct TeraInlinerInterface : public DialectInlinerInterface {
  using DialectInlinerInterface::DialectInlinerInterface;

  bool isLegalToInline(Operation *, Region *, bool, IRMapping &) const final {
    return true;
  }

  bool isLegalToInline(Region *, Region *, bool, IRMapping &) const final {
    return true;
  }
};

}

void TeraDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "Tera/IR/TeraOps.cpp.inc"
      >();
  addInterfaces<TeraInlinerInterface>();
  registerAttributes();
}

Operation *TeraDialect::materializeConstant(OpBuilder &builder, Attribute value,
                                            Type type, Location loc) {
  auto elements = dyn_cast<ElementsAttr>(value);
  auto tensorType = dyn_cast<RankedTensorType>(type);
  if (!elements || !tensorType || elements.getShapedType() != tensorType)
    return nullptr;
  return ConstantOp::create(builder, loc, tensorType, elements);
}

LogicalResult TeraDialect::verifyOperationAttribute(Operation *op,
                                                    NamedAttribute attribute) {
  StringRef name = attribute.getName();
  Attribute value = attribute.getValue();

  auto belongsHere = [&]() -> LogicalResult {
    if (isa<FunctionOpInterface>(op))
      return success();
    return op->emitError() << "'" << name
                           << "' belongs on a function, not on this operation";
  };

  if (name == kDifferentiableAttrName) {
    if (!isa<UnitAttr>(value))
      return op->emitError() << "'" << name << "' must be a unit attribute";
    return belongsHere();
  }

  if (name == kVjpAttrName || name == kFwdAttrName ||
      name == kBwdAttrName || name == kJvpAttrName) {
    if (!isa<FlatSymbolRefAttr>(value))
      return op->emitError()
             << "'" << name << "' must name the derivative, as @symbol";
    return belongsHere();
  }

  if (name == kDiffArgsAttrName) {
    auto positions = dyn_cast<DenseI64ArrayAttr>(value);
    if (!positions)
      return op->emitError() << "'" << name << "' must be an array<i64>";

    int64_t previous = -1;
    for (int64_t position : positions.asArrayRef()) {
      if (position <= previous)
        return op->emitError()
               << "'" << name << "' must be strictly increasing";
      previous = position;
    }
    return belongsHere();
  }

  return op->emitError() << "'" << name
                         << "' is not an attribute of the tera dialect";
}
