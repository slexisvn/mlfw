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

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
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

  if (name == kScheduleAttrName) {
    if (!isa<StringAttr>(value))
      return op->emitError() << "'" << name
                             << "' must name the op it schedules, as a string";
    // This one is the opposite of the others: it belongs on the ops a tera op
    // was lowered into, and never on a tera op itself, which the conversion
    // names through its location instead.
    if (isa<FunctionOpInterface>(op) ||
        op->getDialect() ==
            op->getContext()->getLoadedDialect<TeraDialect>())
      return op->emitError()
             << "'" << name
             << "' names what a tera op became, so it belongs on the lowered "
                "op rather than here";
    return success();
  }

  if (name == kDeviceResidentAttrName)
    return op->emitError() << "'" << name
                           << "' belongs on a function argument, not here";

  return op->emitError() << "'" << name
                         << "' is not an attribute of the tera dialect";
}

LogicalResult TeraDialect::verifyRegionArgAttribute(Operation *op, unsigned,
                                                    unsigned argIndex,
                                                    NamedAttribute attribute) {
  StringRef name = attribute.getName();
  if (name != kDeviceResidentAttrName)
    return op->emitError() << "'" << name
                           << "' is not an argument attribute of the tera "
                              "dialect";

  if (!isa<UnitAttr>(attribute.getValue()))
    return op->emitError() << "'" << name << "' must be a unit attribute";

  auto function = dyn_cast<FunctionOpInterface>(op);
  if (!function)
    return op->emitError() << "'" << name
                           << "' belongs on a function argument";

  if (!isa<ShapedType>(function.getArgumentTypes()[argIndex]))
    return op->emitError()
           << "'" << name << "' is on argument " << argIndex << ", which is "
           << function.getArgumentTypes()[argIndex]
           << " and holds no buffer to leave on the device";
  return success();
}
