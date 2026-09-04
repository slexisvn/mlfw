//===- TeraOps.cpp - Tera dialect op registration ---------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Holds the single expansion of the generated op classes. Op semantics live in
// the per-family files beside this one, matching the TeraOps*.td split.
//
//===----------------------------------------------------------------------===//

#include "Tera/IR/TeraOps.h"

#include "TeraOpsDetail.h"

#include "Tera/IR/TeraDialect.h"

using namespace mlir;
using namespace mlir::tera;

#include "Tera/IR/TeraInterfaces.cpp.inc"

#define GET_OP_CLASSES
#include "Tera/IR/TeraOps.cpp.inc"

LogicalResult mlir::tera::verifySizesClause(Operation *op, ValueRange sizes) {
  int64_t dynamic = 0;
  for (Type result : op->getResultTypes())
    if (auto shaped = dyn_cast<ShapedType>(result))
      dynamic += llvm::count_if(shaped.getShape(), ShapedType::isDynamic);
  if (static_cast<int64_t>(sizes.size()) != dynamic)
    return op->emitOpError()
           << "expects one size per dynamic result extent: " << dynamic
           << " expected, " << sizes.size() << " given";
  return success();
}

Value mlir::tera::createSplat(OpBuilder &builder, Location loc,
                              RankedTensorType type, double value,
                              Value like) {
  Type elementType = type.getElementType();
  if (!type.hasStaticShape()) {
    assert(like && "a dynamic splat needs a value of the shape it wants");
    auto scalarType = RankedTensorType::get({}, elementType);
    Value scalar = createSplat(builder, loc, scalarType, value);
    return BroadcastInDimOp::create(builder, loc, type, scalar,
                                    detail::dynamicExtentsOf(builder, loc, like),
                                    ArrayRef<int64_t>{});
  }
  DenseElementsAttr elements;
  if (auto floatType = dyn_cast<FloatType>(elementType)) {
    APFloat number(value);
    bool lostPrecision = false;
    number.convert(floatType.getFloatSemantics(), APFloat::rmNearestTiesToEven,
                   &lostPrecision);
    elements = DenseElementsAttr::get(type, number);
  } else {
    elements = DenseElementsAttr::get(
        type, APInt(elementType.getIntOrFloatBitWidth(),
                    static_cast<uint64_t>(static_cast<int64_t>(value)),
                    /*isSigned=*/true));
  }
  return ConstantOp::create(builder, loc, type, elements);
}
