//===- TeraOpsConstant.cpp - Value-producing tera ops -----------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Tera/IR/TeraOps.h"

#include "TeraOpsDetail.h"
#include "mlir/IR/BuiltinTypes.h"

using namespace mlir;
using namespace mlir::tera;

OpFoldResult ConstantOp::fold(FoldAdaptor) { return getValue(); }

LogicalResult IotaOp::verify() {
  if (failed(verifySizesClause(*this, getSizes())))
    return failure();
  auto resultType = dyn_cast<RankedTensorType>(getResult().getType());
  if (!resultType)
    return success();
  int64_t dimension = getIotaDimension();
  if (dimension < 0 || dimension >= resultType.getRank())
    return emitOpError() << "iota dimension " << dimension
                         << " is out of range for rank "
                         << resultType.getRank();
  return success();
}

LogicalResult ConstantOp::buildVjp(OpBuilder &, ValueRange,
                                   SmallVectorImpl<Value> &) {
  return success();
}

LogicalResult IotaOp::buildVjp(OpBuilder &, ValueRange,
                               SmallVectorImpl<Value> &operandAdjoints) {
  operandAdjoints.assign(getSizes().size(), Value());
  return success();
}

LogicalResult IotaOp::reifyResultShapes(OpBuilder &builder,
                                        ReifiedRankedShapedTypeDims &reified) {
  auto type = cast<RankedTensorType>(getType());
  SmallVector<OpFoldResult> extents;
  ValueRange sizes = getSizes();
  for (int64_t extent : type.getShape()) {
    if (!ShapedType::isDynamic(extent)) {
      extents.push_back(builder.getIndexAttr(extent));
      continue;
    }
    extents.push_back(detail::sizeAsIndex(builder, getLoc(), sizes.front()));
    sizes = sizes.drop_front();
  }
  reified.push_back(std::move(extents));
  return success();
}
