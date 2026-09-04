//===- TeraOpsDetail.h - Shared shape helpers -------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TERA_LIB_IR_TERAOPSDETAIL_H
#define TERA_LIB_IR_TERAOPSDETAIL_H

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Location.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallBitVector.h"

#include <numeric>
#include <optional>

namespace mlir::tera::detail {
inline bool extentsAgree(int64_t lhs, int64_t rhs) {
  return ShapedType::isDynamic(lhs) || ShapedType::isDynamic(rhs) || lhs == rhs;
}

inline LogicalResult markAxes(std::optional<Location> location,
                              ArrayRef<int64_t> axes, int64_t rank,
                              llvm::SmallBitVector &mask, StringRef what) {
  for (int64_t axis : axes) {
    if (axis < 0 || axis >= rank)
      return emitOptionalError(location, what, " axis ", axis,
                               " is out of range for rank ", rank);
    if (mask.test(axis))
      return emitOptionalError(location, what, " axis ", axis, " is repeated");
    mask.set(axis);
  }
  return success();
}

inline llvm::SmallBitVector claimedAxes(ArrayRef<int64_t> first,
                                       ArrayRef<int64_t> second, int64_t rank) {
  llvm::SmallBitVector mask(rank);
  for (int64_t axis : first)
    mask.set(axis);
  for (int64_t axis : second)
    mask.set(axis);
  return mask;
}

inline SmallVector<int64_t> freeAxes(const llvm::SmallBitVector &mask,
                                     int64_t rank) {
  SmallVector<int64_t> result;
  for (int64_t axis = 0; axis < rank; ++axis)
    if (!mask.test(axis))
      result.push_back(axis);
  return result;
}

inline SmallVector<int64_t> sortedPositions(ArrayRef<int64_t> axes) {
  SmallVector<int64_t> order(axes.size());
  std::iota(order.begin(), order.end(), 0);
  llvm::sort(order,
             [&](int64_t lhs, int64_t rhs) { return axes[lhs] < axes[rhs]; });
  SmallVector<int64_t> positions(axes.size());
  for (auto [position, index] : llvm::enumerate(order))
    positions[index] = position;
  return positions;
}

inline SmallVector<int64_t> axisRange(int64_t begin, int64_t end) {
  SmallVector<int64_t> axes(end - begin);
  std::iota(axes.begin(), axes.end(), begin);
  return axes;
}

SmallVector<Value> dynamicExtentsOf(OpBuilder &builder, Location loc,
                                    Value source);

}

#endif
