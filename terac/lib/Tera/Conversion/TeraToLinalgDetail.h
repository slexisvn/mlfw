//===- TeraToLinalgDetail.h - Shared lowering helpers -----------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Internal to lib/Tera/Conversion. One populate function per TeraOps*.td
// family, matching the split of the dialect itself.
//
//===----------------------------------------------------------------------===//

#ifndef TERA_LIB_CONVERSION_TERATOLINALGDETAIL_H
#define TERA_LIB_CONVERSION_TERATOLINALGDETAIL_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/STLFunctionalExtras.h"

#include <utility>

namespace mlir::tera::detail {

/// dynamicSizes must contain one index per dynamic dimension, in axis order.
Value emptyTensor(OpBuilder &builder, Location loc, RankedTensorType type,
                  ValueRange dynamicSizes = {});

Value filledTensor(OpBuilder &builder, Location loc, RankedTensorType type,
                   TypedAttr init, ValueRange dynamicSizes = {});

/// For each dynamic result axis d, source(d) supplies the value and source
/// axis.
SmallVector<Value>
dynamicExtents(OpBuilder &builder, Location loc, RankedTensorType type,
               function_ref<std::pair<Value, int64_t>(int64_t)> source);

SmallVector<Value> extentsLike(OpBuilder &builder, Location loc,
                               RankedTensorType type, Value operand);

TypedAttr zeroAttr(Type elementType);

/// Pads each axis with low elements before it and spacing - 1 between elements.
/// Remaining positions contain fill. dynamicSizes follows dynamic axis order.
Value spreadInto(OpBuilder &builder, Location loc, RankedTensorType resultType,
                 Value operand, ArrayRef<int64_t> low,
                 ArrayRef<int64_t> spacing, Value fill,
                 ValueRange dynamicSizes = {});

void populateConstantPatterns(RewritePatternSet &patterns);
void populateElementwisePatterns(RewritePatternSet &patterns);
void populateShapePatterns(RewritePatternSet &patterns);
void populateIndexingPatterns(RewritePatternSet &patterns);
void populateWindowPatterns(RewritePatternSet &patterns);
void populateContractionPatterns(RewritePatternSet &patterns);
void populateAutodiffPatterns(RewritePatternSet &patterns);
void populateControlFlowPatterns(RewritePatternSet &patterns);

} // namespace mlir::tera::detail

#endif // TERA_LIB_CONVERSION_TERATOLINALGDETAIL_H
