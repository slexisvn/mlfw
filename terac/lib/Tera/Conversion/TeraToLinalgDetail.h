//===- TeraToLinalgDetail.h - Shared lowering helpers -----------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TERA_LIB_CONVERSION_TERATOLINALGDETAIL_H
#define TERA_LIB_CONVERSION_TERATOLINALGDETAIL_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/STLFunctionalExtras.h"

#include <utility>

namespace mlir::tera::detail {
Value emptyTensor(OpBuilder &builder, Location loc, RankedTensorType type,
                  ValueRange dynamicSizes = {});

Value filledTensor(OpBuilder &builder, Location loc, RankedTensorType type,
                   TypedAttr init, ValueRange dynamicSizes = {});

/// The extents of `op`'s result that its type leaves as `?`, in axis order,
/// which is the form `tensor.empty` and `linalg.fill` want them in.
///
/// The conversion materialises a destination before the op that fills it, so
/// it needs those extents as values before there is a result to read them
/// off. Where they come from is the op's own question -- a window counts them,
/// a contraction takes them from either operand, an elementwise op copies
/// them -- and it is asked through `ReifyRankedShapedTypeOpInterface` rather
/// than answered again here per pattern.
SmallVector<Value> resultExtents(OpBuilder &builder, Location loc,
                                 Operation *op);

/// Every extent of `op`'s result and not only the ones its type leaves as
/// `?`, which is what an op taking a whole shape wants. A static extent comes
/// back as an attribute, so nothing is built for what the type already says.
SmallVector<OpFoldResult> resultShape(OpBuilder &builder, Location loc,
                                      Operation *op);

TypedAttr zeroAttr(Type elementType);

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

}

#endif
