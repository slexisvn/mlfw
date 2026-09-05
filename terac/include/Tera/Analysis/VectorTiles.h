//===- VectorTiles.h - Tile sizes a host vector register holds --*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TERA_ANALYSIS_VECTORTILES_H
#define TERA_ANALYSIS_VECTORTILES_H

#include "Tera/Analysis/TargetModel.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"

namespace mlir::tera {
bool isSliceable(linalg::LinalgOp op);

SmallVector<int64_t> chooseVectorTile(linalg::LinalgOp op,
                                      const HostTargetModel &model);

SmallVector<int64_t> chooseReductionTile(linalg::LinalgOp op);

bool fitsOneVector(linalg::LinalgOp op, const HostTargetModel &model);

}

#endif
