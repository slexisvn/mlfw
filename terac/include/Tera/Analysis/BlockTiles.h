//===- BlockTiles.h - Thread-block shapes a parallel loop takes -*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TERA_ANALYSIS_BLOCKTILES_H
#define TERA_ANALYSIS_BLOCKTILES_H

#include "Tera/Analysis/TargetModel.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

namespace mlir::tera {
SmallVector<int64_t> parallelTripCounts(scf::ParallelOp loop);

SmallVector<int64_t> chooseBlockTile(scf::ParallelOp loop,
                                     const GpuTargetModel &model);

/// The square tile a contraction of these extents is cut to, or zero when the
/// target allows none.
///
/// One tile is chosen for all three axes because that is what lets every
/// thread stage exactly one element of each operand: a block of `tile` by
/// `tile` threads fills a `tile` by `tile` tile of the left operand and one of
/// the right in a single indexed store apiece. `lhsBytes` and `rhsBytes` are
/// the sizes of one element of each, which is what those two tiles cost in
/// shared memory.
int64_t chooseContractionTile(int64_t m, int64_t n, int64_t k,
                              int64_t lhsBytes, int64_t rhsBytes,
                              const GpuTargetModel &model);

bool isExactTile(ArrayRef<int64_t> tripCounts, ArrayRef<int64_t> tile);

}

#endif
