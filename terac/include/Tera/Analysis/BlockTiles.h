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

bool isExactTile(ArrayRef<int64_t> tripCounts, ArrayRef<int64_t> tile);

}

#endif
