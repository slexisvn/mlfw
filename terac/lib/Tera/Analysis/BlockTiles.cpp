//===- BlockTiles.cpp - Thread-block shapes for a parallel loop -*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Tera/Analysis/BlockTiles.h"

#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "llvm/Support/MathExtras.h"

#include <algorithm>

using namespace mlir;
using namespace mlir::tera;

namespace {
int64_t largestDivisor(int64_t count, int64_t cap, int64_t multipleOf) {
  int64_t best = 0;
  auto consider = [&](int64_t candidate) {
    if (candidate <= cap && candidate % multipleOf == 0)
      best = std::max(best, candidate);
  };
  for (int64_t candidate = 1; candidate * candidate <= count; ++candidate) {
    if (count % candidate != 0)
      continue;
    consider(candidate);
    consider(count / candidate);
  }
  return best;
}

}

SmallVector<int64_t> mlir::tera::parallelTripCounts(scf::ParallelOp loop) {
  SmallVector<int64_t> counts;
  for (auto [lower, upper, step] :
       llvm::zip_equal(loop.getLowerBound(), loop.getUpperBound(),
                       loop.getStep())) {
    std::optional<int64_t> from = getConstantIntValue(lower);
    std::optional<int64_t> to = getConstantIntValue(upper);
    std::optional<int64_t> by = getConstantIntValue(step);
    if (!from || !to || !by || *by <= 0 || *to <= *from) {
      counts.push_back(ShapedType::kDynamic);
      continue;
    }
    counts.push_back(llvm::divideCeil(*to - *from, *by));
  }
  return counts;
}

SmallVector<int64_t> mlir::tera::chooseBlockTile(scf::ParallelOp loop,
                                                 const GpuTargetModel &model) {
  SmallVector<int64_t> counts = parallelTripCounts(loop);
  SmallVector<int64_t> tile(counts.size(), 1);
  int64_t mapped = std::min<int64_t>(counts.size(), model.maxBlockExtents.size());
  int64_t budget =
      std::min(model.preferredThreadsPerBlock, model.maxThreadsPerBlock);

  for (int64_t axis = 0; axis < mapped; ++axis) {
    int64_t count = counts[mapped - 1 - axis];
    if (ShapedType::isDynamic(count))
      continue;
    int64_t cap = std::min(budget, model.maxBlockExtents[axis]);
    int64_t size = axis == 0 ? largestDivisor(count, cap, model.warpSize) : 0;
    if (size == 0)
      size = largestDivisor(count, cap, 1);
    tile[mapped - 1 - axis] = size;
    budget /= size;
  }
  return tile;
}

int64_t mlir::tera::chooseContractionTile(int64_t m, int64_t n, int64_t k,
                                          int64_t lhsBytes, int64_t rhsBytes,
                                          const GpuTargetModel &model) {
  if (model.warpSize < 1 || model.maxThreadsPerBlock < 1 ||
      model.sharedMemoryPerBlock < 1)
    return 0;
  if (ShapedType::isDynamic(m) || ShapedType::isDynamic(n) ||
      ShapedType::isDynamic(k))
    return 0;

  int64_t chosen = 0;
  for (int64_t tile = model.warpSize;
       tile * tile <= model.maxThreadsPerBlock &&
       tile <= model.maxBlockExtents[0] && tile <= model.maxBlockExtents[1];
       tile += model.warpSize) {
    if (m % tile != 0 || n % tile != 0 || k % tile != 0)
      continue;
    if (tile * tile * (lhsBytes + rhsBytes) > model.sharedMemoryPerBlock)
      continue;
    chosen = tile;
  }
  return chosen;
}

bool mlir::tera::isExactTile(ArrayRef<int64_t> tripCounts,
                             ArrayRef<int64_t> tile) {
  for (auto [dimension, size] : llvm::enumerate(tile)) {
    if (size < 1)
      return false;
    if (dimension >= tripCounts.size())
      continue;
    int64_t count = tripCounts[dimension];
    if (!ShapedType::isDynamic(count) && count % size != 0)
      return false;
  }
  return true;
}
