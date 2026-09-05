//===- VectorTiles.cpp - Tile sizes a host vector holds ---------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Tera/Analysis/VectorTiles.h"

#include "mlir/Interfaces/IndexingMapOpInterface.h"

using namespace mlir;
using namespace mlir::tera;

namespace {
bool positiveCoefficients(AffineExpr expr) {
  auto binary = dyn_cast<AffineBinaryOpExpr>(expr);
  if (!binary)
    return true;
  if (binary.getKind() == AffineExprKind::Mul) {
    auto coefficient = dyn_cast<AffineConstantExpr>(binary.getRHS());
    if (!coefficient || coefficient.getValue() <= 0)
      return false;
  }
  return positiveCoefficients(binary.getLHS()) &&
         positiveCoefficients(binary.getRHS());
}

SmallVector<int64_t> dimensionsOfKind(ArrayRef<utils::IteratorType> iterators,
                                      utils::IteratorType wanted) {
  SmallVector<int64_t> dimensions;
  for (auto [dimension, kind] : llvm::enumerate(iterators))
    if (kind == wanted)
      dimensions.push_back(dimension);
  return dimensions;
}

/// Whether a loop of this extent is worth cutting to `tile`. A `?` is: the
/// extent it turns out to be is the one the tile was chosen for as often as
/// not, and a loop left whole because nobody knew is one that stays scalar.
bool worthTiling(int64_t extent, int64_t tile) {
  return ShapedType::isDynamic(extent) || extent > tile;
}

bool readIterationSpace(linalg::LinalgOp op, SmallVectorImpl<int64_t> &extents,
                        SmallVectorImpl<utils::IteratorType> &iterators) {
  auto indexed = dyn_cast<IndexingMapOpInterface>(op.getOperation());
  if (!indexed || !isSliceable(op))
    return false;
  extents.assign(indexed.getStaticLoopRanges());
  iterators.assign(op.getIteratorTypesArray());
  return extents.size() == iterators.size();
}

}

bool mlir::tera::isSliceable(linalg::LinalgOp op) {
  auto indexed = dyn_cast<IndexingMapOpInterface>(op.getOperation());
  if (!indexed)
    return false;
  return llvm::all_of(indexed.getIndexingMapsArray(), [](AffineMap map) {
    return llvm::all_of(map.getResults(), positiveCoefficients);
  });
}

SmallVector<int64_t> mlir::tera::chooseVectorTile(linalg::LinalgOp op,
                                                  const HostTargetModel &model) {
  SmallVector<int64_t> extents;
  SmallVector<utils::IteratorType> iterators;
  if (!readIterationSpace(op, extents, iterators))
    return {};

  SmallVector<int64_t> parallel =
      dimensionsOfKind(iterators, utils::IteratorType::parallel);
  if (parallel.empty())
    return {};

  SmallVector<int64_t> sizes(extents.size(), 0);
  for (int64_t dimension : ArrayRef<int64_t>(parallel).drop_back())
    if (worthTiling(extents[dimension], 1))
      sizes[dimension] = 1;
  if (worthTiling(extents[parallel.back()], model.vectorLanes))
    sizes[parallel.back()] = model.vectorLanes;

  if (llvm::all_of(sizes, [](int64_t size) { return size == 0; }))
    return {};
  return sizes;
}

SmallVector<int64_t> mlir::tera::chooseReductionTile(linalg::LinalgOp op) {
  SmallVector<int64_t> extents;
  SmallVector<utils::IteratorType> iterators;
  if (!readIterationSpace(op, extents, iterators))
    return {};

  SmallVector<int64_t> sizes(extents.size(), 0);
  for (int64_t dimension :
       dimensionsOfKind(iterators, utils::IteratorType::reduction))
    if (worthTiling(extents[dimension], 1))
      sizes[dimension] = 1;

  if (llvm::all_of(sizes, [](int64_t size) { return size == 0; }))
    return {};
  return sizes;
}

bool mlir::tera::fitsOneVector(linalg::LinalgOp op,
                               const HostTargetModel &model) {
  auto indexed = dyn_cast<IndexingMapOpInterface>(op.getOperation());
  if (!indexed)
    return false;
  int64_t elements = 1;
  for (int64_t extent : indexed.getStaticLoopRanges()) {
    if (ShapedType::isDynamic(extent))
      return false;
    elements *= extent;
  }
  return elements <= model.maxVectorElements;
}
