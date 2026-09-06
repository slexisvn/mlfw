//===- TileContractionToShared.cpp - Stage operand tiles --------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Tera/Conversion/Passes.h"

#include "Tera/Analysis/BlockTiles.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IRMapping.h"

namespace mlir::tera {
#define GEN_PASS_DEF_TILECONTRACTIONTOSHARED
#include "Tera/Conversion/Passes.h.inc"

namespace {
/// One contraction, resolved down to the pieces the rewrite indexes with: a
/// loop position per axis, the operand each side of the product comes from,
/// and the tile all three axes are cut to.
struct Contraction {
  linalg::LinalgOp op;
  SmallVector<unsigned> batch;
  unsigned m;
  unsigned n;
  unsigned k;
  OpOperand *lhs;
  OpOperand *rhs;
  OpOperand *out;
  int64_t tile;
};

bool usesDim(AffineMap map, unsigned dim) {
  return llvm::any_of(map.getResults(), [&](AffineExpr result) {
    return cast<AffineDimExpr>(result).getPosition() == dim;
  });
}

int64_t elementBytes(OpOperand *operand) {
  Type element = cast<MemRefType>(operand->get().getType()).getElementType();
  return llvm::divideCeil(element.getIntOrFloatBitWidth(), 8);
}

std::optional<Contraction> match(linalg::LinalgOp op,
                                 const GpuTargetModel &model) {
  if (!op.hasPureBufferSemantics() || op.getNumDpsInputs() != 2 ||
      op.getNumDpsInits() != 1)
    return std::nullopt;
  if (!llvm::all_of(op.getIndexingMapsArray(), [](AffineMap map) {
        return map.isProjectedPermutation();
      }))
    return std::nullopt;

  FailureOr<linalg::ContractionDimensions> dimensions =
      linalg::inferContractionDims(op);
  if (failed(dimensions) || dimensions->m.size() != 1 ||
      dimensions->n.size() != 1 || dimensions->k.size() != 1)
    return std::nullopt;
  // The grid has three dimensions and the two the tile takes leave one, so a
  // second batch axis is one more block loop than there is a processor for.
  if (dimensions->batch.size() > 1 ||
      dimensions->batch.size() + 3 != op.getNumLoops())
    return std::nullopt;

  Contraction matched;
  matched.op = op;
  matched.batch.assign(dimensions->batch.begin(), dimensions->batch.end());
  matched.m = dimensions->m.front();
  matched.n = dimensions->n.front();
  matched.k = dimensions->k.front();
  matched.lhs = matched.rhs = nullptr;
  matched.out = op.getDpsInitOperand(0);

  for (OpOperand *input : op.getDpsInputOperands()) {
    AffineMap map = op.getMatchingIndexingMap(input);
    if (usesDim(map, matched.m))
      matched.lhs = input;
    else if (usesDim(map, matched.n))
      matched.rhs = input;
  }
  if (!matched.lhs || !matched.rhs)
    return std::nullopt;

  // An operand already in some other memory space is one this pass did not
  // put there, and a global read is what the staging is written against.
  for (OpOperand &operand : op->getOpOperands())
    if (cast<MemRefType>(operand.get().getType()).getMemorySpace())
      return std::nullopt;

  SmallVector<int64_t> ranges = op.getStaticLoopRanges();
  if (llvm::any_of(ranges, ShapedType::isDynamic))
    return std::nullopt;

  matched.tile = chooseContractionTile(
      ranges[matched.m], ranges[matched.n], ranges[matched.k],
      elementBytes(matched.lhs), elementBytes(matched.rhs), model);
  if (matched.tile == 0)
    return std::nullopt;
  return matched;
}

struct TileContractionToShared
    : public impl::TileContractionToSharedBase<TileContractionToShared> {
  using impl::TileContractionToSharedBase<
      TileContractionToShared>::TileContractionToSharedBase;

  void runOnOperation() final {
    GpuTargetModel model;
    model.warpSize = warpSize;
    model.maxThreadsPerBlock = maxThreadsPerBlock;
    model.sharedMemoryPerBlock = sharedMemoryPerBlock;
    if (model.warpSize < 1 || model.maxThreadsPerBlock < 1 ||
        model.sharedMemoryPerBlock < 1) {
      getOperation().emitError()
          << "was given a thread budget or a shared-memory budget below one, "
             "which leaves no tile to choose";
      return signalPassFailure();
    }

    SmallVector<Contraction> matched;
    getOperation().walk([&](linalg::LinalgOp op) {
      if (std::optional<Contraction> found = match(op, model))
        matched.push_back(*found);
    });

    for (const Contraction &contraction : matched)
      rewrite(contraction);
  }

  /// The subscripts an operand is read at when the loops stand at `ivs`. Every
  /// indexing map here is a projected permutation, so a subscript is one of
  /// the induction variables rather than arithmetic over several.
  static SmallVector<Value> subscripts(const Contraction &contraction,
                                       OpOperand *operand,
                                       ArrayRef<Value> ivs) {
    linalg::LinalgOp op = contraction.op;
    AffineMap map = op.getMatchingIndexingMap(operand);
    SmallVector<Value> found;
    for (AffineExpr result : map.getResults())
      found.push_back(ivs[cast<AffineDimExpr>(result).getPosition()]);
    return found;
  }

  static Value read(OpBuilder &builder, Location loc,
                    const Contraction &contraction, OpOperand *operand,
                    ArrayRef<Value> ivs) {
    return memref::LoadOp::create(builder, loc, operand->get(),
                                  subscripts(contraction, operand, ivs));
  }

  /// Clones the contraction's own body around one element of each operand and
  /// the running accumulator, so what the block computes stays whatever the op
  /// said rather than an assumed multiply-add.
  static Value combine(OpBuilder &builder, Location loc,
                       const Contraction &contraction, Value lhs, Value rhs,
                       Value accumulator) {
    Block &body = contraction.op->getRegion(0).front();
    SmallVector<Value> arguments(body.getNumArguments());
    arguments[contraction.lhs->getOperandNumber()] = lhs;
    arguments[contraction.rhs->getOperandNumber()] = rhs;
    arguments[contraction.out->getOperandNumber()] = accumulator;

    IRMapping mapping;
    mapping.map(body.getArguments(), arguments);
    for (Operation &nested : body.without_terminator())
      builder.clone(nested, mapping);
    return mapping.lookupOrDefault(body.getTerminator()->getOperand(0));
  }

  static MemRefType sharedType(OpOperand *operand, int64_t tile) {
    auto memref = cast<MemRefType>(operand->get().getType());
    return MemRefType::get(
        {tile, tile}, memref.getElementType(), MemRefLayoutAttrInterface(),
        gpu::AddressSpaceAttr::get(memref.getContext(),
                                   gpu::AddressSpace::Workgroup));
  }

  void rewrite(const Contraction &contraction) {
    linalg::LinalgOp op = contraction.op;
    Location loc = op.getLoc();
    OpBuilder builder(op);
    SmallVector<int64_t> ranges = op.getStaticLoopRanges();
    int64_t tile = contraction.tile;

    auto index = [&](int64_t value) {
      return arith::ConstantIndexOp::create(builder, loc, value).getResult();
    };
    Value zero = index(0);
    Value one = index(1);
    Value span = index(tile);
    Value contracted = index(ranges[contraction.k]);

    SmallVector<Value> lowerBounds, upperBounds, steps;
    for (unsigned axis : contraction.batch) {
      lowerBounds.push_back(zero);
      upperBounds.push_back(index(ranges[axis]));
      steps.push_back(one);
    }
    for (unsigned axis : {contraction.m, contraction.n}) {
      lowerBounds.push_back(zero);
      upperBounds.push_back(index(ranges[axis]));
      steps.push_back(span);
    }

    auto blocks =
        scf::ParallelOp::create(builder, loc, lowerBounds, upperBounds, steps);
    builder.setInsertionPointToStart(blocks.getBody());
    SmallVector<Value> blockIvs = blocks.getInductionVars();

    Value sharedLhs =
        memref::AllocOp::create(builder, loc, sharedType(contraction.lhs, tile));
    Value sharedRhs =
        memref::AllocOp::create(builder, loc, sharedType(contraction.rhs, tile));

    auto threads = scf::ParallelOp::create(builder, loc, ValueRange{zero, zero},
                                           ValueRange{span, span},
                                           ValueRange{one, one});
    builder.setInsertionPointToStart(threads.getBody());
    Value row = threads.getInductionVars()[0];
    Value column = threads.getInductionVars()[1];

    // A loop position per axis. The parallel ones stand still for the whole
    // block; each access below writes its own contracted coordinate in.
    SmallVector<Value> ivs(op.getNumLoops(), zero);
    for (auto [position, axis] : llvm::enumerate(contraction.batch))
      ivs[axis] = blockIvs[position];
    ivs[contraction.m] = arith::AddIOp::create(
        builder, loc, blockIvs[blockIvs.size() - 2], row);
    ivs[contraction.n] =
        arith::AddIOp::create(builder, loc, blockIvs.back(), column);
    SmallVector<Value> destination =
        subscripts(contraction, contraction.out, ivs);

    // The accumulator starts at what the destination holds, because a
    // contraction accumulates into it rather than replacing it: whatever
    // filled it beforehand is part of the answer.
    Value initial = memref::LoadOp::create(builder, loc,
                                           contraction.out->get(), destination);
    auto tiles = scf::ForOp::create(builder, loc, zero, contracted, span,
                                    ValueRange{initial});
    builder.setInsertionPointToStart(tiles.getBody());
    Value tileStart = tiles.getInductionVar();

    ivs[contraction.k] = arith::AddIOp::create(builder, loc, tileStart, column);
    memref::StoreOp::create(
        builder, loc, read(builder, loc, contraction, contraction.lhs, ivs),
        sharedLhs, ValueRange{row, column});
    ivs[contraction.k] = arith::AddIOp::create(builder, loc, tileStart, row);
    memref::StoreOp::create(
        builder, loc, read(builder, loc, contraction, contraction.rhs, ivs),
        sharedRhs, ValueRange{row, column});

    gpu::BarrierOp::create(builder, loc);

    auto withinTile = scf::ForOp::create(builder, loc, zero, span, one,
                                     ValueRange{tiles.getRegionIterArgs()[0]});
    builder.setInsertionPointToStart(withinTile.getBody());
    Value step = withinTile.getInductionVar();
    Value lhs =
        memref::LoadOp::create(builder, loc, sharedLhs, ValueRange{row, step});
    Value rhs =
        memref::LoadOp::create(builder, loc, sharedRhs, ValueRange{step, column});
    scf::YieldOp::create(builder, loc,
                         combine(builder, loc, contraction, lhs, rhs,
                                 withinTile.getRegionIterArgs()[0]));

    builder.setInsertionPointAfter(withinTile);
    gpu::BarrierOp::create(builder, loc);
    scf::YieldOp::create(builder, loc, withinTile.getResult(0));

    builder.setInsertionPointAfter(tiles);
    memref::StoreOp::create(builder, loc, tiles.getResult(0),
                            contraction.out->get(), destination);

    op->erase();
  }
};

}
}
