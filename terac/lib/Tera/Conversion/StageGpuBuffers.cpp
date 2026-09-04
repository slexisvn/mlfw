//===- StageGpuBuffers.cpp - Put kernel operands on the device --*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Tera/Conversion/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "llvm/ADT/MapVector.h"

namespace mlir::tera {
#define GEN_PASS_DEF_STAGEGPUBUFFERS
#include "Tera/Conversion/Passes.h.inc"

namespace {
bool isStagingOp(Operation *op) {
  return isa<gpu::AllocOp, gpu::MemcpyOp, gpu::DeallocOp>(op);
}

struct StagedBuffer {
  Value device;
  Value shadow;
  Operation *firstAnchor;
  Operation *lastAnchor;
};

MemRefType contiguousType(MemRefType type) {
  if (type.getLayout().isIdentity())
    return nullptr;
  return MemRefType::get(type.getShape(), type.getElementType());
}

SmallVector<Value> dynamicExtentsOf(OpBuilder &builder, Location loc,
                                    Value buffer) {
  auto type = cast<MemRefType>(buffer.getType());
  SmallVector<Value> extents;
  for (auto [axis, extent] : llvm::enumerate(type.getShape())) {
    if (!ShapedType::isDynamic(extent))
      continue;
    Value index = arith::ConstantIndexOp::create(builder, loc, axis);
    extents.push_back(memref::DimOp::create(builder, loc, buffer, index));
  }
  return extents;
}

struct StageGpuBuffers : impl::StageGpuBuffersBase<StageGpuBuffers> {
  void runOnOperation() override {
    func::FuncOp func = getOperation();

    llvm::MapVector<Value, StagedBuffer> staged;

    WalkResult walked = func.walk([&](gpu::LaunchFuncOp launch) {
      for (OpOperand &use : launch->getOpOperands()) {
        Value host = use.get();
        auto type = dyn_cast<MemRefType>(host.getType());
        if (!type)
          continue;

        Operation *anchor =
            host.getParentBlock()->findAncestorOpInBlock(*launch);

        auto *it = staged.find(host);
        if (it == staged.end()) {
          OpBuilder builder(func.getContext());
          if (Operation *definition = host.getDefiningOp())
            builder.setInsertionPointAfter(definition);
          else
            builder.setInsertionPointToStart(host.getParentBlock());

          Location loc = host.getLoc();
          SmallVector<Value> extents = dynamicExtentsOf(builder, loc, host);
          MemRefType flat = contiguousType(type);
          Value shadow =
              flat ? memref::AllocOp::create(builder, loc, flat, extents)
                   : Value();
          Value device = gpu::AllocOp::create(builder, loc, flat ? flat : type,
                                              /*asyncToken=*/Type(),
                                              /*asyncDependencies=*/ValueRange{},
                                              /*dynamicSizes=*/extents,
                                              /*symbolOperands=*/ValueRange{})
                             .getMemref();
          builder.setInsertionPoint(anchor);
          if (shadow)
            memref::CopyOp::create(builder, loc, host, shadow);
          gpu::MemcpyOp::create(builder, loc, /*asyncToken=*/Type(),
                                /*asyncDependencies=*/ValueRange{}, device,
                                shadow ? shadow : host);
          it = staged.insert({host, {device, shadow, anchor, anchor}}).first;
        }

        it->second.lastAnchor = anchor;

        Value operand = it->second.device;
        if (operand.getType() != type) {
          OpBuilder castBuilder(launch);
          operand = memref::CastOp::create(castBuilder, host.getLoc(), type,
                                           operand);
        }
        use.set(operand);
      }
      return WalkResult::advance();
    });

    if (walked.wasInterrupted())
      return signalPassFailure();

    for (auto &[host, buffer] : staged) {
      auto [device, shadow, firstAnchor, lastAnchor] = buffer;

      for (Operation *user : host.getUsers()) {
        if (isStagingOp(user) || isa<gpu::LaunchFuncOp>(user))
          continue;
        if (isa<memref::DimOp, memref::RankOp>(user))
          continue;
        Operation *sibling =
            host.getParentBlock()->findAncestorOpInBlock(*user);
        if (sibling && (sibling->isBeforeInBlock(firstAnchor) ||
                        lastAnchor->isBeforeInBlock(sibling)))
          continue;
        user->emitError("reads a buffer while kernels around it hold the "
                        "only current copy on the device");
        return signalPassFailure();
      }

      OpBuilder builder(lastAnchor);
      builder.setInsertionPointAfter(lastAnchor);
      Location loc = host.getLoc();
      gpu::MemcpyOp::create(builder, loc, /*asyncToken=*/Type(),
                            /*asyncDependencies=*/ValueRange{},
                            shadow ? shadow : host, device);
      gpu::DeallocOp::create(builder, loc, /*asyncToken=*/Type(),
                             /*asyncDependencies=*/ValueRange{}, device);
      if (shadow) {
        memref::CopyOp::create(builder, loc, shadow, host);
        memref::DeallocOp::create(builder, loc, shadow);
      }
    }
  }
};

}
}
