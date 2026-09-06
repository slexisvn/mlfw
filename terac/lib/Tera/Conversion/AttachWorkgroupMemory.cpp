//===- AttachWorkgroupMemory.cpp - Shared buffers a kernel owns -*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Tera/Conversion/Passes.h"

#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"

namespace mlir::tera {
#define GEN_PASS_DEF_ATTACHWORKGROUPMEMORY
#include "Tera/Conversion/Passes.h.inc"

namespace {
bool isWorkgroup(MemRefType type) {
  auto space = dyn_cast_or_null<gpu::AddressSpaceAttr>(type.getMemorySpace());
  return space && space.getValue() == gpu::AddressSpace::Workgroup;
}

struct AttachWorkgroupMemory
    : public impl::AttachWorkgroupMemoryBase<AttachWorkgroupMemory> {
  using impl::AttachWorkgroupMemoryBase<
      AttachWorkgroupMemory>::AttachWorkgroupMemoryBase;

  void runOnOperation() final {
    WalkResult walked = getOperation().walk([&](gpu::LaunchOp launch) {
      SmallVector<memref::AllocOp> shared;
      launch.walk([&](memref::AllocOp alloc) {
        if (isWorkgroup(alloc.getType()))
          shared.push_back(alloc);
      });
      for (memref::AllocOp alloc : shared)
        if (failed(attach(launch, alloc)))
          return WalkResult::interrupt();
      return WalkResult::advance();
    });
    if (walked.wasInterrupted())
      signalPassFailure();
  }

  static LogicalResult attach(gpu::LaunchOp launch, memref::AllocOp alloc) {
    if (alloc->getParentOp() != launch.getOperation())
      return alloc.emitError()
             << "allocates shared memory inside a region of the kernel rather "
                "than in its body, so it is not one allocation per block that "
                "the kernel could be given";
    if (!alloc.getType().hasStaticShape())
      return alloc.emitError()
             << "allocates shared memory of an extent that is not known here, "
                "but the space a block gets is reserved before the block runs";
    for (Operation *user : alloc.getResult().getUsers())
      if (isa<memref::DeallocOp>(user))
        return user->emitError()
               << "frees shared memory, which is the kernel's own storage and "
                  "is gone when the block ends";

    BlockArgument attribution =
        launch.addWorkgroupAttribution(alloc.getType(), alloc.getLoc());
    alloc.getResult().replaceAllUsesWith(attribution);
    alloc.erase();
    return success();
  }
};

}
}
