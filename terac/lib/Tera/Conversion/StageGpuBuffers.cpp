//===- StageGpuBuffers.cpp - Put kernel operands on the device --*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Tera/Conversion/Passes.h"

#include "Tera/IR/TeraDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"

#include <optional>

namespace mlir::tera {
#define GEN_PASS_DEF_STAGEGPUBUFFERS
#include "Tera/Conversion/Passes.h.inc"

namespace {
struct Access {
  bool reads = false;
  bool writes = false;

  bool any() const { return reads || writes; }

  void merge(Access other) {
    reads |= other.reads;
    writes |= other.writes;
  }
};

/// Where the current contents of a buffer are.
enum class Residence { Nowhere, Host, Device, Both };

/// What happens to one buffer at one op of the block it is defined in. An
/// anchor standing for a region holds the union of what that region does.
struct Event {
  Operation *anchor = nullptr;
  Access host;
  Access device;
  bool launched = false;
  bool frees = false;
};

struct StagedBuffer {
  Value device;
  Value shadow;
};

/// `root` and every memref that a chain of view ops derives from it. A view
/// carries no data of its own, so an access through one is an access to the
/// buffer underneath.
void collectAliases(Value root, SetVector<Value> &aliases) {
  aliases.insert(root);
  for (unsigned index = 0; index < aliases.size(); ++index) {
    Value alias = aliases[index];
    for (Operation *user : alias.getUsers()) {
      auto view = dyn_cast<ViewLikeOpInterface>(user);
      if (!view || view.getViewSource() != alias)
        continue;
      for (Value result : user->getResults())
        if (isa<MemRefType>(result.getType()))
          aliases.insert(result);
    }
  }
}

Access accessBy(Operation *op, Value value) {
  if (op->getNumRegions() != 0)
    return {true, true};
  return {mightHaveEffect<MemoryEffects::Read>(op, value),
          mightHaveEffect<MemoryEffects::Write>(op, value)};
}

/// What everything reachable from `value` does to the buffer behind it.
Access accessesOf(Value value) {
  SetVector<Value> aliases;
  collectAliases(value, aliases);

  Access access;
  for (Value alias : aliases)
    for (Operation *user : alias.getUsers())
      access.merge(accessBy(user, alias));
  return access;
}

/// What the kernel `launch` calls does to the operand held in `use`. Read from
/// the outlined body rather than assumed, because copying a buffer back that
/// no kernel wrote is wasted bandwidth and not copying one back that a kernel
/// did write is a wrong answer.
Access kernelAccess(gpu::LaunchFuncOp launch, OpOperand &use,
                    SymbolTableCollection &symbols) {
  auto kernel = symbols.lookupNearestSymbolFrom<gpu::GPUFuncOp>(
      launch, launch.getKernelAttr());
  if (!kernel || kernel.getBody().empty())
    return {true, true};

  unsigned index = use.getOperandNumber() -
                   launch.getKernelOperands().getBeginOperandIndex();
  if (index >= kernel.getNumArguments())
    return {true, true};
  return accessesOf(kernel.getArgument(index));
}

bool isLocalAllocation(Value buffer) {
  return isa_and_nonnull<memref::AllocOp, memref::AllocaOp>(
      buffer.getDefiningOp());
}

/// The position of `buffer` in `function`'s signature, when the caller
/// promised that argument already points at memory on the device.
std::optional<unsigned> residentArgument(func::FuncOp function, Value buffer) {
  auto argument = dyn_cast<BlockArgument>(buffer);
  if (!argument || argument.getOwner() != &function.getBody().front())
    return std::nullopt;
  if (!function.getArgAttr(argument.getArgNumber(),
                           TeraDialect::kDeviceResidentAttrName))
    return std::nullopt;
  return argument.getArgNumber();
}

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

/// Both directions answer with the last op they made, which is the one the
/// deallocation has to come after.
Operation *copyToDevice(OpBuilder &builder, Location loc, Value host,
                        const StagedBuffer &buffer) {
  if (buffer.shadow)
    memref::CopyOp::create(builder, loc, host, buffer.shadow);
  return gpu::MemcpyOp::create(builder, loc, /*asyncToken=*/Type(),
                               /*asyncDependencies=*/ValueRange{},
                               buffer.device,
                               buffer.shadow ? buffer.shadow : host);
}

Operation *copyToHost(OpBuilder &builder, Location loc, Value host,
                      const StagedBuffer &buffer) {
  Operation *last = gpu::MemcpyOp::create(
      builder, loc, /*asyncToken=*/Type(), /*asyncDependencies=*/ValueRange{},
      buffer.shadow ? buffer.shadow : host, buffer.device);
  if (buffer.shadow)
    last = memref::CopyOp::create(builder, loc, buffer.shadow, host);
  return last;
}

struct StageGpuBuffers : impl::StageGpuBuffersBase<StageGpuBuffers> {
  void runOnOperation() override {
    SymbolTableCollection symbols;
    for (auto func : getOperation().getOps<func::FuncOp>())
      if (failed(stage(func, symbols)))
        return signalPassFailure();
  }

  /// The buffers `func` hands to a kernel, each with every use of it that the
  /// staging has to order against those kernels.
  llvm::MapVector<Value, SmallVector<OpOperand *>>
  stagedBuffers(func::FuncOp func) {
    llvm::MapVector<Value, SmallVector<OpOperand *>> staged;
    func.walk([&](gpu::LaunchFuncOp launch) {
      for (OpOperand &use : launch->getOpOperands())
        if (isa<MemRefType>(use.get().getType()))
          staged[use.get()].push_back(&use);
    });
    return staged;
  }

  /// What happens to `host` across the block it is defined in, in order. An op
  /// holding regions stands for everything inside it, because a copy placed in
  /// there would run once per execution of the region rather than once.
  FailureOr<SmallVector<Event>> timelineOf(Value host,
                                           SymbolTableCollection &symbols) {
    Block *home = host.getParentBlock();
    SetVector<Value> aliases;
    collectAliases(host, aliases);

    llvm::MapVector<Operation *, Event> events;
    for (Value alias : aliases) {
      for (OpOperand &use : alias.getUses()) {
        Operation *user = use.getOwner();
        Operation *anchor = home->findAncestorOpInBlock(*user);
        if (!anchor) {
          user->emitError("reads a staged buffer from a block that cannot be "
                          "ordered against the kernels using it");
          return failure();
        }

        Event &event = events[anchor];
        event.anchor = anchor;
        if (auto launch = dyn_cast<gpu::LaunchFuncOp>(user)) {
          event.launched = true;
          event.device.merge(kernelAccess(launch, use, symbols));
        } else if (isa<memref::DeallocOp, gpu::DeallocOp>(user)) {
          event.frees = true;
        } else if (user->hasTrait<OpTrait::IsTerminator>()) {
          event.host.merge({/*reads=*/true, /*writes=*/false});
        } else {
          event.host.merge(accessBy(user, alias));
        }
      }
    }

    SmallVector<Event> timeline;
    for (auto &[anchor, event] : events) {
      if (event.launched && event.host.any()) {
        anchor->emitError(
            "holds both a kernel and a host access to the same buffer, which "
            "cannot be ordered from the block the buffer lives in");
        return failure();
      }
      timeline.push_back(event);
    }
    llvm::sort(timeline, [](const Event &lhs, const Event &rhs) {
      return lhs.anchor->isBeforeInBlock(rhs.anchor);
    });
    return timeline;
  }

  StagedBuffer openDeviceBuffer(Value host) {
    OpBuilder builder(host.getContext());
    if (Operation *definition = host.getDefiningOp())
      builder.setInsertionPointAfter(definition);
    else
      builder.setInsertionPointToStart(host.getParentBlock());

    Location loc = host.getLoc();
    auto type = cast<MemRefType>(host.getType());
    SmallVector<Value> extents = dynamicExtentsOf(builder, loc, host);
    MemRefType flat = contiguousType(type);

    StagedBuffer buffer;
    if (flat)
      buffer.shadow = memref::AllocOp::create(builder, loc, flat, extents);
    buffer.device = gpu::AllocOp::create(builder, loc, flat ? flat : type,
                                         /*asyncToken=*/Type(),
                                         /*asyncDependencies=*/ValueRange{},
                                         /*dynamicSizes=*/extents,
                                         /*symbolOperands=*/ValueRange{})
                        .getMemref();
    return buffer;
  }

  void closeDeviceBuffer(OpBuilder &builder, Location loc,
                         const StagedBuffer &buffer) {
    gpu::DeallocOp::create(builder, loc, /*asyncToken=*/Type(),
                           /*asyncDependencies=*/ValueRange{}, buffer.device);
    if (buffer.shadow)
      memref::DeallocOp::create(builder, loc, buffer.shadow);
  }

  LogicalResult stage(func::FuncOp func, SymbolTableCollection &symbols) {
    llvm::MapVector<Value, SmallVector<OpOperand *>> staged =
        stagedBuffers(func);

    for (auto &[host, uses] : staged) {
      SetVector<Value> aliases;
      collectAliases(host, aliases);
      for (Value alias : aliases)
        if (alias != host && staged.count(alias)) {
          emitError(alias.getLoc())
              << "is handed to a kernel, and so is the buffer it views, so "
                 "the two would be staged onto the device separately";
          return failure();
        }
    }

    for (auto &[host, uses] : staged) {
      FailureOr<SmallVector<Event>> timeline = timelineOf(host, symbols);
      if (failed(timeline))
        return failure();

      if (std::optional<unsigned> resident = residentArgument(func, host)) {
        for (const Event &event : *timeline) {
          if (!event.host.any())
            continue;
          event.anchor->emitError(
              "reads a buffer the caller left on the device, which the host "
              "cannot address");
          return failure();
        }
        func.removeArgAttr(*resident, TeraDialect::kDeviceResidentAttrName);
        continue;
      }

      StagedBuffer buffer = openDeviceBuffer(host);
      Location loc = host.getLoc();
      Residence state =
          isLocalAllocation(host) ? Residence::Nowhere : Residence::Host;
      Operation *lastUse = nullptr;

      for (const Event &event : *timeline) {
        OpBuilder builder(event.anchor);
        if (event.launched) {
          if (state == Residence::Host) {
            copyToDevice(builder, loc, host, buffer);
            state = Residence::Both;
          }
          if (event.device.writes)
            state = Residence::Device;
          lastUse = event.anchor;
          continue;
        }
        if (event.frees)
          continue;
        if (event.host.reads && state == Residence::Device) {
          lastUse = copyToHost(builder, loc, host, buffer);
          state = Residence::Both;
        }
        if (event.host.writes)
          state = Residence::Host;
      }

      OpBuilder builder(host.getContext());
      builder.setInsertionPointAfter(lastUse);
      if (state == Residence::Device && !isLocalAllocation(host))
        copyToHost(builder, loc, host, buffer);
      closeDeviceBuffer(builder, loc, buffer);

      for (OpOperand *use : uses) {
        Value operand = buffer.device;
        if (operand.getType() != host.getType()) {
          OpBuilder castBuilder(use->getOwner());
          operand =
              memref::CastOp::create(castBuilder, loc, host.getType(), operand);
        }
        use->set(operand);
      }
    }
    return success();
  }
};

}
}
