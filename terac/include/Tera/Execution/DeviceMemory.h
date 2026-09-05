//===- DeviceMemory.h - Memory that outlives one call -----------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TERA_EXECUTION_DEVICEMEMORY_H
#define TERA_EXECUTION_DEVICEMEMORY_H

#include "mlir/ExecutionEngine/ExecutionEngine.h"

#include <cstddef>
#include <cstdint>
#include <memory>

namespace mlir::tera {
/// Memory on the device a compiled module runs on, kept across calls.
///
/// terac links no vendor runtime of its own. The JIT has already loaded the
/// target's runtime library to resolve the launches, and what it exports for
/// allocating and copying is what the generated code calls, so those are the
/// symbols this asks it for. A target whose libraries export none of them has
/// no device memory to hand out, which is what a host-only build looks like.
class DeviceMemory {
public:
  ~DeviceMemory();

  DeviceMemory(const DeviceMemory &) = delete;
  DeviceMemory &operator=(const DeviceMemory &) = delete;

  static std::unique_ptr<DeviceMemory> resolve(ExecutionEngine &engine);

  /// Null when the device has no room left.
  void *allocate(size_t bytes);

  void release(void *pointer);

  void upload(void *device, const void *host, size_t bytes);

  void download(void *host, const void *device, size_t bytes);

private:
  DeviceMemory() = default;

  void *stream = nullptr;
  void *(*memAlloc)(uint64_t, void *, bool) = nullptr;
  void (*memFree)(void *, void *) = nullptr;
  void (*memCopy)(void *, void *, size_t, void *) = nullptr;
  void *(*streamCreate)() = nullptr;
  void (*streamDestroy)(void *) = nullptr;
  void (*streamSynchronize)(void *) = nullptr;
};

}

#endif
