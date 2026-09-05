//===- DeviceMemory.cpp - Memory that outlives one call ---------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Tera/Execution/DeviceMemory.h"

using namespace mlir;
using namespace mlir::tera;

namespace {
template <typename Signature>
Signature find(ExecutionEngine &engine, StringRef name) {
  llvm::Expected<void *> symbol = engine.lookup(name);
  if (!symbol) {
    llvm::consumeError(symbol.takeError());
    return nullptr;
  }
  return reinterpret_cast<Signature>(*symbol);
}

}

std::unique_ptr<DeviceMemory> DeviceMemory::resolve(ExecutionEngine &engine) {
  std::unique_ptr<DeviceMemory> memory(new DeviceMemory);
  memory->memAlloc =
      find<void *(*)(uint64_t, void *, bool)>(engine, "mgpuMemAlloc");
  memory->memFree = find<void (*)(void *, void *)>(engine, "mgpuMemFree");
  memory->memCopy =
      find<void (*)(void *, void *, size_t, void *)>(engine, "mgpuMemcpy");
  memory->streamCreate = find<void *(*)()>(engine, "mgpuStreamCreate");
  memory->streamDestroy = find<void (*)(void *)>(engine, "mgpuStreamDestroy");
  memory->streamSynchronize =
      find<void (*)(void *)>(engine, "mgpuStreamSynchronize");

  if (!memory->memAlloc || !memory->memFree || !memory->memCopy ||
      !memory->streamCreate || !memory->streamDestroy ||
      !memory->streamSynchronize)
    return nullptr;

  memory->stream = memory->streamCreate();
  return memory;
}

DeviceMemory::~DeviceMemory() {
  if (stream)
    streamDestroy(stream);
}

void *DeviceMemory::allocate(size_t bytes) {
  void *pointer = memAlloc(bytes, stream, /*isHostShared=*/false);
  streamSynchronize(stream);
  return pointer;
}

void DeviceMemory::release(void *pointer) {
  if (!pointer)
    return;
  memFree(pointer, stream);
  streamSynchronize(stream);
}

void DeviceMemory::upload(void *device, const void *host, size_t bytes) {
  memCopy(device, const_cast<void *>(host), bytes, stream);
  streamSynchronize(stream);
}

void DeviceMemory::download(void *host, const void *device, size_t bytes) {
  memCopy(host, const_cast<void *>(device), bytes, stream);
  streamSynchronize(stream);
}
