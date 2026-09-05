//===- Targets.cpp - The machines a module can be built for -----*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Tera/Pipelines/TargetBackend.h"

#include "TargetsDetail.h"
#include "llvm/ADT/StringMap.h"

using namespace mlir;
using namespace mlir::tera;

namespace {
llvm::StringMap<std::unique_ptr<TargetBackend>> &backends() {
  static llvm::StringMap<std::unique_ptr<TargetBackend>> registry;
  return registry;
}

}

void mlir::tera::registerTargetBackend(std::unique_ptr<TargetBackend> backend) {
  StringRef name = backend->getName();
  backends()[name] = std::move(backend);
}

const TargetBackend *mlir::tera::lookupTargetBackend(StringRef name) {
  auto found = backends().find(name);
  return found == backends().end() ? nullptr : found->second.get();
}

SmallVector<StringRef> mlir::tera::getTargetBackendNames() {
  SmallVector<StringRef> names;
  for (const auto &entry : backends())
    names.push_back(entry.getKey());
  llvm::sort(names);
  return names;
}

void mlir::tera::registerTeraTargets() {
  detail::registerHostTarget();
  detail::registerCudaTarget();
}
