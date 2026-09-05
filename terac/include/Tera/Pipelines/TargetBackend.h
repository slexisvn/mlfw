//===- TargetBackend.h - The machine a tera module is built for -*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TERA_PIPELINES_TARGETBACKEND_H
#define TERA_PIPELINES_TARGETBACKEND_H

#include "mlir/Support/LLVM.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

#include <memory>

namespace mlir {
class OpPassManager;

namespace tera {
class TargetBackend {
public:
  virtual ~TargetBackend() = default;

  virtual StringRef getName() const = 0;

  virtual StringRef getDescription() const = 0;

  virtual LogicalResult buildPipeline(OpPassManager &pm, StringRef options,
                                      raw_ostream &errorStream) const = 0;

  virtual ArrayRef<StringRef> getRuntimeLibraries() const { return {}; }
};

void registerTargetBackend(std::unique_ptr<TargetBackend> backend);

const TargetBackend *lookupTargetBackend(StringRef name);

SmallVector<StringRef> getTargetBackendNames();

void registerTeraTargets();

}
}

#endif
