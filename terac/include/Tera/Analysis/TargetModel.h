//===- TargetModel.h - What a schedule is chosen for ------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TERA_ANALYSIS_TARGETMODEL_H
#define TERA_ANALYSIS_TARGETMODEL_H

#include "mlir/Support/LLVM.h"
#include "llvm/ADT/SmallVector.h"

#include <cstdint>

namespace mlir::tera {
struct HostTargetModel {
  int64_t vectorLanes = 16;
  int64_t maxVectorElements = 1024;
};

struct GpuTargetModel {
  int64_t warpSize = 32;
  int64_t maxThreadsPerBlock = 1024;
  int64_t preferredThreadsPerBlock = 256;
  SmallVector<int64_t, 3> maxBlockExtents = {1024, 1024, 64};
};

}

#endif
