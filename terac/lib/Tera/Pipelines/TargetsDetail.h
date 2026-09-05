//===- TargetsDetail.h - The backends the registry holds --------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TERA_LIB_PIPELINES_TARGETSDETAIL_H
#define TERA_LIB_PIPELINES_TARGETSDETAIL_H

namespace mlir::tera::detail {
void registerHostTarget();

void registerCudaTarget();

}

#endif
