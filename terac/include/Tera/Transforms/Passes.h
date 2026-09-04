//===- Passes.h - Tera dialect passes ---------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef TERA_TRANSFORMS_PASSES_H
#define TERA_TRANSFORMS_PASSES_H

#include "Tera/IR/TeraDialect.h"
#include "Tera/IR/TeraOps.h"
#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {
namespace tera {
#define GEN_PASS_DECL
#include "Tera/Transforms/Passes.h.inc"

#define GEN_PASS_REGISTRATION
#include "Tera/Transforms/Passes.h.inc"
}
}

#endif
