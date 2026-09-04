//===- Passes.h - Tera conversion passes ------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TERA_CONVERSION_PASSES_H
#define TERA_CONVERSION_PASSES_H

#include "Tera/Conversion/TeraToLinalg.h"
#include "mlir/Pass/Pass.h"

#include <memory>

namespace mlir::tera {
#define GEN_PASS_DECL
#include "Tera/Conversion/Passes.h.inc"

#define GEN_PASS_REGISTRATION
#include "Tera/Conversion/Passes.h.inc"
} // namespace mlir::tera

#endif // TERA_CONVERSION_PASSES_H
