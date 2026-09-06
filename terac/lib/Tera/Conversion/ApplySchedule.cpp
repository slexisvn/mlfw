//===- ApplySchedule.cpp - Schedule from a script ---------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Tera/Conversion/Passes.h"

#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.h"
#include "mlir/Dialect/Transform/Transforms/TransformInterpreterUtils.h"
#include "mlir/IR/BuiltinOps.h"

namespace mlir::tera {
#define GEN_PASS_DEF_APPLYSCHEDULE
#include "Tera/Conversion/Passes.h.inc"

namespace {
struct ApplySchedule : public impl::ApplyScheduleBase<ApplySchedule> {
  using impl::ApplyScheduleBase<ApplySchedule>::ApplyScheduleBase;

  void runOnOperation() final {
    if (schedule.empty())
      return;

    OwningOpRef<ModuleOp> script;
    if (failed(transform::detail::parseTransformModuleFromFile(
            &getContext(), schedule, script)) ||
        !script) {
      getOperation().emitError()
          << "could not read a schedule from '" << schedule << "'";
      return signalPassFailure();
    }

    // `findTransformEntryPoint` looks in the payload first and the script
    // second, and reports for itself when it finds neither.
    transform::TransformOpInterface entry =
        transform::detail::findTransformEntryPoint(getOperation(), *script,
                                                   entryPoint);
    if (!entry)
      return signalPassFailure();

    if (failed(transform::applyTransformNamedSequence(
            getOperation(), entry, *script,
            transform::TransformOptions().enableExpensiveChecks(false))))
      signalPassFailure();
  }
};

}
}
