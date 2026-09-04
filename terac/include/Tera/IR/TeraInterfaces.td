//===- TeraInterfaces.td - Tera op interfaces --------------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TERA_INTERFACES
#define TERA_INTERFACES

include "mlir/IR/OpBase.td"

def Tera_VjpOpInterface : OpInterface<"TeraVjpOpInterface"> {
  let description = [{
    An op that knows its own vector-Jacobian product.

    Reverse-mode autodiff walks a function backwards and asks each op it meets
    for the gradient it passes to its operands. That question is answered here,
    on the op, so `-tera-autodiff` never needs to know which op it is holding.

    An op that does not implement this interface has no derivative, and the pass
    reports that rather than guessing one.
  }];
  let cppNamespace = "::mlir::tera";

  let methods = [
    InterfaceMethod<[{
        Emits the ops computing this op's contribution to the gradient of each
        of its operands, given `adjoints`: one entry per result, holding that
        result's gradient. A rule may read them positionally, because the pass
        supplies a value for every result rather than a hole to test for.

        `builder` is positioned where the new ops belong; the op's own operands
        and results are in scope there, so a rule that needs a forward value
        uses it directly rather than recomputing it.

        On success `operandAdjoints` holds exactly `getNumOperands()` entries,
        in operand order. A null entry means that operand takes no gradient —
        an index, a predicate, or a value behind a `tera.stop_gradient`. A rule
        that cannot differentiate the case in front of it returns failure after
        emitting a diagnostic.
      }],
      "::mlir::LogicalResult", "buildVjp",
      (ins "::mlir::OpBuilder &":$builder,
           "::mlir::ValueRange":$adjoints,
           "::llvm::SmallVectorImpl<::mlir::Value> &":$operandAdjoints)
    >,
  ];
}

#endif // TERA_INTERFACES
