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

def Tera_JvpOpInterface : OpInterface<"TeraJvpOpInterface"> {
  let description = [{
    An op that knows how its result moves when its operands do.

    Forward mode carries a tangent beside every value and asks each op it meets,
    in program order, for the tangent of its results. That is the question here.

    Almost no op has to answer it. `Tera_Linear` says the result is a linear
    function of the operands jointly, so the tangent is the same op over the
    tangents; `Tera_Multilinear` says it is linear in each operand separately,
    so the tangent is the sum of the op with one operand swapped at a time; and
    an elementwise op whose operands have the result's type has a diagonal
    Jacobian, which is its own transpose, so its `buildVjp` already computes
    the forward direction and is reused. What is left over is the handful of
    ops that genuinely compute something different forwards.
  }];
  let cppNamespace = "::mlir::tera";

  let methods = [
    InterfaceMethod<[{
        Emits the ops computing the tangent of each result, given `tangents`:
        one entry per operand, null where that operand does not move.

        On success `resultTangents` holds exactly `getNumResults()` entries. A
        null entry means the result does not move at all, which is what a
        constant, an index, or a `tera.stop_gradient` answers.

        A rule that builds one op computing the result and its tangent together
        -- which is what the region ops do, because running the body twice is
        the cost forward mode exists to avoid -- leaves that op's results in
        `primalResults`, and the copy of this op that the walk made is dropped.
        Leaving it empty means this op's own results are the primal ones.
      }],
      "::mlir::LogicalResult", "buildJvp",
      (ins "::mlir::OpBuilder &":$builder,
           "::mlir::ValueRange":$tangents,
           "::llvm::SmallVectorImpl<::mlir::Value> &":$resultTangents,
           "::llvm::SmallVectorImpl<::mlir::Value> &":$primalResults),
      /*methodBody=*/[{}],
      /*defaultImplementation=*/[{
        return ::mlir::tera::buildStructuralJvp($_op, builder, tangents,
                                                resultTangents);
      }]
    >,
  ];
}

def Tera_LoweringOpInterface : OpInterface<"TeraLoweringOpInterface"> {
  let description = [{
    What an op knows about being lowered to linalg.

    The conversion materialises every destination up front, so an op has to
    say whether a `?` among its operands or results is still an extent it can
    reach, and whether anything else about it puts it out of reach. Both are
    answered here, on the op, so `-convert-tera-to-linalg` holds no op names.
  }];
  let cppNamespace = "::mlir::tera";

  let methods = [
    InterfaceMethod<[{
        Whether this op lowers with a dynamic extent among its operands or its
        results. The default reads the `AcceptsDynamicShapes` trait, which is
        the declarative form of an unconditional yes; an op whose answer
        depends on which axis carries the `?` answers for itself.
      }],
      "bool", "acceptsDynamicShapes", (ins), /*methodBody=*/[{}],
      /*defaultImplementation=*/[{
        return $_op->template hasTrait<::mlir::tera::AcceptsDynamicShapes>();
      }]
    >,
    InterfaceMethod<[{
        Whether this op can be lowered as it stands, emitting on itself the
        reason it cannot. Asked before the conversion runs, so a refusal is
        the op's own sentence rather than a failure to legalize.
      }],
      "::mlir::LogicalResult", "verifyLoweringToLinalg", (ins),
      /*methodBody=*/[{}],
      /*defaultImplementation=*/[{ return ::mlir::success(); }]
    >,
  ];
}

#endif
