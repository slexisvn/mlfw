//===- TeraOpsBase.td - Shared tera op definitions ---------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Type constraints and op families shared by every TeraOps*.td file. Op
// definitions themselves live in the per-family files so that adding an op
// touches one file rather than the whole dialect.
//
//===----------------------------------------------------------------------===//

#ifndef TERA_OPS_BASE
#define TERA_OPS_BASE

include "Tera/IR/TeraDialect.td"
include "Tera/IR/TeraEnums.td"
include "Tera/IR/TeraInterfaces.td"
include "mlir/IR/CommonAttrConstraints.td"
include "mlir/IR/CommonTypeConstraints.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

/// Every tera op reads a rank from its operands, so an unranked tensor is not
/// something the dialect has an answer for. Saying so in the constraint is what
/// stops a builder from assembling one; the shape inference below still reports
/// it as a diagnostic, because inference runs on the parsed operands before any
/// constraint has been checked.
def Tera_Tensor : RankedTensorOf<[AnyFloat, AnySignlessInteger]>;
def Tera_FloatTensor : RankedTensorOf<[AnyFloat]>;
def Tera_BoolTensor : RankedTensorOf<[I1]>;

/// A tensor of positions into another tensor. Separate from `Tera_Tensor`
/// because a float cannot index anything, and an op that takes one reads it
/// rather than computing with it.
def Tera_IndexTensor : RankedTensorOf<[AnySignlessInteger]>;

/// One extent, as a rank-0 tensor rather than an `index`. The dialect is
/// whole-tensor everywhere else and a scalar operand would be the only
/// exception; the lowering reads the value straight back out, so the wrapper
/// costs nothing once the folder has run.
def Tera_ExtentTensor : 0DTensorOf<[I64]> {
  // Buildable, so a `sizes` clause can be printed without repeating the one
  // type every extent has.
  let builderCall =
      "::mlir::RankedTensorType::get({}, $_builder.getIntegerType(64))";
}

/// Carried by an op whose pattern can size a destination that has a `?` in it.
/// `-tera-to-linalg` refuses a dynamic extent on any op without it, so leaving
/// it off is what makes a newly added op opt out until its pattern is known to
/// handle one.
def Tera_AcceptsDynamicShapes : NativeOpTrait<"AcceptsDynamicShapes"> {
  let cppNamespace = "::mlir::tera";
}

/// An op whose result shape and element type follow from its operands and
/// attributes. Implementing `inferReturnTypes` is then the only place the op's
/// shape semantics are written: MLIR derives verification from it, and passes
/// build the op without recomputing anything.
class Tera_InferredOp<string mnemonic, list<Trait> traits = []>
    : Tera_Op<mnemonic, traits # [Pure, InferTypeOpAdaptor]>;

class Tera_BinaryEwiseOp<string mnemonic, list<Trait> traits = []>
    : Tera_Op<mnemonic, traits # [Pure, Elementwise, SameOperandsAndResultType,
                                  DeclareOpInterfaceMethods<Tera_VjpOpInterface>]> {
  let arguments = (ins Tera_Tensor:$lhs, Tera_Tensor:$rhs);
  let results = (outs Tera_Tensor:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($result)";
}

class Tera_UnaryEwiseOp<string mnemonic, Type operandType = Tera_Tensor,
                        list<Trait> traits = []>
    : Tera_Op<mnemonic, traits # [Pure, Elementwise, SameOperandsAndResultType,
                                 DeclareOpInterfaceMethods<Tera_VjpOpInterface>]> {
  let arguments = (ins operandType:$operand);
  let results = (outs operandType:$result);
  let assemblyFormat = "$operand attr-dict `:` type($result)";
}

#endif // TERA_OPS_BASE
