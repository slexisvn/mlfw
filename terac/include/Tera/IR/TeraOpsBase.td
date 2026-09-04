//===- TeraOpsBase.td - Shared tera op definitions ---------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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

def Tera_Tensor : RankedTensorOf<[AnyFloat, AnySignlessInteger]>;
def Tera_FloatTensor : RankedTensorOf<[AnyFloat]>;
def Tera_BoolTensor : RankedTensorOf<[I1]>;

def Tera_IndexTensor : RankedTensorOf<[AnySignlessInteger]>;

def Tera_ExtentTensor : 0DTensorOf<[I64]> {
  let builderCall =
      "::mlir::RankedTensorType::get({}, $_builder.getIntegerType(64))";
}

def Tera_AcceptsDynamicShapes : NativeOpTrait<"AcceptsDynamicShapes"> {
  let cppNamespace = "::mlir::tera";
}

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

#endif
