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

// The extents of a result carrying the shape of the op's first operand.
// Every elementwise op answers this way, and so does every op that only moves
// elements around, so the answer is written once rather than on each of them.
// It is a definition rather than a declaration because the helper it calls is
// private to the dialect library, which the generated header is not.
defvar Tera_ReifyLikeFirstOperand = [{
  ::llvm::LogicalResult $cppClass::reifyResultShapes(
      ::mlir::OpBuilder &builder,
      ::mlir::ReifiedRankedShapedTypeDims &reified) {
    reified.emplace_back(::mlir::tera::detail::reifyExtentsLike(
        builder, getLoc(),
        ::llvm::cast<::mlir::RankedTensorType>(
            (*this)->getResult(0).getType()),
        (*this)->getOperand(0)));
    return ::mlir::success();
  }
}];

// The interface's own default answers `failure()` -- the shape cannot be
// computed -- so the method has to be named to override it rather than
// inherited.
def Tera_Reifies
    : DeclareOpInterfaceMethods<ReifyRankedShapedTypeOpInterface,
                                ["reifyResultShapes"]>;

def Tera_Linear : NativeOpTrait<"LinearOperands"> {
  let cppNamespace = "::mlir::tera";
}

def Tera_Multilinear : NativeOpTrait<"MultilinearOperands"> {
  let cppNamespace = "::mlir::tera";
}

class Tera_InferredOp<string mnemonic, list<Trait> traits = []>
    : Tera_Op<mnemonic, traits # [Pure, InferTypeOpAdaptor]>;

class Tera_BinaryEwiseOp<string mnemonic, list<Trait> traits = []>
    : Tera_Op<mnemonic, traits # [Pure, Elementwise, SameOperandsAndResultType,
                                  Tera_Reifies,
                                  DeclareOpInterfaceMethods<Tera_VjpOpInterface>]> {
  let arguments = (ins Tera_Tensor:$lhs, Tera_Tensor:$rhs);
  let results = (outs Tera_Tensor:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($result)";
  let extraClassDefinition = Tera_ReifyLikeFirstOperand;
}

class Tera_UnaryEwiseOp<string mnemonic, Type operandType = Tera_Tensor,
                        list<Trait> traits = []>
    : Tera_Op<mnemonic, traits # [Pure, Elementwise, SameOperandsAndResultType,
                                 Tera_Reifies,
                                 DeclareOpInterfaceMethods<Tera_VjpOpInterface>]> {
  let arguments = (ins operandType:$operand);
  let results = (outs operandType:$result);
  let assemblyFormat = "$operand attr-dict `:` type($result)";
  let extraClassDefinition = Tera_ReifyLikeFirstOperand;
}

#endif
