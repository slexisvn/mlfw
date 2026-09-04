//===- TeraOpsElementwise.td - Elementwise tera ops --------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TERA_OPS_ELEMENTWISE
#define TERA_OPS_ELEMENTWISE

include "Tera/IR/TeraOpsBase.td"

def Tera_AddOp : Tera_BinaryEwiseOp<"add", [Commutative, Tera_AcceptsDynamicShapes]> {
  let summary = "Elementwise addition";
}

def Tera_SubOp : Tera_BinaryEwiseOp<"sub", [Tera_AcceptsDynamicShapes]> {
  let summary = "Elementwise subtraction";
}

def Tera_MulOp : Tera_BinaryEwiseOp<"mul", [Commutative, Tera_AcceptsDynamicShapes]> {
  let summary = "Elementwise multiplication";
}

def Tera_DivOp : Tera_BinaryEwiseOp<"div", [Tera_AcceptsDynamicShapes]> {
  let summary = "Elementwise division";
}

def Tera_MaximumOp : Tera_BinaryEwiseOp<"maximum", [Commutative, Tera_AcceptsDynamicShapes]> {
  let summary = "Elementwise maximum";
}

def Tera_NegOp : Tera_UnaryEwiseOp<"neg", Tera_Tensor, [Tera_AcceptsDynamicShapes]> {
  let summary = "Elementwise negation";
}

def Tera_ExpOp : Tera_UnaryEwiseOp<"exp", Tera_FloatTensor, [Tera_AcceptsDynamicShapes]> {
  let summary = "Elementwise natural exponential";
}

def Tera_SqrtOp : Tera_UnaryEwiseOp<"sqrt", Tera_FloatTensor> {
  let summary = "Elementwise square root";
}

def Tera_RsqrtOp : Tera_UnaryEwiseOp<"rsqrt", Tera_FloatTensor> {
  let summary = "Elementwise reciprocal square root";
  let description = [{
    Its own op rather than `div` of a `sqrt`, because a normalisation divides by
    a standard deviation once per element and the hardware has the instruction.
  }];
}

def Tera_TanhOp : Tera_UnaryEwiseOp<"tanh", Tera_FloatTensor> {
  let summary = "Elementwise hyperbolic tangent";
}

def Tera_CompareOp
    : Tera_InferredOp<"compare", [Elementwise, SameTypeOperands, DeclareOpInterfaceMethods<Tera_VjpOpInterface>, Tera_AcceptsDynamicShapes]> {
  let summary = "Elementwise comparison producing a boolean tensor";

  let arguments = (ins Tera_Tensor:$lhs, Tera_Tensor:$rhs,
                       Tera_ComparisonDirectionAttr:$direction);
  let results = (outs Tera_BoolTensor:$result);
  let assemblyFormat =
      "$direction `,` $lhs `,` $rhs attr-dict `:` type($lhs) `->` type($result)";
}

def Tera_SelectOp : Tera_Op<"select", [Pure, Elementwise,
                                       SameOperandsAndResultShape,
                                       AllTypesMatch<["on_true", "on_false",
                                                      "result"]>,
                                       DeclareOpInterfaceMethods<Tera_VjpOpInterface>, Tera_AcceptsDynamicShapes]> {
  let summary = "Elementwise choice between two tensors";

  let arguments = (ins Tera_BoolTensor:$pred, Tera_Tensor:$on_true,
                       Tera_Tensor:$on_false);
  let results = (outs Tera_Tensor:$result);
  let assemblyFormat =
      "$pred `,` $on_true `,` $on_false attr-dict `:` type($pred) `,` type($result)";
}

def Tera_ConvertOp : Tera_Op<"convert", [Pure, Elementwise,
                                         SameOperandsAndResultShape,
                                         DeclareOpInterfaceMethods<Tera_VjpOpInterface>, Tera_AcceptsDynamicShapes]> {
  let summary = "Elementwise element type conversion";
  let description = [{
    The target element type is given by the result type, not derived, so this op
    does not infer its result.
  }];

  let arguments = (ins Tera_Tensor:$operand);
  let results = (outs Tera_Tensor:$result);
  let assemblyFormat =
      "$operand attr-dict `:` type($operand) `->` type($result)";
  let hasFolder = 1;
}

#endif
