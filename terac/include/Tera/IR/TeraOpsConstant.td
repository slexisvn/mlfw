//===- TeraOpsConstant.td - Value-producing tera ops -------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TERA_OPS_CONSTANT
#define TERA_OPS_CONSTANT

include "Tera/IR/TeraOpsBase.td"

def Tera_ConstantOp : Tera_Op<"constant", [ConstantLike, Pure,
                                           AllTypesMatch<["value", "result"]>,
                                           DeclareOpInterfaceMethods<Tera_VjpOpInterface>]> {
  let summary = "A constant tensor";
  let description = [{
    The result type is taken from the attribute, so it is never written twice.

    Example:

    ```mlir
    %0 = tera.constant dense<0.000000e+00> : tensor<f32>
    ```
  }];

  let arguments = (ins ElementsAttr:$value);
  let results = (outs Tera_Tensor:$result);
  let assemblyFormat = "attr-dict $value";
  let hasFolder = 1;
}

def Tera_IotaOp : Tera_Op<"iota", [Pure, DeclareOpInterfaceMethods<Tera_VjpOpInterface>, Tera_AcceptsDynamicShapes]> {
  let summary = "A tensor counting up along one axis";
  let description = [{
    Element `[i0, ..., in]` holds `i[iota_dimension]`.

    `sizes` is where a dynamic result extent comes from -- one per `?`, in
    order, and required exactly when the result has one. Unlike `tera.constant`
    this op holds no data, so a shape it was told is a shape it can fill.
  }];

  let arguments = (ins I64Attr:$iota_dimension,
                       Variadic<Tera_ExtentTensor>:$sizes);
  let results = (outs Tera_Tensor:$result);
  let assemblyFormat = "(`sizes` `(` $sizes^ `)`)? attr-dict `:` type($result)";
  let hasVerifier = 1;
}

#endif
