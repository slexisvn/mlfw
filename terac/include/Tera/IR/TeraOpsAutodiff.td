//===- TeraOpsAutodiff.td - Ops that steer autodiff --------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TERA_OPS_AUTODIFF
#define TERA_OPS_AUTODIFF

include "Tera/IR/TeraOpsBase.td"

def Tera_StopGradientOp : Tera_Op<"stop_gradient",
                                  [Pure, Elementwise, SameOperandsAndResultType,
                                   DeclareOpInterfaceMethods<Tera_VjpOpInterface>,
                                   DeclareOpInterfaceMethods<Tera_JvpOpInterface, ["buildJvp"]>,
                                   Tera_Reifies,
                                   Tera_AcceptsDynamicShapes]> {
  let summary = "The identity forwards; a wall to the reverse pass";
  let description = [{
    Forwards its operand unchanged, and reports no gradient for it, so
    `-tera-autodiff` stops there. The stabilising maximum inside a softmax is
    the usual reason: it belongs to the forward value but not to the
    derivative.

    Nothing folds this op away, because folding it away is exactly the bug it
    exists to prevent. It survives until `-convert-tera-to-linalg`, by which
    point the reverse pass has already run.

    Example:

    ```mlir
    %1 = tera.stop_gradient %0 : tensor<4xf32>
    ```
  }];

  let arguments = (ins Tera_Tensor:$operand);
  let results = (outs Tera_Tensor:$result);
  let extraClassDefinition = Tera_ReifyLikeFirstOperand;
  let assemblyFormat = "$operand attr-dict `:` type($result)";
}

#endif
