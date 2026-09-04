//===- TeraOpsContraction.td - Contraction and reduction ---*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TERA_OPS_CONTRACTION
#define TERA_OPS_CONTRACTION

include "Tera/IR/TeraOpsBase.td"

def Tera_DotOp : Tera_InferredOp<"dot", [DeclareOpInterfaceMethods<Tera_VjpOpInterface>, Tera_AcceptsDynamicShapes]> {
  let summary = "General contraction of two tensors";
  let description = [{
    Generalises matmul: any set of axes may be contracted, and any set may be
    carried through as batch axes. The result axes are ordered batch axes first,
    then the free axes of the lhs, then the free axes of the rhs.

    Example:

    ```mlir
    %2 = tera.dot %0, %1 {lhs_batch = array<i64>,
                          lhs_contracting = array<i64: 1>,
                          rhs_batch = array<i64>,
                          rhs_contracting = array<i64: 0>}
        : (tensor<2x4xf32>, tensor<4x2xf32>) -> tensor<2x2xf32>
    ```
  }];

  let arguments = (ins Tera_Tensor:$lhs, Tera_Tensor:$rhs,
                       DenseI64ArrayAttr:$lhs_batch,
                       DenseI64ArrayAttr:$lhs_contracting,
                       DenseI64ArrayAttr:$rhs_batch,
                       DenseI64ArrayAttr:$rhs_contracting);
  let results = (outs Tera_Tensor:$result);
  let assemblyFormat =
      "$lhs `,` $rhs attr-dict `:` functional-type(operands, results)";

  let extraClassDeclaration = [{
    /// The lhs axes that are neither batch nor contracting, in order. They are
    /// the result axes that follow the batch axes.
    ::llvm::SmallVector<int64_t> getLhsFreeAxes();

    /// The rhs axes that are neither batch nor contracting, in order. They are
    /// the trailing result axes.
    ::llvm::SmallVector<int64_t> getRhsFreeAxes();
  }];
}

def Tera_ReduceOp : Tera_InferredOp<"reduce", [DeclareOpInterfaceMethods<Tera_VjpOpInterface>, Tera_AcceptsDynamicShapes]> {
  let summary = "Reduce a tensor over a set of axes with an associative combiner";
  let description = [{
    The combiner is an enum rather than a region. A closed enum keeps the VJP
    rule closed too — the gradient of a `sum` reduction is a broadcast, and that
    is knowable without walking a region body.

    Example:

    ```mlir
    %1 = tera.reduce sum, %0 {dimensions = array<i64: 0, 1>}
        : tensor<2x2xf32> -> tensor<f32>
    ```
  }];

  let arguments = (ins Tera_Tensor:$operand, Tera_ReduceKindAttr:$kind,
                       DenseI64ArrayAttr:$dimensions);
  let results = (outs Tera_Tensor:$result);
  let assemblyFormat =
      "$kind `,` $operand attr-dict `:` type($operand) `->` type($result)";
}

#endif // TERA_OPS_CONTRACTION
