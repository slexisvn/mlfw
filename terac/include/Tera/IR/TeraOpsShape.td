//===- TeraOpsShape.td - Shape and layout tera ops ---------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TERA_OPS_SHAPE
#define TERA_OPS_SHAPE

include "Tera/IR/TeraOpsBase.td"

def Tera_DimOp : Tera_Op<"dim", [Pure, Tera_AcceptsDynamicShapes]> {
  let summary = "The extent of one axis of a tensor";
  let description = [{
    The only op that turns a shape into a value. A dynamic extent is written
    `?` in a type, which says that the number is not known here but not where
    it comes from; this is what says where.

    Example:

    ```mlir
    %0 = tera.dim %x {dimension = 0 : i64} : tensor<?x4xf32> -> tensor<i64>
    ```
  }];

  let arguments = (ins Tera_Tensor:$operand, I64Attr:$dimension);
  let results = (outs Tera_ExtentTensor:$result);
  let assemblyFormat =
      "$operand attr-dict `:` type($operand) `->` type($result)";
  let hasVerifier = 1;
  let hasFolder = 1;
}

def Tera_BroadcastInDimOp
    : Tera_Op<"broadcast_in_dim",
              [Pure, DeclareOpInterfaceMethods<Tera_VjpOpInterface>,
               Tera_Reifies, Tera_AcceptsDynamicShapes, Tera_Linear]> {
  let summary = "Broadcast a tensor into a larger shape";
  let description = [{
    `broadcast_dimensions` maps each axis of the operand to the result axis it
    lands on. The mapping must be strictly increasing, and every operand axis
    must either match the result extent it maps to or have extent 1.

    The target shape is given by the result type rather than a separate
    attribute, so the two can never disagree. That also means the result is not
    inferable from the operand, which is why this op verifies instead.

    This is the one op that decides a shape rather than inheriting one, so it
    is also the one that a dynamic extent cannot reach by itself: a scalar
    broadcast to the shape of an activation has no operand carrying that
    number. `sizes` is where it comes from -- one extent per `?` in the result
    type, in order, and required exactly when the result has one.

    Example:

    ```mlir
    %1 = tera.broadcast_in_dim %0 {broadcast_dimensions = array<i64: 1>}
        : tensor<3xf32> -> tensor<2x3xf32>
    %3 = tera.broadcast_in_dim %0 sizes(%2) {broadcast_dimensions = array<i64: 1>}
        : tensor<3xf32> -> tensor<?x3xf32>
    ```
  }];

  let arguments = (ins Tera_Tensor:$operand,
                       Variadic<Tera_ExtentTensor>:$sizes,
                       DenseI64ArrayAttr:$broadcast_dimensions);
  let results = (outs Tera_Tensor:$result);
  let assemblyFormat =
      "$operand (`sizes` `(` $sizes^ `)`)? attr-dict `:` type($operand) `->` type($result)";
  let hasVerifier = 1;
  let hasFolder = 1;
  let hasCanonicalizer = 1;
}

def Tera_TransposeOp : Tera_InferredOp<"transpose", [DeclareOpInterfaceMethods<Tera_VjpOpInterface>, Tera_Reifies, Tera_AcceptsDynamicShapes, Tera_Linear]> {
  let summary = "Permute the axes of a tensor";
  let description = [{
    `permutation[i]` is the operand axis that becomes result axis `i`.
  }];

  let arguments = (ins Tera_Tensor:$operand, DenseI64ArrayAttr:$permutation);
  let results = (outs Tera_Tensor:$result);
  let assemblyFormat =
      "$operand attr-dict `:` type($operand) `->` type($result)";
  let hasFolder = 1;
  let hasCanonicalizer = 1;
}

def Tera_ReshapeOp : Tera_Op<"reshape",
                             [Pure,
                              DeclareOpInterfaceMethods<Tera_VjpOpInterface>,
                              Tera_Reifies,
                              Tera_AcceptsDynamicShapes, Tera_Linear]> {
  let summary = "Reinterpret a tensor with a new shape and the same element count";
  let description = [{
    The target shape is given by the result type, so this op verifies rather
    than infers.

    `sizes` is where a dynamic result extent comes from -- one per `?`, in
    order, and required exactly when the result has one. A reshape decides its
    result's shape rather than inheriting it, and the element count alone does
    not factorise into extents.
  }];

  let arguments = (ins Tera_Tensor:$operand,
                       Variadic<Tera_ExtentTensor>:$sizes);
  let results = (outs Tera_Tensor:$result);
  let assemblyFormat =
      "$operand (`sizes` `(` $sizes^ `)`)? attr-dict `:` type($operand) `->` type($result)";
  let hasVerifier = 1;
  let hasFolder = 1;
  let hasCanonicalizer = 1;
}

def Tera_SliceOp
    : Tera_InferredOp<"slice",
                      [DeclareOpInterfaceMethods<Tera_VjpOpInterface>,
                       DeclareOpInterfaceMethods<Tera_LoweringOpInterface,
                                                 ["acceptsDynamicShapes"]>,
                       Tera_Linear]> {
  let summary = "Extract a strided rectangular region";
  let description = [{
    Half-open on the high end: axis `i` yields indices
    `start_indices[i], start_indices[i] + strides[i], ...` while strictly below
    `limit_indices[i]`.
  }];

  let arguments = (ins Tera_Tensor:$operand,
                       DenseI64ArrayAttr:$start_indices,
                       DenseI64ArrayAttr:$limit_indices,
                       DenseI64ArrayAttr:$strides);
  let results = (outs Tera_Tensor:$result);
  let assemblyFormat =
      "$operand attr-dict `:` type($operand) `->` type($result)";
}

def Tera_ReverseOp
    : Tera_InferredOp<"reverse",
                      [DeclareOpInterfaceMethods<Tera_VjpOpInterface>,
                       DeclareOpInterfaceMethods<Tera_LoweringOpInterface,
                                                 ["acceptsDynamicShapes"]>,
                       Tera_Reifies, Tera_Linear]> {
  let summary = "Read an axis from the far end";
  let description = [{
    `dimensions` names the axes to reverse; the shape does not change.

    It is here because a convolution runs backwards through a flipped kernel,
    and because it is its own derivative -- the adjoint of a reversed read is
    the reverse of the adjoint.
  }];

  let arguments = (ins Tera_Tensor:$operand, DenseI64ArrayAttr:$dimensions);
  let results = (outs Tera_Tensor:$result);
  let extraClassDefinition = Tera_ReifyLikeFirstOperand;
  let assemblyFormat =
      "$operand attr-dict `:` type($operand) `->` type($result)";
  let hasFolder = 1;
  let hasCanonicalizer = 1;
}

def Tera_PadOp
    : Tera_InferredOp<"pad", [DeclareOpInterfaceMethods<Tera_VjpOpInterface>,
                              Tera_Reifies,
                              Tera_AcceptsDynamicShapes, Tera_Linear]> {
  let summary = "Surround a tensor with a value, and space its elements out";
  let description = [{
    `low` and `high` say how much goes before and after each axis; `interior`
    says how much goes between neighbouring elements of it. An axis of extent
    `n` becomes `low + n + (n - 1) * interior + high`.

    Interior padding is what makes this more than a border. Undoing a strided
    read means putting the adjoint back where it was read from with `stride - 1`
    holes between, and that is the same op with `low` and `high` at zero.

    `interior` is optional because a border is the common case and mlfw leaves
    it out there; absent means no holes. `getSpacing` is how the op is asked
    how far apart its elements sit, and answers 1 either way.

    Example:

    ```mlir
    %2 = tera.pad %0, %1 {low = array<i64: 1, 0>, high = array<i64: 0, 0>,
                          interior = array<i64: 0, 1>}
        : (tensor<2x3xf32>, tensor<f32>) -> tensor<3x5xf32>
    ```
  }];

  let arguments = (ins Tera_Tensor:$operand,
                       Tera_Tensor:$padding_value,
                       DenseI64ArrayAttr:$low,
                       DenseI64ArrayAttr:$high,
                       OptionalAttr<DenseI64ArrayAttr>:$interior);
  let results = (outs Tera_Tensor:$result);
  let assemblyFormat =
      "$operand `,` $padding_value attr-dict `:` functional-type(operands, results)";
  let hasFolder = 1;

  let extraClassDeclaration = [{
    /// How far apart neighbouring elements of each axis sit in the result:
    /// one more than the holes between them, and one where there are none.
    ::llvm::SmallVector<int64_t> getSpacing();
  }];
}

def Tera_ConcatOp : Tera_InferredOp<"concat", [DeclareOpInterfaceMethods<Tera_VjpOpInterface>, Tera_Reifies, Tera_AcceptsDynamicShapes, Tera_Linear]> {
  let summary = "Join tensors along one axis";

  let arguments = (ins Variadic<Tera_Tensor>:$inputs, I64Attr:$dimension);
  let results = (outs Tera_Tensor:$result);
  let assemblyFormat =
      "$inputs attr-dict `:` type($inputs) `->` type($result)";
  let hasFolder = 1;
}

#endif
