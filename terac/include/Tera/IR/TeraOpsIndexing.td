//===- TeraOpsIndexing.td - Gather and scatter tera ops ----*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TERA_OPS_INDEXING
#define TERA_OPS_INDEXING

include "Tera/IR/TeraOpsBase.td"

def Tera_GatherOp
    : Tera_InferredOp<"gather", [DeclareOpInterfaceMethods<Tera_VjpOpInterface>,
                                 Tera_AcceptsDynamicShapes, Tera_Linear]> {
  let summary = "Read a slice of the operand at every position the indices name";
  let description = [{
    An embedding, and every other lookup, written once. The op reads one slice
    of `operand` per position of `indices` and stacks the results.

    `indices` holds one starting position per lookup. `index_vector_dim` says
    which of its axes runs along a single position's coordinates; every other
    axis is a batch axis, one lookup per element. That axis may equal the rank,
    which means the coordinates are one number long and the axis was left
    implicit -- which is what an embedding's indices look like.

    `start_index_map[k]` is the operand axis coordinate `k` addresses; an
    operand axis nothing addresses starts at zero. `slice_sizes` is how wide
    the slice is on each operand axis, `collapsed_slice_dims` names the axes
    whose width is one and that the result drops, and `offset_dims` says which
    result axes the surviving slice axes land on. The result's other axes are
    the batch axes, in order.

    The result shape follows from all of that, so the op infers it. What does
    not follow is whether a position is in range: like mlfw's own lowering the
    op reads where it is told to, and an index outside the operand is undefined
    rather than clamped.

    Example, an embedding of a `2x3` batch of indices into a `10x4` table:

    ```mlir
    %2 = tera.gather %table, %ids {offset_dims = array<i64: 2>,
                                  collapsed_slice_dims = array<i64: 0>,
                                  start_index_map = array<i64: 0>,
                                  slice_sizes = array<i64: 1, 4>,
                                  index_vector_dim = 2 : i64}
        : (tensor<10x4xf32>, tensor<2x3xi32>) -> tensor<2x3x4xf32>
    ```
  }];

  let arguments = (ins Tera_Tensor:$operand,
                       Tera_IndexTensor:$indices,
                       DenseI64ArrayAttr:$offset_dims,
                       DenseI64ArrayAttr:$collapsed_slice_dims,
                       DenseI64ArrayAttr:$start_index_map,
                       DenseI64ArrayAttr:$slice_sizes,
                       I64Attr:$index_vector_dim);
  let results = (outs Tera_Tensor:$result);
  let assemblyFormat =
      "$operand `,` $indices attr-dict `:` functional-type(operands, results)";

  let extraClassDeclaration = [{
    /// The `indices` axes that are not the index vector, in order. There is
    /// one lookup per element of the tensor they span, and they are the result
    /// axes that `offset_dims` does not claim -- so this is where a `?` among
    /// them is found again.
    ::llvm::SmallVector<int64_t> getBatchAxes();
  }];
}

def Tera_ScatterOp
    : Tera_InferredOp<"scatter", [DeclareOpInterfaceMethods<Tera_VjpOpInterface>,
                                  Tera_AcceptsDynamicShapes, Tera_Linear]> {
  let summary = "Add updates into the operand at every position the indices name";
  let description = [{
    The reverse of `tera.gather`, and the mirror of it in every attribute:
    `update_window_dims` is the gather's `offset_dims` read on `updates`,
    `inserted_window_dims` its `collapsed_slice_dims`, and
    `scatter_dims_to_operand_dims` its `start_index_map`. The result is the
    operand with one window of `updates` added in at each position.

    The combiner is addition and is not written down. mlfw's scatter carries it
    as an empty region -- a block with two scalar arguments and no body -- and
    its lowering adds; a scatter that combined some other way would be a
    different op with a different derivative, not this one with an attribute.

    Two updates landing on the same element both count, which is what makes
    this the derivative of a gather that read the same row twice. They are
    added in the order `updates` is laid out, so the answer is the one mlfw's
    loop nest produces down to the rounding.

    Example, the derivative of the embedding above:

    ```mlir
    %3 = tera.scatter %zeros, %ids, %grad
        {update_window_dims = array<i64: 2>,
         inserted_window_dims = array<i64: 0>,
         scatter_dims_to_operand_dims = array<i64: 0>,
         index_vector_dim = 2 : i64}
        : (tensor<10x4xf32>, tensor<2x3xi32>, tensor<2x3x4xf32>)
            -> tensor<10x4xf32>
    ```
  }];

  let arguments = (ins Tera_Tensor:$operand,
                       Tera_IndexTensor:$indices,
                       Tera_Tensor:$updates,
                       DenseI64ArrayAttr:$update_window_dims,
                       DenseI64ArrayAttr:$inserted_window_dims,
                       DenseI64ArrayAttr:$scatter_dims_to_operand_dims,
                       I64Attr:$index_vector_dim);
  let results = (outs Tera_Tensor:$result);
  let assemblyFormat =
      "$operand `,` $indices `,` $updates attr-dict `:` functional-type(operands, results)";

  let extraClassDeclaration = [{
    /// How wide a window is on each operand axis: one on an inserted axis,
    /// and the operand's own extent everywhere else. It is what the gather in
    /// this op's derivative reads back, so it is written here rather than
    /// there.
    ::llvm::SmallVector<int64_t> getWindowSizes();
  }];
}

#endif
