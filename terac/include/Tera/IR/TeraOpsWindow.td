//===- TeraOpsWindow.td - Ops that slide a window ----------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TERA_OPS_WINDOW
#define TERA_OPS_WINDOW

include "Tera/IR/TeraOpsBase.td"

def Tera_ConvOp
    : Tera_InferredOp<"conv", [DeclareOpInterfaceMethods<Tera_VjpOpInterface>,
                               DeclareOpInterfaceMethods<Tera_LoweringOpInterface,
                                                         ["acceptsDynamicShapes"]>,
                               Tera_Reifies, Tera_Multilinear]> {
  let summary = "Slide a kernel over a batch of feature maps";
  let description = [{
    The input is laid out batch, channel, then spatial axes; the kernel is
    output channel, input channel, then spatial axes. mlfw calls those NCHW and
    OIHW and records them as attributes, but only ever emits those two and
    infers only those two, so the dialect writes them into the op instead of
    carrying a name it would refuse every other value of.

    The number of spatial axes is however many `strides` has, so this is a 1-D,
    2-D or 3-D convolution depending on what it is given. `padding` holds two
    entries per spatial axis, the amount before and the amount after, flattened
    rather than nested because a nested attribute is a shape the assembly format
    would have to spell out. `dilation` spaces the kernel out; `groups` splits
    the channels into that many independent convolutions, the first
    `out_channels / groups` outputs reading the first `in_channels / groups`
    inputs and so on.

    A window is only ever taken where it fits, so an axis of extent `n` yields
    `floor((n + pad_low + pad_high - ((k - 1) * dilation + 1)) / stride) + 1`
    windows.

    The batch axis and the spatial axes may be dynamic: the op counts its own
    windows through `ReifyRankedShapedTypeOpInterface`, so a spatial `?` is an
    extent the destination can be told rather than one it has to be given.

    Two `?` are still refused. One anywhere in the kernel sets the reach of
    every window and the size of the reduction, neither of which the result can
    be told from an operand it has not read; and one on the input's channel
    axis is what `groups` cuts into equal parts, which is a division rather
    than a copy and has no answer when the number being divided is not known.

    Example:

    ```mlir
    %2 = tera.conv %0, %1 {strides = array<i64: 1, 1>,
                           padding = array<i64: 0, 0, 0, 0>,
                           dilation = array<i64: 1, 1>,
                           groups = 1 : i64}
        : (tensor<1x1x4x4xf32>, tensor<1x1x2x2xf32>) -> tensor<1x1x3x3xf32>
    ```
  }];

  let arguments = (ins Tera_Tensor:$input,
                       Tera_Tensor:$kernel,
                       DenseI64ArrayAttr:$strides,
                       DenseI64ArrayAttr:$padding,
                       DenseI64ArrayAttr:$dilation,
                       I64Attr:$groups);
  let results = (outs Tera_Tensor:$result);
  let assemblyFormat =
      "$input `,` $kernel attr-dict `:` functional-type(operands, results)";

  let extraClassDeclaration = [{
    /// How many axes the window slides along: the rank less the batch and
    /// channel axes, which is also how long `strides` is.
    int64_t getSpatialRank();

    /// How wide the kernel reaches on each spatial axis once dilation has
    /// spaced it out: `(k - 1) * dilation + 1`.
    ::llvm::SmallVector<int64_t> getReach();

    /// The padding before each spatial axis, then the padding after, unpacked
    /// from the flat attribute the op carries them in.
    ::llvm::SmallVector<int64_t> getPaddingLow();
    ::llvm::SmallVector<int64_t> getPaddingHigh();
  }];
}

def Tera_Pool2dOp
    : Tera_InferredOp<"pool2d",
                      [DeclareOpInterfaceMethods<Tera_VjpOpInterface>,
                       DeclareOpInterfaceMethods<Tera_JvpOpInterface, ["buildJvp"]>,
                       DeclareOpInterfaceMethods<Tera_LoweringOpInterface,
                                                 ["verifyLoweringToLinalg"]>,
                       Tera_Reifies, Tera_AcceptsDynamicShapes]> {
  let summary = "Reduce each window of a batch of feature maps";
  let description = [{
    The traversal `tera.conv` makes, reading a reduction out of each window
    rather than a contraction. `kind` says which reduction: the largest element
    of the window, or their mean.

    The layout is NCHW and the window is two-dimensional, which is what the
    name says and what mlfw's own op infers for. `padding` is flattened the way
    `tera.conv` flattens it.

    An average counts the whole window, padding included, only when
    `count_include_pad` is set; without it a window hanging over the edge
    averages the elements actually inside. A maximum has no such question,
    because the lowering refuses a max pool whose window reaches outside the
    input at all -- there is no value to put there that is not either wrong or
    an infinity the forward pass would have to carry. It is the lowering and
    not the verifier that refuses, so a bridge may still build one and be told
    where it goes wrong rather than what it may not write.

    `ceil_mode` rounds the window count up rather than down, which takes one
    more window along an axis the stride does not divide, and that last window
    hangs over the high edge. What it hangs over is padding: the op is lowered
    with the high edge extended far enough for it, by however much less than a
    stride is missing, and the extension is padding the same way declared
    padding is. So `ceil_mode` is not a separate question from `padding` but
    the same one arrived at differently, and it is answered the same way --
    counted by an average that counts its padding, refused for an average that
    does not and for a maximum, which has nothing to read there.

    That is a choice and not the only one. Frameworks exist that keep the
    overhang out of the divisor, so that the last window of a row is divided
    by how much of it was inside; that is a divisor per window rather than the
    one number the traversal multiplies by, which is the same reason
    `count_include_pad` is required wherever there is padding at all. Here the
    padding is counted, and mlfw's own pooling agrees.

    Any axis may be dynamic. The batch and channel axes the result copies from
    the operand, and the spatial ones it counts the same way it counts them
    from a static extent, so there is no `?` here the shape cannot reach.
    `ceil_mode` over one is refused, though, and this is the one place the op
    needs the extent rather than the shape: how far the last window hangs over
    is what is left of the extent after the stride has divided it, so the
    padding to write depends on a number that is not here.
  }];

  let arguments = (ins Tera_Tensor:$operand,
                       Tera_PoolKindAttr:$kind,
                       DenseI64ArrayAttr:$kernel_size,
                       DenseI64ArrayAttr:$strides,
                       DenseI64ArrayAttr:$padding,
                       DefaultValuedAttr<BoolAttr, "false">:$ceil_mode,
                       DefaultValuedAttr<BoolAttr, "false">:$count_include_pad);
  let results = (outs Tera_Tensor:$result);
  let assemblyFormat =
      "$kind `,` $operand attr-dict `:` type($operand) `->` type($result)";

  let extraClassDeclaration = [{
    /// The padding before each spatial axis, then the padding after.
    ::llvm::SmallVector<int64_t> getPaddingLow();
    ::llvm::SmallVector<int64_t> getPaddingHigh();

    /// How far past the padded input the last window reaches on each spatial
    /// axis, which is what `ceil_mode` buys and what the lowering has to
    /// materialise as more padding. Zero on an axis whose extent is not
    /// known, which is an axis the lowering refuses rather than guesses at.
    ::llvm::SmallVector<int64_t> getWindowOverhang();
  }];
}

#endif
