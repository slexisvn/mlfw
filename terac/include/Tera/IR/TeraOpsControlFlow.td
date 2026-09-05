//===- TeraOpsControlFlow.td - Region-carrying tera ops ----*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TERA_OPS_CONTROL_FLOW
#define TERA_OPS_CONTROL_FLOW

include "Tera/IR/TeraOpsBase.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"

def Tera_YieldOp : Tera_Op<"yield",
        [Pure, Terminator, NativeOpTrait<"ReturnLike">,
         DeclareOpInterfaceMethods<RegionBranchTerminatorOpInterface,
                                   ["getMutableSuccessorOperands"]>,
         ParentOneOf<["::mlir::tera::IfOp", "::mlir::tera::ScanOp"]>, Tera_AcceptsDynamicShapes]> {
  let summary = "Ends the body of a region-carrying tera op";

  let arguments = (ins Variadic<Tera_Tensor>:$results);
  let assemblyFormat = "attr-dict ($results^ `:` type($results))?";

  let builders = [OpBuilder<(ins), [{ /* nothing to do */ }]>];
}

def Tera_IfOp : Tera_Op<"if",
        [RecursiveMemoryEffects, RecursivelySpeculatable, IsolatedFromAbove,
         SingleBlockImplicitTerminator<"::mlir::tera::YieldOp">,
         DeclareOpInterfaceMethods<RegionBranchOpInterface,
                                   ["getEntrySuccessorOperands",
                                    "getSuccessorInputs",
                                    "getRegionInvocationBounds"]>,
         DeclareOpInterfaceMethods<Tera_VjpOpInterface>,
         DeclareOpInterfaceMethods<Tera_JvpOpInterface, ["buildJvp"]>,
         Tera_AcceptsDynamicShapes]> {
  let summary = "Runs one of two bodies on the same inputs";
  let description = [{
    `condition` is a rank-0 boolean tensor. Both bodies take the op's inputs as
    their block arguments and yield the op's results, so the two agree on their
    signature by construction.

    Example:

    ```mlir
    %0 = tera.if %p, %x : (tensor<i1>, tensor<4xf32>) -> tensor<4xf32> {
    ^bb0(%a: tensor<4xf32>):
      %1 = tera.exp %a : tensor<4xf32>
      tera.yield %1 : tensor<4xf32>
    } else {
    ^bb0(%a: tensor<4xf32>):
      tera.yield %a : tensor<4xf32>
    }
    ```
  }];

  let arguments = (ins Tera_BoolTensor:$condition,
                       Variadic<Tera_Tensor>:$inputs);
  let results = (outs Variadic<Tera_Tensor>:$results);
  let regions = (region SizedRegion<1>:$thenBody, SizedRegion<1>:$elseBody);
  let assemblyFormat = [{
    $condition (`,` $inputs^)? attr-dict `:`
    functional-type(operands, results) $thenBody `else` $elseBody
  }];
  let hasVerifier = 1;
}

def Tera_ScanOp : Tera_Op<"scan",
        [RecursiveMemoryEffects, RecursivelySpeculatable, IsolatedFromAbove,
         SingleBlockImplicitTerminator<"::mlir::tera::YieldOp">,
         AttrSizedOperandSegments,
         DeclareOpInterfaceMethods<RegionBranchOpInterface,
                                   ["getEntrySuccessorOperands",
                                    "getSuccessorInputs",
                                    "getRegionInvocationBounds"]>,
         DeclareOpInterfaceMethods<Tera_VjpOpInterface>,
         DeclareOpInterfaceMethods<Tera_JvpOpInterface, ["buildJvp"]>,
         DeclareOpInterfaceMethods<Tera_LoweringOpInterface,
                                   ["verifyLoweringToLinalg"]>,
         Tera_AcceptsDynamicShapes]> {
  let summary = "Threads a carry through a sequence, collecting one output per step";
  let description = [{
    The body runs once per step. It takes the carry, then one slice of each
    input, then each constant unchanged, and yields the next carry followed by
    one slice of each output. Axis 0 of every input and every stacked output is
    the step axis, and its extent is the trip count — which is why at least one
    input is required.

    A constant is what a recurrence's weights are: read every step, sliced by
    none of them. Its gradient is the sum of what every step contributed, so
    the reverse pass accumulates it in a carry rather than stacking it.

    `checkpoint` trades memory in the reverse pass for arithmetic. Without it,
    differentiating keeps the carry from every step, which is `steps` of them.
    With it, only every `checkpoint`-th carry is kept and the ones between are
    recomputed from the nearest one, so the reverse pass holds
    `steps / checkpoint + checkpoint` carries and runs the body one extra time.
    It must divide the trip count.

    Results are the final carries first, then the stacked outputs.

    `sizes` is where a dynamic result extent comes from -- one extent per `?`
    across the results, in order, and required exactly when a result has one.
    A stacked output's shape is decided inside a body that is isolated from
    what surrounds it, so no operand type answers for it and the op has to be
    told; the trip count is not among them, being an input's own leading
    extent.

    `reverse` runs the steps from the last to the first. It does not reverse the
    data: step `t` still reads index `t` of the inputs and writes index `t` of
    the outputs, so a backward pass over a forward scan lines up index for index
    and needs no reversal ops on either side.

    Example, summing a sequence while recording the running total:

    ```mlir
    %sum, %totals = tera.scan init(%zero : tensor<f32>) xs(%xs : tensor<8xf32>)
        -> (tensor<f32>, tensor<8xf32>) {
    ^bb0(%acc: tensor<f32>, %x: tensor<f32>):
      %0 = tera.add %acc, %x : tensor<f32>
      tera.yield %0, %0 : tensor<f32>, tensor<f32>
    }
    ```
  }];

  let arguments = (ins Variadic<Tera_Tensor>:$inits,
                       Variadic<Tera_Tensor>:$inputs,
                       Variadic<Tera_Tensor>:$constants,
                       Variadic<Tera_ExtentTensor>:$sizes, UnitAttr:$reverse,
                       OptionalAttr<ConfinedAttr<I64Attr,
                           [IntMinValue<1>]>>:$checkpoint);
  let results = (outs Variadic<Tera_Tensor>:$results);
  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat = [{
    (`reverse` $reverse^)? `init` `(` $inits `:` type($inits) `)`
    `xs` `(` ($inputs^ `:` type($inputs))? `)`
    (`consts` `(` $constants^ `:` type($constants) `)`)?
    (`sizes` `(` $sizes^ `)`)?
    attr-dict `->` `(` type($results) `)` $body
  }];
  let hasVerifier = 1;

  let extraClassDeclaration = [{
    /// The results holding the carry as it left the last step.
    ::mlir::ResultRange getCarries() {
      return getResults().take_front(getInits().size());
    }

    /// The results holding one slice per step, stacked along axis 0.
    ::mlir::ResultRange getOutputs() {
      return getResults().drop_front(getInits().size());
    }

    /// The number of steps, taken from axis 0 of the first input.
    int64_t getTripCount();

    /// The types the body takes: the carries, then one slice of each input,
    /// then the constants.
    ::llvm::SmallVector<::mlir::Type> getBodyArgumentTypes();

    /// `type` without its leading axis: what one step of a stacked input or
    /// output looks like inside the body.
    static ::mlir::RankedTensorType getSliceType(::mlir::RankedTensorType type);
  }];
}

#endif
