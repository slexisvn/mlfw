//===- Passes.td - Tera dialect passes ---------------------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Passes that rewrite tera into tera. Reverse-mode AD lands here, above the
// lowering in Tera/Conversion, because a closed VJP rule exists for `dot` and
// `reduce` but not for the loop nest they become.
//
//===----------------------------------------------------------------------===//

#ifndef TERA_TRANSFORMS_PASSES
#define TERA_TRANSFORMS_PASSES

include "mlir/Pass/PassBase.td"

def TeraAutodiff : Pass<"tera-autodiff", "::mlir::ModuleOp"> {
  let summary = "Emit the reverse-mode derivative of every differentiable function.";
  let description = [{
    For each `func.func` carrying `tera.differentiable`, adds a function
    computing its vector-Jacobian product and leaves the original alone. Given

    ```mlir
    func.func @f(%x: tensor<4xf32>, %w: tensor<4xf32>) -> tensor<f32>
        attributes {tera.differentiable}
    ```

    the pass adds

    ```mlir
    func.func @f_vjp(%x: tensor<4xf32>, %w: tensor<4xf32>, %seed: tensor<f32>)
        -> (tensor<4xf32>, tensor<4xf32>) attributes {tera.diff_args = array<i64: 0, 1>}
    ```

    The trailing argument is the gradient of the result, so a function whose
    result is not a scalar needs no loss wrapped around it. Results are the
    gradients of the arguments named by `tera.diff_args`, which is every
    argument with a floating-point element type; the pass writes
    `tera.vjp = @f_vjp` back onto the original.

    The body is the original's, cloned so the forward values a rule needs are
    in scope, walked in reverse. Each op is asked for its own contribution
    through `TeraVjpOpInterface`; an op that does not implement it is an error
    rather than a silently missing gradient. Contributions to a value used more
    than once are summed as a tree, so a value with `n` uses costs
    `log n` deep of additions rather than `n`.

    `tera.stop_gradient` reports no gradient for its operand, which is how the
    walk is stopped short of part of the graph.
  }];

  let dependentDialects = [
    "::mlir::func::FuncDialect",
    "::mlir::tera::TeraDialect",
  ];
}

#endif // TERA_TRANSFORMS_PASSES
