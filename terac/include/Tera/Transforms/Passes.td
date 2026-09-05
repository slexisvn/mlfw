//===- Passes.td - Tera dialect passes ---------------------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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

def TeraForwardMode : Pass<"tera-forward-mode", "::mlir::ModuleOp"> {
  let summary = "Emit the forward-mode derivative of every differentiable function.";
  let description = [{
    For each `func.func` carrying `tera.differentiable`, adds a function
    computing its result together with the direction that result moves in,
    given a direction for each argument named by `tera.diff_args`. Given

    ```mlir
    func.func @f(%x: tensor<4xf32>, %w: tensor<4xf32>) -> tensor<f32>
        attributes {tera.differentiable}
    ```

    the pass adds

    ```mlir
    func.func @f_jvp(%x: tensor<4xf32>, %w: tensor<4xf32>,
                     %dx: tensor<4xf32>, %dw: tensor<4xf32>)
        -> (tensor<f32>, tensor<f32>)
        attributes {tera.diff_args = array<i64: 0, 1>}
    ```

    and writes `tera.jvp = @f_jvp` back onto the original. Both the result and
    its tangent come back, because computing them together is the only reason
    to run forwards at all: the primal values a rule needs are already there.

    Nothing is walked backwards, so there is no adjoint to accumulate and
    nothing to save between two halves. One pass over the body answers it.

    Almost no op needs a rule of its own here. An op that is linear in its
    operands moves the way it maps, an op linear in each operand separately
    sums one term per operand, and an elementwise op has a diagonal Jacobian,
    which is its own transpose -- so its `buildVjp` already computes the
    forward direction and is reused rather than restated. What is left is
    `tera.reduce` over a maximum, `tera.pool2d` over a maximum,
    `tera.stop_gradient`, and the two region ops.
  }];

  let dependentDialects = [
    "::mlir::func::FuncDialect",
    "::mlir::tera::TeraDialect",
  ];
}

#endif
