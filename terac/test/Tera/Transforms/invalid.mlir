// RUN: tera-opt %s --tera-autodiff --split-input-file -verify-diagnostics

// An op with no rule is an error, not a silently missing gradient. Nothing
// outside the tera dialect implements the interface, so any op that reaches
// the reverse walk from another dialect stops it.
func.func @foreign(%a: tensor<4xf32>) -> tensor<4xf32>
    attributes {tera.differentiable} {
  // expected-error@+1 {{has no derivative: 'arith.negf' does not implement TeraVjpOpInterface}}
  %0 = arith.negf %a : tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----

// An op with no rule that no gradient reaches is left alone: the walk only
// visits what the result depends on.
func.func @foreign_but_dead(%a: tensor<4xf32>) -> tensor<4xf32>
    attributes {tera.differentiable} {
  %0 = arith.negf %a : tensor<4xf32>
  %1 = tera.exp %a : tensor<4xf32>
  return %1 : tensor<4xf32>
}

// -----

// expected-error@+1 {{must return exactly one tensor to be differentiated, not 2}}
func.func @two_results(%a: tensor<4xf32>) -> (tensor<4xf32>, tensor<4xf32>)
    attributes {tera.differentiable} {
  %0 = tera.exp %a : tensor<4xf32>
  return %0, %0 : tensor<4xf32>, tensor<4xf32>
}

// -----

// expected-error@+1 {{carries no gradient to seed the reverse pass with}}
func.func @integer_result(%a: tensor<4xf32>) -> tensor<4xi32>
    attributes {tera.differentiable} {
  %0 = tera.convert %a : tensor<4xf32> -> tensor<4xi32>
  return %0 : tensor<4xi32>
}

// -----

// expected-error@+1 {{has no argument with a floating-point element type}}
func.func @integer_arguments(%a: tensor<4xi32>) -> tensor<4xf32>
    attributes {tera.differentiable} {
  %0 = tera.convert %a : tensor<4xi32> -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----

// expected-error@+1 {{is marked differentiable but has no body}}
func.func private @declared_only(%a: tensor<4xf32>) -> tensor<4xf32>
    attributes {tera.differentiable}

// -----

// expected-error@+1 {{cannot add 'taken_vjp': that name is already taken}}
func.func @taken(%a: tensor<4xf32>) -> tensor<4xf32>
    attributes {tera.differentiable} {
  %0 = tera.exp %a : tensor<4xf32>
  return %0 : tensor<4xf32>
}
func.func @taken_vjp(%a: tensor<4xf32>) -> tensor<4xf32> {
  return %a : tensor<4xf32>
}

// -----

// Control flow is differentiated through `tera.if` and `tera.scan`, which keep
// it inside one block. A branch between blocks is refused rather than
// differentiated as if it were not there.
// expected-error@+1 {{has control flow between blocks, which -tera-autodiff does not handle; use tera.if or tera.scan}}
func.func @branching(%a: tensor<4xf32>) -> tensor<4xf32>
    attributes {tera.differentiable} {
  cf.br ^next
^next:
  %0 = tera.exp %a : tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----

// A function that names the arguments it wants gradients for is taken at its
// word, so a position it does not have is a mistake worth reporting rather
// than an index to clamp.
// expected-error@+1 {{asks for the gradient of argument 3, which it does not take}}
func.func @out_of_range(%a: tensor<4xf32>) -> tensor<4xf32>
    attributes {tera.differentiable, tera.diff_args = array<i64: 3>} {
  %0 = tera.exp %a : tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----

// Same for an argument that is there but carries no gradient: an integer
// tensor has no derivative to return, and returning a zero of it would answer
// a question that was not asked.
// expected-error@+1 {{asks for the gradient of argument 0, which is 'tensor<4xi32>' and carries none}}
func.func @not_a_gradient(%i: tensor<4xi32>, %a: tensor<4xf32>) -> tensor<4xf32>
    attributes {tera.differentiable, tera.diff_args = array<i64: 0>} {
  %0 = tera.exp %a : tensor<4xf32>
  return %0 : tensor<4xf32>
}
