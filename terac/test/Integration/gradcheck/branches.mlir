// RUN: tera-gradcheck %s --shared-libs=%mlir_c_runner_utils

// Each side of a branch is checked as the side that runs: a constant condition
// keeps finite differences from ever straddling the switch, so the numbers here
// are about the rule and not about which way the coin landed.

func.func @then_side(%x: tensor<4xf64>, %y: tensor<4xf64>) -> tensor<4xf64>
    attributes {tera.differentiable} {
  %0 = tera.constant dense<true> : tensor<i1>
  %1 = tera.if %0, %x, %y : (tensor<i1>, tensor<4xf64>, tensor<4xf64>)
      -> tensor<4xf64> {
  ^bb0(%a: tensor<4xf64>, %b: tensor<4xf64>):
    %2 = tera.mul %a, %b : tensor<4xf64>
    %3 = tera.exp %2 : tensor<4xf64>
    tera.yield %3 : tensor<4xf64>
  } else {
  ^bb0(%a: tensor<4xf64>, %b: tensor<4xf64>):
    %2 = tera.sub %a, %b : tensor<4xf64>
    tera.yield %2 : tensor<4xf64>
  }
  return %1 : tensor<4xf64>
}

// The mirror. `%y` reaches the result on this side and not the other, so the
// two runs together also check that each side reports its own gradients rather
// than the sum of both.
func.func @else_side(%x: tensor<4xf64>, %y: tensor<4xf64>) -> tensor<4xf64>
    attributes {tera.differentiable} {
  %0 = tera.constant dense<false> : tensor<i1>
  %1 = tera.if %0, %x, %y : (tensor<i1>, tensor<4xf64>, tensor<4xf64>)
      -> tensor<4xf64> {
  ^bb0(%a: tensor<4xf64>, %b: tensor<4xf64>):
    %2 = tera.mul %a, %a : tensor<4xf64>
    tera.yield %2 : tensor<4xf64>
  } else {
  ^bb0(%a: tensor<4xf64>, %b: tensor<4xf64>):
    %2 = tera.div %a, %b : tensor<4xf64>
    %3 = tera.neg %2 : tensor<4xf64>
    tera.yield %3 : tensor<4xf64>
  }
  return %1 : tensor<4xf64>
}

// A condition computed from the inputs, which is the shape a real branch has.
// The derivative of a comparison is nothing at all, and that is the right
// answer: moving an input by a hair does not change which side runs, so finite
// differences agree.
func.func @computed_condition(%x: tensor<4xf64>) -> tensor<4xf64>
    attributes {tera.differentiable} {
  %0 = tera.constant dense<0.000000e+00> : tensor<f64>
  %1 = tera.reduce sum, %x {dimensions = array<i64: 0>}
      : tensor<4xf64> -> tensor<f64>
  %2 = tera.compare gt, %1, %0 : tensor<f64> -> tensor<i1>
  %3 = tera.if %2, %x : (tensor<i1>, tensor<4xf64>) -> tensor<4xf64> {
  ^bb0(%a: tensor<4xf64>):
    %4 = tera.mul %a, %a : tensor<4xf64>
    tera.yield %4 : tensor<4xf64>
  } else {
  ^bb0(%a: tensor<4xf64>):
    %4 = tera.neg %a : tensor<4xf64>
    tera.yield %4 : tensor<4xf64>
  }
  return %3 : tensor<4xf64>
}

// A branch inside a loop, so the engine recurses: the scan's rule differentiates
// its body, and the body's rule differentiates two more. The gate rides along as
// a scan constant, and being a boolean it must pick up no gradient accumulator
// at all — an integer one would be arithmetic on something that is not a
// gradient.
func.func @gated_recurrence(%h0: tensor<2xf64>, %xs: tensor<4x2xf64>)
    -> tensor<4x2xf64> attributes {tera.differentiable} {
  %0 = tera.constant dense<true> : tensor<i1>
  %carry, %ys = tera.scan init(%h0 : tensor<2xf64>) xs(%xs : tensor<4x2xf64>)
      consts(%0 : tensor<i1>) -> (tensor<2xf64>, tensor<4x2xf64>) {
  ^bb0(%h: tensor<2xf64>, %x: tensor<2xf64>, %gate: tensor<i1>):
    %1 = tera.if %gate, %h, %x : (tensor<i1>, tensor<2xf64>, tensor<2xf64>)
        -> tensor<2xf64> {
    ^bb0(%a: tensor<2xf64>, %b: tensor<2xf64>):
      %2 = tera.mul %a, %b : tensor<2xf64>
      tera.yield %2 : tensor<2xf64>
    } else {
    ^bb0(%a: tensor<2xf64>, %b: tensor<2xf64>):
      %2 = tera.add %a, %b : tensor<2xf64>
      tera.yield %2 : tensor<2xf64>
    }
    tera.yield %1, %1 : tensor<2xf64>, tensor<2xf64>
  }
  return %ys : tensor<4x2xf64>
}
