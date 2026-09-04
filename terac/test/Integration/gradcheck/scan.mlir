// RUN: tera-gradcheck %s --shared-libs=%mlir_c_runner_utils

// A recurrence, which is what `tera.scan` exists for. The weight is a constant
// rather than a carry or an input: it is read every step and sliced by none of
// them, so its gradient is the sum over steps, and the reverse scan has to
// accumulate it in a carry rather than stack it. Getting that wrong shows up
// here and nowhere else.
//
// The step is a Gaussian rather than something that grows: five steps of an
// expanding recurrence reach 1e34, where a finite difference has no significant
// digits left to compare and the check stops meaning anything.
func.func @recurrence(%h0: tensor<3xf64>, %w: tensor<3x3xf64>,
                      %xs: tensor<5x3xf64>) -> tensor<5x3xf64>
    attributes {tera.differentiable} {
  %carry, %ys = tera.scan init(%h0 : tensor<3xf64>) xs(%xs : tensor<5x3xf64>)
      consts(%w : tensor<3x3xf64>) -> (tensor<3xf64>, tensor<5x3xf64>) {
  ^bb0(%h: tensor<3xf64>, %x: tensor<3xf64>, %weight: tensor<3x3xf64>):
    %0 = tera.dot %h, %weight {lhs_batch = array<i64>,
                               lhs_contracting = array<i64: 0>,
                               rhs_batch = array<i64>,
                               rhs_contracting = array<i64: 0>}
        : (tensor<3xf64>, tensor<3x3xf64>) -> tensor<3xf64>
    %1 = tera.add %0, %x : tensor<3xf64>
    %2 = tera.mul %1, %1 : tensor<3xf64>
    %3 = tera.neg %2 : tensor<3xf64>
    %4 = tera.exp %3 : tensor<3xf64>
    tera.yield %4, %4 : tensor<3xf64>, tensor<3xf64>
  }
  return %ys : tensor<5x3xf64>
}

// The gradient of the final carry alone, with nothing stacked. The reverse
// scan is then seeded entirely from the carry adjoint and every stacked output
// adjoint is absent, which is the other half of the seeding.
func.func @final_state(%h0: tensor<2xf64>, %xs: tensor<4x2xf64>) -> tensor<2xf64>
    attributes {tera.differentiable} {
  %carry = tera.scan init(%h0 : tensor<2xf64>) xs(%xs : tensor<4x2xf64>)
      -> (tensor<2xf64>) {
  ^bb0(%h: tensor<2xf64>, %x: tensor<2xf64>):
    %0 = tera.mul %h, %x : tensor<2xf64>
    %1 = tera.add %0, %x : tensor<2xf64>
    tera.yield %1 : tensor<2xf64>
  }
  return %carry : tensor<2xf64>
}

// A scan already running backwards. Its derivative has to run forwards, and if
// the two directions were not opposites the carry would be threaded the wrong
// way and the gradient would come out as if the sequence were reversed.
func.func @backwards(%h0: tensor<f64>, %xs: tensor<6xf64>) -> tensor<6xf64>
    attributes {tera.differentiable} {
  %carry, %ys = tera.scan reverse init(%h0 : tensor<f64>) xs(%xs : tensor<6xf64>)
      -> (tensor<f64>, tensor<6xf64>) {
  ^bb0(%h: tensor<f64>, %x: tensor<f64>):
    %0 = tera.mul %h, %x : tensor<f64>
    %1 = tera.exp %0 : tensor<f64>
    tera.yield %1, %1 : tensor<f64>, tensor<f64>
  }
  return %ys : tensor<6xf64>
}
