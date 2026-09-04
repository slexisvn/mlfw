// RUN: tera-gradcheck %s --shared-libs=%mlir_c_runner_utils

// The same three shapes test/bench runs, shrunk until finite differences are
// affordable and with every extent left prime.
//
// Everything below the tera level splits a loop nest into tiles, and a tile
// size that happens to divide the trip count hides the remainder it would
// otherwise have to generate. Every model already here is 2, 3, 4, 5 or 6
// wide, so all of them divide by something; 11, 13, 17 and 23 divide by
// nothing, and a nest that gets its epilogue wrong produces a wrong number
// here rather than at a size no gradcheck can afford to run.

// A perceptron whose every matmul extent is prime.
func.func @perceptron(%x: tensor<13x17xf64>, %w1: tensor<17x23xf64>,
                      %b1: tensor<23xf64>, %w2: tensor<23x7xf64>,
                      %b2: tensor<7xf64>) -> tensor<13x7xf64>
    attributes {tera.differentiable} {
  %zero = tera.constant dense<0.000000e+00> : tensor<13x23xf64>
  %0 = tera.dot %x, %w1 {lhs_batch = array<i64>, lhs_contracting = array<i64: 1>,
                         rhs_batch = array<i64>, rhs_contracting = array<i64: 0>}
      : (tensor<13x17xf64>, tensor<17x23xf64>) -> tensor<13x23xf64>
  %1 = tera.broadcast_in_dim %b1 {broadcast_dimensions = array<i64: 1>}
      : tensor<23xf64> -> tensor<13x23xf64>
  %2 = tera.add %0, %1 : tensor<13x23xf64>
  %3 = tera.maximum %2, %zero : tensor<13x23xf64>
  %4 = tera.dot %3, %w2 {lhs_batch = array<i64>, lhs_contracting = array<i64: 1>,
                         rhs_batch = array<i64>, rhs_contracting = array<i64: 0>}
      : (tensor<13x23xf64>, tensor<23x7xf64>) -> tensor<13x7xf64>
  %5 = tera.broadcast_in_dim %b2 {broadcast_dimensions = array<i64: 1>}
      : tensor<7xf64> -> tensor<13x7xf64>
  %6 = tera.add %4, %5 : tensor<13x7xf64>
  return %6 : tensor<13x7xf64>
}

// Causal attention over 11 positions of 5 channels, 3 batches. The reduction
// runs over 11, so a vectorized softmax has three lanes of remainder whatever
// width it picks.
func.func @masked_attention(%q: tensor<3x11x5xf64>, %k: tensor<3x11x5xf64>,
                            %v: tensor<3x11x5xf64>) -> tensor<3x11x5xf64>
    attributes {tera.differentiable} {
  %0 = tera.dot %q, %k {lhs_batch = array<i64: 0>, lhs_contracting = array<i64: 2>,
                        rhs_batch = array<i64: 0>, rhs_contracting = array<i64: 2>}
      : (tensor<3x11x5xf64>, tensor<3x11x5xf64>) -> tensor<3x11x11xf64>
  %1 = tera.constant dense<0.44721359549995793> : tensor<3x11x11xf64>
  %2 = tera.mul %0, %1 : tensor<3x11x11xf64>

  %3 = tera.iota {iota_dimension = 1} : tensor<3x11x11xi64>
  %4 = tera.iota {iota_dimension = 2} : tensor<3x11x11xi64>
  %5 = tera.compare ge, %3, %4 : tensor<3x11x11xi64> -> tensor<3x11x11xi1>
  %6 = tera.constant dense<-1.000000e+30> : tensor<3x11x11xf64>
  %7 = tera.select %5, %2, %6 : tensor<3x11x11xi1>, tensor<3x11x11xf64>

  %8 = tera.reduce maximum, %7 {dimensions = array<i64: 2>}
      : tensor<3x11x11xf64> -> tensor<3x11xf64>
  %9 = tera.stop_gradient %8 : tensor<3x11xf64>
  %10 = tera.broadcast_in_dim %9 {broadcast_dimensions = array<i64: 0, 1>}
      : tensor<3x11xf64> -> tensor<3x11x11xf64>
  %11 = tera.sub %7, %10 : tensor<3x11x11xf64>
  %12 = tera.exp %11 : tensor<3x11x11xf64>
  %13 = tera.reduce sum, %12 {dimensions = array<i64: 2>}
      : tensor<3x11x11xf64> -> tensor<3x11xf64>
  %14 = tera.broadcast_in_dim %13 {broadcast_dimensions = array<i64: 0, 1>}
      : tensor<3x11xf64> -> tensor<3x11x11xf64>
  %15 = tera.div %12, %14 : tensor<3x11x11xf64>

  %16 = tera.dot %15, %v {lhs_batch = array<i64: 0>, lhs_contracting = array<i64: 2>,
                          rhs_batch = array<i64: 0>, rhs_contracting = array<i64: 1>}
      : (tensor<3x11x11xf64>, tensor<3x11x5xf64>) -> tensor<3x11x5xf64>
  return %16 : tensor<3x11x5xf64>
}

// Seven steps of a five-wide recurrence: a trip count no tiling divides, over
// a body whose own contraction is prime on both sides.
func.func @recurrence(%h0: tensor<5xf64>, %w: tensor<5x5xf64>,
                      %xs: tensor<7x5xf64>) -> tensor<7x5xf64>
    attributes {tera.differentiable} {
  %carry, %ys = tera.scan init(%h0 : tensor<5xf64>) xs(%xs : tensor<7x5xf64>)
      consts(%w : tensor<5x5xf64>) -> (tensor<5xf64>, tensor<7x5xf64>) {
  ^bb0(%h: tensor<5xf64>, %x: tensor<5xf64>, %weight: tensor<5x5xf64>):
    %0 = tera.dot %h, %weight {lhs_batch = array<i64>,
                               lhs_contracting = array<i64: 0>,
                               rhs_batch = array<i64>,
                               rhs_contracting = array<i64: 0>}
        : (tensor<5xf64>, tensor<5x5xf64>) -> tensor<5xf64>
    %1 = tera.add %0, %x : tensor<5xf64>
    %2 = tera.mul %1, %1 : tensor<5xf64>
    %3 = tera.neg %2 : tensor<5xf64>
    %4 = tera.exp %3 : tensor<5xf64>
    tera.yield %4, %4 : tensor<5xf64>, tensor<5xf64>
  }
  return %ys : tensor<7x5xf64>
}
