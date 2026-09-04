// RUN: tera-gradcheck %s --shared-libs=%mlir_c_runner_utils

// The same recurrence four ways: keeping every carry, and keeping one in two,
// one in three and one in six. Checkpointing trades memory for arithmetic and
// changes nothing else, so all four are run on the same inputs and all four
// have to land on the same gradients — which is what agreeing with the same
// finite differences to ten digits means.

func.func @plain(%h0: tensor<2xf64>, %w: tensor<2xf64>, %xs: tensor<6x2xf64>)
    -> tensor<6x2xf64> attributes {tera.differentiable} {
  %carry, %ys = tera.scan init(%h0 : tensor<2xf64>) xs(%xs : tensor<6x2xf64>)
      consts(%w : tensor<2xf64>) -> (tensor<2xf64>, tensor<6x2xf64>) {
  ^bb0(%h: tensor<2xf64>, %x: tensor<2xf64>, %weight: tensor<2xf64>):
    %0 = tera.mul %h, %weight : tensor<2xf64>
    %1 = tera.add %0, %x : tensor<2xf64>
    %2 = tera.mul %1, %1 : tensor<2xf64>
    %3 = tera.neg %2 : tensor<2xf64>
    %4 = tera.exp %3 : tensor<2xf64>
    tera.yield %4, %4 : tensor<2xf64>, tensor<2xf64>
  }
  return %ys : tensor<6x2xf64>
}

func.func @every_two(%h0: tensor<2xf64>, %w: tensor<2xf64>, %xs: tensor<6x2xf64>)
    -> tensor<6x2xf64> attributes {tera.differentiable} {
  %carry, %ys = tera.scan init(%h0 : tensor<2xf64>) xs(%xs : tensor<6x2xf64>)
      consts(%w : tensor<2xf64>) {checkpoint = 2 : i64}
      -> (tensor<2xf64>, tensor<6x2xf64>) {
  ^bb0(%h: tensor<2xf64>, %x: tensor<2xf64>, %weight: tensor<2xf64>):
    %0 = tera.mul %h, %weight : tensor<2xf64>
    %1 = tera.add %0, %x : tensor<2xf64>
    %2 = tera.mul %1, %1 : tensor<2xf64>
    %3 = tera.neg %2 : tensor<2xf64>
    %4 = tera.exp %3 : tensor<2xf64>
    tera.yield %4, %4 : tensor<2xf64>, tensor<2xf64>
  }
  return %ys : tensor<6x2xf64>
}

func.func @every_three(%h0: tensor<2xf64>, %w: tensor<2xf64>, %xs: tensor<6x2xf64>)
    -> tensor<6x2xf64> attributes {tera.differentiable} {
  %carry, %ys = tera.scan init(%h0 : tensor<2xf64>) xs(%xs : tensor<6x2xf64>)
      consts(%w : tensor<2xf64>) {checkpoint = 3 : i64}
      -> (tensor<2xf64>, tensor<6x2xf64>) {
  ^bb0(%h: tensor<2xf64>, %x: tensor<2xf64>, %weight: tensor<2xf64>):
    %0 = tera.mul %h, %weight : tensor<2xf64>
    %1 = tera.add %0, %x : tensor<2xf64>
    %2 = tera.mul %1, %1 : tensor<2xf64>
    %3 = tera.neg %2 : tensor<2xf64>
    %4 = tera.exp %3 : tensor<2xf64>
    tera.yield %4, %4 : tensor<2xf64>, tensor<2xf64>
  }
  return %ys : tensor<6x2xf64>
}

// One checkpoint for the whole sequence: nothing is kept but the entry carry,
// and every step is recomputed from it.
func.func @every_six(%h0: tensor<2xf64>, %w: tensor<2xf64>, %xs: tensor<6x2xf64>)
    -> tensor<6x2xf64> attributes {tera.differentiable} {
  %carry, %ys = tera.scan init(%h0 : tensor<2xf64>) xs(%xs : tensor<6x2xf64>)
      consts(%w : tensor<2xf64>) {checkpoint = 6 : i64}
      -> (tensor<2xf64>, tensor<6x2xf64>) {
  ^bb0(%h: tensor<2xf64>, %x: tensor<2xf64>, %weight: tensor<2xf64>):
    %0 = tera.mul %h, %weight : tensor<2xf64>
    %1 = tera.add %0, %x : tensor<2xf64>
    %2 = tera.mul %1, %1 : tensor<2xf64>
    %3 = tera.neg %2 : tensor<2xf64>
    %4 = tera.exp %3 : tensor<2xf64>
    tera.yield %4, %4 : tensor<2xf64>, tensor<2xf64>
  }
  return %ys : tensor<6x2xf64>
}
