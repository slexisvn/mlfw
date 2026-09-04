// RUN: tera-runner %s --entry=rnn --benchmark=1 --warmup=1 --shared-libs=%mlir_c_runner_utils | FileCheck %s
// RUN: tera-opt %s --tera-autodiff | tera-runner - --entry=rnn_vjp --benchmark=1 --warmup=1 --shared-libs=%mlir_c_runner_utils | FileCheck %s --check-prefix=VJP

// A recurrence long enough to be worth measuring: 64 steps over a 256-wide
// state, with a 256 x 256 weight read every step. The derivative of this is
// where the reverse pass does the most work per source op — one scan to
// recover the carries and one to run the body backwards — so it is the model
// that says whether the derivative is paying for anything it does not use.

// CHECK: rnn: 1 runs
// VJP: rnn_vjp: 1 runs

func.func @rnn(%h0: tensor<256xf32>, %w: tensor<256x256xf32>,
               %xs: tensor<64x256xf32>) -> tensor<64x256xf32>
    attributes {tera.differentiable} {
  %carry, %ys = tera.scan init(%h0 : tensor<256xf32>)
      xs(%xs : tensor<64x256xf32>) consts(%w : tensor<256x256xf32>)
      -> (tensor<256xf32>, tensor<64x256xf32>) {
  ^bb0(%h: tensor<256xf32>, %x: tensor<256xf32>, %weight: tensor<256x256xf32>):
    %0 = tera.dot %h, %weight {lhs_batch = array<i64>,
                               lhs_contracting = array<i64: 0>,
                               rhs_batch = array<i64>,
                               rhs_contracting = array<i64: 0>}
        : (tensor<256xf32>, tensor<256x256xf32>) -> tensor<256xf32>
    %1 = tera.add %0, %x : tensor<256xf32>
    %2 = tera.mul %1, %1 : tensor<256xf32>
    %3 = tera.neg %2 : tensor<256xf32>
    %4 = tera.exp %3 : tensor<256xf32>
    tera.yield %4, %4 : tensor<256xf32>, tensor<256xf32>
  }
  return %ys : tensor<64x256xf32>
}
