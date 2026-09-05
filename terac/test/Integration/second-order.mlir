// RUN: tera-opt %s --tera-autodiff --inline --symbol-dce -o %t.first.mlir
// RUN: sed 's/@f_vjp\(.*\)attributes {/@f_vjp\1attributes {tera.differentiable, /' %t.first.mlir > %t.marked.mlir
// RUN: tera-opt %t.marked.mlir --tera-autodiff -o %t.second.mlir
// RUN: tera-runner %t.second.mlir --entry=f_vjp_vjp --shared-libs=%mlir_c_runner_utils --check --data='{"inputs": [{"shape": [3], "data": [1, 2, 3]}, {"shape": [], "data": [2]}, {"shape": [3], "data": [1, 1, 1]}], "output": {"shape": [3], "data": [12, 24, 36]}}'
// RUN: tera-runner %t.second.mlir --entry=f_vjp_vjp --shared-libs=%mlir_c_runner_utils --check --data='{"inputs": [{"shape": [3], "data": [1, 2, 3]}, {"shape": [], "data": [1]}, {"shape": [3], "data": [0.5, 2, -1]}], "output": {"shape": [3], "data": [3, 24, -18]}}'
// RUN: %if cuda %{ tera-runner %t.second.mlir --entry=f_vjp_vjp --target=cuda --shared-libs=%mlir_cuda_runtime --shared-libs=%mlir_c_runner_utils --check --data='{"inputs": [{"shape": [3], "data": [1, 2, 3]}, {"shape": [], "data": [2]}, {"shape": [3], "data": [1, 1, 1]}], "output": {"shape": [3], "data": [12, 24, 36]}}' %}

// The derivative of a derivative, taken by running the pass twice.
//
// Nothing special happens the second time. `-tera-autodiff` writes `@f_vjp` as
// an ordinary function, `-inline` leaves it holding tera ops and no call, and
// marking that one differentiable is the same request as marking the first.
// What makes it possible is that a derivative takes a seed per result and
// gives a gradient per argument, so it is the same shape of thing the pass
// already knows how to walk -- which is why the only change this needed was
// letting a function with more than one result be differentiated at all.
//
// `@f_vjp_vjp(x, s, v)` is the Hessian of `f` contracted with `s` and `v`.
// Here `f(x) = sum(x^3)`, so `@f_vjp(x, s) = 3x^2 s` and the second derivative
// is `6x s v` -- exact in f32 at these values, so the checks are the answer
// rather than a tolerance around it.

func.func @f(%x: tensor<3xf32>) -> tensor<f32> attributes {tera.differentiable} {
  %0 = tera.mul %x, %x : tensor<3xf32>
  %1 = tera.mul %0, %x : tensor<3xf32>
  %2 = tera.reduce sum, %1 {dimensions = array<i64: 0>}
      : tensor<3xf32> -> tensor<f32>
  return %2 : tensor<f32>
}
