// RUN: tera-runner %s --shared-libs=%mlir_c_runner_utils --entry=running_total --check --data='{"inputs": [{"shape": [], "data": [0]}, {"shape": [4], "data": [1, 2, 3, 4]}], "output": {"shape": [4], "data": [1, 3, 6, 10]}}'
// RUN: tera-runner %s --shared-libs=%mlir_c_runner_utils --entry=running_total_backwards --check --data='{"inputs": [{"shape": [], "data": [0]}, {"shape": [4], "data": [1, 2, 3, 4]}], "output": {"shape": [4], "data": [10, 9, 7, 4]}}'
// RUN: tera-runner %s --shared-libs=%mlir_c_runner_utils --entry=final_carry --check --data='{"inputs": [{"shape": [], "data": [1]}, {"shape": [4], "data": [1, 2, 3, 4]}], "output": {"shape": [], "data": [24]}}'
// RUN: tera-runner %s --shared-libs=%mlir_c_runner_utils --entry=matrix_steps --check --data='{"inputs": [{"shape": [2], "data": [1, 1]}, {"shape": [3, 2], "data": [1, 2, 3, 4, 5, 6]}], "output": {"shape": [3, 2], "data": [2, 3, 5, 7, 10, 13]}}'
// RUN: tera-runner %s --shared-libs=%mlir_c_runner_utils --entry=branch_on_sign --check --data='{"inputs": [{"shape": [4], "data": [1, 2, 3, 4]}], "output": {"shape": [4], "data": [1, 4, 9, 16]}}'
// RUN: tera-runner %s --shared-libs=%mlir_c_runner_utils --entry=branch_on_sign --check --data='{"inputs": [{"shape": [4], "data": [-1, -2, -3, -4]}], "output": {"shape": [4], "data": [1, 2, 3, 4]}}'

// A FileCheck test proves the shape of the loop. Only running it proves the
// carry threads through in the right order, that `reverse` visits the steps
// backwards without moving the data, and that a branch picks a side.

// Running sums of [1,2,3,4] from zero: [1,3,6,10].
func.func @running_total(%init: tensor<f32>, %xs: tensor<4xf32>) -> tensor<4xf32> {
  %carry, %ys = tera.scan init(%init : tensor<f32>) xs(%xs : tensor<4xf32>)
      -> (tensor<f32>, tensor<4xf32>) {
  ^bb0(%acc: tensor<f32>, %x: tensor<f32>):
    %0 = tera.add %acc, %x : tensor<f32>
    tera.yield %0, %0 : tensor<f32>, tensor<f32>
  }
  return %ys : tensor<4xf32>
}

// The same body run from the last step to the first. Output `t` still lands at
// index `t`, so this is the suffix sum: [10,9,7,4]. A `reverse` that flipped
// the data instead of the visiting order would answer [4,7,9,10].
func.func @running_total_backwards(%init: tensor<f32>, %xs: tensor<4xf32>) -> tensor<4xf32> {
  %carry, %ys = tera.scan reverse init(%init : tensor<f32>) xs(%xs : tensor<4xf32>)
      -> (tensor<f32>, tensor<4xf32>) {
  ^bb0(%acc: tensor<f32>, %x: tensor<f32>):
    %0 = tera.add %acc, %x : tensor<f32>
    tera.yield %0, %0 : tensor<f32>, tensor<f32>
  }
  return %ys : tensor<4xf32>
}

// A scan with no stacked output at all: 1*1*2*3*4 = 24.
func.func @final_carry(%init: tensor<f32>, %xs: tensor<4xf32>) -> tensor<f32> {
  %carry = tera.scan init(%init : tensor<f32>) xs(%xs : tensor<4xf32>)
      -> (tensor<f32>) {
  ^bb0(%acc: tensor<f32>, %x: tensor<f32>):
    %0 = tera.mul %acc, %x : tensor<f32>
    tera.yield %0 : tensor<f32>
  }
  return %carry : tensor<f32>
}

// A carry with a shape, so the slicing has a non-trivial inner extent:
// [1,1] + [1,2] = [2,3], + [3,4] = [5,7], + [5,6] = [10,13].
func.func @matrix_steps(%init: tensor<2xf32>, %xs: tensor<3x2xf32>) -> tensor<3x2xf32> {
  %carry, %ys = tera.scan init(%init : tensor<2xf32>) xs(%xs : tensor<3x2xf32>)
      -> (tensor<2xf32>, tensor<3x2xf32>) {
  ^bb0(%acc: tensor<2xf32>, %x: tensor<2xf32>):
    %0 = tera.add %acc, %x : tensor<2xf32>
    tera.yield %0, %0 : tensor<2xf32>, tensor<2xf32>
  }
  return %ys : tensor<3x2xf32>
}

// The condition is computed rather than passed in, which is the shape a real
// branch has. Both sides are reached by the two runs above.
func.func @branch_on_sign(%x: tensor<4xf32>) -> tensor<4xf32> {
  %0 = tera.constant dense<0.000000e+00> : tensor<f32>
  %1 = tera.reduce sum, %x {dimensions = array<i64: 0>}
      : tensor<4xf32> -> tensor<f32>
  %2 = tera.compare gt, %1, %0 : tensor<f32> -> tensor<i1>
  %3 = tera.if %2, %x : (tensor<i1>, tensor<4xf32>) -> tensor<4xf32> {
  ^bb0(%a: tensor<4xf32>):
    %4 = tera.mul %a, %a : tensor<4xf32>
    tera.yield %4 : tensor<4xf32>
  } else {
  ^bb0(%a: tensor<4xf32>):
    %4 = tera.neg %a : tensor<4xf32>
    tera.yield %4 : tensor<4xf32>
  }
  return %3 : tensor<4xf32>
}
