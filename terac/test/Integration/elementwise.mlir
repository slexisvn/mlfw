// RUN: tera-runner %s --shared-libs=%mlir_c_runner_utils --entry=arithmetic --check --data='{"inputs": [{"shape": [4], "data": [1, 2, 3, 4]}, {"shape": [4], "data": [4, 3, 2, 1]}], "output": {"shape": [4], "data": [-4, -3, -3, -4]}}'
// RUN: tera-runner %s --shared-libs=%mlir_c_runner_utils --entry=integer_arithmetic --check --data='{"inputs": [{"shape": [4], "data": [10, 20, 30, 40]}, {"shape": [4], "data": [4, 3, 2, 1]}], "output": {"shape": [4], "data": [-4, -6, -15, -40]}}'
// RUN: tera-runner %s --shared-libs=%mlir_c_runner_utils --entry=select_min --check --data='{"inputs": [{"shape": [4], "data": [1, 2, 3, 4]}, {"shape": [4], "data": [4, 3, 2, 1]}], "output": {"shape": [4], "data": [1, 2, 2, 1]}}'
// RUN: tera-runner %s --shared-libs=%mlir_c_runner_utils --entry=exponential --check --data='{"inputs": [{"shape": [3], "data": [0, 1, 2]}], "output": {"shape": [3], "data": [1.0, 2.718281828, 7.389056099]}}'

// The FileCheck tests beside this file prove each op reaches the right linalg
// op. Only running the result proves the body computes the right thing.

// sub undoes add and div undoes mul, so this collapses to -max(a, b):
// a = [1,2,3,4], b = [4,3,2,1] -> max = [4,3,3,4] -> neg = [-4,-3,-3,-4].
func.func @arithmetic(%a: tensor<4xf32>, %b: tensor<4xf32>) -> tensor<4xf32> {
  %0 = tera.add %a, %b : tensor<4xf32>
  %1 = tera.sub %0, %b : tensor<4xf32>
  %2 = tera.mul %1, %b : tensor<4xf32>
  %3 = tera.div %2, %b : tensor<4xf32>
  %4 = tera.maximum %3, %b : tensor<4xf32>
  %5 = tera.neg %4 : tensor<4xf32>
  return %5 : tensor<4xf32>
}

// The same ops on integers, where division truncates towards zero and negation
// has to be synthesised from a subtraction: a = [10,20,30,40], b = [4,3,2,1]
// gives a/b = [2,6,15,40], max with b = [4,6,15,40], negated [-4,-6,-15,-40].
func.func @integer_arithmetic(%a: tensor<4xi32>, %b: tensor<4xi32>) -> tensor<4xi32> {
  %0 = tera.div %a, %b : tensor<4xi32>
  %1 = tera.maximum %0, %b : tensor<4xi32>
  %2 = tera.neg %1 : tensor<4xi32>
  return %2 : tensor<4xi32>
}

// compare + select together are a minimum, and convert truncates it to i32:
// a = [1,2,3,4], b = [4,3,2,1] -> a<b = [1,1,0,0] -> [1,2,2,1].
func.func @select_min(%a: tensor<4xf32>, %b: tensor<4xf32>) -> tensor<4xi32> {
  %0 = tera.compare lt, %a, %b : tensor<4xf32> -> tensor<4xi1>
  %1 = tera.select %0, %a, %b : tensor<4xi1>, tensor<4xf32>
  %2 = tera.convert %1 : tensor<4xf32> -> tensor<4xi32>
  return %2 : tensor<4xi32>
}

func.func @exponential(%a: tensor<3xf32>) -> tensor<3xf32> {
  %0 = tera.exp %a : tensor<3xf32>
  return %0 : tensor<3xf32>
}
