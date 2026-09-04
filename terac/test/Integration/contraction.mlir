// RUN: tera-runner %s --shared-libs=%mlir_c_runner_utils --entry=matmul --check --data='{"inputs": [{"shape": [2, 3], "data": [1, 2, 3, 4, 5, 6]}, {"shape": [3, 2], "data": [1, 0, 0, 1, 1, 1]}], "output": {"shape": [2, 2], "data": [4, 5, 10, 11]}}'
// RUN: tera-runner %s --shared-libs=%mlir_c_runner_utils --entry=batched_matmul --check --data='{"inputs": [{"shape": [2, 2, 3], "data": [1, 2, 3, 4, 5, 6, 1, 1, 1, 2, 2, 2]}, {"shape": [2, 3, 2], "data": [1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1]}], "output": {"shape": [2, 2, 2], "data": [4, 5, 10, 11, 2, 2, 4, 4]}}'
// RUN: tera-runner %s --shared-libs=%mlir_c_runner_utils --entry=transposed_matmul --check --data='{"inputs": [{"shape": [3, 2], "data": [1, 4, 2, 5, 3, 6]}, {"shape": [3, 2], "data": [1, 0, 0, 1, 1, 1]}], "output": {"shape": [2, 2], "data": [4, 5, 10, 11]}}'
// RUN: tera-runner %s --shared-libs=%mlir_c_runner_utils --entry=inner_product --check --data='{"inputs": [{"shape": [4], "data": [1, 2, 3, 4]}, {"shape": [4], "data": [10, 20, 30, 40]}], "output": {"shape": [], "data": [300]}}'
// RUN: tera-runner %s --shared-libs=%mlir_c_runner_utils --entry=outer_product --check --data='{"inputs": [{"shape": [2], "data": [1, 2]}, {"shape": [3], "data": [10, 20, 30]}], "output": {"shape": [2, 3], "data": [10, 20, 30, 20, 40, 60]}}'
// RUN: tera-runner %s --shared-libs=%mlir_c_runner_utils --entry=row_sum --check --data='{"inputs": [{"shape": [2, 3], "data": [1, 2, 3, 4, 5, 6]}], "output": {"shape": [2], "data": [6, 15]}}'
// RUN: tera-runner %s --shared-libs=%mlir_c_runner_utils --entry=row_product --check --data='{"inputs": [{"shape": [2, 3], "data": [1, 2, 3, 4, 5, 6]}], "output": {"shape": [2], "data": [6, 120]}}'
// RUN: tera-runner %s --shared-libs=%mlir_c_runner_utils --entry=maximum_of_negatives --check --data='{"inputs": [{"shape": [4], "data": [-5, -2, -9, -1]}], "output": {"shape": [], "data": [-1]}}'
// RUN: tera-runner %s --shared-libs=%mlir_c_runner_utils --entry=minimum_of_positives --check --data='{"inputs": [{"shape": [4], "data": [5, 2, 9, 1]}], "output": {"shape": [], "data": [1]}}'
// RUN: tera-runner %s --shared-libs=%mlir_c_runner_utils --entry=unsorted_reduction --check --data='{"inputs": [{"shape": [2, 2, 2], "data": [1, 2, 3, 4, 5, 6, 7, 8]}], "output": {"shape": [2], "data": [14, 22]}}'

// [[1,2,3],[4,5,6]] . [[1,0],[0,1],[1,1]] = [[4,5],[10,11]].
func.func @matmul(%a: tensor<2x3xf32>, %b: tensor<3x2xf32>) -> tensor<2x2xf32> {
  %0 = tera.dot %a, %b {lhs_batch = array<i64>,
                        lhs_contracting = array<i64: 1>,
                        rhs_batch = array<i64>,
                        rhs_contracting = array<i64: 0>}
      : (tensor<2x3xf32>, tensor<3x2xf32>) -> tensor<2x2xf32>
  return %0 : tensor<2x2xf32>
}

// The second batch is [[1,1,1],[2,2,2]] against the same matrix, giving
// [[2,2],[4,4]]. A batch axis mapped to the wrong iterator would mix them.
func.func @batched_matmul(%a: tensor<2x2x3xf32>, %b: tensor<2x3x2xf32>) -> tensor<2x2x2xf32> {
  %0 = tera.dot %a, %b {lhs_batch = array<i64: 0>,
                        lhs_contracting = array<i64: 2>,
                        rhs_batch = array<i64: 0>,
                        rhs_contracting = array<i64: 1>}
      : (tensor<2x2x3xf32>, tensor<2x3x2xf32>) -> tensor<2x2x2xf32>
  return %0 : tensor<2x2x2xf32>
}

// Contracting the lhs on axis 0 instead of axis 1 is the same product with the
// lhs stored transposed, and must give the same answer as @matmul.
func.func @transposed_matmul(%a: tensor<3x2xf32>, %b: tensor<3x2xf32>) -> tensor<2x2xf32> {
  %0 = tera.dot %a, %b {lhs_batch = array<i64>,
                        lhs_contracting = array<i64: 0>,
                        rhs_batch = array<i64>,
                        rhs_contracting = array<i64: 0>}
      : (tensor<3x2xf32>, tensor<3x2xf32>) -> tensor<2x2xf32>
  return %0 : tensor<2x2xf32>
}

// Everything contracted, nothing free: 10 + 40 + 90 + 160 = 300.
func.func @inner_product(%a: tensor<4xf32>, %b: tensor<4xf32>) -> tensor<f32> {
  %0 = tera.dot %a, %b {lhs_batch = array<i64>,
                        lhs_contracting = array<i64: 0>,
                        rhs_batch = array<i64>,
                        rhs_contracting = array<i64: 0>}
      : (tensor<4xf32>, tensor<4xf32>) -> tensor<f32>
  return %0 : tensor<f32>
}

// Nothing contracted: the reduction loop is empty and the result is the full
// outer product, lhs axes before rhs axes.
func.func @outer_product(%a: tensor<2xf32>, %b: tensor<3xf32>) -> tensor<2x3xf32> {
  %0 = tera.dot %a, %b {lhs_batch = array<i64>,
                        lhs_contracting = array<i64>,
                        rhs_batch = array<i64>,
                        rhs_contracting = array<i64>}
      : (tensor<2xf32>, tensor<3xf32>) -> tensor<2x3xf32>
  return %0 : tensor<2x3xf32>
}

func.func @row_sum(%a: tensor<2x3xf32>) -> tensor<2xf32> {
  %0 = tera.reduce sum, %a {dimensions = array<i64: 1>}
      : tensor<2x3xf32> -> tensor<2xf32>
  return %0 : tensor<2xf32>
}

func.func @row_product(%a: tensor<2x3xf32>) -> tensor<2xf32> {
  %0 = tera.reduce product, %a {dimensions = array<i64: 1>}
      : tensor<2x3xf32> -> tensor<2xf32>
  return %0 : tensor<2xf32>
}

// All-negative input: an accumulator seeded with zero rather than -inf would
// answer 0 here.
func.func @maximum_of_negatives(%a: tensor<4xf32>) -> tensor<f32> {
  %0 = tera.reduce maximum, %a {dimensions = array<i64: 0>}
      : tensor<4xf32> -> tensor<f32>
  return %0 : tensor<f32>
}

// The mirror of the above for the minimum identity.
func.func @minimum_of_positives(%a: tensor<4xf32>) -> tensor<f32> {
  %0 = tera.reduce minimum, %a {dimensions = array<i64: 0>}
      : tensor<4xf32> -> tensor<f32>
  return %0 : tensor<f32>
}

// Reducing axes 2 and 0, written out of order. The surviving axis is 1, so the
// answer is [1+2+5+6, 3+4+7+8] = [14, 22].
func.func @unsorted_reduction(%a: tensor<2x2x2xf32>) -> tensor<2xf32> {
  %0 = tera.reduce sum, %a {dimensions = array<i64: 2, 0>}
      : tensor<2x2x2xf32> -> tensor<2xf32>
  return %0 : tensor<2xf32>
}
