// RUN: tera-runner %s --shared-libs=%mlir_c_runner_utils --entry=counting --check --data='{"inputs": [], "output": {"shape": [2, 3], "data": [0, 1, 2, 0, 1, 2]}}'
// RUN: tera-runner %s --shared-libs=%mlir_c_runner_utils --entry=broadcast_new_axis --check --data='{"inputs": [{"shape": [3], "data": [1, 2, 3]}], "output": {"shape": [2, 3], "data": [1, 2, 3, 1, 2, 3]}}'
// RUN: tera-runner %s --shared-libs=%mlir_c_runner_utils --entry=broadcast_stretch --check --data='{"inputs": [{"shape": [2, 1], "data": [1, 2]}], "output": {"shape": [2, 3], "data": [1, 1, 1, 2, 2, 2]}}'
// RUN: tera-runner %s --shared-libs=%mlir_c_runner_utils --entry=transpose --check --data='{"inputs": [{"shape": [2, 3], "data": [1, 2, 3, 4, 5, 6]}], "output": {"shape": [3, 2], "data": [1, 4, 2, 5, 3, 6]}}'
// RUN: tera-runner %s --shared-libs=%mlir_c_runner_utils --entry=transpose_then_reshape --check --data='{"inputs": [{"shape": [2, 3], "data": [1, 2, 3, 4, 5, 6]}], "output": {"shape": [6], "data": [1, 4, 2, 5, 3, 6]}}'
// RUN: tera-runner %s --shared-libs=%mlir_c_runner_utils --entry=reshape_to_scalar --check --data='{"inputs": [{"shape": [1, 1], "data": [7]}], "output": {"shape": [], "data": [7]}}'
// RUN: tera-runner %s --shared-libs=%mlir_c_runner_utils --entry=strided_slice --check --data='{"inputs": [{"shape": [8], "data": [0, 1, 2, 3, 4, 5, 6, 7]}], "output": {"shape": [3], "data": [1, 3, 5]}}'
// RUN: tera-runner %s --shared-libs=%mlir_c_runner_utils --entry=concat --check --data='{"inputs": [{"shape": [2, 2], "data": [1, 2, 3, 4]}, {"shape": [1, 2], "data": [5, 6]}], "output": {"shape": [3, 2], "data": [1, 2, 3, 4, 5, 6]}}'
// RUN: tera-runner %s --shared-libs=%mlir_c_runner_utils --entry=concat_inner --check --data='{"inputs": [{"shape": [2, 2], "data": [1, 2, 3, 4]}, {"shape": [2, 1], "data": [5, 6]}], "output": {"shape": [2, 3], "data": [1, 2, 5, 3, 4, 6]}}'

func.func @counting() -> tensor<2x3xf32> {
  %0 = tera.iota {iota_dimension = 1 : i64} : tensor<2x3xf32>
  return %0 : tensor<2x3xf32>
}

func.func @broadcast_new_axis(%a: tensor<3xf32>) -> tensor<2x3xf32> {
  %0 = tera.broadcast_in_dim %a {broadcast_dimensions = array<i64: 1>}
      : tensor<3xf32> -> tensor<2x3xf32>
  return %0 : tensor<2x3xf32>
}

// The stretched axis is the one that would be read out of bounds if its
// indexing map were the loop index instead of the constant zero.
func.func @broadcast_stretch(%a: tensor<2x1xf32>) -> tensor<2x3xf32> {
  %0 = tera.broadcast_in_dim %a {broadcast_dimensions = array<i64: 0, 1>}
      : tensor<2x1xf32> -> tensor<2x3xf32>
  return %0 : tensor<2x3xf32>
}

func.func @transpose(%a: tensor<2x3xf32>) -> tensor<3x2xf32> {
  %0 = tera.transpose %a {permutation = array<i64: 1, 0>}
      : tensor<2x3xf32> -> tensor<3x2xf32>
  return %0 : tensor<3x2xf32>
}

// Reshaping after a transpose is the case that separates a real data movement
// from a relabelling: the flat order must be the transposed one.
func.func @transpose_then_reshape(%a: tensor<2x3xf32>) -> tensor<6xf32> {
  %0 = tera.transpose %a {permutation = array<i64: 1, 0>}
      : tensor<2x3xf32> -> tensor<3x2xf32>
  %1 = tera.reshape %0 : tensor<3x2xf32> -> tensor<6xf32>
  return %1 : tensor<6xf32>
}

func.func @reshape_to_scalar(%a: tensor<1x1xf32>) -> tensor<f32> {
  %0 = tera.reshape %a : tensor<1x1xf32> -> tensor<f32>
  return %0 : tensor<f32>
}

func.func @strided_slice(%a: tensor<8xf32>) -> tensor<3xf32> {
  %0 = tera.slice %a {start_indices = array<i64: 1>,
                      limit_indices = array<i64: 7>,
                      strides = array<i64: 2>}
      : tensor<8xf32> -> tensor<3xf32>
  return %0 : tensor<3xf32>
}

func.func @concat(%a: tensor<2x2xf32>, %b: tensor<1x2xf32>) -> tensor<3x2xf32> {
  %0 = tera.concat %a, %b {dimension = 0 : i64}
      : tensor<2x2xf32>, tensor<1x2xf32> -> tensor<3x2xf32>
  return %0 : tensor<3x2xf32>
}

// Concatenating along the inner axis interleaves the two inputs in the flat
// result, so an offset applied to the wrong axis shows up here and not above.
func.func @concat_inner(%a: tensor<2x2xf32>, %b: tensor<2x1xf32>) -> tensor<2x3xf32> {
  %0 = tera.concat %a, %b {dimension = 1 : i64}
      : tensor<2x2xf32>, tensor<2x1xf32> -> tensor<2x3xf32>
  return %0 : tensor<2x3xf32>
}
