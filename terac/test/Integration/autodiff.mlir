// RUN: tera-runner %s --shared-libs=%mlir_c_runner_utils --entry=passes_values_through --check --data='{"inputs": [{"shape": [4], "data": [1, 2, 3, 4]}], "output": {"shape": [4], "data": [2, 4, 6, 8]}}'

// A FileCheck test proves the op disappears; only running it proves the value
// it was carrying arrives on the other side.
func.func @passes_values_through(%a: tensor<4xf32>) -> tensor<4xf32> {
  %0 = tera.stop_gradient %a : tensor<4xf32>
  %1 = tera.add %0, %a : tensor<4xf32>
  return %1 : tensor<4xf32>
}
