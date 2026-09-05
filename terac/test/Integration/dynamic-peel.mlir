// RUN: tera-runner %s --entry=f --shared-libs=%mlir_c_runner_utils --data=%S/dynamic-peel-whole.json --check
// RUN: tera-runner %s --entry=f --shared-libs=%mlir_c_runner_utils --data=%S/dynamic-peel-short.json --check
// RUN: tera-runner %s --entry=summed --shared-libs=%mlir_c_runner_utils --data=%S/dynamic-peel-reduced.json --check
// RUN: %if cuda %{ tera-runner %s --entry=f --target=cuda --shared-libs=%mlir_cuda_runtime --shared-libs=%mlir_c_runner_utils --data=%S/dynamic-peel-short.json --check %}

// A loop over a `?` is cut to a vector and then split, so nearly every element
// goes through the vectorized loop and the few left over go through a scalar
// one. Both halves have to compute the same thing, and the seam between them
// is where they would stop doing so.
//
// 32 elements is two whole tiles and no remainder; 37 is two whole tiles and
// five elements that only the peeled-off iteration reaches. The reduction runs
// over 19 rows, which is neither.
//
// Every value is a dyadic rational and every operation is exact in f32, so
// `--check` compares bit for bit rather than within a tolerance that could
// hide a seam off by one element.

func.func @f(%x: tensor<?xf32>, %y: tensor<?xf32>) -> tensor<?xf32> {
  %0 = tera.mul %x, %y : tensor<?xf32>
  %1 = tera.add %0, %x : tensor<?xf32>
  %2 = tera.maximum %1, %y : tensor<?xf32>
  return %2 : tensor<?xf32>
}

// The reduction loop is cut too, and to one element rather than to a vector,
// so what peeling has to leave alone is the loop carrying the accumulator.

func.func @summed(%x: tensor<?x4xf32>) -> tensor<4xf32> {
  %0 = tera.reduce sum, %x {dimensions = array<i64: 0>}
      : tensor<?x4xf32> -> tensor<4xf32>
  return %0 : tensor<4xf32>
}
