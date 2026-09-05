// RUN: tera-opt %s --split-input-file --convert-tera-to-linalg | FileCheck %s
// RUN: tera-runner %s --shared-libs=%mlir_c_runner_utils --entry=dynamic_batch --check --data='{"inputs": [{"shape": [3, 4], "data": [1, 2, 3, 4, 5, 6, 7, 8, -1, -2, -3, -4]}, {"shape": [4, 2], "data": [1, 0, 0, 1, 1, 1, 0, 0]}], "output": {"shape": [3, 2], "data": [4, 5, 12, 13, 0, 0]}}'
// The same module at a different batch. One compile, two shapes: that is the
// whole point of a dynamic extent, and running it twice is the only thing that
// says the extent is read rather than baked in.
// RUN: tera-runner %s --shared-libs=%mlir_c_runner_utils --entry=dynamic_batch --check --data='{"inputs": [{"shape": [1, 4], "data": [1, 2, 3, 4]}, {"shape": [4, 2], "data": [1, 0, 0, 1, 1, 1, 0, 0]}], "output": {"shape": [1, 2], "data": [4, 5]}}'
// RUN: %if cuda %{ tera-runner %s --target=cuda --shared-libs=%mlir_cuda_runtime --shared-libs=%mlir_c_runner_utils --entry=dynamic_batch --check --data='{"inputs": [{"shape": [3, 4], "data": [1, 2, 3, 4, 5, 6, 7, 8, -1, -2, -3, -4]}, {"shape": [4, 2], "data": [1, 0, 0, 1, 1, 1, 0, 0]}], "output": {"shape": [3, 2], "data": [4, 5, 12, 13, 0, 0]}}' %}

// A batch nobody has decided yet. This is what mlfw's tracer emits for
// `dynamicShapes: [new Set([0]), null]`, printed by its own printer.
//
// Everything here is shaped like an operand except one op. An elementwise
// result has its operands' extents, and a `dot`'s free axes have its operands',
// so both find a dynamic extent with `tensor.dim` on something they were
// already given. `tera.broadcast_in_dim` decides a shape rather than inheriting
// one -- a scalar zero has nothing to say how tall the activation it is being
// broadcast onto is -- and that is what the `sizes` clause carries, one extent
// per `?` in the result type.
//
// `tera.dim` is where an extent comes from, and the only op in the dialect that
// turns a shape into a value. It folds to a constant over a static extent, so a
// module written with it everywhere lowers exactly like one written without.

// CHECK-LABEL: func @dynamic_batch
func.func @dynamic_batch(%x: tensor<?x4xf32>, %w: tensor<4x2xf32>) -> tensor<?x2xf32> {
  // The contraction's destination is `?x2`, and the `?` is the lhs's own.
  // CHECK: %[[M:.*]] = tensor.dim
  // CHECK: tensor.empty(%[[M]])
  // CHECK: linalg.generic
  %0 = tera.dot %x, %w {lhs_batch = array<i64>, lhs_contracting = array<i64: 1>,
                        rhs_batch = array<i64>, rhs_contracting = array<i64: 0>}
      : (tensor<?x4xf32>, tensor<4x2xf32>) -> tensor<?x2xf32>

  %1 = tera.constant dense<0.0> : tensor<f32>

  // The extent, as a value. Nothing else in the dialect produces one.
  // CHECK: %[[DIM:.*]] = tensor.dim
  // CHECK: tensor.from_elements %{{.*}}
  %2 = tera.dim %0 {dimension = 0 : i64} : tensor<?x2xf32> -> tensor<i64>

  // And read back out of the rank-0 tensor it travelled in, to size the one
  // destination in the program that no operand's type could have sized.
  // CHECK: %[[E:.*]] = tensor.extract
  // CHECK: %[[I:.*]] = arith.index_cast %[[E]]
  // CHECK: tensor.empty(%[[I]])
  %3 = tera.broadcast_in_dim %1 sizes(%2) {broadcast_dimensions = array<i64>}
      : tensor<f32> -> tensor<?x2xf32>

  // CHECK: linalg.map
  %4 = tera.maximum %0, %3 : tensor<?x2xf32>
  return %4 : tensor<?x2xf32>
}

// -----

// `tera.concat` is the one op whose result extent is not any operand's: along
// the axis it joins it is the sum of them, and that sum is also the offset each
// band is written at. Every other axis is an input's own.

// CHECK-LABEL: func @concat_dynamic
func.func @concat_dynamic(%a: tensor<?x2xf32>, %b: tensor<?x2xf32>) -> tensor<?x2xf32> {
  // CHECK: %[[A:.*]] = tensor.dim %arg0
  // CHECK: %[[B:.*]] = tensor.dim %arg1
  // CHECK: %[[N:.*]] = arith.addi %[[A]], %[[B]]
  // CHECK: tensor.empty(%[[N]])
  // CHECK: tensor.insert_slice
  // CHECK: tensor.insert_slice
  %0 = tera.concat %a, %b {dimension = 0 : i64}
      : tensor<?x2xf32>, tensor<?x2xf32> -> tensor<?x2xf32>
  return %0 : tensor<?x2xf32>
}

// -----

// A scan whose step axis is static and whose batch is not. The carries are the
// operands' own shape, so the loop needs telling nothing about them; a stacked
// output is a destination decided inside a body that is isolated from what
// surrounds it, and `sizes` is what it is told.

// CHECK-LABEL: func @scan_dynamic_batch
func.func @scan_dynamic_batch(%init: tensor<?x2xf32>, %xs: tensor<3x?x2xf32>)
    -> (tensor<?x2xf32>, tensor<3x?x2xf32>) {
  %n = tera.dim %init {dimension = 0 : i64} : tensor<?x2xf32> -> tensor<i64>
  // One `sizes` operand per `?` across the results, so the carry has one too
  // even though the loop takes its shape from the value it was handed. The
  // stacked output is the one that needs it.
  // CHECK: tensor.extract
  // CHECK: arith.index_cast
  // CHECK: tensor.empty(%{{.*}}) : tensor<3x?x2xf32>
  // CHECK: scf.for
  // CHECK: tensor.extract_slice
  // CHECK: tensor.insert_slice
  %carry, %ys = tera.scan init(%init : tensor<?x2xf32>) xs(%xs : tensor<3x?x2xf32>)
      sizes(%n, %n) -> (tensor<?x2xf32>, tensor<3x?x2xf32>) {
  ^bb0(%acc: tensor<?x2xf32>, %x: tensor<?x2xf32>):
    %0 = tera.add %acc, %x : tensor<?x2xf32>
    tera.yield %0, %0 : tensor<?x2xf32>, tensor<?x2xf32>
  }
  return %carry, %ys : tensor<?x2xf32>, tensor<3x?x2xf32>
}

// -----

// `tera.reshape` decides its result's shape rather than inheriting it, so it is
// told the extents the same way a broadcast is. Folding adjacent axes together
// is a view, and a view is also the only form the memref lowering can take a
// dynamic extent through.

// CHECK-LABEL: func @reshape_dynamic
func.func @reshape_dynamic(%x: tensor<?x2x4xf32>) -> tensor<?x8xf32> {
  %n = tera.dim %x {dimension = 0 : i64} : tensor<?x2x4xf32> -> tensor<i64>
  // CHECK: tensor.collapse_shape %arg0 {{\[}}[0], [1, 2]] : tensor<?x2x4xf32> into tensor<?x8xf32>
  %0 = tera.reshape %x sizes(%n) : tensor<?x2x4xf32> -> tensor<?x8xf32>
  return %0 : tensor<?x8xf32>
}

// -----

// `tera.iota` holds no data, so a shape it was told is a shape it can fill --
// which is what separates it from `tera.constant`, whose type comes from an
// attribute and so carries a count.

// CHECK-LABEL: func @iota_dynamic
func.func @iota_dynamic(%n: tensor<i64>) -> tensor<?x3xf32> {
  // CHECK: %[[E:.*]] = tensor.extract
  // CHECK: %[[I:.*]] = arith.index_cast %[[E]]
  // CHECK: tensor.empty(%[[I]])
  // CHECK: linalg.index
  %0 = tera.iota sizes(%n) {iota_dimension = 1 : i64} : tensor<?x3xf32>
  return %0 : tensor<?x3xf32>
}

// -----

// A branch needs no telling at all: its results are its bodies', and both
// bodies were handed the op's own inputs.

// CHECK-LABEL: func @if_dynamic
func.func @if_dynamic(%p: tensor<i1>, %x: tensor<?x2xf32>) -> tensor<?x2xf32> {
  // CHECK: scf.if
  %0 = tera.if %p, %x : (tensor<i1>, tensor<?x2xf32>) -> tensor<?x2xf32> {
  ^bb0(%a: tensor<?x2xf32>):
    %1 = tera.mul %a, %a : tensor<?x2xf32>
    tera.yield %1 : tensor<?x2xf32>
  } else {
  ^bb0(%a: tensor<?x2xf32>):
    %2 = tera.neg %a : tensor<?x2xf32>
    tera.yield %2 : tensor<?x2xf32>
  }
  return %0 : tensor<?x2xf32>
}

// -----

// A `?` on an axis `tera.reverse` does NOT name is not its problem: the axes
// it reverses are all static, and the dynamic one is carried through by the
// destination the conversion builds from the operand.

// CHECK-LABEL: func @reverse_static_axis
// CHECK: linalg.generic
func.func @reverse_static_axis(%x: tensor<?x4xf32>) -> tensor<?x4xf32> {
  %0 = tera.reverse %x {dimensions = array<i64: 1>}
      : tensor<?x4xf32> -> tensor<?x4xf32>
  return %0 : tensor<?x4xf32>
}

// -----

// `sqrt`, `rsqrt` and `tanh` lower through the same map as `exp` and `neg`,
// which take a dynamic extent; they were refused one only because the trait
// saying so had not been written on them.

// CHECK-LABEL: func @dynamic_unary
// CHECK: linalg.map { math.sqrt }
// CHECK: linalg.map { math.rsqrt }
// CHECK: linalg.map { math.tanh }
func.func @dynamic_unary(%x: tensor<?xf32>) -> tensor<?xf32> {
  %0 = tera.sqrt %x : tensor<?xf32>
  %1 = tera.rsqrt %0 : tensor<?xf32>
  %2 = tera.tanh %1 : tensor<?xf32>
  return %2 : tensor<?xf32>
}
