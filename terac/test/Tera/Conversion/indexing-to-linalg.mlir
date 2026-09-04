// RUN: tera-opt %s --convert-tera-to-linalg --split-input-file | FileCheck %s

// A gather writes each result element once, so the loop nest is the result and
// linalg.generic supplies it. What no indexing map can say is where the read
// comes from, so the body says it: linalg.index recovers the position being
// written, tensor.extract reads the coordinate out of the index tensor, and a
// second tensor.extract reads the element that coordinate names. The generic
// takes no inputs at all -- both tensors are read from inside the body.
// CHECK-LABEL: func @gather_rows
// CHECK: %[[DEST:.*]] = tensor.empty() : tensor<2x3x4xf32>
// CHECK: linalg.generic
// CHECK-SAME: iterator_types = ["parallel", "parallel", "parallel"]
// CHECK-SAME: outs(%[[DEST]]
// CHECK: %[[B0:.*]] = linalg.index 0
// CHECK: %[[B1:.*]] = linalg.index 1
// CHECK: %[[OFF:.*]] = linalg.index 2
// CHECK: %[[AT:.*]] = tensor.extract %arg1[%[[B0]], %[[B1]]]
// CHECK: %[[ROW:.*]] = arith.index_cast %[[AT]]
// CHECK: %[[VALUE:.*]] = tensor.extract %arg0[%[[ROW]], %[[OFF]]]
// CHECK: linalg.yield %[[VALUE]]
func.func @gather_rows(%table: tensor<10x4xf32>, %ids: tensor<2x3xi32>) -> tensor<2x3x4xf32> {
  %0 = tera.gather %table, %ids {offset_dims = array<i64: 2>,
                                 collapsed_slice_dims = array<i64: 0>,
                                 start_index_map = array<i64: 0>,
                                 slice_sizes = array<i64: 1, 4>,
                                 index_vector_dim = 2 : i64}
      : (tensor<10x4xf32>, tensor<2x3xi32>) -> tensor<2x3x4xf32>
  return %0 : tensor<2x3x4xf32>
}

// -----

// Two coordinates rather than one, so the index tensor is read twice and each
// read pins the index vector axis to the coordinate it is fetching.
// CHECK-LABEL: func @gather_elements
// CHECK: %[[I:.*]] = linalg.index 0
// CHECK: %[[J:.*]] = linalg.index 1
// CHECK: %[[C0:.*]] = arith.constant 0 : index
// CHECK: %[[FIRST:.*]] = tensor.extract %arg1[%[[I]], %[[J]], %[[C0]]]
// CHECK: %[[ROW:.*]] = arith.index_cast %[[FIRST]]
// CHECK: %[[C1:.*]] = arith.constant 1 : index
// CHECK: %[[SECOND:.*]] = tensor.extract %arg1[%[[I]], %[[J]], %[[C1]]]
// CHECK: %[[COLUMN:.*]] = arith.index_cast %[[SECOND]]
// An axis with no window index to offset from takes the coordinate as it is.
// CHECK-NOT: arith.addi
// CHECK: tensor.extract %arg0[%[[ROW]], %[[COLUMN]]]
func.func @gather_elements(%x: tensor<3x4xf32>, %at: tensor<3x3x2xi32>) -> tensor<3x3xf32> {
  %0 = tera.gather %x, %at {offset_dims = array<i64>,
                            collapsed_slice_dims = array<i64: 0, 1>,
                            start_index_map = array<i64: 0, 1>,
                            slice_sizes = array<i64: 1, 1>,
                            index_vector_dim = 2 : i64}
      : (tensor<3x4xf32>, tensor<3x3x2xi32>) -> tensor<3x3xf32>
  return %0 : tensor<3x3xf32>
}

// -----

// A scatter is not one write per element: two updates may land on the same
// place and both have to count, so the writes happen in an order. That order is
// a loop nest over the updates carrying the tensor from step to step, starting
// from a copy of the operand.
//
// The copy is written down rather than left to bufferization, which would be
// entitled to conclude that the operand is dead after the scatter and write
// into it -- true inside the function and false across calls, because the
// buffer is the caller's. It is an alloc_tensor rather than a linalg.copy
// because a linalg.copy is a parallel op and becomes a kernel on the GPU
// target, which would leave this host-side nest reading a buffer the launches
// around it hold on the device.
// CHECK-LABEL: func @scatter_rows
// CHECK-NOT: linalg.copy
// CHECK: %[[COPY:.*]] = bufferization.alloc_tensor() copy(%arg0)
// CHECK: %[[OUTER:.*]] = scf.for %[[B0:.*]] = %{{.*}} iter_args(%[[A0:.*]] = %[[COPY]])
// CHECK: %[[MIDDLE:.*]] = scf.for %[[B1:.*]] = %{{.*}} iter_args(%[[A1:.*]] = %[[A0]])
// CHECK: %[[INNER:.*]] = scf.for %[[W:.*]] = %{{.*}} iter_args(%[[ACC:.*]] = %[[A1]])
// CHECK: %[[AT:.*]] = tensor.extract %arg1[%[[B0]], %[[B1]]]
// CHECK: %[[ROW:.*]] = arith.index_cast %[[AT]]
// CHECK: %[[STANDING:.*]] = tensor.extract %[[ACC]][%[[ROW]], %[[W]]]
// CHECK: %[[UPDATE:.*]] = tensor.extract %arg2[%[[B0]], %[[B1]], %[[W]]]
// CHECK: %[[SUM:.*]] = arith.addf %[[STANDING]], %[[UPDATE]]
// CHECK: %[[NEXT:.*]] = tensor.insert %[[SUM]] into %[[ACC]][%[[ROW]], %[[W]]]
// CHECK: scf.yield %[[NEXT]]
// CHECK: scf.yield %[[INNER]]
// CHECK: scf.yield %[[MIDDLE]]
// CHECK: return %[[OUTER]]
func.func @scatter_rows(%table: tensor<10x4xf32>, %ids: tensor<2x3xi32>,
                        %updates: tensor<2x3x4xf32>) -> tensor<10x4xf32> {
  %0 = tera.scatter %table, %ids, %updates
      {update_window_dims = array<i64: 2>,
       inserted_window_dims = array<i64: 0>,
       scatter_dims_to_operand_dims = array<i64: 0>,
       index_vector_dim = 2 : i64}
      : (tensor<10x4xf32>, tensor<2x3xi32>, tensor<2x3x4xf32>) -> tensor<10x4xf32>
  return %0 : tensor<10x4xf32>
}

// -----

// An integer scatter accumulates with the integer add. The combiner is the op,
// not an attribute, so this is the only thing the element type changes.
// CHECK-LABEL: func @scatter_integers
// CHECK: arith.addi
// CHECK-NOT: arith.addf
func.func @scatter_integers(%x: tensor<3x4xi32>, %at: tensor<3x3x2xi32>,
                            %updates: tensor<3x3xi32>) -> tensor<3x4xi32> {
  %0 = tera.scatter %x, %at, %updates
      {update_window_dims = array<i64>,
       inserted_window_dims = array<i64: 0, 1>,
       scatter_dims_to_operand_dims = array<i64: 0, 1>,
       index_vector_dim = 2 : i64}
      : (tensor<3x4xi32>, tensor<3x3x2xi32>, tensor<3x3xi32>) -> tensor<3x4xi32>
  return %0 : tensor<3x4xi32>
}

// -----

// Neither op decides a shape, so neither carries a sizes clause and both take
// a dynamic extent from a tensor they were given. A gather reads its batch
// extents off the index tensor -- its slice extents are attributes and cannot
// be dynamic -- and a scatter reads its loop bounds off the updates and hands
// back the operand it was given.
// CHECK-LABEL: func @dynamic_batch
// CHECK: %[[N:.*]] = tensor.dim %arg1, %{{.*}} : tensor<?xi32>
// CHECK: tensor.empty(%[[N]]) : tensor<?x4xf32>
// CHECK: linalg.generic
func.func @dynamic_batch(%table: tensor<10x4xf32>, %ids: tensor<?xi32>) -> tensor<?x4xf32> {
  %0 = tera.gather %table, %ids {offset_dims = array<i64: 1>,
                                 collapsed_slice_dims = array<i64: 0>,
                                 start_index_map = array<i64: 0>,
                                 slice_sizes = array<i64: 1, 4>,
                                 index_vector_dim = 1 : i64}
      : (tensor<10x4xf32>, tensor<?xi32>) -> tensor<?x4xf32>
  return %0 : tensor<?x4xf32>
}

// -----

// CHECK-LABEL: func @dynamic_updates
// CHECK: %[[N:.*]] = tensor.dim %arg2, %{{.*}} : tensor<?x4xf32>
// CHECK: scf.for %{{.*}} to %[[N]] {{.*}} iter_args(
func.func @dynamic_updates(%table: tensor<10x4xf32>, %ids: tensor<?xi32>,
                           %updates: tensor<?x4xf32>) -> tensor<10x4xf32> {
  %0 = tera.scatter %table, %ids, %updates
      {update_window_dims = array<i64: 1>,
       inserted_window_dims = array<i64: 0>,
       scatter_dims_to_operand_dims = array<i64: 0>,
       index_vector_dim = 1 : i64}
      : (tensor<10x4xf32>, tensor<?xi32>, tensor<?x4xf32>) -> tensor<10x4xf32>
  return %0 : tensor<10x4xf32>
}
