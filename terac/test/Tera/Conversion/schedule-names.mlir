// RUN: tera-opt %s --convert-tera-to-linalg --split-input-file | FileCheck %s

// A schedule addresses the op it schedules by name, and the name comes from
// the tera op rather than from where the lowered op ended up. Two dots and an
// elementwise chain between them: the second dot is `dot.1` whatever is
// written between them, which is the property that lets a tuned schedule be
// cached against a program and still mean the same op the next time.

func.func @named(%x: tensor<8x16xf32>, %w: tensor<16x4xf32>,
                 %v: tensor<4x4xf32>) -> tensor<8x4xf32> {
  %0 = tera.dot %x, %w {lhs_batch = array<i64>, lhs_contracting = array<i64: 1>,
                        rhs_batch = array<i64>, rhs_contracting = array<i64: 0>}
      : (tensor<8x16xf32>, tensor<16x4xf32>) -> tensor<8x4xf32>
  %1 = tera.exp %0 : tensor<8x4xf32>
  %2 = tera.dot %1, %v {lhs_batch = array<i64>, lhs_contracting = array<i64: 1>,
                        rhs_batch = array<i64>, rhs_contracting = array<i64: 0>}
      : (tensor<8x4xf32>, tensor<4x4xf32>) -> tensor<8x4xf32>
  return %2 : tensor<8x4xf32>
}

// The fill each contraction accumulates into is left unnamed: it is the
// destination and not a schedule, and the schedule that does apply folds it in.
// CHECK-LABEL: func.func @named
// CHECK:         linalg.fill
// CHECK-NOT:       tera.schedule
// CHECK:         linalg.generic
// CHECK-SAME:      iterator_types = ["parallel", "parallel", "reduction"]
// CHECK-SAME:      {tera.schedule = "named.dot.0"}
// CHECK:         linalg.map { math.exp }
// CHECK-SAME:      {tera.schedule = "named.exp.0"}
// CHECK:         linalg.fill
// CHECK-NOT:       tera.schedule
// CHECK:         linalg.generic
// CHECK-SAME:      iterator_types = ["parallel", "parallel", "reduction"]
// CHECK-SAME:      {tera.schedule = "named.dot.1"}

// -----

// The count is per function and per mnemonic, so the same program in another
// function gets its own names and an op of another kind does not push a dot's
// number along.

func.func @elsewhere(%x: tensor<8x16xf32>, %w: tensor<16x4xf32>)
    -> tensor<8x4xf32> {
  %0 = tera.dot %x, %w {lhs_batch = array<i64>, lhs_contracting = array<i64: 1>,
                        rhs_batch = array<i64>, rhs_contracting = array<i64: 0>}
      : (tensor<8x16xf32>, tensor<16x4xf32>) -> tensor<8x4xf32>
  return %0 : tensor<8x4xf32>
}

// CHECK-LABEL: func.func @elsewhere
// CHECK:         tera.schedule = "elsewhere.dot.0"

// -----

// One tera op becomes several linalg ops and they all carry the name, which
// says where they came from rather than which of them to schedule. A padded
// average pool is three of them -- the border it reads through, the traversal
// and the divide -- and a script that wants the traversal narrows the match
// the way it would narrow any other.

func.func @several(%x: tensor<1x1x4x4xf32>) -> tensor<1x1x2x2xf32> {
  %0 = tera.pool2d average, %x {kernel_size = array<i64: 2, 2>,
                                strides = array<i64: 2, 2>,
                                padding = array<i64: 1, 0, 1, 0>,
                                count_include_pad = true}
      : tensor<1x1x4x4xf32> -> tensor<1x1x2x2xf32>
  return %0 : tensor<1x1x2x2xf32>
}

// CHECK-LABEL: func.func @several
// CHECK:         linalg.generic
// CHECK-SAME:      {tera.schedule = "several.pool2d.0"}
// CHECK:         linalg.generic
// CHECK-SAME:      iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction"]
// CHECK-SAME:      {tera.schedule = "several.pool2d.0"}
// CHECK:         linalg.generic
// CHECK-SAME:      {tera.schedule = "several.pool2d.0"}
