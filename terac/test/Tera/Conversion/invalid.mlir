// RUN: tera-opt %s --split-input-file --verify-diagnostics --convert-tera-to-linalg

// A destination is materialised from the extents the pattern can reach. Most
// ops are shaped like an operand and reach them through `tensor.dim`; the ones
// that decide a shape rather than inheriting one are handed the extents they
// decided in a `sizes` clause. What is left below cannot do either, and each is
// refused for its own reason rather than the idea being refused.

// `tera.constant` cannot carry a dynamic shape at all, and not for want of a
// clause: its type comes from its attribute, and an attribute holding a tensor
// holds a count. So this is refused by the op's own type rule and never reaches
// the lowering. The dialect spells a splat at a shape nobody has decided as a
// rank-0 constant and a `broadcast_in_dim`, which is what mlfw's tracer emits.
func.func @constant_holds_a_count() -> tensor<?x3xf32> {
  // expected-error @+1 {{all of {value, result} have same type}}
  %0 = "tera.constant"() <{value = dense<0.000000e+00> : tensor<1x3xf32>}>
      : () -> tensor<?x3xf32>
  return %0 : tensor<?x3xf32>
}

// -----

// `tera.slice` computes its result extents from static bounds, so a `?` there
// is a range its attributes cannot name. Its own shape inference says so
// before the lowering sees it.
func.func @slice_cannot_name_a_dynamic_range(%a: tensor<?x4xf32>) -> tensor<?x2xf32> {
  // expected-error @+2 {{are incompatible with return type}}
  // expected-error @+1 {{failed to infer returned types}}
  %0 = tera.slice %a {start_indices = array<i64: 0, 0>,
                      limit_indices = array<i64: 3, 2>,
                      strides = array<i64: 1, 1>}
      : tensor<?x4xf32> -> tensor<?x2xf32>
  return %0 : tensor<?x2xf32>
}

// -----

// A `tera.scan` over a dynamic *step* axis is refused where one over a dynamic
// anything-else is not: every other extent is a destination's width, which the
// `sizes` clause carries, and this one is a trip count the loop bound would
// have to read at run time.
func.func @scan_over_a_dynamic_step_axis(%init: tensor<f32>, %xs: tensor<?xf32>)
    -> tensor<f32> {
  // expected-error @+1 {{cannot be lowered to linalg with a dynamic step axis}}
  %carry = tera.scan init(%init : tensor<f32>) xs(%xs : tensor<?xf32>)
      -> (tensor<f32>) {
  ^bb0(%acc: tensor<f32>, %x: tensor<f32>):
    %0 = tera.add %acc, %x : tensor<f32>
    tera.yield %0 : tensor<f32>
  }
  return %carry : tensor<f32>
}

// -----

// What a window reads outside its input is the identity of what the window
// does, and a maximum has none that is not also an answer. A window made only
// of padding would come out holding it.
func.func @pool_max_padded(%x: tensor<1x1x5x5xf32>) -> tensor<1x1x3x3xf32> {
  // expected-error @+1 {{cannot be lowered with padding: a maximum has no value to read outside the input that a window made only of padding would not then answer with}}
  %0 = tera.pool2d max, %x {kernel_size = array<i64: 2, 2>,
                            strides = array<i64: 2, 2>,
                            padding = array<i64: 1, 0, 1, 0>}
      : tensor<1x1x5x5xf32> -> tensor<1x1x3x3xf32>
  return %0 : tensor<1x1x3x3xf32>
}

// -----

// An average that leaves the padding out of its count divides each window by
// how much of it fell inside, and that is a different number per window.
func.func @pool_average_excludes_pad(%x: tensor<1x1x5x5xf32>) -> tensor<1x1x3x3xf32> {
  // expected-error @+1 {{cannot be lowered without counting the padding}}
  %0 = tera.pool2d average, %x {kernel_size = array<i64: 2, 2>,
                                strides = array<i64: 2, 2>,
                                padding = array<i64: 1, 0, 1, 0>}
      : tensor<1x1x5x5xf32> -> tensor<1x1x3x3xf32>
  return %0 : tensor<1x1x3x3xf32>
}

// -----

// `ceil_mode` takes one more window than fits, and what it reads past the end
// is padding the op was never told about.
func.func @pool_hangs_over(%x: tensor<1x1x5x5xf32>) -> tensor<1x1x3x3xf32> {
  // expected-error @+1 {{cannot be lowered with a window that hangs over axis 0}}
  %0 = tera.pool2d average, %x {kernel_size = array<i64: 2, 2>,
                                strides = array<i64: 2, 2>,
                                padding = array<i64: 0, 0, 0, 0>,
                                ceil_mode = true,
                                count_include_pad = true}
      : tensor<1x1x5x5xf32> -> tensor<1x1x3x3xf32>
  return %0 : tensor<1x1x3x3xf32>
}
