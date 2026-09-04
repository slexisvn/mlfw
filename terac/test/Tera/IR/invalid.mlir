// RUN: tera-opt %s --split-input-file --verify-diagnostics

// Ops that infer their result report twice on failure: the diagnostic raised
// inside inferReturnTypes, then the interface's own "failed to infer returned
// types". Both are expected.

func.func @broadcast_wrong_arity(%a: tensor<2x3xf32>) -> tensor<2x3xf32> {
  // expected-error @+1 {{expects one broadcast dimension per operand axis: 2 expected, 1 given}}
  %0 = tera.broadcast_in_dim %a {broadcast_dimensions = array<i64: 0>}
      : tensor<2x3xf32> -> tensor<2x3xf32>
  return %0 : tensor<2x3xf32>
}

// -----

func.func @broadcast_not_increasing(%a: tensor<2x3xf32>) -> tensor<3x2xf32> {
  // expected-error @+1 {{broadcast dimensions must be strictly increasing}}
  %0 = tera.broadcast_in_dim %a {broadcast_dimensions = array<i64: 1, 0>}
      : tensor<2x3xf32> -> tensor<3x2xf32>
  return %0 : tensor<3x2xf32>
}

// -----

func.func @broadcast_bad_extent(%a: tensor<3xf32>) -> tensor<2x4xf32> {
  // expected-error @+1 {{cannot broadcast extent 3 at operand axis 0 to 4}}
  %0 = tera.broadcast_in_dim %a {broadcast_dimensions = array<i64: 1>}
      : tensor<3xf32> -> tensor<2x4xf32>
  return %0 : tensor<2x4xf32>
}

// -----

func.func @transpose_repeated_axis(%a: tensor<2x3xf32>) -> tensor<2x2xf32> {
  // expected-error @+2 {{permutation axis 0 is repeated}}
  // expected-error @+1 {{failed to infer returned types}}
  %0 = tera.transpose %a {permutation = array<i64: 0, 0>}
      : tensor<2x3xf32> -> tensor<2x2xf32>
  return %0 : tensor<2x2xf32>
}

// -----

func.func @transpose_wrong_length(%a: tensor<2x3xf32>) -> tensor<2x3xf32> {
  // expected-error @+2 {{expects a permutation of length 2, got 3}}
  // expected-error @+1 {{failed to infer returned types}}
  %0 = tera.transpose %a {permutation = array<i64: 0, 1, 2>}
      : tensor<2x3xf32> -> tensor<2x3xf32>
  return %0 : tensor<2x3xf32>
}

// -----

// The declared result contradicts the inferred one. This case exists only
// because the result type is written in the assembly; inference is the source
// of truth and the interface checks the two agree.
func.func @transpose_bad_result(%a: tensor<2x3xf32>) -> tensor<2x3xf32> {
  // expected-error @+2 {{are incompatible with return type(s) of operation}}
  // expected-error @+1 {{failed to infer returned types}}
  %0 = tera.transpose %a {permutation = array<i64: 1, 0>}
      : tensor<2x3xf32> -> tensor<2x3xf32>
  return %0 : tensor<2x3xf32>
}

// -----

func.func @reshape_element_count(%a: tensor<2x3xf32>) -> tensor<5xf32> {
  // expected-error @+1 {{changes the element count from 6 to 5}}
  %0 = tera.reshape %a : tensor<2x3xf32> -> tensor<5xf32>
  return %0 : tensor<5xf32>
}

// -----

func.func @slice_zero_stride(%a: tensor<8xf32>) -> tensor<4xf32> {
  // expected-error @+2 {{stride 0 at axis 0 must be positive}}
  // expected-error @+1 {{failed to infer returned types}}
  %0 = tera.slice %a {start_indices = array<i64: 0>,
                      limit_indices = array<i64: 4>,
                      strides = array<i64: 0>}
      : tensor<8xf32> -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----

func.func @slice_out_of_bounds(%a: tensor<8xf32>) -> tensor<4xf32> {
  // expected-error @+2 {{limit 12 at axis 0 exceeds the operand extent 8}}
  // expected-error @+1 {{failed to infer returned types}}
  %0 = tera.slice %a {start_indices = array<i64: 8>,
                      limit_indices = array<i64: 12>,
                      strides = array<i64: 1>}
      : tensor<8xf32> -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----

func.func @concat_bad_sum(%a: tensor<2x3xf32>, %b: tensor<4x3xf32>) -> tensor<5x3xf32> {
  // expected-error @+2 {{are incompatible with return type(s) of operation}}
  // expected-error @+1 {{failed to infer returned types}}
  %0 = tera.concat %a, %b {dimension = 0 : i64}
      : tensor<2x3xf32>, tensor<4x3xf32> -> tensor<5x3xf32>
  return %0 : tensor<5x3xf32>
}

// -----

func.func @concat_disagreeing_axis(%a: tensor<2x3xf32>, %b: tensor<4x5xf32>) -> tensor<6x3xf32> {
  // expected-error @+2 {{input 1 disagrees with input 0 at axis 1}}
  // expected-error @+1 {{failed to infer returned types}}
  %0 = tera.concat %a, %b {dimension = 0 : i64}
      : tensor<2x3xf32>, tensor<4x5xf32> -> tensor<6x3xf32>
  return %0 : tensor<6x3xf32>
}

// -----

func.func @iota_out_of_range() -> tensor<2x3xi32> {
  // expected-error @+1 {{iota dimension 2 is out of range for rank 2}}
  %0 = tera.iota {iota_dimension = 2 : i64} : tensor<2x3xi32>
  return %0 : tensor<2x3xi32>
}

// -----

func.func @dot_contracting_mismatch(%a: tensor<2x4xf32>, %b: tensor<3x2xf32>) -> tensor<2x2xf32> {
  // expected-error @+2 {{contracting axis pair 0 has mismatched extents}}
  // expected-error @+1 {{failed to infer returned types}}
  %0 = tera.dot %a, %b {lhs_batch = array<i64>,
                        lhs_contracting = array<i64: 1>,
                        rhs_batch = array<i64>,
                        rhs_contracting = array<i64: 0>}
      : (tensor<2x4xf32>, tensor<3x2xf32>) -> tensor<2x2xf32>
  return %0 : tensor<2x2xf32>
}

// -----

func.func @dot_axis_used_twice(%a: tensor<2x4xf32>, %b: tensor<4x2xf32>) -> tensor<2x2xf32> {
  // expected-error @+2 {{lhs_contracting axis 0 is repeated}}
  // expected-error @+1 {{failed to infer returned types}}
  %0 = tera.dot %a, %b {lhs_batch = array<i64: 0>,
                        lhs_contracting = array<i64: 0>,
                        rhs_batch = array<i64: 0>,
                        rhs_contracting = array<i64: 0>}
      : (tensor<2x4xf32>, tensor<4x2xf32>) -> tensor<2x2xf32>
  return %0 : tensor<2x2xf32>
}

// -----

func.func @dot_wrong_result_rank(%a: tensor<2x4xf32>, %b: tensor<4x2xf32>) -> tensor<2xf32> {
  // expected-error @+2 {{are incompatible with return type(s) of operation}}
  // expected-error @+1 {{failed to infer returned types}}
  %0 = tera.dot %a, %b {lhs_batch = array<i64>,
                        lhs_contracting = array<i64: 1>,
                        rhs_batch = array<i64>,
                        rhs_contracting = array<i64: 0>}
      : (tensor<2x4xf32>, tensor<4x2xf32>) -> tensor<2xf32>
  return %0 : tensor<2xf32>
}

// -----

func.func @reduce_axis_out_of_range(%a: tensor<2x3xf32>) -> tensor<2xf32> {
  // expected-error @+2 {{reduction axis 5 is out of range for rank 2}}
  // expected-error @+1 {{failed to infer returned types}}
  %0 = tera.reduce sum, %a {dimensions = array<i64: 5>}
      : tensor<2x3xf32> -> tensor<2xf32>
  return %0 : tensor<2xf32>
}

// -----

func.func @reduce_wrong_result(%a: tensor<2x3xf32>) -> tensor<3xf32> {
  // expected-error @+2 {{are incompatible with return type(s) of operation}}
  // expected-error @+1 {{failed to infer returned types}}
  %0 = tera.reduce sum, %a {dimensions = array<i64: 1>}
      : tensor<2x3xf32> -> tensor<3xf32>
  return %0 : tensor<3xf32>
}

// -----

func.func @compare_unranked(%a: tensor<*xf32>, %b: tensor<*xf32>) -> tensor<*xi1> {
  // expected-error @+2 {{expected builtin.tensor, but found 'tensor<*xf32>'}}
  %0 = tera.compare lt, %a, %b
      : tensor<*xf32> -> tensor<*xi1>
  return %0 : tensor<*xi1>
}

// -----

func.func @condition_is_not_scalar(%p: tensor<2xi1>, %x: tensor<4xf32>) -> tensor<4xf32> {
  // expected-error@+1 {{expects a rank-0 condition}}
  %0 = tera.if %p, %x : (tensor<2xi1>, tensor<4xf32>) -> tensor<4xf32> {
  ^bb0(%a: tensor<4xf32>):
    tera.yield %a : tensor<4xf32>
  } else {
  ^bb0(%a: tensor<4xf32>):
    tera.yield %a : tensor<4xf32>
  }
  return %0 : tensor<4xf32>
}

// -----

func.func @branch_takes_the_wrong_shape(%p: tensor<i1>, %x: tensor<4xf32>) -> tensor<4xf32> {
  // expected-error@+1 {{the else body argument 0 is 'tensor<2xf32>', expected 'tensor<4xf32>'}}
  %0 = tera.if %p, %x : (tensor<i1>, tensor<4xf32>) -> tensor<4xf32> {
  ^bb0(%a: tensor<4xf32>):
    tera.yield %a : tensor<4xf32>
  } else {
  ^bb0(%a: tensor<2xf32>):
    %1 = tera.constant dense<0.000000e+00> : tensor<4xf32>
    tera.yield %1 : tensor<4xf32>
  }
  return %0 : tensor<4xf32>
}

// -----

func.func @branches_disagree(%p: tensor<i1>, %x: tensor<4xf32>) -> tensor<4xf32> {
  // expected-error@+1 {{the else body yields 2 values, expected 1}}
  %0 = tera.if %p, %x : (tensor<i1>, tensor<4xf32>) -> tensor<4xf32> {
  ^bb0(%a: tensor<4xf32>):
    tera.yield %a : tensor<4xf32>
  } else {
  ^bb0(%a: tensor<4xf32>):
    tera.yield %a, %a : tensor<4xf32>, tensor<4xf32>
  }
  return %0 : tensor<4xf32>
}

// -----

func.func @scan_without_a_sequence(%init: tensor<f32>) -> tensor<f32> {
  // expected-error@+1 {{expects at least one input, which is what sets the trip count}}
  %0 = tera.scan init(%init : tensor<f32>) xs() -> (tensor<f32>) {
  ^bb0(%acc: tensor<f32>):
    tera.yield %acc : tensor<f32>
  }
  return %0 : tensor<f32>
}

// -----

func.func @sequences_of_different_lengths(%init: tensor<f32>, %a: tensor<4xf32>, %b: tensor<3xf32>) -> tensor<f32> {
  // expected-error@+1 {{input 1 runs for 3 steps, but input 0 runs for 4}}
  %0 = tera.scan init(%init : tensor<f32>) xs(%a, %b : tensor<4xf32>, tensor<3xf32>)
      -> (tensor<f32>) {
  ^bb0(%acc: tensor<f32>, %x: tensor<f32>, %y: tensor<f32>):
    tera.yield %acc : tensor<f32>
  }
  return %0 : tensor<f32>
}

// -----

func.func @carry_changes_shape(%init: tensor<f32>, %xs: tensor<4xf32>) -> tensor<2xf32> {
  // expected-error@+1 {{carry 0 leaves as 'tensor<2xf32>' but entered as 'tensor<f32>'}}
  %0 = tera.scan init(%init : tensor<f32>) xs(%xs : tensor<4xf32>)
      -> (tensor<2xf32>) {
  ^bb0(%acc: tensor<f32>, %x: tensor<f32>):
    %1 = tera.constant dense<0.000000e+00> : tensor<2xf32>
    tera.yield %1 : tensor<2xf32>
  }
  return %0 : tensor<2xf32>
}

// -----

func.func @output_stacks_the_wrong_count(%init: tensor<f32>, %xs: tensor<4xf32>) -> tensor<3xf32> {
  // expected-error@+1 {{output 0 stacks 3 steps, but the scan runs for 4}}
  %0, %1 = tera.scan init(%init : tensor<f32>) xs(%xs : tensor<4xf32>)
      -> (tensor<f32>, tensor<3xf32>) {
  ^bb0(%acc: tensor<f32>, %x: tensor<f32>):
    tera.yield %acc, %acc : tensor<f32>, tensor<f32>
  }
  return %1 : tensor<3xf32>
}

// -----

func.func @body_misses_the_constant(%init: tensor<f32>, %xs: tensor<4xf32>, %w: tensor<f32>) -> tensor<f32> {
  // expected-error@+1 {{the body takes 2 arguments, expected 3}}
  %0 = tera.scan init(%init : tensor<f32>) xs(%xs : tensor<4xf32>)
      consts(%w : tensor<f32>) -> (tensor<f32>) {
  ^bb0(%acc: tensor<f32>, %x: tensor<f32>):
    tera.yield %acc : tensor<f32>
  }
  return %0 : tensor<f32>
}

// -----

func.func @ragged_checkpoints(%init: tensor<f32>, %xs: tensor<5xf32>) -> tensor<f32> {
  // expected-error@+1 {{checkpoints every 2 steps, which does not divide the 5 steps it runs for}}
  %0 = tera.scan init(%init : tensor<f32>) xs(%xs : tensor<5xf32>)
      {checkpoint = 2 : i64} -> (tensor<f32>) {
  ^bb0(%acc: tensor<f32>, %x: tensor<f32>):
    %1 = tera.add %acc, %x : tensor<f32>
    tera.yield %1 : tensor<f32>
  }
  return %0 : tensor<f32>
}

// -----

// Every op reads a rank from its operands, so the constraint refuses what it
// has no answer for rather than the lowering discovering it much later.

func.func @unranked_operand(%a: tensor<*xf32>, %b: tensor<*xf32>) -> tensor<*xf32> {
  // expected-error @+1 {{operand #0 must be ranked tensor of floating-point or signless integer values}}
  %0 = tera.dot %a, %b {lhs_batch = array<i64>, lhs_contracting = array<i64: 1>,
                        rhs_batch = array<i64>, rhs_contracting = array<i64: 0>}
      : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
  return %0 : tensor<*xf32>
}

// -----

// A body whose last op is some other terminator. The op's own verifier reads
// the yield to compare types against, and runs before any block has been
// checked, so this is the case where reading it as a yield would abort the
// compiler instead of diagnosing.

func.func @body_without_a_yield(%init: tensor<f32>, %xs: tensor<4xf32>) -> tensor<f32> {
  // expected-error @+2 {{expects regions to end with 'tera.yield', found 'llvm.unreachable'}}
  // expected-note @+1 {{the absence of terminator implies 'tera.yield'}}
  %0 = tera.scan init(%init : tensor<f32>) xs(%xs : tensor<4xf32>)
      -> (tensor<f32>) {
  ^bb0(%acc: tensor<f32>, %x: tensor<f32>):
    llvm.unreachable
  }
  return %0 : tensor<f32>
}

// -----

// The autodiff attributes sit on `func.func`, which cannot check them. Nothing
// else will either unless the dialect does, and a misspelling that is silently
// ignored reads exactly like a function nobody asked to differentiate.

// expected-error @+1 {{'tera.differentable' is not an attribute of the tera dialect}}
func.func @misspelt_marker(%x: tensor<4xf32>) -> tensor<4xf32>
    attributes {tera.differentable} {
  %0 = tera.exp %x : tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----

// expected-error @+1 {{'tera.differentiable' must be a unit attribute}}
func.func @marker_with_a_value(%x: tensor<4xf32>) -> tensor<4xf32>
    attributes {tera.differentiable = 1 : i64} {
  %0 = tera.exp %x : tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----

// expected-error @+1 {{'tera.vjp' must name the derivative, as @symbol}}
func.func @derivative_is_not_a_symbol(%x: tensor<4xf32>) -> tensor<4xf32>
    attributes {tera.vjp = 42 : i64} {
  %0 = tera.exp %x : tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----

// expected-error @+1 {{'tera.diff_args' must be an array<i64>}}
func.func @positions_are_not_an_array(%x: tensor<4xf32>, %seed: tensor<4xf32>)
    -> tensor<4xf32> attributes {tera.diff_args = "the first one"} {
  return %x : tensor<4xf32>
}

// -----

// expected-error @+1 {{'tera.diff_args' must be strictly increasing}}
func.func @positions_out_of_order(%x: tensor<4xf32>, %w: tensor<4xf32>,
                                  %seed: tensor<4xf32>)
    -> (tensor<4xf32>, tensor<4xf32>) attributes {tera.diff_args = array<i64: 1, 0>} {
  return %w, %x : tensor<4xf32>, tensor<4xf32>
}

// -----

func.func @marker_on_an_operation(%x: tensor<4xf32>) -> tensor<4xf32> {
  // expected-error @+1 {{'tera.differentiable' belongs on a function, not on this operation}}
  %0 = tera.exp %x {tera.differentiable} : tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----

func.func @gather_wrong_slice_arity(%t: tensor<10x4xf32>, %i: tensor<2xi32>) -> tensor<2x4xf32> {
  // expected-error @+2 {{expects one slice size per operand axis: 2 expected, 1 given}}
  // expected-error @+1 {{failed to infer returned types}}
  %0 = tera.gather %t, %i {offset_dims = array<i64: 1>,
                           collapsed_slice_dims = array<i64: 0>,
                           start_index_map = array<i64: 0>,
                           slice_sizes = array<i64: 1>,
                           index_vector_dim = 1 : i64}
      : (tensor<10x4xf32>, tensor<2xi32>) -> tensor<2x4xf32>
  return %0 : tensor<2x4xf32>
}

// -----

func.func @gather_collapses_a_wide_slice(%t: tensor<10x4xf32>, %i: tensor<2xi32>) -> tensor<2x4xf32> {
  // expected-error @+2 {{collapsed axis 1 has slice size 4, expected 1}}
  // expected-error @+1 {{failed to infer returned types}}
  %0 = tera.gather %t, %i {offset_dims = array<i64: 1>,
                           collapsed_slice_dims = array<i64: 1>,
                           start_index_map = array<i64: 0>,
                           slice_sizes = array<i64: 1, 4>,
                           index_vector_dim = 1 : i64}
      : (tensor<10x4xf32>, tensor<2xi32>) -> tensor<2x4xf32>
  return %0 : tensor<2x4xf32>
}

// -----

func.func @gather_slice_past_the_operand(%t: tensor<10x4xf32>, %i: tensor<2xi32>) -> tensor<2x6xf32> {
  // expected-error @+2 {{slice size 6 at axis 1 exceeds the operand extent 4}}
  // expected-error @+1 {{failed to infer returned types}}
  %0 = tera.gather %t, %i {offset_dims = array<i64: 1>,
                           collapsed_slice_dims = array<i64: 0>,
                           start_index_map = array<i64: 0>,
                           slice_sizes = array<i64: 1, 6>,
                           index_vector_dim = 1 : i64}
      : (tensor<10x4xf32>, tensor<2xi32>) -> tensor<2x6xf32>
  return %0 : tensor<2x6xf32>
}

// -----

// One start_index_map entry per coordinate a position holds, and the index
// vector axis is what says how many that is.
func.func @gather_map_disagrees_with_the_index_vector(
    %t: tensor<10x4xf32>, %i: tensor<2x2xi32>) -> tensor<2x4xf32> {
  // expected-error @+2 {{expects one start_index_map entry per index coordinate: 2 expected, 1 given}}
  // expected-error @+1 {{failed to infer returned types}}
  %0 = tera.gather %t, %i {offset_dims = array<i64: 1>,
                           collapsed_slice_dims = array<i64: 0>,
                           start_index_map = array<i64: 0>,
                           slice_sizes = array<i64: 1, 4>,
                           index_vector_dim = 1 : i64}
      : (tensor<10x4xf32>, tensor<2x2xi32>) -> tensor<2x4xf32>
  return %0 : tensor<2x4xf32>
}

// -----

func.func @gather_index_vector_out_of_range(%t: tensor<10x4xf32>,
                                            %i: tensor<2xi32>) -> tensor<2x4xf32> {
  // expected-error @+2 {{index_vector_dim 2 is out of range for indices of rank 1}}
  // expected-error @+1 {{failed to infer returned types}}
  %0 = tera.gather %t, %i {offset_dims = array<i64: 1>,
                           collapsed_slice_dims = array<i64: 0>,
                           start_index_map = array<i64: 0>,
                           slice_sizes = array<i64: 1, 4>,
                           index_vector_dim = 2 : i64}
      : (tensor<10x4xf32>, tensor<2xi32>) -> tensor<2x4xf32>
  return %0 : tensor<2x4xf32>
}

// -----

func.func @gather_too_many_offset_dims(%t: tensor<10x4xf32>, %i: tensor<2xi32>) -> tensor<2x4xf32> {
  // expected-error @+2 {{expects one offset dimension per surviving slice axis: 1 expected, 2 given}}
  // expected-error @+1 {{failed to infer returned types}}
  %0 = tera.gather %t, %i {offset_dims = array<i64: 0, 1>,
                           collapsed_slice_dims = array<i64: 0>,
                           start_index_map = array<i64: 0>,
                           slice_sizes = array<i64: 1, 4>,
                           index_vector_dim = 1 : i64}
      : (tensor<10x4xf32>, tensor<2xi32>) -> tensor<2x4xf32>
  return %0 : tensor<2x4xf32>
}

// -----

func.func @scatter_window_leaves_an_operand_axis_out(%t: tensor<10x4xf32>, %i: tensor<2xi32>,
                                                     %u: tensor<2x4xf32>) -> tensor<10x4xf32> {
  // expected-error @+2 {{expects one window axis per operand axis, inserted or not: 2 expected, 1 given}}
  // expected-error @+1 {{failed to infer returned types}}
  %0 = tera.scatter %t, %i, %u {update_window_dims = array<i64: 1>,
                                inserted_window_dims = array<i64>,
                                scatter_dims_to_operand_dims = array<i64: 0>,
                                index_vector_dim = 1 : i64}
      : (tensor<10x4xf32>, tensor<2xi32>, tensor<2x4xf32>) -> tensor<10x4xf32>
  return %0 : tensor<10x4xf32>
}

// -----

func.func @scatter_batch_axes_disagree(%t: tensor<10x4xf32>, %i: tensor<2xi32>,
                                       %u: tensor<2x3x4xf32>) -> tensor<10x4xf32> {
  // expected-error @+2 {{expects 1 update axes outside the window, one per index batch axis, but the window leaves 2}}
  // expected-error @+1 {{failed to infer returned types}}
  %0 = tera.scatter %t, %i, %u {update_window_dims = array<i64: 2>,
                                inserted_window_dims = array<i64: 0>,
                                scatter_dims_to_operand_dims = array<i64: 0>,
                                index_vector_dim = 1 : i64}
      : (tensor<10x4xf32>, tensor<2xi32>, tensor<2x3x4xf32>) -> tensor<10x4xf32>
  return %0 : tensor<10x4xf32>
}

// -----

func.func @scatter_window_wider_than_the_axis(%t: tensor<10x4xf32>, %i: tensor<2xi32>,
                                              %u: tensor<2x6xf32>) -> tensor<10x4xf32> {
  // expected-error @+2 {{window of width 6 does not fit operand axis 1 of extent 4}}
  // expected-error @+1 {{failed to infer returned types}}
  %0 = tera.scatter %t, %i, %u {update_window_dims = array<i64: 1>,
                                inserted_window_dims = array<i64: 0>,
                                scatter_dims_to_operand_dims = array<i64: 0>,
                                index_vector_dim = 1 : i64}
      : (tensor<10x4xf32>, tensor<2xi32>, tensor<2x6xf32>) -> tensor<10x4xf32>
  return %0 : tensor<10x4xf32>
}

// -----

func.func @scatter_element_type_disagrees(%t: tensor<10x4xf32>, %i: tensor<2xi32>,
                                          %u: tensor<2x4xf64>) -> tensor<10x4xf32> {
  // expected-error @+2 {{scatters 'f64' into 'f32'}}
  // expected-error @+1 {{failed to infer returned types}}
  %0 = tera.scatter %t, %i, %u {update_window_dims = array<i64: 1>,
                                inserted_window_dims = array<i64: 0>,
                                scatter_dims_to_operand_dims = array<i64: 0>,
                                index_vector_dim = 1 : i64}
      : (tensor<10x4xf32>, tensor<2xi32>, tensor<2x4xf64>) -> tensor<10x4xf32>
  return %0 : tensor<10x4xf32>
}

// -----

// The same operand axis addressed twice is a lookup with two answers, not a
// shape that happens not to work out.
func.func @scatter_repeats_an_operand_axis(%t: tensor<10x4xf32>, %i: tensor<2x2xi32>,
                                           %u: tensor<2xf32>) -> tensor<10x4xf32> {
  // expected-error @+2 {{scatter_dims_to_operand_dims axis 0 is repeated}}
  // expected-error @+1 {{failed to infer returned types}}
  %0 = tera.scatter %t, %i, %u {update_window_dims = array<i64>,
                                inserted_window_dims = array<i64: 0, 1>,
                                scatter_dims_to_operand_dims = array<i64: 0, 0>,
                                index_vector_dim = 1 : i64}
      : (tensor<10x4xf32>, tensor<2x2xi32>, tensor<2xf32>) -> tensor<10x4xf32>
  return %0 : tensor<10x4xf32>
}

// -----

func.func @conv_wrong_padding_arity(%x: tensor<1x1x5x5xf32>, %k: tensor<1x1x2x2xf32>)
    -> tensor<1x1x4x4xf32> {
  // expected-error @+2 {{expects a low and a high padding per spatial axis: 4 expected, 2 given}}
  // expected-error @+1 {{failed to infer returned types}}
  %0 = tera.conv %x, %k {strides = array<i64: 1, 1>,
                         padding = array<i64: 0, 0>,
                         dilation = array<i64: 1, 1>,
                         groups = 1 : i64}
      : (tensor<1x1x5x5xf32>, tensor<1x1x2x2xf32>) -> tensor<1x1x4x4xf32>
  return %0 : tensor<1x1x4x4xf32>
}

// -----

// A crop is a slice, not a convolution that pads by a negative amount.
func.func @conv_negative_padding(%x: tensor<1x1x5x5xf32>, %k: tensor<1x1x2x2xf32>)
    -> tensor<1x1x2x2xf32> {
  // expected-error @+2 {{pads axis 0 by a negative amount; cropping is a slice}}
  // expected-error @+1 {{failed to infer returned types}}
  %0 = tera.conv %x, %k {strides = array<i64: 1, 1>,
                         padding = array<i64: -1, -1, -1, -1>,
                         dilation = array<i64: 1, 1>,
                         groups = 1 : i64}
      : (tensor<1x1x5x5xf32>, tensor<1x1x2x2xf32>) -> tensor<1x1x2x2xf32>
  return %0 : tensor<1x1x2x2xf32>
}

// -----

func.func @conv_rank_disagrees(%x: tensor<1x1x5xf32>, %k: tensor<1x1x2x2xf32>)
    -> tensor<1x1x4x4xf32> {
  // expected-error @+2 {{expects a batch axis, a channel axis and 2 spatial axes on both operands, but they have rank 3 and 4}}
  // expected-error @+1 {{failed to infer returned types}}
  %0 = tera.conv %x, %k {strides = array<i64: 1, 1>,
                         padding = array<i64: 0, 0, 0, 0>,
                         dilation = array<i64: 1, 1>,
                         groups = 1 : i64}
      : (tensor<1x1x5xf32>, tensor<1x1x2x2xf32>) -> tensor<1x1x4x4xf32>
  return %0 : tensor<1x1x4x4xf32>
}

// -----

func.func @conv_channels_do_not_divide(%x: tensor<1x5x5x5xf32>, %k: tensor<6x2x2x2xf32>)
    -> tensor<1x6x4x4xf32> {
  // expected-error @+2 {{splits 5 input channels into 2 groups}}
  // expected-error @+1 {{failed to infer returned types}}
  %0 = tera.conv %x, %k {strides = array<i64: 1, 1>,
                         padding = array<i64: 0, 0, 0, 0>,
                         dilation = array<i64: 1, 1>,
                         groups = 2 : i64}
      : (tensor<1x5x5x5xf32>, tensor<6x2x2x2xf32>) -> tensor<1x6x4x4xf32>
  return %0 : tensor<1x6x4x4xf32>
}

// -----

// The kernel is as deep as one group of the input, so a kernel as deep as the
// whole of it is a kernel for a different number of groups.
func.func @conv_kernel_too_deep(%x: tensor<1x4x5x5xf32>, %k: tensor<6x4x2x2xf32>)
    -> tensor<1x6x4x4xf32> {
  // expected-error @+2 {{reads 4 input channels per group, but 2 groups of 4 is 2}}
  // expected-error @+1 {{failed to infer returned types}}
  %0 = tera.conv %x, %k {strides = array<i64: 1, 1>,
                         padding = array<i64: 0, 0, 0, 0>,
                         dilation = array<i64: 1, 1>,
                         groups = 2 : i64}
      : (tensor<1x4x5x5xf32>, tensor<6x4x2x2xf32>) -> tensor<1x6x4x4xf32>
  return %0 : tensor<1x6x4x4xf32>
}

// -----

func.func @conv_zero_stride(%x: tensor<1x1x5x5xf32>, %k: tensor<1x1x2x2xf32>)
    -> tensor<1x1x4x4xf32> {
  // expected-error @+2 {{stride 0 and dilation 1 at axis 0 must both be positive}}
  // expected-error @+1 {{failed to infer returned types}}
  %0 = tera.conv %x, %k {strides = array<i64: 0, 1>,
                         padding = array<i64: 0, 0, 0, 0>,
                         dilation = array<i64: 1, 1>,
                         groups = 1 : i64}
      : (tensor<1x1x5x5xf32>, tensor<1x1x2x2xf32>) -> tensor<1x1x4x4xf32>
  return %0 : tensor<1x1x4x4xf32>
}

// -----

func.func @pool_not_four_axes(%x: tensor<1x5x5xf32>) -> tensor<1x2x2xf32> {
  // expected-error @+2 {{pools a batch, a channel and two spatial axes, so expects rank 4, got 3}}
  // expected-error @+1 {{failed to infer returned types}}
  %0 = tera.pool2d max, %x {kernel_size = array<i64: 2, 2>,
                            strides = array<i64: 2, 2>,
                            padding = array<i64: 0, 0, 0, 0>}
      : tensor<1x5x5xf32> -> tensor<1x2x2xf32>
  return %0 : tensor<1x2x2xf32>
}

// -----

func.func @pool_window_is_not_two_axes(%x: tensor<1x1x5x5xf32>) -> tensor<1x1x2x2xf32> {
  // expected-error @+2 {{expects a two-axis window, got 3 extents}}
  // expected-error @+1 {{failed to infer returned types}}
  %0 = tera.pool2d max, %x {kernel_size = array<i64: 2, 2, 2>,
                            strides = array<i64: 2, 2>,
                            padding = array<i64: 0, 0, 0, 0>}
      : tensor<1x1x5x5xf32> -> tensor<1x1x2x2xf32>
  return %0 : tensor<1x1x2x2xf32>
}

// -----

func.func @pool_empty_window(%x: tensor<1x1x5x5xf32>) -> tensor<1x1x5x5xf32> {
  // expected-error @+2 {{window extent 0 at axis 0 must be positive}}
  // expected-error @+1 {{failed to infer returned types}}
  %0 = tera.pool2d max, %x {kernel_size = array<i64: 0, 1>,
                            strides = array<i64: 1, 1>,
                            padding = array<i64: 0, 0, 0, 0>}
      : tensor<1x1x5x5xf32> -> tensor<1x1x5x5xf32>
  return %0 : tensor<1x1x5x5xf32>
}

// -----

func.func @pad_wrong_arity(%x: tensor<2x3xf32>, %v: tensor<f32>) -> tensor<3x3xf32> {
  // expected-error @+2 {{expects low, high and any interior to each have 2 entries}}
  // expected-error @+1 {{failed to infer returned types}}
  %0 = tera.pad %x, %v {low = array<i64: 1>, high = array<i64: 0, 0>}
      : (tensor<2x3xf32>, tensor<f32>) -> tensor<3x3xf32>
  return %0 : tensor<3x3xf32>
}

// -----

func.func @pad_crops(%x: tensor<2x3xf32>, %v: tensor<f32>) -> tensor<1x3xf32> {
  // expected-error @+2 {{pads axis 0 by a negative amount; cropping is a slice}}
  // expected-error @+1 {{failed to infer returned types}}
  %0 = tera.pad %x, %v {low = array<i64: -1, 0>, high = array<i64: 0, 0>}
      : (tensor<2x3xf32>, tensor<f32>) -> tensor<1x3xf32>
  return %0 : tensor<1x3xf32>
}

// -----

// The value a pad fills with is one number, not a tensor of them.
func.func @pad_with_a_tensor(%x: tensor<2x3xf32>, %v: tensor<2xf32>) -> tensor<3x3xf32> {
  // expected-error @+2 {{pads with a rank-1 tensor, which is not a single value}}
  // expected-error @+1 {{failed to infer returned types}}
  %0 = tera.pad %x, %v {low = array<i64: 1, 0>, high = array<i64: 0, 0>}
      : (tensor<2x3xf32>, tensor<2xf32>) -> tensor<3x3xf32>
  return %0 : tensor<3x3xf32>
}

// -----

func.func @reverse_repeats_an_axis(%x: tensor<2x3xf32>) -> tensor<2x3xf32> {
  // expected-error @+2 {{dimensions axis 1 is repeated}}
  // expected-error @+1 {{failed to infer returned types}}
  %0 = tera.reverse %x {dimensions = array<i64: 1, 1>}
      : tensor<2x3xf32> -> tensor<2x3xf32>
  return %0 : tensor<2x3xf32>
}
