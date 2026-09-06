// The thread block `-tera-tile-parallel-loops` cuts a parallel loop into,
// written in the transform dialect. The two arrive at the same launch by
// different routes and the test beside this file compares the launch rather
// than the module: the pass tiles `scf.parallel` after bufferization and lets
// `convert-parallel-loops-to-gpu` read the block off the nesting, and this
// tiles the linalg op into `scf.forall` and hands the mapping to the GPU
// transform ops, which substitute ids rather than nest loops.
//
// 256 lanes across and one row down is what the pass derives for this shape
// from its target model: the innermost axis takes the thread budget first
// because that is the axis `mapping-policy=innermost-first` gives to
// `thread_x`, and 256 divides 512.

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %op = transform.structured.match interface{LinalgOp} in %root
        : (!transform.any_op) -> !transform.any_op
    %blocks, %block_loop = transform.structured.tile_using_forall %op
        tile_sizes [1, 256]
        ( mapping = [#gpu.block<y>, #gpu.block<x>] )
        : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    %threads, %thread_loop = transform.structured.tile_using_forall %blocks
        tile_sizes [1, 1]
        ( mapping = [#gpu.thread<y>, #gpu.thread<x>] )
        : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    %fn = transform.structured.match ops{["func.func"]} in %root
        : (!transform.any_op) -> !transform.any_op
    %launch = transform.gpu.map_forall_to_blocks %fn generate_gpu_launch
        : (!transform.any_op) -> !transform.any_op
    transform.gpu.map_nested_forall_to_threads %launch block_dims = [256, 1, 1]
        : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}
