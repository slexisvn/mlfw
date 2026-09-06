// A schedule for the contraction in pool-free-matmul, written from outside the
// compiler and handed to it. It is what a tuned schedule looks like: the op is
// named rather than counted, the numbers in it are the search's answer rather
// than a heuristic's, and nothing about it is a pass.
//
// The block is 16 by 16 and each thread takes one element of it, which is a
// different schedule from the one terac would have chosen -- its own model
// picks 32 by 32 for these extents -- so the numbers coming out prove that the
// script was used and not merely accepted.

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %dot = transform.structured.match
        attributes{tera.schedule = "scheduled.dot.0"} in %root
        : (!transform.any_op) -> !transform.any_op
    %blocks, %block_loop = transform.structured.tile_using_forall %dot
        tile_sizes [16, 16]
        ( mapping = [#gpu.block<y>, #gpu.block<x>] )
        : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    %threads, %thread_loop = transform.structured.tile_using_forall %blocks
        tile_sizes [1, 1]
        ( mapping = [#gpu.thread<y>, #gpu.thread<x>] )
        : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    // The `linalg.fill` that seeds the accumulator is left alone, and so runs
    // on the host. That is not the schedule anyone wants; it is where the
    // vocabulary runs out. `transform.gpu.map_forall_to_blocks` builds its
    // launch from the one top-level `scf.forall` it can find and refuses two,
    // so a script gets one launch, and on buffers the fill cannot be folded
    // into the contraction the way it is on tensors -- they are two writes to
    // the same memory rather than a value and its consumer. A schedule for
    // both would need either a second entry point or a way to say `launch` on
    // its own, and neither is here.

    %fn = transform.structured.match ops{["func.func"]} in %root
        : (!transform.any_op) -> !transform.any_op
    %launch = transform.gpu.map_forall_to_blocks %fn generate_gpu_launch
        : (!transform.any_op) -> !transform.any_op
    transform.gpu.map_nested_forall_to_threads %launch block_dims = [16, 16, 1]
        : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}
