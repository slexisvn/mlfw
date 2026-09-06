// The schedule `-tera-tile-and-fuse` applies, written in the transform
// dialect instead of C++. Not an approximation of it: the test beside this
// file requires the two to produce the same module, character for character.
//
// Three linalg ops reach here -- the fill, the contraction and the elementwise
// consumer -- and the handles are split positionally because nothing yet
// carries a name from the tera op each came from. That is what makes this a
// fixture for one program rather than a schedule for any program, and it is
// the gap a tuned script closes by naming the op it schedules.

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %all = transform.structured.match interface{LinalgOp} in %root
        : (!transform.any_op) -> !transform.any_op
    %fill, %dot, %ewise = transform.split_handle %all
        : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)

    // The consumer is cut first and takes no producer with it, which is what
    // the pass does when it walks the ops in reverse: the contraction it reads
    // is a reduction, and a reduction cannot be finished inside a tile of its
    // consumer.
    %tiled_ewise, %rows, %lanes = transform.structured.tile_using_for %ewise
        tile_sizes [1, 16]
        : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)

    // The contraction is cut on its parallel axes and pulls the fill that
    // seeds it into the tile, so the destination is zeroed a tile at a time
    // rather than all at once beforehand.
    %tiled_dot, %dot_rows, %dot_lanes = transform.structured.fuse %dot
        tile_sizes [1, 16, 0]
        : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)

    // Then the contracted axis, to one element, which leaves exactly one
    // vector and a loop carrying the accumulator.
    %reduced, %steps = transform.structured.tile_using_for %tiled_dot
        tile_sizes [0, 0, 1]
        : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}
