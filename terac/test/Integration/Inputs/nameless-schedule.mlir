// A transform module with no `__transform_main` in it. The entry point is what
// the pass is told to run, and a module that does not have one is a schedule
// that cannot be applied rather than a schedule that does nothing.

module attributes {transform.with_named_sequence} {
  transform.named_sequence @tune(%root: !transform.any_op {transform.readonly}) {
    transform.yield
  }
}
