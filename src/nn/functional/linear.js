import * as ops from '../../tensor/ops/ops.js';
import { transpose as viewTranspose } from '../../tensor/view/view_ops.js';
import { SymbolicTensor } from '../../tracing/symbolic_tensor.js';

export function linear(input, weight, bias) {
  let wt;
  if (input instanceof SymbolicTensor || (input.isSymbolic)) {
    wt = ops.transpose_op(weight, 0, 1);
  } else {
    wt = viewTranspose(weight, 0, 1);
  }
  const output = ops.matmul(input, wt);
  if (bias) return ops.add(output, bias);
  return output;
}
