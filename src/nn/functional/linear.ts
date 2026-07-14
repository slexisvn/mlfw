import * as ops from '../../tensor/ops/ops.js';
import { transpose as viewTranspose } from '../../tensor/ops/ops.js';
import { SymbolicTensor } from '../../tracing/symbolic_tensor.js';
import type { NNTensor, OptionalTensor } from '../types.js';

export function linear(input: NNTensor, weight: NNTensor, bias?: OptionalTensor): NNTensor {
  let wt: NNTensor;
  if (input instanceof SymbolicTensor || (input.isSymbolic)) {
    wt = ops.transpose(weight, 0, 1) as NNTensor;
  } else {
    wt = viewTranspose(weight, 0, 1) as NNTensor;
  }
  const output = ops.matmul(input, wt) as NNTensor;
  if (bias) return ops.add(output, bias) as NNTensor;
  return output;
}
