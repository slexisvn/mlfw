import { getActiveTracer } from './tracer.js';
import { Tensor } from '../tensor/core/tensor.js';
import type { TensorOutput } from './types.js';

type BranchValue = TensorOutput | TensorOutput[];
type BranchFn = () => BranchValue;

export function cond(predicate: TensorOutput, onTrue: BranchFn, onFalse: BranchFn): BranchValue {
  const tracer = getActiveTracer();
  if (!tracer) {
    return Number((predicate as Tensor).item()) !== 0 ? onTrue() : onFalse();
  }
  return tracer.cond(predicate, onTrue, onFalse);
}
