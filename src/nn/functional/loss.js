import * as ops from '../../tensor/ops/ops.js';
import { full } from '../../tensor/factory/creation_ops.js';
import { log_softmax } from './activation.js';

export function mse_loss(input, target, reduction = 'mean') {
  const diff = ops.sub(input, target);
  const sq = ops.mul(diff, diff);
  return _reduce(sq, reduction);
}

export function nll_loss(input, target, reduction = 'mean') {
  const batchSize = input.shape[0];
  const data = input._impl.storage.data;
  const targetData = target._impl.storage.data;
  const inputOffset = input._impl.storageOffset;
  const targetOffset = target._impl.storageOffset;
  const numClasses = input.shape[1];

  let total = 0;
  for (let i = 0; i < batchSize; i++) {
    const cls = targetData[targetOffset + i] | 0;
    total -= data[inputOffset + i * numClasses + cls];
  }

  const { scalar } = require('../../tensor/factory/from_ops.js');
  if (reduction === 'mean') return scalar(total / batchSize, { dtype: input.dtype, device: input.device });
  if (reduction === 'sum') return scalar(total, { dtype: input.dtype, device: input.device });
  throw new Error('nll_loss only supports mean/sum reduction with this implementation');
}

export function cross_entropy(input, target, reduction = 'mean') {
  const logProbs = log_softmax(input, -1);
  return nll_loss(logProbs, target, reduction);
}

export function binary_cross_entropy(input, target, reduction = 'mean') {
  const eps = full(input.shape, 1e-7, { dtype: input.dtype, device: input.device });
  const one = full(input.shape, 1, { dtype: input.dtype, device: input.device });
  const logInput = ops.log(ops.add(input, eps));
  const logOneMinusInput = ops.log(ops.add(ops.sub(one, input), eps));
  const loss = ops.neg(ops.add(ops.mul(target, logInput), ops.mul(ops.sub(one, target), logOneMinusInput)));
  return _reduce(loss, reduction);
}

function _reduce(tensor, reduction) {
  if (reduction === 'mean') return ops.mean(tensor);
  if (reduction === 'sum') return ops.sum(tensor);
  if (reduction === 'none') return tensor;
  throw new Error(`Unknown reduction: ${reduction}`);
}
