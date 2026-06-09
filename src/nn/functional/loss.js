import * as ops from '../../tensor/ops/ops.js';
import { full } from '../../tensor/factory/creation_ops.js';
import { tensor as fromArray } from '../../tensor/factory/from_ops.js';

export function mse_loss(input, target, reduction = 'mean') {
  const diff = ops.sub(input, target);
  const sq = ops.mul(diff, diff);
  return _reduce(sq, reduction);
}

function _logSoftmaxAutograd(input, dim) {
  const maxVal = _dimMax(input, dim);
  const shifted = ops.sub(input, maxVal);
  const expShifted = ops.exp(shifted);
  const sumExp = _dimSum(expShifted, dim);
  const logSumExp = ops.log(sumExp);
  return ops.sub(shifted, logSumExp);
}

function _dimMax(input, dim) {
  const actualDim = dim < 0 ? input.ndim + dim : dim;
  const maxT = ops.max(input, actualDim, true);
  return maxT;
}

function _dimSum(input, dim) {
  const actualDim = dim < 0 ? input.ndim + dim : dim;
  return ops.sum(input, actualDim, true);
}

function _buildOneHot(batchSize, numClasses, target, dtype, device) {
  const targetData = target._impl.storage.data;
  const targetOffset = target._impl.storageOffset;
  const oneHotData = new Float32Array(batchSize * numClasses);
  for (let i = 0; i < batchSize; i++) {
    oneHotData[i * numClasses + (targetData[targetOffset + i] | 0)] = 1;
  }
  return fromArray(oneHotData, { shape: [batchSize, numClasses], dtype, device });
}

export function nll_loss(input, target, reduction = 'mean') {
  const batchSize = input.shape[0];
  const numClasses = input.shape[1];
  const oneHot = _buildOneHot(batchSize, numClasses, target, input.dtype, input.device);
  const gathered = ops.mul(input, oneHot);
  const totalNeg = ops.neg(ops.sum(gathered));
  if (reduction === 'mean') {
    const n = full(totalNeg.shape, batchSize, { dtype: input.dtype, device: input.device });
    return ops.div(totalNeg, n);
  }
  if (reduction === 'sum') return totalNeg;
  throw new Error(`nll_loss: unknown reduction '${reduction}'`);
}

export function cross_entropy(input, target, reduction = 'mean') {
  const logProbs = _logSoftmaxAutograd(input, -1);
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
