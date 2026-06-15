import * as ops from '../../tensor/ops/ops.js';
import { full } from '../../tensor/factory/creation_ops.js';
import { select } from '../../tensor/view/view_ops.js';
import { computeNumel } from '../../tensor/utils/shape_utils.js';

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

export function nll_loss(input, target, reduction = 'mean', ignoreIndex = null) {
  const lastDim = input.ndim - 1;
  const numClasses = input.shape[lastDim];
  const oneHot = ops.one_hot(target, numClasses);
  const perRow = ops.sum(ops.mul(input, oneHot), lastDim);

  let masked = perRow;
  let denom = null;
  if (ignoreIndex !== null) {
    const ignored = select(oneHot, lastDim, ignoreIndex);
    const maskF = ops.add(ops.neg(ignored), 1);
    masked = ops.mul(perRow, maskF);
    denom = ops.sum(maskF);
  }

  const totalNeg = ops.neg(ops.sum(masked));
  if (reduction === 'sum') return totalNeg;
  if (reduction === 'mean') {
    if (denom !== null) return ops.div(totalNeg, denom);
    return ops.div(totalNeg, computeNumel(target.shape));
  }
  throw new Error(`nll_loss: unknown reduction '${reduction}'`);
}

export function cross_entropy(input, target, reduction = 'mean', ignoreIndex = null) {
  const logProbs = _logSoftmaxAutograd(input, -1);
  return nll_loss(logProbs, target, reduction, ignoreIndex);
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
