import * as ops from '../../tensor/ops/ops.js';
import { full } from '../../tensor/factory/creation_ops.js';
import { select } from '../../tensor/ops/ops.js';
import { computeNumel } from '../../tensor/utils/shape_utils.js';
import type { Tensor } from '../../tensor/core/tensor.js';

export type LossReduction = 'mean' | 'sum' | 'none';

export function mse_loss(input: Tensor, target: Tensor, reduction: LossReduction = 'mean'): Tensor {
  const diff = ops.sub(input, target);
  const sq = ops.mul(diff, diff);
  return _reduce(sq, reduction);
}

function _logSoftmaxAutograd(input: Tensor, dim: number): Tensor {
  const maxVal = _dimMax(input, dim);
  const shifted = ops.sub(input, maxVal);
  const expShifted = ops.exp(shifted);
  const sumExp = _dimSum(expShifted, dim);
  const logSumExp = ops.log(sumExp);
  return ops.sub(shifted, logSumExp);
}

function _dimMax(input: Tensor, dim: number): Tensor {
  const actualDim = dim < 0 ? input.ndim + dim : dim;
  const maxT = ops.max(input, actualDim, true);
  return maxT;
}

function _dimSum(input: Tensor, dim: number): Tensor {
  const actualDim = dim < 0 ? input.ndim + dim : dim;
  return ops.sum(input, actualDim, true);
}

export function nll_loss(input: Tensor, target: Tensor, reduction: LossReduction = 'mean', ignoreIndex: number | null = null): Tensor {
  const lastDim = input.ndim - 1;
  const numClasses = input.shape[lastDim];
  const oneHot = ops.one_hot(target, numClasses);
  const perRow = ops.sum(ops.mul(input, oneHot), lastDim);

  let masked = perRow;
  let denom: Tensor | null = null;
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

export function cross_entropy(input: Tensor, target: Tensor, reduction: LossReduction = 'mean', ignoreIndex: number | null = null): Tensor {
  const logProbs = _logSoftmaxAutograd(input, -1);
  return nll_loss(logProbs, target, reduction, ignoreIndex);
}

export function binary_cross_entropy(input: Tensor, target: Tensor, reduction: LossReduction = 'mean'): Tensor {
  const eps = full(input.shape, 1e-7, { dtype: input.dtype, device: input.device });
  const one = full(input.shape, 1, { dtype: input.dtype, device: input.device });
  const logInput = ops.log(ops.add(input, eps));
  const logOneMinusInput = ops.log(ops.add(ops.sub(one, input), eps));
  const loss = ops.neg(ops.add(ops.mul(target, logInput), ops.mul(ops.sub(one, target), logOneMinusInput)));
  return _reduce(loss, reduction);
}

function _reduce(tensor: Tensor, reduction: LossReduction): Tensor {
  if (reduction === 'mean') return ops.mean(tensor);
  if (reduction === 'sum') return ops.sum(tensor);
  if (reduction === 'none') return tensor;
  throw new Error(`Unknown reduction: ${reduction}`);
}
