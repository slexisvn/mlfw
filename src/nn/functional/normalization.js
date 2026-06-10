import * as ops from '../../tensor/ops/ops.js';
import { full } from '../../tensor/factory/creation_ops.js';
import { SymbolicTensor } from '../../tracing/symbolic_tensor.js';

export function layer_norm(input, normalizedShape, weight, bias, eps = 1e-5) {
  if ((input instanceof SymbolicTensor || input.isSymbolic) && weight && bias) {
    const axis = input.ndim - normalizedShape.length;
    return ops.layer_norm(input, weight, bias, axis, eps);
  }
  const axis = input.ndim - normalizedShape.length;
  const dims = [];
  for (let i = axis; i < input.ndim; i++) dims.push(i);

  let meanVal = input;
  for (let i = dims.length - 1; i >= 0; i--) {
    meanVal = ops.mean(meanVal, dims[i], true);
  }

  const centered = ops.sub(input, meanVal);
  const sq = ops.mul(centered, centered);

  let variance = sq;
  for (let i = dims.length - 1; i >= 0; i--) {
    variance = ops.mean(variance, dims[i], true);
  }

  const epsT = full([], eps);
  const invStd = ops.div(full([], 1), ops.sqrt(ops.add(variance, epsT)));
  let normalized = ops.mul(centered, invStd);

  if (weight) normalized = ops.mul(normalized, weight);
  if (bias) normalized = ops.add(normalized, bias);
  return normalized;
}

export function batch_norm(input, runningMean, runningVar, weight, bias, training = true, eps = 1e-5) {
  return ops.batch_norm(input, weight, bias, runningMean, runningVar, 1, eps);
}
