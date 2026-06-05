import * as ops from '../../tensor/ops/ops.js';

export function layer_norm(input, normalizedShape, weight, bias, eps = 1e-5) {
  const axis = input.ndim - normalizedShape.length;
  return ops.layer_norm(input, weight, bias, axis, eps);
}

export function batch_norm(input, runningMean, runningVar, weight, bias, training = true, eps = 1e-5) {
  return ops.batch_norm(input, weight, bias, runningMean, runningVar, 1, eps);
}
