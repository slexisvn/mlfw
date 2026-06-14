import * as ops from '../../tensor/ops/ops.js';

import { full } from '../../tensor/factory/creation_ops.js';

export function relu(input) {
  return ops.relu(input);
}

export function gelu(input) {
  return ops.gelu(input);
}

export function silu(input) {
  return ops.silu(input);
}

export function sigmoid(input) {
  return ops.sigmoid(input);
}

export function tanh(input) {
  return ops.tanh(input);
}

export function softmax(input, dim = -1) {
  return ops.softmax(input, dim);
}

export function log_softmax(input, dim = -1) {
  return ops.log_softmax(input, dim);
}

export function leaky_relu(input, negativeSlope = 0.01) {
  const scaled = ops.mul(input, full(input.shape, negativeSlope, { dtype: input.dtype, device: input.device }));
  return ops.maximum(input, scaled);
}

export function elu(input, alpha = 1.0) {
  const zero = full(input.shape, 0, { dtype: input.dtype, device: input.device });
  const one = full(input.shape, 1, { dtype: input.dtype, device: input.device });
  const alphaT = full(input.shape, alpha, { dtype: input.dtype, device: input.device });
  const mask = ops.gt(input, zero);
  const negative = ops.mul(alphaT, ops.sub(ops.exp(input), one));
  return ops.where(mask, input, negative);
}
