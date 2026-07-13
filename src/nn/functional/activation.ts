import * as ops from '../../tensor/ops/ops.js';

import { full } from '../../tensor/factory/creation_ops.js';
import type { Tensor } from '../../tensor/core/tensor.js';

export function relu(input: Tensor): Tensor {
  return ops.relu(input);
}

export function gelu(input: Tensor): Tensor {
  return ops.gelu(input);
}

export function silu(input: Tensor): Tensor {
  return ops.silu(input);
}

export function sigmoid(input: Tensor): Tensor {
  return ops.sigmoid(input);
}

export function tanh(input: Tensor): Tensor {
  return ops.tanh(input);
}

export function softmax(input: Tensor, dim = -1): Tensor {
  return ops.softmax(input, dim);
}

export function log_softmax(input: Tensor, dim = -1): Tensor {
  return ops.log_softmax(input, dim);
}

export function leaky_relu(input: Tensor, negativeSlope = 0.01): Tensor {
  const scaled = ops.mul(input, full(input.shape, negativeSlope, { dtype: input.dtype, device: input.device }));
  return ops.maximum(input, scaled);
}

export function elu(input: Tensor, alpha = 1.0): Tensor {
  const zero = full(input.shape, 0, { dtype: input.dtype, device: input.device });
  const one = full(input.shape, 1, { dtype: input.dtype, device: input.device });
  const alphaT = full(input.shape, alpha, { dtype: input.dtype, device: input.device });
  const mask = ops.gt(input, zero);
  const negative = ops.mul(alphaT, ops.sub(ops.exp(input), one));
  return ops.where(mask, input, negative);
}
