import * as ops from '../../tensor/ops/ops.js';

export function embedding(weight, indices) {
  const dim = weight.shape[weight.shape.length - 1];
  const flatLen = indices.shape.reduce((a, b) => a * b, 1);
  const selected = ops.index_select(weight, 0, indices.reshape([flatLen]));
  return selected.reshape([...indices.shape, dim]);
}
