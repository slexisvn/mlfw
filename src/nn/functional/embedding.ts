import * as ops from '../../tensor/ops/ops.js';
import type { Tensor } from '../../tensor/core/tensor.js';

type ReshapableTensor = Tensor & { reshape: (shape: readonly number[]) => Tensor };

export function embedding(weight: Tensor, indices: Tensor): Tensor {
  const dim = weight.shape[weight.shape.length - 1];
  const flatLen = indices.shape.reduce((a, b) => a * b, 1);
  const selected = ops.index_select(weight, 0, (indices as ReshapableTensor).reshape([flatLen]));
  return (selected as ReshapableTensor).reshape([...indices.shape, dim]);
}
