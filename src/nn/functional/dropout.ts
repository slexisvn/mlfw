import * as ops from '../../tensor/ops/ops.js';
import { full, empty } from '../../tensor/factory/creation_ops.js';
import type { Tensor } from '../../tensor/core/tensor.js';
import type { DType } from '../../tensor/types/dtype.js';
import type { Device } from '../../tensor/types/device.js';

export function dropout(input: Tensor, p = 0.5, training = true): Tensor {
  if (!training || p === 0) return input;
  if (p === 1) return full(input.shape, 0, { dtype: input.dtype, device: input.device });

  const mask = _bernoulliMask(input.shape, 1 - p, input.dtype, input.device);
  const scale = full(input.shape, 1 / (1 - p), { dtype: input.dtype, device: input.device });
  return ops.mul(ops.mul(input, mask), scale);
}

function _bernoulliMask(shape: readonly number[], prob: number, dtype: DType, device: Device): Tensor {
  const t = empty(shape, { dtype, device });
  const data = t._impl.storage.data;
  if (data) {
    for (let i = 0; i < data.length; i++) {
      data[i] = Math.random() < prob ? 1 : 0;
    }
  }
  return t;
}
