import { empty, zeros, ones, full, randn } from './creation_ops.js';
import type { TensorOptions } from '../types/options.js';
import type { Tensor } from '../core/tensor.js';

function _likeOpts(tensor: Tensor, opts?: TensorOptions): TensorOptions {
  return {
    dtype: opts?.dtype ?? tensor.dtype,
    device: opts?.device ?? tensor.device,
    requiresGrad: opts?.requiresGrad ?? false,
  };
}

export function emptyLike(tensor: Tensor, opts?: TensorOptions): Tensor {
  return empty(tensor.shape, _likeOpts(tensor, opts));
}

export function zerosLike(tensor: Tensor, opts?: TensorOptions): Tensor {
  return zeros(tensor.shape, _likeOpts(tensor, opts));
}

export function onesLike(tensor: Tensor, opts?: TensorOptions): Tensor {
  return ones(tensor.shape, _likeOpts(tensor, opts));
}

export function fullLike(tensor: Tensor, value: number | bigint, opts?: TensorOptions): Tensor {
  return full(tensor.shape, value, _likeOpts(tensor, opts));
}

export function randnLike(tensor: Tensor, opts?: TensorOptions): Tensor {
  return randn(tensor.shape, _likeOpts(tensor, opts));
}
