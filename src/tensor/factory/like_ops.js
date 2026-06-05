import { empty, zeros, ones, full, randn } from './creation_ops.js';

function _likeOpts(tensor, opts) {
  return {
    dtype: opts?.dtype ?? tensor.dtype,
    device: opts?.device ?? tensor.device,
    requiresGrad: opts?.requiresGrad ?? false,
  };
}

export function emptyLike(tensor, opts) {
  return empty(tensor.shape, _likeOpts(tensor, opts));
}

export function zerosLike(tensor, opts) {
  return zeros(tensor.shape, _likeOpts(tensor, opts));
}

export function onesLike(tensor, opts) {
  return ones(tensor.shape, _likeOpts(tensor, opts));
}

export function fullLike(tensor, value, opts) {
  return full(tensor.shape, value, _likeOpts(tensor, opts));
}

export function randnLike(tensor, opts) {
  return randn(tensor.shape, _likeOpts(tensor, opts));
}
