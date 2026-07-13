import { _dispatch } from './ops.js';
import type { Tensor } from '../core/tensor.js';

export function svd(a: Tensor): { U: Tensor; S: Tensor; V: Tensor } {
  const [U, S, V] = _dispatch('svd', a) as [Tensor, Tensor, Tensor];
  return { U, S, V };
}

export function eigh(a: Tensor): { values: Tensor; vectors: Tensor } {
  const [values, vectors] = _dispatch('eigh', a) as [Tensor, Tensor];
  return { values, vectors };
}

export function qr(a: Tensor): { Q: Tensor; R: Tensor } {
  const [Q, R] = _dispatch('qr', a) as [Tensor, Tensor];
  return { Q, R };
}

export const cholesky = (a: Tensor): Tensor => _dispatch('cholesky', a) as Tensor;
export const inv = (a: Tensor): Tensor => _dispatch('inv', a) as Tensor;
export const pinv = (a: Tensor): Tensor => _dispatch('pinv', a) as Tensor;
export const cov = (a: Tensor): Tensor => _dispatch('cov', a) as Tensor;
export const solve = (a: Tensor, b: Tensor): Tensor => _dispatch('solve', a, b) as Tensor;
export const lstsq = (a: Tensor, b: Tensor): Tensor => _dispatch('lstsq', a, b) as Tensor;
export const det = (a: Tensor): number | bigint => (_dispatch('det', a) as Tensor).item();
