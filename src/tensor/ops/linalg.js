import { _dispatch } from './ops.js';

export function svd(a) {
  const [U, S, V] = _dispatch('svd', a);
  return { U, S, V };
}

export function eigh(a) {
  const [values, vectors] = _dispatch('eigh', a);
  return { values, vectors };
}

export const cholesky = (a) => _dispatch('cholesky', a);
export const inv = (a) => _dispatch('inv', a);
export const pinv = (a) => _dispatch('pinv', a);
export const cov = (a) => _dispatch('cov', a);
export const solve = (a, b) => _dispatch('solve', a, b);
export const lstsq = (a, b) => _dispatch('lstsq', a, b);
export const det = (a) => _dispatch('det', a).item();
