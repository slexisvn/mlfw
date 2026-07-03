import { _dispatch } from '../tensor/ops/ops.js';

export const fft = (x) => _dispatch('fft', x);
export const ifft = (x) => _dispatch('ifft', x);
