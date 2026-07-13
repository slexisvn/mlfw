import { _dispatch } from '../tensor/ops/ops.js';
import type { Tensor } from '../tensor/core/tensor.js';

export const fft = (x: Tensor): Tensor => _dispatch('fft', x) as Tensor;
export const ifft = (x: Tensor): Tensor => _dispatch('ifft', x) as Tensor;
