import type { Tensor } from './tensor.js';

export function isTensor(value: unknown): value is Tensor {
  return typeof value === 'object' && value !== null && '_impl' in value;
}
