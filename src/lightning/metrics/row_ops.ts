import type { NumericTypedArray } from '../../tensor/types/dtype.js';

export function argmaxRow(data: NumericTypedArray, row: number, cols: number): number {
  let best = 0;
  let bestVal = data[row * cols];
  for (let j = 1; j < cols; j++) {
    const v = data[row * cols + j];
    if (v > bestVal) { bestVal = v; best = j; }
  }
  return best;
}
