import type { NumericTypedArray } from '../types/dtype.js';

export function addAt(data: NumericTypedArray, index: number, value: number | bigint): void {
  if (typeof value === 'bigint') {
    (data as BigInt64Array)[index] += value;
  } else {
    (data as Exclude<NumericTypedArray, BigInt64Array>)[index] += value;
  }
}
