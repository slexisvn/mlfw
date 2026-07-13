import type { Device } from './device.js';
import type { DType } from './dtype.js';

export type TensorOptions = {
  shape?: readonly number[];
  dtype?: DType;
  device?: Device;
  requiresGrad?: boolean;
  offset?: number;
};

export type TensorDataOptions = Pick<TensorOptions, 'dtype' | 'device'>;

export type MutableNumericArray = {
  length: number;
  fill(value: number | bigint): unknown;
  [index: number]: number | bigint;
};

export type NumericSettable = {
  set(array: ArrayLike<number | bigint>, offset?: number): void;
};
