import type { Tensor } from '../tensor/core/tensor.js';
import type { Device } from '../tensor/types/device.js';
import type { DType } from '../tensor/types/dtype.js';
export type { TensorDataOptions as TensorOptions } from '../tensor/types/options.js';

export type NumericVectorInput = Tensor | number | ArrayLike<number>;
export type NumericMatrixInput = Tensor | ReadonlyArray<ArrayLike<number>>;
export type NumericElementInput = Tensor | number;
export type NumericArrayInput = Tensor | ArrayLike<number>;
export type NumericShape = number | readonly number[];
export type ScalarFn = (x: number) => number;
export type VectorFn = (x: number[]) => number;
export type GradientFn = (x: number[]) => number[];
export type ResidualFn = (x: number[]) => number[];
export type JacobianFn = (x: number[], m: number) => number[][];
export type Bounds = ReadonlyArray<readonly [number, number] | null | undefined>;

export type HostVector = {
  data: Float64Array;
  dtype: DType;
  device: Device;
};

export type HostGrid = HostVector & {
  rows: number;
  cols: number;
};

export type OptimizeResult = {
  x: number[];
  fun: number;
  iterations: number;
  converged: boolean;
};

export type TestResult = {
  statistic: number;
  pvalue: number;
  df?: number;
};
