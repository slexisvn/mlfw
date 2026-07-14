import type { Tensor } from '../tensor/core/tensor.js';
import type { DType } from '../tensor/types/dtype.js';
import type { Device } from '../tensor/types/device.js';

export type MLTensor = Tensor & {
  reshape(shape: readonly number[]): MLTensor;
  transpose(dim0: number, dim1: number): MLTensor;
  narrow(dim: number, start: number, length: number): MLTensor;
  toArray(): ArrayLike<number | bigint>;
};

export type HostMatrix = { data: Float64Array; rows: number; cols: number };
export type HostVector = { data: Float64Array; n: number };
export type Rng = () => number;
export type FitPredictEstimator = {
  fit(X: MLTensor, y: MLTensor): FitPredictEstimator;
  predict(X: MLTensor): MLTensor;
  score?(X: MLTensor, y: MLTensor): number;
};
export type TensorOptionsLike = { dtype: DType; device: Device };

