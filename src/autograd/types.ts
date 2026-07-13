import type { DType } from '../tensor/types/dtype.js';
import type { Tensor } from '../tensor/core/tensor.js';

export type InputMetadata = {
  shape: readonly number[];
  dtype: DType;
};

export type OpArgs = readonly unknown[] | null;
export type GradOutputList = readonly Tensor[];
export type GradInput = Tensor | null;
export type GradInputList = GradInput[];
export type BackwardApply = (gradOutputs: GradOutputList) => GradInputList | null;
