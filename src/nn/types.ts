import type { Tensor } from '../tensor/core/tensor.js';

export type NNTensor = {
  isSymbolic?: boolean;
  reshape(shape: readonly number[]): NNTensor;
  transpose(dim0: number, dim1: number): NNTensor;
  permute(dims: readonly number[]): NNTensor;
  unsqueeze(dim: number): NNTensor;
  detach(): NNTensor;
} & Tensor;

export type OptionalTensor = NNTensor | null;
export type TensorPair = [NNTensor, NNTensor];
