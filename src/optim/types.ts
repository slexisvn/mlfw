import type { Tensor } from '../tensor/core/tensor.js';
import type { NumericTypedArray } from '../tensor/types/dtype.js';

export type OptimizerParam = Tensor;
export type OptimizerStateValue =
  | number
  | boolean
  | string
  | readonly number[]
  | NumericTypedArray
  | undefined;
export type OptimizerState = Record<string, OptimizerStateValue>;

export type OptimizerParamGroup = {
  params: OptimizerParam[];
  [key: string]: OptimizerStateValue | OptimizerParam[];
};

export type ParamGroupInput = {
  params: Iterable<OptimizerParam> | OptimizerParam[];
  [key: string]: OptimizerStateValue | Iterable<OptimizerParam> | OptimizerParam[];
};

export type OptimizerParams = Iterable<OptimizerParam> | OptimizerParam[] | ParamGroupInput[];

export type OptimizerStateDict = {
  state: Map<number, OptimizerState>;
  paramGroups: Array<Record<string, OptimizerStateValue>>;
};

export type NumberTypedArray = Exclude<NumericTypedArray, BigInt64Array>;
export type NumberTypedArrayConstructor = {
  new(length: number): NumberTypedArray;
};
