import type { NumericTypedArray } from '../tensor/types/dtype.js';

export type MetricResult = number | number[] | number[][];
export type MetricValue = number | { item(): number | bigint };
export type UnknownRecord = Record<string, unknown>;
export type NumericMetricRecord = Record<string, number>;
export type HyperparameterRecord = UnknownRecord;
export type LoggerOptions = {
  name?: string;
  version?: number | null;
};
export type ClassificationTask = 'binary' | 'multiclass';
export type AccuracyTask = ClassificationTask | 'multilabel';
export type AverageMode = 'macro' | 'micro' | 'weighted' | 'none';

export type TensorLike = {
  readonly shape: readonly number[];
  readonly _impl: {
    readonly storage: {
      readonly data: NumericTypedArray;
    };
  };
};
