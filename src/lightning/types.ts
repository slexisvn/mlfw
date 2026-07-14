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
export type TrainerLike = {
  state: {
    epoch: number;
    globalStep: number;
    maxEpochs?: number;
    numTrainingBatches?: number;
    numValBatches?: number;
    _progBarMetrics?: Map<string, unknown>;
    epochMetrics?: {
      computeAll(): Record<string, number | undefined>;
    };
  };
  accumulateGradBatches?: number;
  limitTrainBatches?: number | null;
  limitValBatches?: number | null;
  shouldStop?: boolean;
  callbackConnector?: {
    dispatch(hook: string, ...args: unknown[]): void;
  };
};
export type OptimizerGroupLike = {
  lr: number;
  momentum?: number;
};
export type OptimizerLike = {
  paramGroups: OptimizerGroupLike[];
};
export type LightningModuleLike = {
  _currentOptimizers?: OptimizerLike[];
  log(name: string, value: unknown, options?: UnknownRecord): void;
  stateDict?(): unknown;
  loadStateDict?(state: unknown): void;
};

export type TensorLike = {
  readonly shape: readonly number[];
  readonly _impl: {
    readonly storage: {
      readonly data: NumericTypedArray;
    };
  };
};
