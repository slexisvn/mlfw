import type { NumericTypedArray } from '../tensor/types/dtype.js';
import type { Tensor } from '../tensor/core/tensor.js';
import type { OptimizerState } from '../optim/types.js';
import type { Device } from '../tensor/types/device.js';
import type { CudaHandle, DevicePtr, EagerGraph } from '../runtime/io.js';
import type { Callback } from './callbacks/callback.js';
import type { CallbackConnector, LoggerConnector } from './core/hooks.js';
import type { SchedulerConfig } from './core/module.js';
import type { SingleDeviceStrategy, TrainerState } from './core/state.js';
import type { Logger } from './loggers/logger.js';

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
  [key: string]: unknown;
};
export type OptimizerLike = {
  _state?: Map<number, OptimizerState>;
  paramGroups: OptimizerGroupLike[];
  defaults?: Record<string, unknown>;
  step(): void;
  zeroGrad(): void;
  stateDict?(): unknown;
};
export type LightningModuleLike = {
  _cudaGraphPhase?: string;
  _currentOptimizers?: OptimizerLike[];
  _logBuffer: Map<string, unknown>;
  _trainer?: unknown;
  _device?: Device | null;
  __compiledTrainStep?: CompiledTrainStep;
  __eagerGraphRunner?: EagerGraphRunner;
  automaticOptimization: boolean;
  log(name: string, value: unknown, options?: UnknownRecord): void;
  stateDict?(): unknown;
  loadStateDict?(state: unknown): void;
  parameters?(): Iterable<Tensor>;
  train(): void;
  eval(): void;
  forward(...args: unknown[]): unknown;
  trainingStep(batch: unknown, batchIdx: number): unknown;
  validationStep(batch: unknown, batchIdx: number): unknown;
  testStep(batch: unknown, batchIdx: number): unknown;
  predictStep(batch: unknown, batchIdx: number): unknown;
  configureOptimizers(): unknown;
  onTrainEpochStart(): void;
  onTrainEpochEnd(): void;
  onValidationEpochStart(): void;
  onValidationEpochEnd(): void;
  onTestEpochStart(): void;
  onTestEpochEnd(): void;
};

export type DataLoaderLike = Iterable<unknown> & {
  length: number;
};

export type TrainerCoreLike = {
  state: TrainerState;
  strategy: SingleDeviceStrategy;
  callbackConnector: CallbackConnector;
  loggerConnector: LoggerConnector;
  fitLoop: {
    validationLoop: {
      run(model: LightningModuleLike, dataLoader: DataLoaderLike, trainer: TrainerCoreLike, schedulerConfigs: Array<SchedulerConfig | null> | null): Promise<NumericMetricRecord>;
    };
  };
  checkValEveryNEpoch: number;
  accumulateGradBatches: number;
  limitTrainBatches: number | null;
  limitValBatches: number | null;
  limitTestBatches: number | null;
  logEveryNSteps: number;
  gradientClipVal: number | null;
  gradientClipAlgorithm: string;
  compile: boolean;
  compileMode: string;
  cudaGraph: boolean;
  cudaGraphWarmupSteps: number;
  _flushEagerInference(): Promise<void>;
};

export type TrainerOptions = {
  maxEpochs?: number;
  maxSteps?: number;
  accelerator?: 'auto' | 'cpu' | 'gpu' | 'wasm' | 'webgpu';
  precision?: string;
  callbacks?: Callback[];
  logger?: boolean | Logger | Logger[] | null;
  enableCheckpointing?: boolean;
  enableProgress?: boolean;
  gradientClipVal?: number | null;
  gradientClipAlgorithm?: string;
  accumulateGradBatches?: number;
  limitTrainBatches?: number | null;
  limitValBatches?: number | null;
  limitTestBatches?: number | null;
  valCheckInterval?: number;
  checkValEveryNEpoch?: number;
  logEveryNSteps?: number;
  deterministic?: boolean;
  fastDevRun?: boolean | number;
  defaultRootDir?: string;
  compile?: boolean;
  compileMode?: string;
  cudaGraph?: boolean;
  cudaGraphWarmupSteps?: number;
};

export type CompiledTrainStep = {
  (...args: unknown[]): unknown;
  capturedParams(): Array<{ grad: unknown }>;
  backward(seed: unknown): unknown;
};

export type EagerGraphRunner = {
  phase: 'warmup' | 'disabled' | 'replay';
  seen: number;
  inputs?: Array<{ dptr: DevicePtr | null }>;
  captured?: EagerGraph;
  exec?: CudaHandle | null;
  lossDptr?: DevicePtr | null;
  lossScratch?: Float32Array;
  captureError?: unknown;
};

export type TensorLike = {
  readonly shape: readonly number[];
  readonly _impl: {
    readonly storage: {
      readonly data: NumericTypedArray;
    };
  };
};
