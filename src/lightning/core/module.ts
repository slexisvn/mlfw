import { Module } from '../../nn/module.js';
import type { Device } from '../../tensor/types/device.js';
import type { Logger } from '../loggers/logger.js';
import type { MetricValue, OptimizerLike } from '../types.js';
import type { ReduceFnName, Stage } from './state.js';

type LogOptions = {
  onStep?: boolean | null;
  onEpoch?: boolean | null;
  reduceFx?: ReduceFnName;
  progBar?: boolean;
};

type LogEntry = {
  value: MetricValue;
  onStep: boolean;
  onEpoch: boolean;
  reduceFx: ReduceFnName;
  progBar: boolean;
};

type TrainerRef = {
  state: {
    epoch: number;
    globalStep: number;
    stage: Stage | string;
  };
  logger: Logger | null;
  loggers: Logger[];
  strategy?: {
    backward(loss: { backward(): void }): void;
  };
};

type SchedulerLike = {
  step(...args: unknown[]): void;
};

export type SchedulerConfig = {
  scheduler: SchedulerLike;
  interval: 'step' | 'epoch';
  frequency: number;
  monitor: string | null;
};

type OptimizerConfig = {
  optimizer?: OptimizerLike;
  lrScheduler?: SchedulerConfigInput | SchedulerLike | null;
};

type SchedulerConfigInput = {
  scheduler?: SchedulerLike;
  interval?: 'step' | 'epoch';
  frequency?: number;
  monitor?: string | null;
};

export type ParsedOptimizersConfig = {
  optimizers: OptimizerLike[];
  schedulerConfigs: Array<SchedulerConfig | null>;
};

export class LightningModule extends Module {
  _trainer: TrainerRef | null;
  _logBuffer: Map<string, LogEntry>;
  _automaticOptimization: boolean;
  _currentOptimizers: OptimizerLike[];
  _device: Device | null;

  constructor() {
    super();
    this._trainer = null;
    this._logBuffer = new Map();
    this._automaticOptimization = true;
    this._currentOptimizers = [];
    this._device = null;
  }

  get trainer(): TrainerRef | null {
    return this._trainer;
  }

  get currentEpoch(): number {
    return this._trainer ? this._trainer.state.epoch : 0;
  }

  get globalStep(): number {
    return this._trainer ? this._trainer.state.globalStep : 0;
  }

  get device(): Device | null {
    return this._device;
  }

  get logger(): Logger | null {
    return this._trainer ? this._trainer.logger : null;
  }

  get loggers(): Logger[] {
    return this._trainer ? this._trainer.loggers : [];
  }

  get automaticOptimization(): boolean {
    return this._automaticOptimization;
  }

  set automaticOptimization(value: boolean) {
    this._automaticOptimization = value;
  }

  get optimizers(): OptimizerLike[] {
    return this._currentOptimizers;
  }

  trainingStep(_batch: unknown, _batchIdx: number): unknown {
    throw new Error(`${this.constructor.name}.trainingStep() not implemented`);
  }

  validationStep(_batch: unknown, _batchIdx: number): unknown { return undefined; }

  testStep(_batch: unknown, _batchIdx: number): unknown { return undefined; }

  predictStep(batch: unknown, _batchIdx: number): unknown {
    return this.forward(Array.isArray(batch) && batch.length === 1 ? batch[0] : batch);
  }

  configureOptimizers(): unknown {
    throw new Error(`${this.constructor.name}.configureOptimizers() not implemented`);
  }

  onTrainEpochStart(): void {}
  onTrainEpochEnd(): void {}
  onValidationEpochStart(): void {}
  onValidationEpochEnd(): void {}
  onTestEpochStart(): void {}
  onTestEpochEnd(): void {}

  log(name: string, value: MetricValue, {
    onStep = null,
    onEpoch = null,
    reduceFx = 'mean',
    progBar = false,
  }: LogOptions = {}): void {
    const stage = this._trainer ? this._trainer.state.stage : 'training';
    const stepDefault = stage === 'training';
    const epochDefault = stage !== 'training';
    this._logBuffer.set(name, {
      value,
      onStep: onStep !== null ? onStep : stepDefault,
      onEpoch: onEpoch !== null ? onEpoch : epochDefault,
      reduceFx,
      progBar,
    });
  }

  logDict(dict: Record<string, MetricValue>, opts: LogOptions = {}): void {
    for (const key of Object.keys(dict)) {
      this.log(key, dict[key], opts);
    }
  }

  manualBackward(loss: { backward(): void }): void {
    if (this._trainer && this._trainer.strategy) {
      this._trainer.strategy.backward(loss);
    } else {
      loss.backward();
    }
  }
}

export function parseOptimizersConfig(result: unknown): ParsedOptimizersConfig {
  if (!result) throw new Error('configureOptimizers() returned null/undefined');

  if (Array.isArray(result)) {
    const optimizers: OptimizerLike[] = [];
    const schedulerConfigs: Array<SchedulerConfig | null> = [];
    for (let i = 0; i < result.length; i++) {
      const item = result[i];
      if (isOptimizerConfig(item)) {
        optimizers.push(item.optimizer);
        schedulerConfigs.push(normalizeSchedulerConfig(item.lrScheduler));
      } else {
        optimizers.push(item as OptimizerLike);
        schedulerConfigs.push(null);
      }
    }
    return { optimizers, schedulerConfigs };
  }

  const maybeConfig = result as OptimizerConfig;
  if (maybeConfig.optimizer) {
    return {
      optimizers: [maybeConfig.optimizer],
      schedulerConfigs: [normalizeSchedulerConfig(maybeConfig.lrScheduler)],
    };
  }

  if (isOptimizerLike(result)) {
    return { optimizers: [result], schedulerConfigs: [null] };
  }

  throw new Error('configureOptimizers() returned an unrecognized format');
}

function normalizeSchedulerConfig(config: SchedulerConfigInput | SchedulerLike | null | undefined): SchedulerConfig | null {
  if (!config) return null;
  if ('scheduler' in config && config.scheduler) {
    return {
      scheduler: config.scheduler,
      interval: config.interval || 'epoch',
      frequency: config.frequency || 1,
      monitor: config.monitor || null,
    };
  }
  if (isSchedulerLike(config)) {
    return {
      scheduler: config,
      interval: 'epoch',
      frequency: 1,
      monitor: null,
    };
  }
  return null;
}

function isSchedulerLike(value: unknown): value is SchedulerLike {
  return typeof value === 'object' && value !== null && typeof (value as { step?: unknown }).step === 'function';
}

function isOptimizerLike(value: unknown): value is OptimizerLike {
  return isSchedulerLike(value);
}

function isOptimizerConfig(value: unknown): value is OptimizerConfig & { optimizer: OptimizerLike } {
  return typeof value === 'object' && value !== null && 'optimizer' in value;
}
