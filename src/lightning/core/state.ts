import type { Device } from '../../tensor/types/device.js';
import type { MetricValue, NumericMetricRecord } from '../types.js';

export enum Stage {
  IDLE = 'idle',
  TRAINING = 'training',
  VALIDATING = 'validating',
  TESTING = 'testing',
  PREDICTING = 'predicting',
}

export type ReduceFnName = 'mean' | 'sum' | 'min' | 'max' | 'last';

type Accumulator = {
  sum: number;
  count: number;
  min: number;
  max: number;
  last: number;
};

const REDUCE_FNS: Record<ReduceFnName, (acc: Accumulator) => number> = {
  mean: acc => acc.count === 0 ? 0 : acc.sum / acc.count,
  sum: acc => acc.sum,
  min: acc => acc.min,
  max: acc => acc.max,
  last: acc => acc.last,
};

function createAccumulator(): Accumulator {
  return { sum: 0, count: 0, min: Infinity, max: -Infinity, last: 0 };
}

function updateAccumulator(acc: Accumulator, value: MetricValue): void {
  const v = typeof value === 'number' ? value : value.item();
  acc.sum += v as number;
  acc.count += 1;
  if (v < acc.min) acc.min = v as number;
  if (v > acc.max) acc.max = v as number;
  acc.last = v as number;
}

export class MetricAccumulator {
  private _accumulators: Map<string, Accumulator>;
  private _reduceFns: Map<string, ReduceFnName>;

  constructor() {
    this._accumulators = new Map();
    this._reduceFns = new Map();
  }

  update(name: string, value: MetricValue, reduceFx: ReduceFnName = 'mean'): void {
    if (!this._accumulators.has(name)) {
      this._accumulators.set(name, createAccumulator());
      this._reduceFns.set(name, reduceFx);
    }
    updateAccumulator(this._accumulators.get(name)!, value);
  }

  compute(name: string): number | undefined {
    const acc = this._accumulators.get(name);
    if (!acc) return undefined;
    const fn = REDUCE_FNS[this._reduceFns.get(name)!];
    return fn(acc);
  }

  computeAll(): NumericMetricRecord {
    const result: NumericMetricRecord = {};
    for (const [name] of this._accumulators) {
      result[name] = this.compute(name)!;
    }
    return result;
  }

  reset(): void {
    this._accumulators.clear();
    this._reduceFns.clear();
  }

  has(name: string): boolean {
    return this._accumulators.has(name);
  }

  get size(): number {
    return this._accumulators.size;
  }
}

export class TrainerState {
  stage: Stage;
  epoch: number;
  globalStep: number;
  maxEpochs: number;
  maxSteps: number;
  shouldStop: boolean;
  stepMetrics: MetricAccumulator;
  epochMetrics: MetricAccumulator;
  numTrainingBatches?: number;
  numValBatches?: number;
  _progBarMetrics?: Map<string, number>;

  constructor() {
    this.stage = Stage.IDLE;
    this.epoch = 0;
    this.globalStep = 0;
    this.maxEpochs = 0;
    this.maxSteps = -1;
    this.shouldStop = false;
    this.stepMetrics = new MetricAccumulator();
    this.epochMetrics = new MetricAccumulator();
  }

  resetEpochMetrics(): void {
    this.epochMetrics.reset();
  }

  resetStepMetrics(): void {
    this.stepMetrics.reset();
  }
}

export class SingleDeviceStrategy {
  device: Device | null;

  constructor() {
    this.device = null;
  }

  setup(model: unknown, device: Device): void {
    this.device = device;
    if ((device.type === 'gpu' || device.type === 'webgpu') && hasMethod(model, 'to')) model.to(device);
  }

  toDevice(value: unknown): unknown {
    if (!this.device || (this.device.type !== 'gpu' && this.device.type !== 'webgpu')) return value;
    if (hasDeviceTo(value)) return value.to(this.device);
    if (Array.isArray(value)) {
      const moved = new Array(value.length);
      for (let i = 0; i < value.length; i++) moved[i] = this.toDevice(value[i]);
      return moved;
    }
    return value;
  }

  backward(loss: { backward(): void }): void {
    loss.backward();
  }

  optimizerStep(optimizer: { step(): void }): void {
    optimizer.step();
  }
}

function hasMethod<T extends string>(value: unknown, method: T): value is Record<T, (...args: unknown[]) => unknown> {
  return typeof value === 'object' && value !== null && typeof (value as Record<T, unknown>)[method] === 'function';
}

function hasDeviceTo(value: unknown): value is { device: unknown; to(device: Device): unknown } {
  return typeof value === 'object'
    && value !== null
    && 'device' in value
    && typeof (value as { to?: unknown }).to === 'function';
}
